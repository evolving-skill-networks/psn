"""
Parameter Extraction Module (Facade)

Extracts parameter metadata from code, including types, defaults, and
semantic inference.

v5.0 architecture reorganization — migrated from graph_manager_impl.py.
v5.1 supports the new ParameterSemantic format (direction + transform_hint).
v5.2 split into facade + 2 mixins (_param_extraction, _param_semantic);
     constants moved to _param_constants.py to avoid circular imports.
"""

import logging
from typing import Any, Dict, List, Optional

# Re-export constants for backward compatibility (external callers import from here)
from ._param_constants import (  # noqa: F401
    QUANTITY_PARAM_NAMES,
    TARGET_TOTAL_PATTERNS,
    DELTA_FUNC_PREFIXES,
    ENSURE_PREFIXES,
    INPUT_DIRECTION_INDICATORS,
    OUTPUT_DIRECTION_INDICATORS,
    CONFIG_DIRECTION_INDICATORS,
    FUEL_PARAM_NAMES,
    SEMANTIC_INFERENCE_PROMPT,
    determine_delta_prefix,
)

from ._param_extraction import ParameterExtractionMixin
from ._param_semantic import ParameterSemanticMixin

logger = logging.getLogger(__name__)


# ============================================================================
# Plan v3-rev Fix 2B: Object Schema Format Helpers
# ============================================================================

def _is_planner_format(schema: Any) -> bool:
    """Whether `schema` is in the format the planner's
    `_build_object_parameter_value` (parameter_resolution.py:615-655) expects.

    Planner-format:    {fieldName: {source_hint?, type?, default?, ...}, ...}
                       (top-level keys are field names; values are dicts)
    Babel-format:      {destructured: bool, fields: [{name, type, default, ...}], ...}
                       (top-level keys are 'destructured' / 'fields' literal)
    Empty/None/non-dict: not planner-format
    """
    if not isinstance(schema, dict) or not schema:
        return False
    # Babel's structured-extraction format is its own shape, not planner-format
    if "destructured" in schema or "fields" in schema:
        return False
    # Heuristic: planner-format has dict-valued entries (field metadata).
    # A bare {fieldName: "string"} would not qualify, which is fine — those
    # need re-enrichment to add source_hint.
    return any(isinstance(v, dict) for v in schema.values())


# Deterministic Babel→planner translation table. Maps a normalized field
# name (lowercase, no underscores) to a (effect_field, transform) pair that
# the planner's _build_object_parameter_value can use to extract values from
# target_effects. Conservative set of well-known patterns; novel field names
# fall through to LLM enrichment when strategy=babel_then_llm.
DETERMINISTIC_HINTS = {
    "targettotal":   ("count", None),
    "targetcount":   ("count", None),
    "count":         ("count", None),
    "amount":        ("count", None),
    "quantity":      ("count", None),
    "n":             ("count", None),
    "num":           ("count", None),
    "item":          ("item", None),
    "itemname":      ("item", None),
    "resourcetype":  ("item", None),
    "resource":      ("item", None),
    "blockname":     ("item", None),
    "block":         ("item", None),
    "target":        ("item", None),
    "targetitem":    ("item", None),
    "type":          ("item", None),
}


def _normalize_field_name(name: str) -> str:
    """Lowercase, strip underscores/hyphens. Used for DETERMINISTIC_HINTS lookup."""
    return name.lower().replace("_", "").replace("-", "") if name else ""


# ============================================================================
# ParameterExtractor Class
# ============================================================================

class ParameterExtractor(ParameterExtractionMixin, ParameterSemanticMixin):
    """
    Parameter extractor (facade).

    Responsibilities:
    1. Extract parameter metadata from code (Babel or LLM) — ParameterExtractionMixin.
    2. Infer parameter semantics (target_total vs delta) — ParameterSemanticMixin.
    3. Extract inner schema for object-typed parameters — ParameterExtractionMixin.
    4. Validate consistency between function name and parameter semantics — ParameterSemanticMixin.
    """

    def __init__(
        self,
        llm,  # LangChain LLM
        extraction_mode: str = "babel_then_llm",
        semantic_inference_enabled: bool = True,
        semantic_llm_fallback: bool = True,
        semantic_default: str = "delta",
        quantity_param_names: Optional[List[str]] = None,
        custom_logger: Optional[logging.Logger] = None,
        entry_parameter_name: str = "bot",
        object_schema_strategy: str = "llm_always",
    ):
        """
        Initialize the parameter extractor.

        Args:
            llm: LangChain LLM instance
            extraction_mode: extraction mode, "babel_then_llm" or "llm_only"
            semantic_inference_enabled: whether to enable parameter-semantic inference
            semantic_llm_fallback: whether to enable LLM-assisted inference (disabled = rules only)
            semantic_default: default when inference fails: "delta" or "target_total"
            quantity_param_names: list of quantity parameter names (customizable)
            custom_logger: optional custom logger
            entry_parameter_name: name of the implicit first parameter (e.g. 'bot' for Minecraft)
            object_schema_strategy: how to enrich object-parameter schemas
                (Plan v3-rev Fix 2B). One of:
                - "llm_always" (default): always call LLM, overwriting Babel
                - "babel_then_llm": deterministic Babel->planner first, LLM fallback
                - "babel_only": no LLM enrichment (test/debug only)
        """
        self.llm = llm
        self.extraction_mode = extraction_mode
        self.semantic_inference_enabled = semantic_inference_enabled
        self.semantic_llm_fallback = semantic_llm_fallback
        self.semantic_default = semantic_default
        self.quantity_param_names = quantity_param_names or QUANTITY_PARAM_NAMES
        self.logger = custom_logger or logger
        self.entry_parameter_name = entry_parameter_name
        self.object_schema_strategy = object_schema_strategy

    def extract(
        self,
        code: str,
        description: Optional[str] = None,
        skill_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract parameter info (main method; chooses extraction approach per config).

        Args:
            code: skill code
            description: skill description (optional)
            skill_name: skill name (optional, used to match the main function)

        Returns:
            Parameter metadata dict, format:
            {
                "paramName": {
                    "type": "number|string|boolean|array",
                    "default": default_value,
                    "description": "parameter description",
                    "semantic": "target_total|delta|config"
                }
            }
        """
        if self.extraction_mode == "llm_only":
            self.logger.info(
                f"[Parameter Extraction] Using LLM-only mode for {skill_name or 'unknown'}"
            )
            # Mods 1.4 & 1.5: add a layered fallback chain
            parameters = self.extract_llm(code, description, skill_name)
            if parameters is None:
                self.logger.warning("Full LLM extraction failed, trying simplified")
                parameters = self.extract_llm_simplified(code, skill_name)
            if parameters is None:
                self.logger.warning("Simplified extraction failed, trying regex emergency")
                parameters = self.extract_regex_emergency(code, skill_name)
            return parameters if parameters is not None else {}
        else:
            self.logger.info(
                f"[Parameter Extraction] Using Babel-then-LLM mode for {skill_name or 'unknown'}"
            )
            parameters = self.extract_babel(code, description, skill_name)
            if parameters is not None:
                # {} is valid (0 user-facing params, e.g. bot-only function)
                if parameters:
                    self.logger.info(
                        "[Parameter Extraction] Successfully extracted parameters using Babel"
                    )
                else:
                    self.logger.info(
                        "[Parameter Extraction] Babel: 0 user-facing parameters (bot-only function)"
                    )

                # Plan v3-rev Fix 2B: check whether any object parameters need schema enrichment.
                # The old gate `not info.get("schema")` skipped enrichment when Babel produced
                # a non-planner schema format `{destructured, fields:[...]}`, which made the
                # schema unusable by the planner's consumer `_build_object_parameter_value`
                # (it expects {fieldName: {...}}). The new gate detects non-planner formats
                # and re-enriches via the model-aware object_schema_strategy.
                object_params = [
                    name for name, info in parameters.items()
                    if info.get("type") == "object"
                    and not _is_planner_format(info.get("schema"))
                ]
                if object_params:
                    self.logger.info(
                        f"[Parameter Extraction] Found object params needing planner-format schema "
                        f"(strategy={self.object_schema_strategy}): {object_params}"
                    )
                    schemas = self._enrich_object_schemas(code, parameters, object_params)
                    for param_name, schema in schemas.items():
                        if param_name in parameters and _is_planner_format(schema):
                            parameters[param_name]["schema"] = schema
                            self.logger.info(
                                f"[Parameter Extraction] Added schema for '{param_name}': "
                                f"{list(schema.keys())}"
                            )

                return parameters
            else:
                self.logger.warning(
                    "[Parameter Extraction] Babel extraction error, falling back to LLM"
                )
                # Mods 1.4 & 1.5: add a layered fallback chain
                parameters = self.extract_llm(code, description, skill_name)
                if parameters is None:
                    self.logger.warning("Full LLM extraction failed, trying simplified")
                    parameters = self.extract_llm_simplified(code, skill_name)
                if parameters is None:
                    self.logger.warning("Simplified extraction failed, trying regex emergency")
                    parameters = self.extract_regex_emergency(code, skill_name)
                return parameters if parameters is not None else {}

    # ========================================================================
    # Plan v3-rev Fix 2B: Object Schema Enrichment (model-aware)
    # ========================================================================

    def _enrich_object_schemas(
        self,
        code: str,
        parameters: Dict[str, Dict[str, Any]],
        object_params: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Dispatch object-schema enrichment based on object_schema_strategy.

        Returns a dict mapping param_name to a planner-format schema
        (`{fieldName: {source_hint?, type?, default?, ...}, ...}`).

        Strategies:
          - "llm_always":     call extract_object_schemas_llm for all params;
                              fall back to deterministic Babel translation for
                              params the LLM returns empty for (it strictly
                              looks for destructuring; property-access patterns
                              like `options.targetTotal` may yield empty).
          - "babel_then_llm": deterministic translation first; LLM fallback
                              for params whose translation yields nothing
          - "babel_only":     deterministic only; no LLM call
        """
        strategy = self.object_schema_strategy

        if strategy == "llm_always":
            llm_schemas = self.extract_object_schemas_llm(code, object_params)
            # Normalize: gpt-5-mini sometimes returns the Babel-shaped
            # {fields:[{name,type,default,source_hint},...]} instead of the
            # planner-format {fieldName: {source_hint?, type?, ...}}. Convert
            # any such fields-list output into planner-format using the same
            # deterministic translator + DETERMINISTIC_HINTS lookup.
            llm_schemas = {
                p: self._normalize_llm_schema_to_planner_format(p, schema)
                for p, schema in llm_schemas.items()
            }
            # Backstop: for params the LLM returned empty (e.g. when it didn't
            # detect any property-access patterns at all), fall back to
            # deterministic translation from Babel's fields list.
            unresolved = [
                p for p in object_params
                if not _is_planner_format(llm_schemas.get(p))
            ]
            if unresolved:
                det_schemas = self._translate_babel_schemas_deterministic(
                    parameters, unresolved
                )
                # det_schemas wins for unresolved; LLM wins for the rest
                return {**llm_schemas, **det_schemas}
            return llm_schemas

        if strategy in ("babel_then_llm", "babel_only"):
            det_schemas = self._translate_babel_schemas_deterministic(
                parameters, object_params
            )
            unresolved = [
                p for p in object_params
                if not _is_planner_format(det_schemas.get(p))
            ]
            if strategy == "babel_only" or not unresolved:
                return det_schemas
            # babel_then_llm: LLM fills in the unresolved
            llm_schemas = self.extract_object_schemas_llm(code, unresolved)
            return {**det_schemas, **llm_schemas}

        # Unknown strategy — default to LLM
        self.logger.warning(
            f"[Parameter Extraction] Unknown object_schema_strategy "
            f"'{strategy}', defaulting to llm_always"
        )
        return self.extract_object_schemas_llm(code, object_params)

    def _normalize_llm_schema_to_planner_format(
        self,
        param_name: str,
        schema: Any,
    ) -> Dict[str, Any]:
        """If `schema` is already planner-format, pass through. Otherwise
        normalize from one of the LLM-quirk formats:
          - `{fields: [{name, type, default, source_hint}, ...]}` (list)
          - `{fields: {fieldName: {type, default, source_hint, ...}}}` (dict)
          - `{type: 'object', required: bool, fields: ...}` (wrapper around fields)
          - `{fieldName: {type, default, source_hint: "JS snippet string"}}` —
             top-level planner-shaped but with non-dict source_hint

        Returns {} for empty / unrecognized schemas.
        """
        if not isinstance(schema, dict) or not schema:
            return {}

        # First pass: extract a working "fields_iter" — list of (name, field_info_dict).
        fields_iter = None

        if "fields" in schema:
            raw_fields = schema["fields"]
            if isinstance(raw_fields, list):
                fields_iter = [
                    (f.get("name"), f) for f in raw_fields
                    if isinstance(f, dict) and f.get("name")
                ]
            elif isinstance(raw_fields, dict):
                fields_iter = [
                    (k, v) for k, v in raw_fields.items()
                    if isinstance(v, dict)
                ]

        # If no fields key, the schema might itself be a planner-format-ish
        # mapping {fieldName: {...}} but with non-dict source_hint values.
        # Distinguish from {type, required, default} wrapper keys.
        if fields_iter is None:
            wrapper_keys = {"type", "required", "default", "description"}
            non_wrapper_entries = [
                (k, v) for k, v in schema.items()
                if k not in wrapper_keys and isinstance(v, dict)
            ]
            if non_wrapper_entries:
                fields_iter = non_wrapper_entries

        if not fields_iter:
            return {}

        planner_schema = {}
        for fname, field_info in fields_iter:
            if not fname or not isinstance(field_info, dict):
                continue
            entry = {
                "type": field_info.get("type", "unknown"),
                "default": field_info.get("default"),
            }
            src_hint = field_info.get("source_hint")
            if isinstance(src_hint, dict) and "effect_field" in src_hint:
                entry["source_hint"] = {
                    "effect_field": src_hint.get("effect_field"),
                    "transform": src_hint.get("transform"),
                }
            else:
                # Source hint is missing / string / non-conforming.
                # Look up via DETERMINISTIC_HINTS using normalized field name.
                hint = DETERMINISTIC_HINTS.get(_normalize_field_name(fname))
                if hint:
                    effect_field, transform = hint
                    entry["source_hint"] = {
                        "effect_field": effect_field,
                        "transform": transform,
                    }
            planner_schema[fname] = entry

        # Only return as planner-format if at least one field has source_hint
        if any(
            isinstance(v, dict) and "source_hint" in v
            for v in planner_schema.values()
        ):
            return planner_schema
        return {}

    def _translate_babel_schemas_deterministic(
        self,
        parameters: Dict[str, Dict[str, Any]],
        object_params: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Translate Babel-format `{destructured, fields:[...]}` schemas
        into planner-format `{fieldName: {source_hint?, type?, default?}, ...}`
        using DETERMINISTIC_HINTS for source_hint discovery.

        Returns one entry per object_param. Params whose Babel schema has
        no recognizable field names get an empty dict (caller treats as
        unresolved for LLM fallback).
        """
        result = {}
        for pname in object_params:
            babel_schema = (parameters.get(pname, {}) or {}).get("schema") or {}
            fields = babel_schema.get("fields") or []
            planner_schema = {}
            for field in fields:
                if not isinstance(field, dict):
                    continue
                field_name = field.get("name")
                if not field_name:
                    continue
                normalized = _normalize_field_name(field_name)
                hint = DETERMINISTIC_HINTS.get(normalized)
                entry = {
                    "type": field.get("type", "unknown"),
                    "default": field.get("default"),
                }
                if hint:
                    effect_field, transform = hint
                    entry["source_hint"] = {
                        "effect_field": effect_field,
                        "transform": transform,
                    }
                planner_schema[field_name] = entry
            # Only return planner-format if at least one field has source_hint;
            # otherwise the schema is useless to the planner (it would fall
            # through to LLM anyway). Returning empty dict marks unresolved.
            if any(
                isinstance(v, dict) and "source_hint" in v
                for v in planner_schema.values()
            ):
                result[pname] = planner_schema
            else:
                result[pname] = {}
        return result
