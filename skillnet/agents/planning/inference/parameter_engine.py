"""
Parameter Inference Engine

This module provides the unified parameter inference engine for the Graph Planner.
It uses a strategy chain to infer parameter values with fallback mechanisms.

Strategy chain depends on mode:

**LLM-first mode** (default, llm_first=True):
1. Learned rules - Previously successful inference patterns
2. PureLLM - Rich-context LLM inference using all InferenceContext fields
3. State mapping - Structural extraction from effects (fallback)
4. Semantic/Effect/Context - Additional structural strategies (fallback)
5. Default value - Fall back to parameter default

**Structural mode** (llm_first=False, or no LLM available):
1. Learned rules - Previously successful inference patterns
2. State mapping - Structural extraction from effects
3. Semantic inference - Based on ParameterSemantic metadata
4. Effect extraction - Direct extraction from target_effects
5. Context inference - From precondition_context
6. LLM inference - AI-assisted inference (minimal prompt)
7. Default value - Fall back to parameter default
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .parameter_semantic import (
    QuantitySemantic,
    DirectionSemantic,
    TransformHint,
    ParameterSemantic,
)
from .config_validator import (
    CONFIG_INDICATORS,
    item_matches_config_context,
    is_config_parameter,
    is_input_material_param,
    get_fuel_items_from_inventory,
    extract_base_type,
    map_output_to_input_material,
)
logger = logging.getLogger("skillnet.planner.inference.parameter_engine")

# ── Domain knowledge injection ──

from skillnet.core.dk_registry import get_domain_knowledge
from skillnet.utils.stats_tracker import record_llm_usage


def _get_transform_fn(name):
    """Get a named transform function from domain knowledge.

    Returns the callable or None if no domain knowledge is available
    or the named transform is not provided.
    """
    dk = get_domain_knowledge()
    if dk:
        transforms = dk.get_item_transform_functions()
        return transforms.get(name)
    return None


# ============================================================================
# Enums and Data Classes
# ============================================================================

class InferenceStrategy(Enum):
    """Strategy used for parameter inference."""
    LEARNED_RULES = "learned_rules"
    SPECIALIZED_WHEN = "specialized_when"  # Layer 1.5: refactor-derived param binding
    # Plan v3-rev Fix 2C: schema-based object resolution for object params.
    # Slots BEFORE LLM in the chain so a planner-format schema with source_hint
    # entries wins over PureLLMResolver (which returns raw effect field names).
    SCHEMA_OBJECT_RESOLUTION = "schema_object_resolution"
    STATE_MAPPING = "state_mapping"
    SEMANTIC = "semantic"
    EFFECT_EXTRACTION = "effect_extraction"
    CONTEXT = "context"
    LLM = "llm"
    DEFAULT = "default"
    NONE = "none"


@dataclass
class InferenceContext:
    """
    Context for parameter inference.

    Contains all information needed to infer a parameter value.
    """
    # Required fields
    skill_name: str
    param_name: str
    param_info: Dict[str, Any]  # Parameter metadata from skill node
    target_effects: List[Dict[str, Any]]

    # Optional fields
    current_inventory: Dict[str, int] = field(default_factory=dict)
    current_task: Optional[str] = None
    precondition_context: Optional[Dict[str, Any]] = None
    semantic: Optional[ParameterSemantic] = None

    # P0 fields: Call chain support
    upstream_output: Optional[str] = None  # Output from upstream skill
    caller_skill: Optional[str] = None  # Skill that called this one
    output_to_input_mapping: Optional[Dict[str, str]] = None

    # Expected inventory (what will be produced by dependencies)
    expected_inventory: Optional[Dict[str, int]] = None

    # Environment context (from current_state)
    nearby_blocks: Optional[List[str]] = None    # Block names within 8-block radius
    biome: Optional[str] = None                  # Current biome name
    position: Optional[Dict[str, float]] = None  # x, y, z coordinates
    equipment: Optional[Dict[str, str]] = None   # Equipped items by slot

    # Optimizer parameter corrections (task-scoped, from Phase 1 analysis)
    parameter_corrections: Optional[List[Dict[str, Any]]] = None

    # Layer 1.5: candidate skill's expected_effects (carries specialized_when
    # annotations propagated by sibling refactor). When present, the
    # SPECIALIZED_WHEN strategy uses these to recover param bindings the
    # refactor knew about (e.g. wooden_axe → toolType='axe').
    skill_expected_effects: List[Any] = field(default_factory=list)

    def get_param_type(self) -> str:
        """Get parameter type from param_info."""
        return self.param_info.get("type", "string")

    def get_default_value(self) -> Any:
        """Get default value from param_info."""
        return self.param_info.get("default")

    def get_supported_values(self) -> Optional[List[str]]:
        """Get supported values from param_info."""
        return self.param_info.get("supported_values")

    def is_flexible(self) -> bool:
        """Check if this is a flexible requirement (any accepted item works)."""
        if self.precondition_context:
            return self.precondition_context.get("is_flexible", False)
        return False


@dataclass
class InferenceResult:
    """
    Result of parameter inference.

    Contains the inferred value and metadata about how it was obtained.
    """
    value: str  # JS literal format (e.g., '"oak_log"', '8', '["item1"]')
    strategy_used: InferenceStrategy
    confidence: float  # 0.0 to 1.0
    explanation: str

    # Optional metadata
    transform_applied: Optional[str] = None  # Transform that was applied
    raw_value: Any = None  # Python value before JS conversion
    validation_passed: bool = True  # Whether value passed validation

    def is_successful(self) -> bool:
        """Check if inference was successful."""
        return (
            self.strategy_used != InferenceStrategy.NONE
            and self.confidence > 0.0
            and self.value is not None
        )


# ============================================================================
# Parameter Value Helper
# ============================================================================

@dataclass
class ParameterValue:
    """
    Type-safe parameter value representation.

    Ensures values are correctly formatted for JavaScript consumption.
    """
    raw_value: Any  # Python value
    js_literal: str  # JavaScript literal string
    value_type: str  # "string" | "number" | "boolean" | "array" | "object" | "null"

    @classmethod
    def from_python(cls, value: Any, param_type: str) -> "ParameterValue":
        """
        Create ParameterValue from Python value.

        Args:
            value: Python value
            param_type: Expected parameter type

        Returns:
            ParameterValue with correct JS formatting
        """
        if value is None:
            return cls(raw_value=None, js_literal="null", value_type="null")

        if param_type == "number":
            try:
                num_value = float(value) if isinstance(value, str) else value
                # Use int if it's a whole number
                if isinstance(num_value, float) and num_value.is_integer():
                    num_value = int(num_value)
                return cls(
                    raw_value=num_value,
                    js_literal=str(num_value),
                    value_type="number"
                )
            except (ValueError, TypeError):
                return cls(raw_value=value, js_literal="null", value_type="null")

        elif param_type == "boolean":
            bool_value = bool(value) if not isinstance(value, bool) else value
            return cls(
                raw_value=bool_value,
                js_literal="true" if bool_value else "false",
                value_type="boolean"
            )

        elif param_type == "array":
            if isinstance(value, str):
                # Try to parse as JSON array
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return cls(
                            raw_value=parsed,
                            js_literal=json.dumps(parsed),
                            value_type="array"
                        )
                except json.JSONDecodeError:
                    # Treat as single-item array
                    return cls(
                        raw_value=[value],
                        js_literal=json.dumps([value]),
                        value_type="array"
                    )
            elif isinstance(value, list):
                return cls(
                    raw_value=value,
                    js_literal=json.dumps(value),
                    value_type="array"
                )
            else:
                return cls(
                    raw_value=[value],
                    js_literal=json.dumps([value]),
                    value_type="array"
                )

        elif param_type == "object":
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    return cls(
                        raw_value=parsed,
                        js_literal=json.dumps(parsed),
                        value_type="object"
                    )
                except json.JSONDecodeError:
                    return cls(raw_value=value, js_literal="null", value_type="null")
            elif isinstance(value, dict):
                return cls(
                    raw_value=value,
                    js_literal=json.dumps(value),
                    value_type="object"
                )
            else:
                return cls(raw_value=value, js_literal="null", value_type="null")

        else:  # string or unknown
            str_value = str(value) if not isinstance(value, str) else value
            # Escape quotes in the string
            escaped = str_value.replace('\\', '\\\\').replace('"', '\\"')
            return cls(
                raw_value=str_value,
                js_literal=f'"{escaped}"',
                value_type="string"
            )

    @classmethod
    def undefined(cls) -> "ParameterValue":
        """Create an undefined value (let JS use default)."""
        return cls(raw_value=None, js_literal="undefined", value_type="null")


# ============================================================================
# Parameter Inference Engine
# ============================================================================

class ParameterInferenceEngine:
    """
    Unified parameter inference engine.

    Uses a strategy chain to infer parameter values with fallback mechanisms.
    """

    def __init__(
        self,
        skill_graph_manager=None,
        llm=None,
        ckpt_dir: str = "ckpt",
        custom_logger: Optional[logging.Logger] = None,
        llm_first: bool = True,
    ):
        """
        Initialize the inference engine.

        Args:
            skill_graph_manager: SkillGraphManager instance (for learned rules)
            llm: LLM instance for AI-assisted inference
            ckpt_dir: Checkpoint directory for rule persistence
            custom_logger: Custom logger
            llm_first: If True (default), use PureLLM as primary strategy
                       before structural strategies. If False, use the legacy
                       chain where LLM is a late fallback.
        """
        self.skill_graph_manager = skill_graph_manager
        self.llm = llm
        self.ckpt_dir = ckpt_dir
        self.logger = custom_logger or logger
        self._llm_first = llm_first

        # Learned rules storage
        self.rules_file = os.path.join(ckpt_dir, "planner", "parameter_rules.json")
        self.learned_rules: Dict[str, Any] = self._load_rules()

        # Failure tracking for learning
        self._failures: List[Dict[str, Any]] = []

        # PureLLM resolver for rich-context inference
        self._pure_llm_resolver = None
        if llm and llm_first:
            from .pure_llm_resolver import PureLLMParameterResolver
            self._pure_llm_resolver = PureLLMParameterResolver(llm, self.logger)

    # ========================================================================
    # Main Interface
    # ========================================================================

    def infer_parameter_value(self, context: InferenceContext) -> InferenceResult:
        """
        Infer a parameter value using the strategy chain.

        When llm_first=True and LLM is available:
        1. Learned rules → 2. PureLLM → 3-6. Structural fallback → 7. Default

        When llm_first=False or no LLM:
        1. Learned rules → 2-5. Structural → 6. LLM (minimal) → 7. Default

        Args:
            context: Inference context with all required information

        Returns:
            InferenceResult with the inferred value and metadata
        """
        self.logger.debug(
            f"[Inference] Starting inference for {context.skill_name}.{context.param_name}"
        )

        # Build strategy chain based on mode
        strategies = [
            (InferenceStrategy.LEARNED_RULES, self._try_learned_rules),
            # Layer 1.5: refactor-derived param bindings from specialized_when
            # annotations on propagated effects. High-confidence direct lookup,
            # short-circuits LLM call for merge_siblings-refactored skills.
            (InferenceStrategy.SPECIALIZED_WHEN, self._infer_from_specialized_when),
            # Plan v3-rev Fix 2C: object params with planner-format schema get
            # resolved deterministically from source_hint mappings. Beats LLM
            # for cases like ensureResource(bot, type, {targetTotal:...}) where
            # the LLM would return raw effect field names (item/count) instead
            # of schema-mapped names (targetTotal). Returns None for non-object
            # params or schemas lacking source_hint, falling through cleanly.
            (InferenceStrategy.SCHEMA_OBJECT_RESOLUTION,
             self._infer_from_schema_object_resolution),
        ]

        if self._llm_first and self._pure_llm_resolver:
            # LLM-first mode: PureLLM as primary strategy
            strategies.append((InferenceStrategy.LLM, self._infer_with_pure_llm))

        # Structural strategies (always present as fallback or primary)
        strategies.extend([
            (InferenceStrategy.STATE_MAPPING, self._infer_from_state_mapping),
            (InferenceStrategy.SEMANTIC, self._infer_from_semantic),
            (InferenceStrategy.EFFECT_EXTRACTION, self._extract_from_effects),
            (InferenceStrategy.CONTEXT, self._infer_from_context),
        ])

        if not (self._llm_first and self._pure_llm_resolver):
            # Structural mode: old LLM as late fallback
            strategies.append((InferenceStrategy.LLM, self._infer_with_llm))

        strategies.append((InferenceStrategy.DEFAULT, self._use_default_value))

        # Try each strategy
        for strategy_type, strategy_func in strategies:
            try:
                result = strategy_func(context)
                if result and result.is_successful():
                    # Phase 3.5: type-adapter validation
                    result = self._apply_type_validation(context, result)

                    self.logger.info(
                        f"[Inference] {context.skill_name}.{context.param_name}: "
                        f"'{result.value}' via {strategy_type.value} "
                        f"(confidence: {result.confidence:.2f})"
                    )
                    return result
            except Exception as e:
                self.logger.warning(
                    f"[Inference] Strategy {strategy_type.value} failed: {e}"
                )
                self._record_failure(strategy_type.value, context, str(e))

        # All strategies failed
        self.logger.warning(
            f"[Inference] All strategies failed for {context.skill_name}.{context.param_name}"
        )
        return InferenceResult(
            value="undefined",
            strategy_used=InferenceStrategy.NONE,
            confidence=0.0,
            explanation="All inference strategies failed",
        )

    # ========================================================================
    # Strategy Implementations
    # ========================================================================

    def _try_learned_rules(self, context: InferenceContext) -> Optional[InferenceResult]:
        """
        Try to apply learned rules for this parameter.

        Args:
            context: Inference context

        Returns:
            InferenceResult if rule matches, None otherwise
        """
        rule_key = f"{context.skill_name}.{context.param_name}"

        if rule_key not in self.learned_rules:
            return None

        rule = self.learned_rules[rule_key]

        # Extract value using rule pattern
        if "effect_field" in rule:
            field = rule["effect_field"]
            for effect in context.target_effects:
                if field in effect:
                    raw_value = effect[field]

                    # Apply transform if specified
                    if "transform" in rule:
                        transformed = self._apply_rule_transform(raw_value, rule["transform"])
                        if transformed:
                            raw_value = transformed

                    # Validate if semantic is config
                    if context.semantic and context.semantic.direction == DirectionSemantic.CONFIG:
                        if not item_matches_config_context(context.param_name, str(raw_value)):
                            return None

                    pv = ParameterValue.from_python(raw_value, context.get_param_type())
                    return InferenceResult(
                        value=pv.js_literal,
                        strategy_used=InferenceStrategy.LEARNED_RULES,
                        confidence=0.9,
                        explanation=f"Applied learned rule: {rule_key}",
                        raw_value=pv.raw_value,
                    )

        return None

    def _infer_from_specialized_when(
        self, context: InferenceContext
    ) -> Optional[InferenceResult]:
        """
        Layer 1.5 strategy: extract param value from a propagated effect's
        `specialized_when` annotation (set by Layer 1 sibling refactor).

        For each target_effect (from task), find the matching skill effect
        by item identity. If that effect carries
        ``specialized_when[param_name]``, return it as the inferred value.

        Restricted to ``is_primary=True`` effects to avoid wrong bindings on
        shared secondary effects (e.g. oak_planks produced by both craftAxe
        and craftWoodenPickaxe wrappers - only one survives Layer 1's dedup,
        carrying its source's specialized_when which would mis-bind for the
        other sibling's task).

        Returns None when:
          - no skill_expected_effects available (caller didn't pass them)
          - no target effect matches by item
          - matched effect is not primary
          - matched effect lacks specialized_when[param_name]
        Falls through to remaining strategies in those cases.
        """
        if not context.skill_expected_effects:
            return None
        for target_eff in context.target_effects:
            target_item = target_eff.get("item")
            if not target_item:
                continue
            for sk_eff in context.skill_expected_effects:
                if not getattr(sk_eff, 'is_primary', False):
                    continue
                sr = getattr(sk_eff, 'state_representation', None) or {}
                if sr.get("item") != target_item:
                    continue
                specialized_when = getattr(sk_eff, 'specialized_when', None)
                if specialized_when and context.param_name in specialized_when:
                    value = specialized_when[context.param_name]
                    pv = ParameterValue.from_python(value, context.get_param_type())
                    return InferenceResult(
                        value=pv.js_literal,
                        strategy_used=InferenceStrategy.SPECIALIZED_WHEN,
                        confidence=0.95,
                        explanation=(
                            f"Layer 1.5: specialized_when[{context.param_name}]"
                            f"={value!r} from {context.skill_name}'s primary effect "
                            f"for {target_item}"
                        ),
                        raw_value=pv.raw_value,
                    )
        return None

    def _infer_from_schema_object_resolution(
        self, context: InferenceContext
    ) -> Optional[InferenceResult]:
        """Plan v3-rev Fix 2C: build an object literal from a planner-format
        schema by mapping each field's source_hint to context.target_effects.

        Planner-format schema example (the format `_build_object_parameter_value`
        in parameter_resolution.py:615-655 expects):
            {
              "targetTotal": {"type": "number",
                              "source_hint": {"effect_field": "count", "transform": null}},
              "maxAttempts": {...},
            }

        Returns None when:
          - param is not type=object
          - schema is missing, empty, or in Babel-format (has "destructured"/"fields" keys)
          - no field's source_hint resolves to a value in target_effects
        """
        param_info = context.param_info or {}
        if param_info.get("type") != "object":
            return None
        schema = param_info.get("schema") or {}
        if not schema:
            return None
        # Reject Babel-format schemas - Fix 2B should convert them upstream.
        if isinstance(schema, dict) and (
            "destructured" in schema or "fields" in schema
        ):
            return None
        if not context.target_effects:
            return None

        obj_fields = {}
        for field_name, field_info in schema.items():
            if not isinstance(field_info, dict):
                continue
            source_hint = field_info.get("source_hint") or {}
            effect_field = source_hint.get("effect_field")
            if not effect_field:
                continue
            for eff in context.target_effects:
                if not isinstance(eff, dict):
                    continue
                raw_value = eff.get(effect_field)
                if raw_value is None:
                    continue
                # JS literal formatting - strings get quoted, numbers/bools raw.
                # Transforms are not applied here (parameter_resolution.py's
                # _build_object_parameter_value has a transform pipeline; if
                # transforms become common we should pull that into shared
                # utility, but for the {count→targetTotal} case no transform
                # is needed).
                if isinstance(raw_value, str):
                    obj_fields[field_name] = f'"{raw_value}"'
                elif isinstance(raw_value, bool):
                    obj_fields[field_name] = "true" if raw_value else "false"
                else:
                    obj_fields[field_name] = str(raw_value)
                break

        if not obj_fields:
            return None

        obj_literal = "{ " + ", ".join(
            f"{k}: {v}" for k, v in obj_fields.items()
        ) + " }"
        return InferenceResult(
            value=obj_literal,
            strategy_used=InferenceStrategy.SCHEMA_OBJECT_RESOLUTION,
            confidence=0.95,
            explanation=(
                f"Fix 2C: built {context.param_name}={obj_literal} from schema "
                f"source_hints ({len(obj_fields)} fields resolved)"
            ),
            raw_value=obj_fields,
        )

    def _infer_from_state_mapping(self, context: InferenceContext) -> Optional[InferenceResult]:
        """Resolve parameter value using explicit state-space mapping.

        Uses StateMapping to extract values from target_effects and optionally
        apply transforms (e.g., infer_input to map output→input material).

        Returns None when:
        - No semantic or no state_mapping present
        - source is "config" (let downstream strategies handle config params)
        - No matching field found in target_effects
        """
        semantic = context.semantic
        if not semantic or not semantic.state_mapping:
            return None

        mapping = semantic.state_mapping

        # Config params are not derived from effects/preconditions
        if mapping.source == "config":
            return None

        # Extract raw value from target_effects
        raw_value = None
        for effect in context.target_effects:
            if mapping.field in effect:
                raw_value = effect[mapping.field]
                break

        if raw_value is None:
            return None

        # Apply transform if specified
        if mapping.transform:
            transformed = self._apply_rule_transform(raw_value, mapping.transform)
            if transformed:
                # Resolve material prefix to specific item if possible
                final_value = self._resolve_material_value(context, transformed, raw_value)
                pv = ParameterValue.from_python(final_value, context.get_param_type())
                return InferenceResult(
                    value=pv.js_literal,
                    strategy_used=InferenceStrategy.STATE_MAPPING,
                    confidence=0.85,
                    explanation=(
                        f"State mapping: {mapping.source}.{mapping.field} "
                        f"+ transform '{mapping.transform}': '{raw_value}' → '{final_value}'"
                    ),
                    raw_value=pv.raw_value,
                    transform_applied=mapping.transform,
                )
            # Transform failed - fall through to use raw value

        pv = ParameterValue.from_python(raw_value, context.get_param_type())
        return InferenceResult(
            value=pv.js_literal,
            strategy_used=InferenceStrategy.STATE_MAPPING,
            confidence=0.9,
            explanation=f"State mapping: direct from {mapping.source}.{mapping.field}",
            raw_value=pv.raw_value,
        )

    def _resolve_material_value(
        self,
        context: InferenceContext,
        material_prefix: str,
        original_product: str,
    ) -> str:
        """Resolve a material prefix (e.g., 'wooden', 'iron') to a specific item.

        Resolution order:
        1. supported_values from parameter metadata
        2. expected_inventory from PlanningState
        3. precondition_context required_item
        4. default value from parameter metadata
        5. Return prefix as-is (still better than raw product)
        """
        # 1. Check supported_values (e.g., ["oak_planks", "birch_planks"])
        supported = context.get_supported_values()
        if supported:
            for val in supported:
                if material_prefix in val:
                    return val

        # 2. Check expected_inventory
        if context.expected_inventory:
            for item_name in context.expected_inventory:
                if material_prefix in item_name:
                    return item_name

        # 3. Check precondition_context
        if context.precondition_context:
            required = context.precondition_context.get("required_item")
            if required and material_prefix in str(required):
                return required

        # 4. Check default value
        default = context.get_default_value()
        if default and isinstance(default, str) and material_prefix in default:
            return default

        # 5. Return prefix as-is
        return material_prefix

    def _infer_from_semantic(self, context: InferenceContext) -> Optional[InferenceResult]:
        """
        Infer value based on parameter semantic metadata.

        Args:
            context: Inference context

        Returns:
            InferenceResult if semantic inference succeeds, None otherwise
        """
        semantic = context.semantic
        if not semantic:
            # Try to infer semantic from param name and skill name
            semantic = self._infer_semantic(context)

        if not semantic:
            return None

        # Handle fuel parameters specially
        if semantic.is_fuel_param:
            return self._infer_fuel_value(context)

        # Handle quantity parameters
        if semantic.is_quantity_param():
            return self._infer_quantity_value(context, semantic)

        # Handle type parameters based on direction
        if semantic.direction == DirectionSemantic.INPUT:
            return self._infer_input_material(context, semantic)
        elif semantic.direction == DirectionSemantic.OUTPUT:
            return self._infer_output_product(context, semantic)

        return None

    def _extract_from_effects(self, context: InferenceContext) -> Optional[InferenceResult]:
        """
        Extract value directly from target_effects.

        Args:
            context: Inference context

        Returns:
            InferenceResult if extraction succeeds, None otherwise
        """
        param_type = context.get_param_type()
        param_name = context.param_name.lower()

        for effect in context.target_effects:
            # Try to match by field name
            for field, value in effect.items():
                field_lower = field.lower()

                # Direct name match
                if field_lower == param_name or field_lower.replace("_", "") == param_name.replace("_", ""):
                    pv = ParameterValue.from_python(value, param_type)
                    return InferenceResult(
                        value=pv.js_literal,
                        strategy_used=InferenceStrategy.EFFECT_EXTRACTION,
                        confidence=0.85,
                        explanation=f"Direct extraction from effect field '{field}'",
                        raw_value=pv.raw_value,
                    )

            # Try semantic matching for count/quantity
            if param_type == "number" and "count" in effect:
                if any(q in param_name for q in ["count", "num", "quantity", "amount", "total"]):
                    pv = ParameterValue.from_python(effect["count"], "number")
                    return InferenceResult(
                        value=pv.js_literal,
                        strategy_used=InferenceStrategy.EFFECT_EXTRACTION,
                        confidence=0.8,
                        explanation="Extracted count from effect",
                        raw_value=pv.raw_value,
                    )

            # Try semantic matching for item type
            if param_type == "string" and "item" in effect:
                if any(t in param_name for t in ["type", "item", "name"]):
                    item_value = effect["item"]
                    transformed = None  # Initialize to avoid referencing an unassigned variable

                    # Check if we need to transform (input vs output)
                    should_transform = is_input_material_param(param_name)
                    # Defense-in-depth: semantic direction also triggers transform
                    if not should_transform and context.semantic and context.semantic.direction == DirectionSemantic.INPUT:
                        should_transform = True
                    if should_transform:
                        _infer_fn = _get_transform_fn("infer_input_from_output")
                        transformed = _infer_fn(item_value) if _infer_fn else None
                        if transformed:
                            item_value = transformed

                    pv = ParameterValue.from_python(item_value, "string")
                    return InferenceResult(
                        value=pv.js_literal,
                        strategy_used=InferenceStrategy.EFFECT_EXTRACTION,
                        confidence=0.75,
                        explanation="Extracted item from effect",
                        raw_value=pv.raw_value,
                        transform_applied="input_from_output" if transformed else None,
                    )

        return None

    def _infer_from_context(self, context: InferenceContext) -> Optional[InferenceResult]:
        """
        Infer value from precondition_context.

        Args:
            context: Inference context

        Returns:
            InferenceResult if context inference succeeds, None otherwise
        """
        if not context.precondition_context:
            return None

        pc = context.precondition_context
        param_type = context.get_param_type()
        param_name = context.param_name.lower()

        # Handle flexible requirements
        if context.is_flexible():
            # For flexible requirements, use default value if flexible_compatible
            semantic = context.semantic
            if semantic and semantic.flexible_compatible:
                default = context.get_default_value()
                if default is not None:
                    pv = ParameterValue.from_python(default, param_type)
                    return InferenceResult(
                        value=pv.js_literal,
                        strategy_used=InferenceStrategy.CONTEXT,
                        confidence=0.7,
                        explanation="Using default for flexible requirement",
                        raw_value=pv.raw_value,
                    )

        # Try required_item
        if "required_item" in pc:
            required_item = pc["required_item"]

            # For type parameters
            if any(t in param_name for t in ["type", "item", "name"]):
                # Check if we need to transform
                if is_input_material_param(param_name):
                    _infer_fn = _get_transform_fn("infer_input_from_output")
                    transformed = _infer_fn(required_item) if _infer_fn else None
                    if transformed:
                        required_item = transformed

                pv = ParameterValue.from_python(required_item, "string")
                return InferenceResult(
                    value=pv.js_literal,
                    strategy_used=InferenceStrategy.CONTEXT,
                    confidence=0.75,
                    explanation="Extracted from precondition_context.required_item",
                    raw_value=pv.raw_value,
                )

        # Try required_count
        if "required_count" in pc:
            if param_type == "number" and any(q in param_name for q in ["count", "num", "total", "amount"]):
                pv = ParameterValue.from_python(pc["required_count"], "number")
                return InferenceResult(
                    value=pv.js_literal,
                    strategy_used=InferenceStrategy.CONTEXT,
                    confidence=0.75,
                    explanation="Extracted from precondition_context.required_count",
                    raw_value=pv.raw_value,
                )

        return None

    def _infer_with_pure_llm(self, context: InferenceContext) -> Optional[InferenceResult]:
        """Use rich-context LLM for parameter inference (primary strategy).

        Delegates to PureLLMParameterResolver which uses ALL InferenceContext
        fields for maximum context (14 fields vs the old 6-field prompt).
        """
        if not self._pure_llm_resolver:
            return None
        return self._pure_llm_resolver.resolve(context)

    def _infer_with_llm(self, context: InferenceContext) -> Optional[InferenceResult]:
        """
        Use LLM for complex parameter inference (legacy, minimal prompt).

        Args:
            context: Inference context

        Returns:
            InferenceResult if LLM inference succeeds, None otherwise
        """
        if not self.llm:
            return None

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            # Build prompt
            system_prompt = """You are a parameter inference expert for Minecraft automation.
Given a skill name, parameter info, and target effects, infer the correct parameter value.

RULES:
1. For count parameters: Extract from effect's count field
2. For item type parameters (logType, oreType, preferredLogs, etc.):
   - Return the SPECIFIC Minecraft item name from the effect's 'item' field
   - Examples: "oak_log", "iron_ore", "cobblestone"
   - If item is generic (e.g., "log", "ore"), return null to use default
   - NEVER return abstract concepts like "INPUT", "OUTPUT", "material"
3. For fuel parameters: Return a valid fuel item like "coal", "charcoal", "oak_planks"
4. Return ONLY the value in JSON format: {"value": <inferred_value>}
"""

            user_prompt = f"""Skill: {context.skill_name}
Parameter: {context.param_name}
Parameter type: {context.get_param_type()}
Parameter default: {context.get_default_value()}
Target effects: {json.dumps(context.target_effects, indent=2)}
Current task: {context.current_task or "Unknown"}

What value should this parameter have?"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="planning_param_inference", function_name="parameter_engine._infer_with_llm", task=context.current_task, skill_name=context.skill_name)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            value = self._parse_llm_response(response_text, context.get_param_type())

            # Validation: reject placeholder values
            PLACEHOLDER_VALUES = {"INPUT", "OUTPUT", "material", "item", "type", "value"}
            if isinstance(value, str) and value.upper() in {v.upper() for v in PLACEHOLDER_VALUES}:
                self.logger.warning(
                    f"[Inference] LLM returned placeholder '{value}', falling back to default"
                )
                return None  # Trigger fallback to default value

            if value is not None:
                pv = ParameterValue.from_python(value, context.get_param_type())
                return InferenceResult(
                    value=pv.js_literal,
                    strategy_used=InferenceStrategy.LLM,
                    confidence=0.6,
                    explanation="LLM inference",
                    raw_value=pv.raw_value,
                )

        except Exception as e:
            self.logger.warning(f"[Inference] LLM inference failed: {e}")

        return None

    def _use_default_value(self, context: InferenceContext) -> Optional[InferenceResult]:
        """
        Fall back to parameter default value.

        Args:
            context: Inference context

        Returns:
            InferenceResult with default value, or undefined
        """
        default = context.get_default_value()

        if default is not None:
            pv = ParameterValue.from_python(default, context.get_param_type())
            return InferenceResult(
                value=pv.js_literal,
                strategy_used=InferenceStrategy.DEFAULT,
                confidence=0.5,
                explanation="Using parameter default value",
                raw_value=pv.raw_value,
            )

        # type="nullable" means JS had `param = null` - respect the developer's intent
        # null is an intentional default, not an absence of default
        if context.get_param_type() == "nullable":
            return InferenceResult(
                value="null",
                strategy_used=InferenceStrategy.DEFAULT,
                confidence=0.65,
                explanation="Parameter has explicit null default (nullable type)",
                raw_value=None,
            )

        # No default value at all - return undefined to let JS handle it
        return InferenceResult(
            value="undefined",
            strategy_used=InferenceStrategy.DEFAULT,
            confidence=0.3,
            explanation="No default value, returning undefined",
        )

    # ========================================================================
    # Specialized Inference Methods
    # ========================================================================

    def _infer_fuel_value(self, context: InferenceContext) -> Optional[InferenceResult]:
        """
        Infer value for fuel parameters.

        Prioritizes available fuel in inventory.

        Args:
            context: Inference context

        Returns:
            InferenceResult with fuel value
        """
        # Check inventory for available fuels
        fuel_items = get_fuel_items_from_inventory(context.current_inventory)

        if fuel_items:
            # Use highest priority available fuel
            fuel = fuel_items[0]
            pv = ParameterValue.from_python(fuel, "string")
            return InferenceResult(
                value=pv.js_literal,
                strategy_used=InferenceStrategy.SEMANTIC,
                confidence=0.85,
                explanation=f"Selected fuel '{fuel}' from inventory",
                raw_value=pv.raw_value,
            )

        # Check expected_inventory for upcoming fuels
        if context.expected_inventory:
            fuel_items = get_fuel_items_from_inventory(context.expected_inventory)
            if fuel_items:
                fuel = fuel_items[0]
                pv = ParameterValue.from_python(fuel, "string")
                return InferenceResult(
                    value=pv.js_literal,
                    strategy_used=InferenceStrategy.SEMANTIC,
                    confidence=0.75,
                    explanation=f"Selected expected fuel '{fuel}'",
                    raw_value=pv.raw_value,
                )

        # Fall back to default
        default = context.get_default_value()
        if default:
            pv = ParameterValue.from_python(default, "string")
            return InferenceResult(
                value=pv.js_literal,
                strategy_used=InferenceStrategy.SEMANTIC,
                confidence=0.6,
                explanation="Using default fuel value",
                raw_value=pv.raw_value,
            )

        return None

    def _infer_quantity_value(
        self,
        context: InferenceContext,
        semantic: ParameterSemantic
    ) -> Optional[InferenceResult]:
        """
        Infer value for quantity parameters.

        Args:
            context: Inference context
            semantic: Parameter semantic

        Returns:
            InferenceResult with quantity value

        v7.8 fix: prefer extracting count from precondition_context to avoid prerequisite
        skills (e.g. craftStonePickaxe) using the task target's count instead of the
        actually required quantity.
        """
        # Check precondition_context first (highest priority)
        # Example: task "Ensure 16 cobblestone" with precondition "Has 1 stone_pickaxe"
        # should use required_count=1, not the task target's 16
        if context.precondition_context and "required_count" in context.precondition_context:
            count = context.precondition_context["required_count"]
            pv = ParameterValue.from_python(count, "number")
            return InferenceResult(
                value=pv.js_literal,
                strategy_used=InferenceStrategy.CONTEXT,
                confidence=0.95,
                explanation="Extracted count from precondition_context (highest priority for prerequisite skills)",
                raw_value=pv.raw_value,
            )

        # Try to extract from effects (for directly executed tasks, not preconditions)
        for effect in context.target_effects:
            if "count" in effect:
                count = effect["count"]

                # Adjust based on semantic
                if semantic.quantity_semantic == QuantitySemantic.TARGET_TOTAL:
                    # Use count directly as target total
                    pv = ParameterValue.from_python(count, "number")
                    return InferenceResult(
                        value=pv.js_literal,
                        strategy_used=InferenceStrategy.SEMANTIC,
                        confidence=0.85,
                        explanation="Extracted count as target_total",
                        raw_value=pv.raw_value,
                    )
                elif semantic.quantity_semantic == QuantitySemantic.DELTA:
                    # For delta, we might need to adjust
                    # But typically the effect count is already the delta
                    pv = ParameterValue.from_python(count, "number")
                    return InferenceResult(
                        value=pv.js_literal,
                        strategy_used=InferenceStrategy.SEMANTIC,
                        confidence=0.85,
                        explanation="Extracted count as delta",
                        raw_value=pv.raw_value,
                    )

        return None

    def _infer_input_material(
        self,
        context: InferenceContext,
        semantic: ParameterSemantic
    ) -> Optional[InferenceResult]:
        """
        Infer value for input material parameters.

        Transforms output products to input materials when needed.

        Args:
            context: Inference context
            semantic: Parameter semantic

        Returns:
            InferenceResult with input material value
        """
        # Try to get from effects
        for effect in context.target_effects:
            if "item" in effect:
                output_item = effect["item"]

                # Transform to input material
                _infer_fn = _get_transform_fn("infer_input_from_output")
                input_item = _infer_fn(output_item) if _infer_fn else None
                if input_item:
                    # Validate against supported_values if available
                    supported = context.get_supported_values()
                    if supported and input_item not in supported:
                        # Try variants
                        for variant in [input_item, f"stripped_{input_item}", input_item.replace("stripped_", "")]:
                            if variant in supported:
                                input_item = variant
                                break

                    pv = ParameterValue.from_python(input_item, "string")
                    return InferenceResult(
                        value=pv.js_literal,
                        strategy_used=InferenceStrategy.SEMANTIC,
                        confidence=0.8,
                        explanation=f"Transformed '{output_item}' to input '{input_item}'",
                        raw_value=pv.raw_value,
                        transform_applied="infer_input_from_output",
                    )

        return None

    def _infer_output_product(
        self,
        context: InferenceContext,
        semantic: ParameterSemantic
    ) -> Optional[InferenceResult]:
        """
        Infer value for output product parameters.

        Extracts output products directly from effects.

        Args:
            context: Inference context
            semantic: Parameter semantic

        Returns:
            InferenceResult with output product value
        """
        # Number params can't be output product types (item names are strings)
        if context.get_param_type() == "number":
            return None

        # For output parameters, extract directly
        for effect in context.target_effects:
            if "item" in effect:
                output_item = effect["item"]
                pv = ParameterValue.from_python(output_item, "string")
                return InferenceResult(
                    value=pv.js_literal,
                    strategy_used=InferenceStrategy.SEMANTIC,
                    confidence=0.85,
                    explanation=f"Extracted output product '{output_item}'",
                    raw_value=pv.raw_value,
                )

        return None

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _infer_semantic(self, context: InferenceContext) -> Optional[ParameterSemantic]:
        """
        Infer parameter semantic from name and skill context.

        Args:
            context: Inference context

        Returns:
            Inferred ParameterSemantic or None
        """
        param_name = context.param_name.lower()
        skill_name = context.skill_name.lower()
        param_type = context.get_param_type()

        # Infer quantity semantic
        quantity_semantic = QuantitySemantic.NONE
        if param_type == "number":
            if any(q in param_name for q in ["count", "num", "total", "amount", "quantity"]):
                # Check skill name prefix
                if skill_name.startswith("ensure"):
                    quantity_semantic = QuantitySemantic.TARGET_TOTAL
                elif skill_name.startswith("craft") or skill_name.startswith("mine"):
                    quantity_semantic = QuantitySemantic.DELTA
                else:
                    quantity_semantic = QuantitySemantic.TARGET_TOTAL  # Default

        # Infer direction semantic
        direction = DirectionSemantic.CONFIG

        # Number parameters are never INPUT/OUTPUT direction (those are item type semantics).
        # Keep CONFIG default and let quantity_semantic (already set above) handle the semantics.
        if param_type != "number":
            # Check for input indicators
            if any(p in param_name for p in ["input", "source", "log_type", "ore_type", "material", "raw"]):
                direction = DirectionSemantic.INPUT
            # Check for output indicators
            elif any(p in param_name for p in ["output", "result", "product"]):
                direction = DirectionSemantic.OUTPUT
            # "target" only implies OUTPUT for non-quantity string params
            elif "target" in param_name and not any(q in param_name for q in ["total", "count", "num", "amount", "quantity"]):
                direction = DirectionSemantic.OUTPUT
            # Check skill name for type parameters
            elif "type" in param_name:
                if skill_name.startswith("craft"):
                    direction = DirectionSemantic.INPUT
                elif skill_name.startswith("ensure"):
                    direction = DirectionSemantic.OUTPUT

        # Check for fuel parameter
        is_fuel = "fuel" in param_name

        # Check for config parameter
        if is_config_parameter(param_name):
            direction = DirectionSemantic.CONFIG

        return ParameterSemantic(
            quantity_semantic=quantity_semantic,
            _direction_fallback=direction,
            is_fuel_param=is_fuel,
            inferred_by="parameter_engine",
            confidence=0.7,
        )

    def _apply_rule_transform(self, value: Any, transform: str) -> Optional[Any]:
        """
        Apply a learned rule transform to a value.

        Args:
            value: Value to transform
            transform: Transform type

        Returns:
            Transformed value or None
        """
        if not value or not transform:
            return None

        str_value = str(value)

        # Map transform type to base type
        transform_mappings = {
            "extract_ore_base": lambda v: extract_base_type(v, "ore"),
            "extract_log_base": lambda v: extract_base_type(v, "log"),
            "extract_planks_base": lambda v: extract_base_type(v, "planks"),
            "extract_ingot_base": lambda v: extract_base_type(v, "ingot"),
            "extract_raw_base": lambda v: extract_base_type(v, "raw"),
            "infer_input": lambda v: (_get_transform_fn("infer_input_from_output") or (lambda x: None))(v),
            "infer_output": lambda v: (_get_transform_fn("infer_output_from_input") or (lambda x: None))(v),
        }

        transform_func = transform_mappings.get(transform)
        if transform_func:
            return transform_func(str_value)

        return None

    def _parse_llm_response(self, response: str, param_type: str) -> Optional[Any]:
        """
        Parse LLM response to extract parameter value.

        Args:
            response: LLM response text
            param_type: Expected parameter type

        Returns:
            Extracted value or None
        """
        # Try to find JSON in response
        try:
            # Try direct JSON parse
            data = json.loads(response)
            if isinstance(data, dict) and "value" in data:
                return data["value"]
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        json_match = re.search(r'```json\n(.+?)\n```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "value" in data:
                    return data["value"]
            except json.JSONDecodeError:
                pass

        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*"value"\s*:\s*([^{}]+)\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict) and "value" in data:
                    return data["value"]
            except json.JSONDecodeError:
                pass

        # Try to extract value directly for simple types
        if param_type == "number":
            num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
            if num_match:
                return float(num_match.group(1))

        if param_type == "string":
            # Try quoted string
            str_match = re.search(r'"([^"]+)"', response)
            if str_match:
                return str_match.group(1)

        return None

    # ========================================================================
    # Rule Learning and Persistence
    # ========================================================================

    def _load_rules(self) -> Dict[str, Any]:
        """Load learned rules from file."""
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, 'r') as f:
                    rules = json.load(f)
                self.logger.info(f"[Rules] Loaded {len(rules)} learned rules")
                return rules
            except Exception as e:
                self.logger.warning(f"[Rules] Failed to load rules: {e}")
                return {}
        return {}

    def _save_rules(self):
        """Save learned rules to file."""
        try:
            os.makedirs(os.path.dirname(self.rules_file), exist_ok=True)
            with open(self.rules_file, 'w') as f:
                json.dump(self.learned_rules, f, indent=2)
            self.logger.info(f"[Rules] Saved {len(self.learned_rules)} rules")
        except Exception as e:
            self.logger.warning(f"[Rules] Failed to save rules: {e}")

    def _record_failure(self, strategy: str, context: InferenceContext, error: str):
        """Record a strategy failure for later analysis."""
        self._failures.append({
            "strategy": strategy,
            "skill_name": context.skill_name,
            "param_name": context.param_name,
            "error": error,
        })

    # ========================================================================
    # Type Validation
    # ========================================================================

    def _apply_type_validation(
        self,
        context: InferenceContext,
        result: InferenceResult
    ) -> InferenceResult:
        """
        v7.3 Phase 3.5: apply type-adapter validation.

        Detects whether the inferred value matches the type expected by the
        parameter semantic. If a mismatch is detected, lower confidence and
        mark ``validation_passed=False``.

        Args:
            context: inference context
            result: current inference result

        Returns:
            Potentially modified InferenceResult
        """
        from skillnet.agents.optimizer.validators.type_adapter import TypeMismatchDetector

        try:
            # Obtain or infer the parameter semantic
            semantic = self._infer_semantic(context)
            if not semantic:
                return result

            # Use TypeMismatchDetector to detect type mismatches
            detector = TypeMismatchDetector(domain_knowledge=get_domain_knowledge())
            # str() conversion to handle the case where value is an integer
            inferred_val = str(result.raw_value) if result.raw_value is not None else str(result.value).strip('"\'')
            issues = detector.detect(
                param_name=context.param_name,
                param_semantic=semantic,
                inferred_value=inferred_val,
                skill_name=context.skill_name
            )

            if issues:
                # Type mismatch detected: lower the confidence and mark
                new_confidence = result.confidence * 0.7  # Reduce by 30%
                return InferenceResult(
                    value=result.value,
                    strategy_used=result.strategy_used,
                    confidence=new_confidence,
                    explanation=f"{result.explanation}; type validation issues: {len(issues)}",
                    raw_value=result.raw_value,
                    transform_applied=result.transform_applied,
                    validation_passed=False
                )

        except Exception as e:
            self.logger.warning(f"[TypeValidation] Failed: {e}")

        return result
