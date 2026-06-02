"""
Parameter Semantic Inference Mixin

Quantity, direction, and transform hint inference for parameter metadata.
Split from parameters.py for maintainability.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage

from skillnet.agents.planning.inference import (
    QuantitySemantic,
    DirectionSemantic,
    TransformHint,
    StateMapping,
    ParameterSemantic,
)

from ._param_constants import (
    QUANTITY_PARAM_NAMES,
    TARGET_TOTAL_PATTERNS,
    DELTA_FUNC_PREFIXES,
    ENSURE_PREFIXES,
    INPUT_DIRECTION_INDICATORS,
    OUTPUT_DIRECTION_INDICATORS,
    CONFIG_DIRECTION_INDICATORS,
    FUEL_PARAM_NAMES,
    _get_func_prefix_to_direction,
    _get_param_suffix_to_transform,
    SEMANTIC_INFERENCE_PROMPT,
    determine_delta_prefix,
)

logger = logging.getLogger(__name__)


class ParameterSemanticMixin:
    """Mixin: Semantic inference methods for parameter enrichment."""

    def infer_full_semantic(
        self,
        param_name: str,
        func_name: str,
        code: str,
        param_type: str = "number",
        param_info: Optional[Dict[str, Any]] = None,
    ) -> ParameterSemantic:
        """
        Infer the full parameter semantic (including direction and transform_hint)

        Args:
            param_name: parameter name
            func_name: function name
            code: function code
            param_type: parameter type
            param_info: parameter info dict (optional, contains LLM-extracted _llm_state_mapping)

        Returns:
            The complete ParameterSemantic object
        """
        # 1. Infer quantity semantic
        # If LLM already provided a valid semantic string, prefer it
        llm_quantity = None
        if param_info:
            raw_sem = param_info.get("semantic")
            if isinstance(raw_sem, str) and raw_sem in ("target_total", "delta"):
                llm_quantity = QuantitySemantic.from_string(raw_sem)

        if llm_quantity is not None:
            quantity_semantic = llm_quantity
        else:
            quantity_semantic = self._infer_quantity_semantic(
                param_name, func_name, code, param_type
            )

        # 2. Build StateMapping from LLM extraction (if available)
        state_mapping = None
        if param_info and "_llm_state_mapping" in param_info:
            sm = StateMapping.from_dict(param_info["_llm_state_mapping"])
            if sm:
                state_mapping = self._validate_state_mapping(
                    sm, param_name, param_type
                )

        # 3. Infer direction semantic (only when no state_mapping — direction is derived from it)
        direction = DirectionSemantic.CONFIG
        if not state_mapping:
            direction = self._infer_direction(param_name, func_name, param_type)

        # 4. Generate transform hint (only when no state_mapping — it has its own transform)
        transform_hint = None
        if not state_mapping:
            transform_hint = self._generate_transform_hint(param_name, direction)

        # 5. Detect special parameter types
        config_context = self._detect_config_context(param_name)
        is_fuel_param = self._is_fuel_param(param_name)

        return ParameterSemantic(
            state_mapping=state_mapping,
            quantity_semantic=quantity_semantic,
            _direction_fallback=direction,
            transform_hint=transform_hint,
            config_context=config_context,
            is_fuel_param=is_fuel_param,
            inferred_by="llm" if state_mapping else "parameter_extractor",
            confidence=0.85 if state_mapping else 0.8,
        )

    def _validate_state_mapping(
        self,
        sm: StateMapping,
        param_name: str,
        param_type: str,
    ) -> Optional[StateMapping]:
        """Validate LLM-provided StateMapping for basic sanity.

        Returns the StateMapping if valid, None if rejected.
        """
        # field="item" but param is numeric → likely wrong
        if sm.field == "item" and param_type in ("number", "integer"):
            self.logger.warning(
                f"[StateMapping] Rejecting {param_name}: field='item' but param_type='{param_type}'"
            )
            return None

        # field="count" but param is string → likely wrong
        if sm.field == "count" and param_type == "string":
            self.logger.warning(
                f"[StateMapping] Rejecting {param_name}: field='count' but param_type='string'"
            )
            return None

        # source="config" with transform → doesn't make sense
        if sm.source == "config" and sm.transform:
            self.logger.warning(
                f"[StateMapping] {param_name}: source='config' with transform='{sm.transform}', "
                "clearing transform"
            )
            sm = StateMapping(
                source=sm.source, field=sm.field,
                state_type=sm.state_type, transform=None,
            )

        return sm

    def _infer_quantity_semantic(
        self,
        param_name: str,
        func_name: str,
        code: str,
        param_type: str,
    ) -> QuantitySemantic:
        """Infer the quantity semantic (target_total vs delta)"""
        # Non-quantity parameter -> NONE
        if param_name.lower() not in [p.lower() for p in self.quantity_param_names]:
            return QuantitySemantic.NONE

        # Non-numeric type -> NONE
        if param_type not in ["number", "integer", "unknown"]:
            return QuantitySemantic.NONE

        # Code-pattern detection
        for pattern in TARGET_TOTAL_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return QuantitySemantic.TARGET_TOTAL

        # Parameter-name check
        param_pattern = r"if\s*\([^)]*>=\s*" + re.escape(param_name) + r"[^)]*\)\s*{?\s*(return|bot\.chat)"
        if re.search(param_pattern, code, re.IGNORECASE):
            return QuantitySemantic.TARGET_TOTAL

        # Function-name check
        if func_name.lower().startswith("ensure"):
            return QuantitySemantic.TARGET_TOTAL

        # delta indicators
        delta_indicators = ["delta", "additional", "extra", "more"]
        if any(indicator in param_name.lower() for indicator in delta_indicators):
            return QuantitySemantic.DELTA

        # Function-name prefix
        if any(func_name.lower().startswith(prefix) for prefix in DELTA_FUNC_PREFIXES):
            return QuantitySemantic.DELTA

        # Default
        if self.semantic_default == "target_total":
            return QuantitySemantic.TARGET_TOTAL
        return QuantitySemantic.DELTA

    def _infer_direction(
        self,
        param_name: str,
        func_name: str,
        param_type: str,
    ) -> DirectionSemantic:
        """
        Infer the direction semantic of a parameter (INPUT vs OUTPUT vs CONFIG)

        Inference rules:
        1. Quantity parameter -> CONFIG (not a type parameter)
        2. Parameter name contains a CONFIG indicator -> CONFIG
        3. Parameter name contains an INPUT indicator -> INPUT
        4. Parameter name contains an OUTPUT indicator -> OUTPUT
        5. Use the function-name prefix to infer the direction of type parameters
        6. Default -> CONFIG
        """
        param_lower = param_name.lower().replace("_", "")

        # Rule 1: a quantity parameter is not a type parameter
        # Bug 1 fix: use more precise matching logic
        # - Exact match: count, amount, num, etc.
        # - Suffix match: targetTotal, itemCount, etc. (the name ends with a quantity word)
        # - Exclude: targetBlock, targetEntity, etc. (here "target" refers to the target object, not a quantity)
        quantity_suffixes = ["total", "count", "num", "amount", "quantity", "number"]
        for qty_name in self.quantity_param_names:
            qty_lower = qty_name.lower()
            # Exact match (e.g., count, amount)
            if param_lower == qty_lower:
                return DirectionSemantic.CONFIG
            # Suffix match (e.g., targetTotal, itemCount) - only for non-"target" quantity words
            if qty_lower in quantity_suffixes and param_lower.endswith(qty_lower):
                return DirectionSemantic.CONFIG

        # Rule 2: CONFIG indicator
        for indicator in CONFIG_DIRECTION_INDICATORS:
            if indicator.replace("_", "") in param_lower:
                return DirectionSemantic.CONFIG

        # Rule 3: INPUT indicator
        for indicator in INPUT_DIRECTION_INDICATORS:
            if indicator.replace("_", "") in param_lower:
                return DirectionSemantic.INPUT

        # Rule 4: OUTPUT indicator
        for indicator in OUTPUT_DIRECTION_INDICATORS:
            if indicator.replace("_", "") in param_lower:
                return DirectionSemantic.OUTPUT

        # Rule 5: infer type parameters from the function-name prefix
        if "type" in param_lower:
            func_lower = func_name.lower()
            for prefix, direction in _get_func_prefix_to_direction().items():
                if func_lower.startswith(prefix):
                    self.logger.debug(
                        f"[Direction] {func_name}.{param_name}: inferred as {direction.value} "
                        f"(rule: func_prefix_{prefix})"
                    )
                    return direction

        # Rule 6: default
        return DirectionSemantic.CONFIG

    def _generate_transform_hint(
        self,
        param_name: str,
        direction: DirectionSemantic,
    ) -> Optional[TransformHint]:
        """
        Generate a TransformHint (for input-material parameters)

        Only INPUT-direction parameters need a transform hint
        """
        if direction != DirectionSemantic.INPUT:
            return None

        param_lower = param_name.lower().replace("_", "")

        # Check predefined transforms
        for suffix, hint in _get_param_suffix_to_transform().items():
            if suffix.replace("_", "") in param_lower:
                return hint

        return None

    def _detect_config_context(self, param_name: str) -> Optional[str]:
        """Detect the context type of a config parameter"""
        param_lower = param_name.lower()

        if "fuel" in param_lower:
            return "fuel"
        if "log" in param_lower:
            return "log"
        if "tool" in param_lower:
            return "tool"
        if "block" in param_lower:
            return "block"

        return None

    def _is_fuel_param(self, param_name: str) -> bool:
        """Detect whether this is a fuel parameter"""
        param_lower = param_name.lower().replace("_", "")
        return any(
            fuel.replace("_", "") in param_lower
            for fuel in FUEL_PARAM_NAMES
        )

    def _infer_type_from_usage(
        self,
        param_name: str,
        code: str,
        initial_type: str,
    ) -> str:
        """
        Infer the actual parameter type from its usage in the code.

        Triggered when the initial type is "nullable" or "unknown".

        Inference rules:
        1. .filter() / .map() / .forEach() / for-of -> "array"
        2. .toLowerCase() / .trim() / .split() -> "string"
        3. Parameter-name semantic inference (fallback)

        Args:
            param_name: parameter name
            code: function code
            initial_type: initial type ("nullable" | "unknown")

        Returns:
            The inferred type; returns initial_type if it cannot be inferred
        """
        if initial_type not in ("nullable", "unknown"):
            return initial_type

        # Array-method detection
        array_patterns = [
            rf'{re.escape(param_name)}\s*\.\s*(filter|map|forEach|find|some|every|reduce|includes)\s*\(',
            rf'{re.escape(param_name)}\s*\.\s*length\b',
            rf'{re.escape(param_name)}\s*\[\d+\]',  # index access
            rf'for\s*\([^)]*\s+of\s+{re.escape(param_name)}\s*\)',  # for-of loop
        ]
        for pattern in array_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                self.logger.info(
                    f"[Type Inference] {param_name}: {initial_type} -> array (rule: usage_pattern)"
                )
                return "array"

        # String-method detection
        string_patterns = [
            rf'{re.escape(param_name)}\s*\.\s*(toLowerCase|toUpperCase|trim|split|replace|startsWith|endsWith)\s*\(',
            rf'{re.escape(param_name)}\s*\.\s*charAt\s*\(',
        ]
        for pattern in string_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                self.logger.info(
                    f"[Type Inference] {param_name}: {initial_type} -> string (rule: usage_pattern)"
                )
                return "string"

        # Numeric-operation detection
        number_patterns = [
            rf'{re.escape(param_name)}\s*[+\-*/%]\s*\d',
            rf'\d\s*[+\-*/%]\s*{re.escape(param_name)}',
            rf'{re.escape(param_name)}\s*(>=|<=|>|<)\s*\d',
        ]
        for pattern in number_patterns:
            if re.search(pattern, code):
                self.logger.info(
                    f"[Type Inference] {param_name}: {initial_type} -> number (rule: usage_pattern)"
                )
                return "number"

        # Parameter-name semantic inference (fallback).
        # Tokenize by camelCase boundary and underscore, then match against hint
        # words EXACTLY. Substring matching here used to false-positive on
        # names like itemType (contains 'items') or toolType (contains 'types'),
        # mistyping single-value parameters as arrays and corrupting downstream
        # parameter-inference calls — see the verification report's Phase 4
        # known-issue entry for the full trace.
        ARRAY_HINT_TOKENS = {"types", "list", "items", "names", "options"}
        tokens = {
            t.lower()
            for t in re.split(r"(?<=[a-z])(?=[A-Z])|_", param_name)
            if t
        }
        if tokens & ARRAY_HINT_TOKENS:
            self.logger.info(
                f"[Type Inference] {param_name}: {initial_type} -> array (rule: param_name_hint, tokens={tokens})"
            )
            return "array"

        return initial_type

    def validate_function_name_semantic_consistency(
        self,
        func_name: str,
        parameters: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Validate consistency between the function name and parameter semantics.

        Check rules:
        1. Function name starts with ensure*, but the parameter semantic is delta → warn
        2. Function name starts with craft/mine/collect, but the parameter semantic is target_total → warn

        Args:
            func_name: function name
            parameters: parameter-metadata dict

        Returns:
            List of warnings
        """
        warnings = []
        func_name_lower = func_name.lower()
        quantity_param_names_lower = [p.lower() for p in self.quantity_param_names]

        for param_name, param_info in parameters.items():
            if param_name.lower() not in quantity_param_names_lower:
                continue

            # Get the semantic (handle both new and old formats)
            semantic_data = param_info.get("semantic")
            if isinstance(semantic_data, dict):
                # New format: {"quantity_semantic": "delta", "direction": "config", ...}
                semantic = semantic_data.get("quantity_semantic", "none")
            elif isinstance(semantic_data, str):
                # Old format: "delta" | "target_total" | "config"
                semantic = semantic_data
            else:
                continue

            if not semantic or semantic in ["config", "none"]:
                continue

            # Check ensure* function name with delta semantic
            is_ensure_name = any(func_name_lower.startswith(prefix) for prefix in ENSURE_PREFIXES)
            if is_ensure_name and semantic == "delta":
                new_prefix = determine_delta_prefix(func_name)
                suffix = func_name[6:] if len(func_name) > 6 else func_name
                suggested_name = f"{new_prefix}{suffix}"
                warnings.append({
                    "param_name": param_name,
                    "semantic": semantic,
                    "expected": "target_total",
                    "suggestion": f"Function name starts with 'ensure' but semantic is delta; suggest renaming to '{suggested_name}'",
                    "suggested_name": suggested_name
                })
                self.logger.warning(
                    f"[Semantic Validation] {func_name}.{param_name}: function name implies target_total but semantic is delta"
                )

            # Check craft/mine/collect function name with target_total semantic
            is_delta_name = any(func_name_lower.startswith(prefix) for prefix in DELTA_FUNC_PREFIXES)
            if is_delta_name and semantic == "target_total":
                # Extract the original prefix length
                original_prefix = ""
                for prefix in DELTA_FUNC_PREFIXES:
                    if func_name_lower.startswith(prefix):
                        original_prefix = prefix
                        break
                suffix = func_name[len(original_prefix):] if original_prefix else func_name
                suggested_name = f"ensure{suffix}"
                warnings.append({
                    "param_name": param_name,
                    "semantic": semantic,
                    "expected": "delta",
                    "suggestion": f"Function name implies delta but semantic is target_total; suggest renaming to '{suggested_name}'",
                    "suggested_name": suggested_name
                })
                self.logger.warning(
                    f"[Semantic Validation] {func_name}.{param_name}: function name implies delta but semantic is target_total"
                )

        return warnings
