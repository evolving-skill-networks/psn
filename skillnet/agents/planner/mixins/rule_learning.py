"""Rule Learning Mixin - Loading, saving, and learning parameter mapping rules."""

from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Any
from datetime import datetime

import skillnet.utils as U
from skillnet.agents.planning import item_matches_config_context
from .._types import TRANSFORM_TO_PATTERN

if TYPE_CHECKING:
    from ..graph_planner import GraphPlanner


class RuleLearningMixin:
    """Rule Learning Mixin - Loading, saving, and learning parameter mapping rules.

    Methods:
        _load_parameter_rules: Load learned parameter rules from file
        _save_parameter_rules: Save parameter rules to file
        _try_learned_rules: Try to extract parameter value using learned rules
        _try_object_rule: Apply learned object-type parameter rules
        _apply_field_rule: Apply a field rule to extract value from effects
        _learn_parameter_rule: Learn rule from successful parameter inference
        _learn_object_parameter_rule: Learn field rules for object-type parameters
        _effect_matches_pattern: Check if an effect matches a rule pattern
        _extract_value_by_rule: Extract value from effect using rule
        _validate_learned_value: Validate a learned value is valid
        _item_matches_config_context: Check if item matches config context
        _can_extract_from_effect: Check if value can be extracted from effect

    Note:
        These methods depend on self.parameter_rules_file, self.parameter_rules
        attributes initialized in GraphPlanner.__init__().
    """

    def _load_parameter_rules(self) -> Dict[str, Any]:
        """Load learned parameter-mapping rules."""
        if os.path.exists(self.parameter_rules_file):
            try:
                rules = U.load_json(self.parameter_rules_file)
                print(f"\033[36m[Parameter Rules] loaded {len(rules)} learned rule(s)\033[0m")
                return rules
            except (IOError, OSError, json.JSONDecodeError, ValueError) as e:
                print(f"\033[33m[Parameter Rules] failed to load rules: {e}\033[0m")
                return {}
        return {}

    def _save_parameter_rules(self):
        """Save parameter-mapping rules to file."""
        try:
            os.makedirs(os.path.dirname(self.parameter_rules_file), exist_ok=True)
            U.dump_json(self.parameter_rules, self.parameter_rules_file)
            print(f"\033[36m[Parameter Rules] saved {len(self.parameter_rules)} rule(s)\033[0m")
        except (IOError, OSError) as e:
            print(f"\033[33m[Parameter Rules] failed to save rules: {e}\033[0m")

    def _try_learned_rules(
        self,
        param_name: str,
        skill_name: str,
        target_effects: List[Dict[str, Any]],
        param_type: str
    ) -> Optional[str]:
        """
        Try extracting a parameter value using learned rules.

        Returns:
            Parameter value string, or None if no rule matches.
        """
        # Build rule key: skill_name.param_name
        rule_key = f"{skill_name}.{param_name}"

        if rule_key not in self.parameter_rules:
            return None

        rule = self.parameter_rules[rule_key]

        # Check whether this is an object-typed rule
        if rule.get("param_type") == "object":
            result = self._try_object_rule(rule, target_effects)
            if result:
                print(f"\033[36m[Parameter Rules] extracted '{param_name}' using a learned object rule: {result}\033[0m")
            return result

        # Original rule-matching logic
        rule_pattern = rule.get("pattern")  # Effect pattern
        rule_extraction = rule.get("extraction")  # How to extract the value from the effect

        # Try matching target_effects
        for effect in target_effects:
            # Check whether the effect matches the rule pattern
            if self._effect_matches_pattern(effect, rule_pattern):
                # Use the rule to extract the value
                value = self._extract_value_by_rule(effect, rule_extraction, param_type)
                if value:
                    # NEW: Validate learned rule before using
                    if not self._validate_learned_value(param_name, value, param_type):
                        print(f"\033[33m[Parameter Rules] discarding invalid learned rule: {param_name} <- {value}\033[0m")
                        # Remove the invalid rule from storage
                        del self.parameter_rules[rule_key]
                        self._save_parameter_rules()
                        return None
                    print(f"\033[36m[Parameter Rules] extracted '{param_name}' using learned rule: {value}\033[0m")
                    return value

        return None

    def _try_object_rule(
        self,
        rule: Dict[str, Any],
        target_effects: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Use a learned object rule to extract a parameter value.

        Args:
            rule: object-typed rule, e.g.:
                {
                    "param_type": "object",
                    "field_rules": {
                        "fieldName": {"effect_field": "item", "transform": "extract_ore_base"}
                    }
                }
            target_effects: list of target effects

        Returns:
            Constructed object parameter string, or None if it cannot be built.
        """
        field_rules = rule.get("field_rules", {})
        if not field_rules:
            return None

        obj_fields = {}
        for field_name, field_rule in field_rules.items():
            value = self._apply_field_rule(field_rule, target_effects)
            if value is not None:
                obj_fields[field_name] = value

        if obj_fields:
            props = [f'{k}: {v}' for k, v in obj_fields.items()]
            return "{" + ", ".join(props) + "}"

        return None

    def _apply_field_rule(
        self,
        field_rule: Dict[str, Any],
        target_effects: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Apply a field rule to extract a value.

        Args:
            field_rule: field rule, format:
                {"effect_field": "item", "transform": "extract_ore_base"}
            target_effects: list of target effects

        Returns:
            Extracted value string, or None if it cannot be extracted.
        """
        effect_field = field_rule.get("effect_field")
        transform = field_rule.get("transform")

        if not effect_field:
            return None

        for effect in target_effects:
            if effect_field in effect:
                value = effect[effect_field]

                if transform and transform.startswith("extract_"):
                    # Handle via _extract_base_type
                    pattern_type = TRANSFORM_TO_PATTERN.get(transform, "generic")
                    base = self._extract_base_type(str(value), pattern_type)
                    if base:
                        return f'"{base}"'
                elif transform is None:
                    # Use the value directly
                    if isinstance(value, str):
                        return f'"{value}"'
                    else:
                        return str(value)

        return None

    def _learn_object_parameter_rule(
        self,
        param_name: str,
        skill_name: str,
        target_effects: List[Dict[str, Any]],
        extracted_value: str,
        schema: Dict[str, Any]
    ):
        """
        Learn a mapping rule for an object-typed parameter.

        Called after successfully constructing an object parameter so that
        future extractions can use the learned rule directly.

        Args:
            param_name: parameter name
            skill_name: skill name
            target_effects: list of target effects
            extracted_value: successfully constructed parameter value
            schema: schema of the object parameter
        """
        rule_key = f"{skill_name}.{param_name}"
        field_rules = {}

        for field_name, field_info in schema.items():
            # Use the source_hint from the schema directly as the rule
            source_hint = field_info.get("source_hint")
            if source_hint:
                field_rules[field_name] = {
                    "effect_field": source_hint.get("effect_field"),
                    "transform": source_hint.get("transform")
                }

        if field_rules:
            self.parameter_rules[rule_key] = {
                "param_type": "object",
                "field_rules": field_rules,
                "learned_at": datetime.now().isoformat(),
                "usage_count": 1
            }

            print(f"\033[36m[Parameter Rules] learned object rule: {rule_key}\033[0m")
            print(f"\033[36m[Parameter Rules]   field rules: {list(field_rules.keys())}\033[0m")

            self._save_parameter_rules()

    def _effect_matches_pattern(self, effect: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check whether an effect matches a rule pattern."""
        if not pattern:
            return True

        # Simple pattern match: check that effect contains the pattern's key/value pairs
        for key, value in pattern.items():
            if key not in effect:
                return False
            if isinstance(value, dict):
                # Nested match
                if not isinstance(effect[key], dict):
                    return False
                if not self._effect_matches_pattern(effect[key], value):
                    return False
            elif effect[key] != value:
                return False

        return True

    def _extract_value_by_rule(self, effect: Dict[str, Any], extraction: Dict[str, Any], param_type: str) -> Optional[str]:
        """Extract a value from an effect using a rule."""
        if not extraction:
            return None

        # Extraction path: e.g. {"path": "item"} or {"path": "count"}
        path = extraction.get("path")
        if not path:
            return None

        # Get the value from the effect
        value = effect.get(path)
        if value is None:
            return None

        # Format the value based on parameter type
        if param_type == "array":
            if isinstance(value, list):
                return json.dumps(value)
            else:
                return json.dumps([value])
        elif param_type == "string":
            return f'"{value}"'
        elif param_type == "number":
            return str(value)
        else:
            return str(value)

    def _learn_parameter_rule(
        self,
        param_name: str,
        skill_name: str,
        target_effects: List[Dict[str, Any]],
        extracted_value: str,
        param_type: str
    ):
        """
        Learn a parameter-mapping rule.

        Called after the LLM successfully extracts a parameter value to learn the mapping.
        """
        # NEW: Validate the value before learning
        if not self._validate_learned_value(param_name, extracted_value, param_type):
            print(f"\033[33m[Parameter Rules] refusing to learn invalid rule: {param_name} <- {extracted_value}\033[0m")
            return

        # Find a matching effect (used for learning the pattern)
        matched_effect = None
        for effect in target_effects:
            # Check whether extracted_value can be extracted from this effect
            if self._can_extract_from_effect(effect, extracted_value, param_type):
                matched_effect = effect
                break

        if not matched_effect:
            return

        # Build the rule
        rule_key = f"{skill_name}.{param_name}"

        # Extract pattern (simplified: record key fields only)
        pattern = {}
        extraction = {}

        # If the effect has count, record the count pattern
        if "count" in matched_effect:
            pattern["count"] = matched_effect["count"]
            if "count" in param_name.lower():
                extraction["path"] = "count"

        # If the effect has item, record the item pattern
        if "item" in matched_effect:
            pattern["item"] = matched_effect["item"]
            if "item" in param_name.lower() or "type" in param_name.lower():
                extraction["path"] = "item"

        # If the effect has type, record the type pattern
        if "type" in matched_effect:
            pattern["type"] = matched_effect["type"]
            if "type" in param_name.lower():
                extraction["path"] = "type"

        # Save the rule
        existing_rule = self.parameter_rules.get(rule_key, {})
        self.parameter_rules[rule_key] = {
            "pattern": pattern,
            "extraction": extraction,
            "param_type": param_type,
            "learned_at": datetime.now().isoformat(),
            "usage_count": existing_rule.get("usage_count", 0) + 1
        }

        print(f"\033[36m[Parameter Rules] learned new rule: {rule_key}\033[0m")
        print(f"\033[36m[Parameter Rules]   pattern: {pattern}\033[0m")
        print(f"\033[36m[Parameter Rules]   extraction: {extraction}\033[0m")

        # Persist the rule (could be optimized to batch or delayed saves)
        self._save_parameter_rules()

    def _item_matches_config_context(self, param_name: str, item: str) -> bool:
        """
        Check if item semantically matches the config parameter's expected type.

        delegates to the pure function item_matches_config_context.
        """
        return item_matches_config_context(param_name, item)

    def _validate_learned_value(self, param_name: str, extracted_value: str, param_type: str) -> bool:
        """
        Validate if the extracted value is reasonable for this parameter.

        This prevents learning incorrect rules like mapping iron_ingot to fuelPriority.

        Args:
            param_name: Parameter name
            extracted_value: The value extracted by LLM or rules
            param_type: Parameter type (array, string, number, etc.)

        Returns:
            bool: True if valid, False if should be rejected
        """
        param_lower = param_name.lower()

        # Config parameter indicators - these need special validation
        config_indicators = [
            "priority", "preference", "prefer", "option", "options",
            "config", "timeout", "distance", "fallback", "default",
            "selection", "choices", "allowed"
        ]

        is_config_param = any(ind in param_lower for ind in config_indicators)

        if not is_config_param:
            # Non-config params don't need this validation
            return True

        # Parse the value to get the item(s)
        try:
            if param_type == "array":
                items = json.loads(extracted_value)
                if not items or not isinstance(items, list):
                    return True  # Empty or non-list is OK
                # Check all items in the array
                for item in items:
                    if isinstance(item, str) and not self._item_matches_config_context(param_name, item):
                        return False
                return True
            elif param_type == "string":
                item = extracted_value.strip('"\'')
                return self._item_matches_config_context(param_name, item)
            else:
                return True  # Non-item params don't need validation
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            return True  # If parsing fails, allow it (conservative)

    def _can_extract_from_effect(self, effect: Dict[str, Any], extracted_value: str, param_type: str) -> bool:
        """Check whether extracted_value can be obtained from the effect."""
        # Parse extracted_value
        try:
            if param_type == "array":
                # Try parsing a JSON array
                parsed = json.loads(extracted_value)
                if isinstance(parsed, list) and len(parsed) > 0:
                    item = parsed[0]
                    return effect.get("item") == item
            elif param_type == "string":
                # Strip quotes
                value = extracted_value.strip('"\'')
                return effect.get("item") == value or effect.get("type") == value
            elif param_type == "number":
                value = int(extracted_value)
                return effect.get("count") == value
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return False
