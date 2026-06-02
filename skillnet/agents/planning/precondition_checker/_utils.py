"""
Precondition Checker Utilities (Layer A)

Pure-function and constants layer: stateless helpers for tool tiers, generic types,
condition evaluation, etc.

extracted from precondition_checker.py.

Functions:
- get_tool_tier: extract a tool tier
- extract_tool_type: classify a tool type
- check_tool_requirement: validate a tool requirement
- extract_generic_type: extract a generic type
- is_generic_item_type: detect a generic type
- condition_matches: evaluate condition matching
- extract_conditions_from_state_repr: recursively extract conditions
- normalize_shorthand_condition: normalize shorthand format (extracted from a PreconditionChecker method)

Constants:
- TOOL_TIERS: tool-tier mapping
- MATERIAL_PREFIXES: list of material prefixes
- GENERIC_PATTERNS: generic-type patterns
"""

import re
from typing import Any, Dict, List, Optional

from skillnet.core.dk_registry import get_domain_knowledge


# ============================================================================
# Constants (fallback defaults)
# ============================================================================

_FALLBACK_TOOL_TIERS = {}

_FALLBACK_MATERIAL_PREFIXES = []

_FALLBACK_GENERIC_PATTERNS = []


# ============================================================================
# Domain-aware accessor functions
# ============================================================================

def get_tool_tiers():
    """Return tool tier mapping from domain or fallback."""
    dk = get_domain_knowledge()
    if dk:
        config = dk.get_tool_tier_config()
        if config and "tool_tiers" in config:
            return config["tool_tiers"]
    return _FALLBACK_TOOL_TIERS


def get_material_prefixes():
    """Return material prefix list from domain or fallback."""
    dk = get_domain_knowledge()
    if dk:
        config = dk.get_tool_tier_config()
        if config and "material_prefixes" in config:
            return config["material_prefixes"]
    return _FALLBACK_MATERIAL_PREFIXES


def get_generic_patterns():
    """Return generic item patterns from domain or fallback."""
    dk = get_domain_knowledge()
    if dk:
        config = dk.get_tool_tier_config()
        if config and "generic_patterns" in config:
            return config["generic_patterns"]
    return _FALLBACK_GENERIC_PATTERNS


# ============================================================================
# Tool functions
# ============================================================================

def get_tool_tier(item_name: str) -> int:
    """
    Get a tool tier

    Args:
        item_name: item name

    Returns:
        int: tool tier (1-6); unknown tools return 0
    """
    item_lower = item_name.lower()
    for tier, level in get_tool_tiers().items():
        if tier in item_lower:
            return level
    return 0


def extract_tool_type(item_name: str) -> Optional[str]:
    """
    Extract the tool type from an item name

    Args:
        item_name: item name

    Returns:
        Optional[str]: tool type (pickaxe, axe, sword, shovel, hoe) or None
    """
    item_lower = item_name.lower()
    if "pickaxe" in item_lower:
        return "pickaxe"
    elif "axe" in item_lower and "pickaxe" not in item_lower:
        return "axe"
    elif "sword" in item_lower:
        return "sword"
    elif "shovel" in item_lower:
        return "shovel"
    elif "hoe" in item_lower:
        return "hoe"
    return None


def check_tool_requirement(
    item_name: str,
    operation: str,
    inventory: Dict[str, int],
    tool_tier: str = None
) -> bool:
    """
    Check whether a tool requirement is satisfied

    Args:
        item_name: required item name
        operation: operation type ("require" or "require_or_better")
        inventory: current inventory
        tool_tier: tool tier (optional)

    Returns:
        bool: whether the requirement is satisfied
    """
    # If tool_tier is not provided, extract it from item_name
    tiers = get_tool_tiers()
    if not tool_tier:
        item_lower = item_name.lower()
        for tier in tiers.keys():
            if tier in item_lower:
                tool_tier = tier
                break

    required_tier_level = tiers.get(tool_tier, 0) if tool_tier else 0

    # Extract tool type
    tool_type = extract_tool_type(item_name)

    if not tool_type:
        # If not a tool, use exact match
        item_count = inventory.get(item_name, 0)
        return item_count >= 1

    item_lower = item_name.lower()

    # Check whether the inventory has a tool that meets the requirement
    for inv_item, count in inventory.items():
        if count <= 0:
            continue
        inv_item_lower = inv_item.lower()

        # Check whether it's the same kind of tool
        if tool_type not in inv_item_lower:
            continue

        if operation == "require":
            # Exact match
            if inv_item_lower == item_lower:
                return True
        elif operation == "require_or_better":
            # Check whether the tier meets the requirement or better
            item_tier_level = get_tool_tier(inv_item)
            if required_tier_level > 0 and item_tier_level >= required_tier_level:
                return True
            # If the tier cannot be determined, check for an exact match
            elif inv_item_lower == item_lower:
                return True

    return False


# ============================================================================
# Generic-type functions
# ============================================================================

def extract_generic_type(item_list: List[str]) -> str:
    """
    Extract a generic type name from an item list

    For example:
    - ["oak_planks", "birch_planks", "spruce_planks"] → "planks"
    - ["coal", "charcoal", "oak_log", "oak_planks"] → "fuel"
    - ["oak_log", "birch_log", "spruce_log"] → "log"

    Args:
        item_list: list of item names

    Returns:
        generic type name
    """
    if not item_list:
        return "unknown"

    # Keep only valid string items
    valid_items = [item.lower() for item in item_list if isinstance(item, str)]
    if not valid_items:
        return "unknown"

    # Try to find a common suffix
    suffixes = ["_planks", "_log", "_ore", "_ingot", "_pickaxe", "_axe", "_sword", "_shovel", "_hoe"]
    for suffix in suffixes:
        if all(item.endswith(suffix) for item in valid_items):
            return suffix.lstrip("_")  # Return "planks", "log", etc.

    # Check whether it's a fuel list
    fuel_items = {"coal", "charcoal", "stick", "blaze_rod", "lava_bucket"}
    fuel_suffixes = ["_planks", "_log", "_wood"]
    is_fuel_list = all(
        item in fuel_items or any(item.endswith(s) for s in fuel_suffixes)
        for item in valid_items
    )
    if is_fuel_list:
        return "fuel"

    # Unrecognized; return the first item (preserve compatibility)
    return valid_items[0] if valid_items else "unknown"


def is_generic_item_type(item: str) -> bool:
    """
    Check whether an item name is a generic type (does not specify a specific variant)

    For example:
    - "planks" → True (generic planks; does not specify oak/birch/etc.)
    - "oak_planks" → False (specifically oak planks)
    - "log" → True (generic log)
    - "fuel" → True (generic fuel)

    Args:
        item: item name

    Returns:
        whether it is a generic type
    """
    item_lower = item.lower()

    # If it's a pure generic type name
    if item_lower in get_generic_patterns():
        return True

    # If it contains a material prefix, it's not a generic type
    for prefix in get_material_prefixes():
        if item_lower.startswith(prefix + "_"):
            return False

    # Default True (conservative: allow defaults)
    return True


# ============================================================================
# Condition-evaluation functions
# ============================================================================

def condition_matches(
    condition: Dict[str, Any],
    param_values: Dict[str, Any]
) -> bool:
    """
    Check whether a condition matches parameter values

    Used for conditional preconditions/effects: only when parameter values match the condition does
    the precondition/effect apply.

    Args:
        condition: condition expression, e.g. {"targetBlockNames": "diamond_ore"}
                  or {"targetBlockNames": ["diamond_ore", "gold_ore"]}
        param_values: current parameter values, e.g. {"targetBlockNames": "diamond_ore", "count": 3}

    Returns:
        bool: whether the condition matches
    """
    if condition is None:
        return True  # No condition; always matches

    if param_values is None:
        return True  # No parameter values; cannot evaluate, default to match

    for param_name, expected_value in condition.items():
        actual_value = param_values.get(param_name)
        if actual_value is None:
            continue  # Parameter value unknown; skip check (do not exclude this precondition)

        # Support a single value or a list
        if isinstance(expected_value, list):
            if actual_value not in expected_value:
                return False
        else:
            if actual_value != expected_value:
                return False

    return True


def extract_conditions_from_state_repr(state_repr) -> List[Dict]:
    """
    Extract all conditions from a state_representation

    Supported formats:
    1. Old format (list): [{"type": "inventory", "item": "oak_log", ...}, ...]
    2. New format: {"logic": "OR", "conditions": [...]}
    3. Single condition: {"type": "inventory", "item": "oak_log", ...}

    Args:
        state_repr: state_representation (dict, list, or nested structure)

    Returns:
        list of conditions
    """
    if isinstance(state_repr, list):
        return state_repr
    elif isinstance(state_repr, dict):
        if 'logic' in state_repr and 'conditions' in state_repr:
            # Recursively extract all conditions
            conditions = []
            for cond in state_repr.get('conditions', []):
                conditions.extend(extract_conditions_from_state_repr(cond))
            return conditions
        else:
            # Single condition
            return [state_repr]
    return []


def normalize_shorthand_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    """
    convert shorthand conditions into the standard format

    Shorthand: {"item": "stone_pickaxe", "min_count": 1}
    Standard: {"type": "inventory", "item": "stone_pickaxe", "count": 1, "operation": "require"}

    extracted from PreconditionChecker._normalize_shorthand_condition into a
    module-level pure function (the original method did not use self).

    Args:
        condition: condition in shorthand format

    Returns:
        condition in standard format
    """
    if not isinstance(condition, dict):
        return condition

    normalized = dict(condition)

    # If type is missing, default to inventory
    if "type" not in normalized:
        normalized["type"] = "inventory"

    # min_count -> count
    if "min_count" in normalized and "count" not in normalized:
        normalized["count"] = normalized.pop("min_count")

    # If operation is missing, default to require
    if "operation" not in normalized:
        normalized["operation"] = "require"

    return normalized
