"""
Config Validator Module

This module provides validation utilities for config parameters:
- Config context matching (fuel, tool, log, etc.)
- Item compatibility checking
- Parameter name pattern detection

Migrated from parameter_resolver.py for use in the unified inference engine.

Domain injection:
    Domain knowledge is accessed via the central ``dk_registry``.  When no
    domain is registered, the module falls back to empty defaults.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("skillnet.planner.inference.config_validator")


# ============================================================================
# Domain injection (via central registry)
# ============================================================================

from skillnet.core.dk_registry import get_domain_knowledge


# ============================================================================
# Fallback constants (Minecraft defaults)
# ============================================================================

_FALLBACK_CONFIG_CONTEXTS = {}

# Config parameter indicator keywords
# Parameters containing these words are likely config parameters
CONFIG_INDICATORS = [
    "priority", "preference", "prefer", "option", "options",
    "config", "timeout", "distance", "fallback", "default",
    "selection", "choices", "allowed", "setting", "mode"
]

_FALLBACK_INPUT_MATERIAL_PARAMS = []

# Transform type to regex pattern mapping
TRANSFORM_TO_PATTERN = {
    "extract_ore_base": "ore",
    "extract_log_base": "log",
    "extract_planks_base": "planks",
    "extract_ingot_base": "ingot",
    "extract_raw_base": "raw",
    "extract_base": "generic",
}

_FALLBACK_PRODUCT_TO_MATERIAL_MAPPINGS = []


# ============================================================================
# Domain-aware accessors
# ============================================================================

def _get_config_contexts() -> dict:
    """Return config contexts from domain or fallback."""
    dk = get_domain_knowledge()
    if dk is not None:
        cfg = dk.get_crafting_chain_config()
        if cfg.get("config_contexts"):
            return cfg["config_contexts"]
    return _FALLBACK_CONFIG_CONTEXTS


def _get_input_material_params() -> list:
    """Return input material param patterns from domain or fallback."""
    dk = get_domain_knowledge()
    if dk is not None:
        cfg = dk.get_crafting_chain_config()
        if cfg.get("input_material_params"):
            return cfg["input_material_params"]
    return _FALLBACK_INPUT_MATERIAL_PARAMS


def _get_product_to_material_mappings() -> list:
    """Return product-to-material mappings from domain or fallback."""
    dk = get_domain_knowledge()
    if dk is not None:
        cfg = dk.get_crafting_chain_config()
        if cfg.get("product_to_material_mappings"):
            return cfg["product_to_material_mappings"]
    return _FALLBACK_PRODUCT_TO_MATERIAL_MAPPINGS


# ============================================================================
# Validation Functions
# ============================================================================

def item_matches_config_context(param_name: str, item: str) -> bool:
    """
    Check if item semantically matches the config parameter's expected type.

    This validates whether an item makes sense for a config parameter.
    For example:
    - fuelPriority + coal -> True (coal is a fuel)
    - fuelPriority + iron_ingot -> False (ingot is not a fuel)
    - toolPreference + iron_pickaxe -> True (pickaxe is a tool)

    Args:
        param_name: Parameter name (e.g., "fuelPriority")
        item: Item name to check (e.g., "iron_ingot")

    Returns:
        bool: True if item matches the config context, False otherwise
    """
    param_lower = param_name.lower()
    item_lower = item.lower()

    for context, patterns in _get_config_contexts().items():
        if context in param_lower:
            # Check if item matches any pattern for this context
            for pattern in patterns:
                if pattern in item_lower or item_lower == pattern.strip("_"):
                    return True
            # Context found but item doesn't match - reject
            return False

    # Unknown config param - be permissive
    return True


def is_config_parameter(param_name: str) -> bool:
    """
    Check if parameter name indicates a config parameter.

    Config parameters are configuration options rather than item types.

    Args:
        param_name: Parameter name to check

    Returns:
        bool: True if this appears to be a config parameter
    """
    param_lower = param_name.lower()
    return any(indicator in param_lower for indicator in CONFIG_INDICATORS)


def is_input_material_param(param_name: str) -> bool:
    """
    Check if parameter name indicates an input material parameter.

    Input material parameters expect raw materials or source items,
    not crafted products or outputs.

    Args:
        param_name: Parameter name to check

    Returns:
        bool: True if this appears to be an input material parameter
    """
    param_lower = param_name.lower().replace("_", "")
    return any(
        p.replace("_", "") in param_lower
        for p in _get_input_material_params()
    )


# ============================================================================
# Type Extraction Functions
# ============================================================================

def extract_base_type(item_name: str, pattern_type: str = "ore") -> Optional[str]:
    """
    Extract base type from an item name.

    Args:
        item_name: Item name (e.g., "iron_ore", "oak_log", "oak_planks")
        pattern_type: Pattern type:
            - "ore": iron_ore -> iron, deepslate_iron_ore -> iron
            - "log": oak_log -> oak, stripped_oak_log -> oak
            - "planks": oak_planks -> oak
            - "ingot": iron_ingot -> iron
            - "raw": raw_iron -> iron
            - "generic": xxx_yyy -> xxx (extract first part)

    Returns:
        Extracted base type, or None if no match
    """
    patterns = {
        "ore": r'^(?:deepslate_)?(\w+)_ore$',
        "log": r'^(?:stripped_)?(\w+)_log$',
        "planks": r'^(\w+)_planks$',
        "ingot": r'^(\w+)_ingot$',
        "raw": r'^raw_(\w+)$',
        "generic": r'^(\w+?)_\w+$'  # Non-greedy, extract first part
    }

    pattern = patterns.get(pattern_type, patterns["generic"])
    match = re.match(pattern, item_name)
    return match.group(1) if match else None


def map_output_to_input_material(
    param_name: str,
    output_item: str,
    skill_name: str,
    supported_values: Optional[List[str]] = None,
    semantic: Optional[str] = None,
    custom_logger: Optional[logging.Logger] = None
) -> str:
    """
    Map output product to input material.

    Examples:
    - craftPlanks.logType: birch_planks -> birch_log
    - craftSticks.plankType: stick -> oak_planks
    - smeltRawIron.inputItem: iron_ingot -> raw_iron

    Args:
        param_name: Parameter name (used to determine if mapping needed)
        output_item: Target product name
        skill_name: Skill name
        supported_values: List of supported values (for validation)
        semantic: Parameter semantic type (if "config", no mapping)
        custom_logger: Custom logger

    Returns:
        Mapped input material name, or original value if no mapping needed
    """
    log = custom_logger or logger
    param_lower = param_name.lower()

    # If parameter semantic is "config", don't map
    if semantic == "config":
        log.debug(f"[Mapping] Parameter '{param_name}' is config type, skipping output->input mapping")
        return output_item

    # Check if this is an input material parameter
    if not is_input_material_param(param_name):
        return output_item

    # Try mapping
    mapped_item = output_item
    for product_suffix, material_suffix in _get_product_to_material_mappings():
        if output_item.endswith(product_suffix):
            if output_item == product_suffix and material_suffix:
                mapped_item = material_suffix
                break
            else:
                base_name = output_item[:-len(product_suffix)] if product_suffix else output_item
                mapped_item = base_name + material_suffix
                break

    # If supported_values provided, validate mapping result
    if supported_values and mapped_item not in supported_values:
        possible_variants = [
            mapped_item,
            f"stripped_{mapped_item}",
            mapped_item.replace("stripped_", ""),
        ]
        for variant in possible_variants:
            if variant in supported_values:
                mapped_item = variant
                break

    # Log mapping result
    if mapped_item != output_item:
        log.info(f"[Mapping] Mapped output '{output_item}' to input '{mapped_item}' for param '{param_name}'")

    return mapped_item


# ============================================================================
# Function Parameter Parsing
# ============================================================================

def parse_function_parameters(params_str: str) -> List[str]:
    """
    Parse function parameter string, handling arrays, objects, and destructuring.

    Args:
        params_str: Parameter string, e.g.:
            - "bot, count = 1, logTypes = [\"oak_log\", \"birch_log\"]"
            - "bot, { toolName = \"wooden_axe\", count = 1 } = {}"

    Returns:
        List of parameter strings
    """
    if not params_str or not params_str.strip():
        return []

    params = []
    current_param = ""
    bracket_depth = 0  # []
    brace_depth = 0    # {}
    paren_depth = 0    # ()
    in_string = False
    string_char = None

    i = 0
    while i < len(params_str):
        char = params_str[i]

        # Handle string literals
        if char in ['"', "'"] and (i == 0 or params_str[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
            current_param += char
        elif in_string:
            current_param += char
        # Handle braces (objects/destructuring)
        elif char == '{':
            brace_depth += 1
            current_param += char
        elif char == '}':
            brace_depth -= 1
            current_param += char
        # Handle brackets (arrays)
        elif char == '[':
            bracket_depth += 1
            current_param += char
        elif char == ']':
            bracket_depth -= 1
            current_param += char
        # Handle parentheses
        elif char == '(':
            paren_depth += 1
            current_param += char
        elif char == ')':
            paren_depth -= 1
            current_param += char
        # Handle parameter separator (comma)
        elif char == ',':
            if bracket_depth == 0 and brace_depth == 0 and paren_depth == 0:
                # This is a real parameter separator
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
            else:
                # This is inside an array, object, or function call
                current_param += char
        else:
            current_param += char

        i += 1

    # Add last parameter
    if current_param.strip():
        params.append(current_param.strip())

    return params


def is_destructured_parameter(param_str: str) -> bool:
    """
    Check if parameter is a destructured parameter (starts with {).

    Args:
        param_str: Parameter string

    Returns:
        bool: True if destructured parameter
    """
    return param_str.strip().startswith('{')


# ============================================================================
# Fuel Parameter Helpers
# ============================================================================

def get_fuel_items_from_inventory(inventory: Dict[str, int]) -> List[str]:
    """
    Get fuel items from inventory, sorted by priority.

    Priority (high to low):
    1. Coal, charcoal (most common fuels)
    2. Planks (renewable, easy to get)
    3. Logs (can be used directly)
    4. Other burnable items

    Args:
        inventory: Current inventory {item_name: count}

    Returns:
        List of fuel item names sorted by priority
    """
    fuel_patterns = _get_config_contexts().get("fuel", [])
    fuel_items = []

    # Priority order
    priority_order = ["coal", "charcoal", "_planks", "_log", "stick", "wood"]

    for item in inventory:
        if inventory[item] <= 0:
            continue
        for pattern in fuel_patterns:
            if pattern in item.lower() or item.lower() == pattern.strip("_"):
                # Calculate priority (lower is better)
                priority = len(priority_order)  # Default: lowest priority
                for i, p in enumerate(priority_order):
                    if p in item.lower():
                        priority = i
                        break
                fuel_items.append((item, priority))
                break

    # Sort by priority and return item names
    fuel_items.sort(key=lambda x: x[1])
    return [item for item, _ in fuel_items]


def is_fuel_item(item: str) -> bool:
    """
    Check if an item is a fuel item.

    Args:
        item: Item name to check

    Returns:
        bool: True if item is a fuel
    """
    item_lower = item.lower()
    fuel_patterns = _get_config_contexts().get("fuel", [])

    for pattern in fuel_patterns:
        if pattern in item_lower or item_lower == pattern.strip("_"):
            return True
    return False
