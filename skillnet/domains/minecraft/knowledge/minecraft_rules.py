"""
Minecraft-Specific Mapping Rules

This module contains Minecraft-specific rules for parameter inference:
- Item type mappings (ore -> raw, planks -> log, etc.)
- Special case lookups for items that don't follow patterns
- Regex pattern mappings for complex transformations

Key transformations:
- OUTPUT -> INPUT: When we need the raw material for a product
  - oak_planks -> oak_log
  - iron_ingot -> raw_iron
  - deepslate_iron_ore -> raw_iron

- INPUT -> OUTPUT: When we need the product from raw material
  - oak_log -> oak_planks
  - raw_iron -> iron_ingot
"""

import re
from typing import Callable, Dict, List, Optional, Tuple, Union


# ============================================================================
# Regex Pattern Mappings
# ============================================================================

# Precompiled regex patterns for performance
_COMPILED_PATTERNS = {}


def _get_compiled_pattern(pattern: str) -> re.Pattern:
    """Get or compile a regex pattern."""
    if pattern not in _COMPILED_PATTERNS:
        _COMPILED_PATTERNS[pattern] = re.compile(pattern)
    return _COMPILED_PATTERNS[pattern]


# Regex mappings for complex item transformations
# Format: {name: {"pattern": regex_str, "replacement": str_or_callable}}
REGEX_MAPPINGS = {
    # Ore to raw material (handles deepslate variants)
    "ore_to_raw": {
        "pattern": r"^(?:deepslate_)?(\w+)_ore$",
        "replacement": "raw_{1}",
        "description": "Convert ore to raw material (e.g., deepslate_iron_ore -> raw_iron)"
    },

    # Log base extraction (handles stripped variants)
    "log_base": {
        "pattern": r"^(?:stripped_)?(\w+)_log$",
        "replacement": "{1}",
        "description": "Extract wood type from log (e.g., stripped_oak_log -> oak)"
    },

    # Planks base extraction
    "planks_base": {
        "pattern": r"^(\w+)_planks$",
        "replacement": "{1}",
        "description": "Extract wood type from planks (e.g., oak_planks -> oak)"
    },

    # Ingot to ore
    "ingot_to_ore": {
        "pattern": r"^(\w+)_ingot$",
        "replacement": "{1}_ore",
        "description": "Convert ingot to ore (e.g., iron_ingot -> iron_ore)"
    },

    # Raw material base extraction
    "raw_base": {
        "pattern": r"^raw_(\w+)$",
        "replacement": "{1}",
        "description": "Extract material from raw item (e.g., raw_iron -> iron)"
    },

    # Planks to log
    "planks_to_log": {
        "pattern": r"^(\w+)_planks$",
        "replacement": "{1}_log",
        "description": "Convert planks to log (e.g., oak_planks -> oak_log)"
    },

    # Log to planks
    "log_to_planks": {
        "pattern": r"^(?:stripped_)?(\w+)_log$",
        "replacement": "{1}_planks",
        "description": "Convert log to planks (e.g., oak_log -> oak_planks)"
    },

    # Block to base material
    "block_to_base": {
        "pattern": r"^(\w+)_block$",
        "replacement": "{1}",
        "description": "Extract material from block (e.g., iron_block -> iron)"
    },
}


# ============================================================================
# Special Case Mappings
# ============================================================================

# Special cases that don't follow regular patterns
# Format: {output_item: input_item}
SPECIAL_MAPPINGS = {
    # Smelting products -> raw materials
    "iron_ingot": "raw_iron",
    "gold_ingot": "raw_gold",
    "copper_ingot": "raw_copper",

    # Crafting special cases
    "stick": "oak_planks",  # Default wood type
    "torch": "coal",
    "charcoal": "oak_log",  # Smelting result

    # Food special cases
    "cooked_beef": "beef",
    "cooked_porkchop": "porkchop",
    "cooked_chicken": "chicken",
    "cooked_mutton": "mutton",
    "cooked_rabbit": "rabbit",
    "cooked_cod": "cod",
    "cooked_salmon": "salmon",
    "baked_potato": "potato",

    # Glass and smelting
    "glass": "sand",
    "stone": "cobblestone",
    "smooth_stone": "stone",

    # Bricks
    "brick": "clay_ball",
    "nether_brick": "netherrack",
}

# Reverse special mappings (input -> output)
SPECIAL_MAPPINGS_REVERSE = {v: k for k, v in SPECIAL_MAPPINGS.items()}


# ============================================================================
# Suffix Mappings
# ============================================================================

# Simple suffix replacement rules
# Format: {output_suffix: input_suffix}
SUFFIX_MAPPINGS = {
    "_planks": "_log",
    "_plank": "_log",
    "_slab": "_planks",  # Slabs come from planks
    "_stairs": "_planks",  # Stairs come from planks
    "_fence": "_planks",  # Fences come from planks
    "_door": "_planks",  # Doors come from planks
    "_boat": "_planks",  # Boats come from planks
}

# Reverse suffix mappings
SUFFIX_MAPPINGS_REVERSE = {v: k for k, v in SUFFIX_MAPPINGS.items()}


# Tool/armor suffixes for material type extraction
# CRITICAL: sorted by length descending to prevent false suffix matches
# e.g., "_pickaxe" must come before "_axe" because "wooden_pickaxe".endswith("_axe") is True
TOOL_ARMOR_SUFFIXES = [
    "_chestplate",  # 11 chars
    "_leggings",    # 9 chars
    "_pickaxe",     # 8 chars
    "_shovel",      # 7 chars
    "_helmet",      # 7 chars
    "_sword",       # 6 chars
    "_boots",       # 6 chars
    "_axe",         # 4 chars
    "_hoe",         # 4 chars
]


# ============================================================================
# Wood Type Constants
# ============================================================================

# All Minecraft wood types
WOOD_TYPES = [
    "oak", "spruce", "birch", "jungle", "acacia", "dark_oak",
    "mangrove", "cherry", "bamboo", "crimson", "warped"
]

# Nether "wood" types (technically fungi)
NETHER_WOOD_TYPES = ["crimson", "warped"]

# Regular overworld wood types
OVERWORLD_WOOD_TYPES = [w for w in WOOD_TYPES if w not in NETHER_WOOD_TYPES]


# ============================================================================
# Ore Type Constants
# ============================================================================

# Ores that produce raw materials when mined
RAW_ORE_TYPES = ["iron", "gold", "copper"]

# Ores that drop themselves or other items
DIRECT_DROP_ORES = ["diamond", "emerald", "coal", "lapis", "redstone", "quartz"]


# ============================================================================
# Transform Functions
# ============================================================================

def apply_regex_transform(item: str, mapping_name: str) -> Optional[str]:
    """
    Apply a named regex transform to an item.

    Args:
        item: Item name to transform
        mapping_name: Name of the regex mapping (e.g., "ore_to_raw")

    Returns:
        Transformed item name, or None if pattern doesn't match
    """
    if mapping_name not in REGEX_MAPPINGS:
        return None

    mapping = REGEX_MAPPINGS[mapping_name]
    pattern = _get_compiled_pattern(mapping["pattern"])
    match = pattern.match(item)

    if not match:
        return None

    replacement = mapping["replacement"]

    # Handle group substitution: {1}, {2}, etc.
    result = replacement
    for i, group in enumerate(match.groups(), 1):
        result = result.replace(f"{{{i}}}", group or "")

    return result


def apply_minecraft_transform(item: str, transform_type: str = "auto") -> Optional[str]:
    """
    Apply a Minecraft-specific transform to get input material from output.

    Args:
        item: Item name (typically an output/product)
        transform_type: Type of transform:
            - "auto": Try all transforms in priority order
            - "special": Only try special mappings
            - "regex": Only try regex mappings
            - "suffix": Only try suffix mappings
            - Specific mapping name (e.g., "ore_to_raw")

    Returns:
        Transformed item name (input material), or None if no transform applies
    """
    if transform_type == "auto":
        # Try in priority order: special -> regex -> suffix
        result = SPECIAL_MAPPINGS.get(item)
        if result:
            return result

        # Try each regex mapping
        for name in ["ore_to_raw", "planks_to_log", "ingot_to_ore"]:
            result = apply_regex_transform(item, name)
            if result:
                return result

        # Try suffix mapping
        for suffix, replacement in SUFFIX_MAPPINGS.items():
            if item.endswith(suffix):
                return item[:-len(suffix)] + replacement

        # Tool/armor material type extraction (e.g., wooden_pickaxe -> wooden)
        for suffix in TOOL_ARMOR_SUFFIXES:
            if item.endswith(suffix):
                material = item[:-len(suffix)]
                if material:  # guard against empty string
                    return material

        return None

    elif transform_type == "special":
        return SPECIAL_MAPPINGS.get(item)

    elif transform_type == "suffix":
        for suffix, replacement in SUFFIX_MAPPINGS.items():
            if item.endswith(suffix):
                return item[:-len(suffix)] + replacement
        return None

    elif transform_type == "regex":
        for name in REGEX_MAPPINGS:
            result = apply_regex_transform(item, name)
            if result:
                return result
        return None

    else:
        # Specific mapping name
        if transform_type in REGEX_MAPPINGS:
            return apply_regex_transform(item, transform_type)
        elif transform_type in SPECIAL_MAPPINGS:
            return SPECIAL_MAPPINGS.get(transform_type)
        return None


def infer_input_from_output(output_item: str) -> Optional[str]:
    """
    Infer the input/raw material needed to produce an output item.

    Examples:
        - oak_planks -> oak_log
        - iron_ingot -> raw_iron
        - cooked_beef -> beef
        - deepslate_iron_ore -> raw_iron

    Args:
        output_item: The output/product item

    Returns:
        The input/raw material, or None if unknown
    """
    return apply_minecraft_transform(output_item, "auto")


def infer_output_from_input(input_item: str) -> Optional[str]:
    """
    Infer the output/product from an input material.

    Examples:
        - oak_log -> oak_planks
        - raw_iron -> iron_ingot
        - beef -> cooked_beef

    Args:
        input_item: The input/raw material

    Returns:
        The output/product, or None if unknown
    """
    # Try reverse special mappings
    result = SPECIAL_MAPPINGS_REVERSE.get(input_item)
    if result:
        return result

    # Try log_to_planks
    result = apply_regex_transform(input_item, "log_to_planks")
    if result:
        return result

    # Try reverse suffix mappings
    for suffix, replacement in SUFFIX_MAPPINGS_REVERSE.items():
        if input_item.endswith(suffix):
            return input_item[:-len(suffix)] + replacement

    return None


