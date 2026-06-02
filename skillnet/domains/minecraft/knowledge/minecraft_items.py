"""
Minecraft Items Constants - Unified resource definitions for PSN

This module provides centralized constants for Minecraft item groups, resource aliases,
and item-to-block mappings. These are used by:
- PSN Curriculum's ResourceTracker for inventory tracking
- Skill Graph Optimizer for resource validation and analysis
- Any other component needing Minecraft item knowledge

Benefits:
- Single source of truth for item definitions
- Eliminates duplicate definitions across modules
- Easy to update when Minecraft versions change
"""

from typing import Dict, List, Optional, Set


# =============================================================================
# ITEM GROUPS - Functionally equivalent items that can substitute for each other
# =============================================================================

ITEM_GROUPS: Dict[str, List[str]] = {
    # All log types are interchangeable for crafting planks
    "logs": [
        "oak_log", "birch_log", "spruce_log", "jungle_log",
        "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log",
        # Stripped variants
        "stripped_oak_log", "stripped_birch_log", "stripped_spruce_log",
        "stripped_jungle_log", "stripped_acacia_log", "stripped_dark_oak_log",
    ],

    # All plank types are interchangeable for crafting
    "planks": [
        "oak_planks", "birch_planks", "spruce_planks", "jungle_planks",
        "acacia_planks", "dark_oak_planks", "mangrove_planks", "cherry_planks",
        "crimson_planks", "warped_planks", "bamboo_planks",
    ],

    # Cooked food items - any provides similar nutrition
    "cooked_food": [
        "cooked_beef", "cooked_porkchop", "cooked_chicken", "cooked_mutton",
        "cooked_rabbit", "cooked_cod", "cooked_salmon", "baked_potato",
        "bread", "golden_carrot", "golden_apple",
    ],

    # Fuel sources - all can power furnaces
    # When ENABLE_WOOD_AS_FUEL is True, logs are also counted as fuel
    # (matches Minecraft game logic where logs can be used as furnace fuel)
    "fuel": [
        "coal", "charcoal", "coal_block",
    ],  # NOTE: Extended dynamically below if RobustnessConfig.ENABLE_WOOD_AS_FUEL

    # Raw ores that smelt to iron
    "raw_iron_source": [
        "raw_iron", "iron_ore", "deepslate_iron_ore",
    ],

    # Raw ores that smelt to gold
    "raw_gold_source": [
        "raw_gold", "gold_ore", "deepslate_gold_ore", "nether_gold_ore",
    ],

    # Pickaxes of any tier
    "pickaxes": [
        "wooden_pickaxe", "stone_pickaxe", "iron_pickaxe",
        "golden_pickaxe", "diamond_pickaxe", "netherite_pickaxe",
    ],

    # Axes of any tier
    "axes": [
        "wooden_axe", "stone_axe", "iron_axe",
        "golden_axe", "diamond_axe", "netherite_axe",
    ],

    # Swords of any tier
    "swords": [
        "wooden_sword", "stone_sword", "iron_sword",
        "golden_sword", "diamond_sword", "netherite_sword",
    ],

    # Shovels of any tier
    "shovels": [
        "wooden_shovel", "stone_shovel", "iron_shovel",
        "golden_shovel", "diamond_shovel", "netherite_shovel",
    ],

    # Hoes of any tier
    "hoes": [
        "wooden_hoe", "stone_hoe", "iron_hoe",
        "golden_hoe", "diamond_hoe", "netherite_hoe",
    ],
}

# Group aliases: when counting a group, also include items from these other groups.
# This avoids duplicating items in ITEM_GROUPS (which breaks ITEM_TO_GROUP reverse mapping).
# Instead, the counting logic (_get_group_count) checks aliases at runtime.
#
# In Minecraft, logs can be used as furnace fuel (1 log smelts 1.5 items).
# When ENABLE_WOOD_AS_FUEL is True, counting "fuel" also counts "logs".
GROUP_ALIASES: Dict[str, List[str]] = {}

try:
    from skillnet.config.robustness_config import RobustnessConfig
    if RobustnessConfig.ENABLE_WOOD_AS_FUEL:
        GROUP_ALIASES["fuel"] = ["logs"]  # "fuel" count also includes "logs" group
except ImportError:
    pass  # Config not available, no aliases

# Reverse mapping: item -> group name (for quick lookup)
ITEM_TO_GROUP: Dict[str, str] = {}
for _group_name, _items in ITEM_GROUPS.items():
    for _item in _items:
        ITEM_TO_GROUP[_item] = _group_name

# ITEM_CATEGORIES - Alias for ITEM_GROUPS with additional singular->plural mappings
# Bug 4 fix: used for category matching during effect validation
ITEM_CATEGORIES: Dict[str, List[str]] = {
    **ITEM_GROUPS,
    # Singular-form aliases
    "log": ITEM_GROUPS["logs"],
    "plank": ITEM_GROUPS["planks"],
    "pickaxe": ITEM_GROUPS["pickaxes"],
    "axe": ITEM_GROUPS["axes"],
    "sword": ITEM_GROUPS["swords"],
    "shovel": ITEM_GROUPS["shovels"],
    "hoe": ITEM_GROUPS["hoes"],
}


# =============================================================================
# RESOURCE ALIASES - Mapping from skill name parts to actual resource names
# Used by optimizer to extract resources from skill names like "mineCoal"
# =============================================================================

RESOURCE_ALIASES: Dict[str, List[str]] = {
    # Stone variants
    "cobblestone": ["cobblestone", "stone"],
    "stone": ["stone", "cobblestone"],

    # Ores and their forms
    "coal": ["coal", "coal_ore", "charcoal"],
    "iron": ["iron", "iron_ore", "raw_iron", "iron_ingot"],
    "ironore": ["iron_ore", "deepslate_iron_ore"],
    "gold": ["gold", "gold_ore", "raw_gold", "gold_ingot"],
    "goldore": ["gold_ore", "deepslate_gold_ore", "nether_gold_ore"],
    "diamond": ["diamond", "diamond_ore"],
    "diamondore": ["diamond_ore", "deepslate_diamond_ore"],
    "emerald": ["emerald", "emerald_ore"],
    "emeraldore": ["emerald_ore", "deepslate_emerald_ore"],
    "copper": ["copper", "copper_ore", "raw_copper", "copper_ingot"],
    "copperore": ["copper_ore", "deepslate_copper_ore"],
    "lapis": ["lapis_lazuli", "lapis_ore"],
    "lapisore": ["lapis_ore", "deepslate_lapis_ore"],
    "redstone": ["redstone", "redstone_ore"],
    "redstoneore": ["redstone_ore", "deepslate_redstone_ore"],

    # Wood types (use logs group for dynamic lookup)
    "logs": ITEM_GROUPS["logs"],
    "log": ITEM_GROUPS["logs"],
    "wood_log": ITEM_GROUPS["logs"],  # LLM sometimes extracts "wood_log" as a generic name
    "woodlog": ITEM_GROUPS["logs"],
    "wood": ITEM_GROUPS["logs"] + ITEM_GROUPS["planks"],
    "oaklog": ["oak_log"],
    "birchlog": ["birch_log"],
    "sprucelog": ["spruce_log"],
    "junglelog": ["jungle_log"],
    "acacialog": ["acacia_log"],
    "darkoaklog": ["dark_oak_log"],

    # Planks
    "planks": ITEM_GROUPS["planks"],
    "oakplanks": ["oak_planks"],
    "birchplanks": ["birch_planks"],
    "spruceplanks": ["spruce_planks"],

    # Crafting stations
    "furnace": ["furnace"],
    "craftingtable": ["crafting_table"],
    "anvil": ["anvil"],
    "enchantingtable": ["enchanting_table"],
    "smithingtable": ["smithing_table"],
    "stonecutter": ["stonecutter"],

    # Other common items
    "stick": ["stick"],
    "torch": ["torch"],
    "chest": ["chest"],
    "bed": ["red_bed", "white_bed", "blue_bed", "green_bed", "yellow_bed",
            "black_bed", "brown_bed", "cyan_bed", "gray_bed", "light_blue_bed",
            "light_gray_bed", "lime_bed", "magenta_bed", "orange_bed", "pink_bed", "purple_bed"],
}


# =============================================================================
# ITEM TO BLOCK MAPPING - Maps item names to their corresponding block names
# Used to fix common code errors like blocksByName['coal'] -> blocksByName['coal_ore']
# =============================================================================

ITEM_TO_BLOCK_MAP: Dict[str, str] = {
    # Ores
    "coal": "coal_ore",
    "diamond": "diamond_ore",
    "emerald": "emerald_ore",
    "iron": "iron_ore",
    "gold": "gold_ore",
    "copper": "copper_ore",
    "redstone": "redstone_ore",
    "lapis": "lapis_ore",
    "lapis_lazuli": "lapis_ore",

    # Raw items to ore blocks
    "raw_iron": "iron_ore",
    "raw_gold": "gold_ore",
    "raw_copper": "copper_ore",
}


# =============================================================================
# BLOCK SETS - Common block categories for validation
# =============================================================================

# Common minable blocks (subset for fallback validation)
COMMON_BLOCKS: Set[str] = {
    # Ores
    "coal_ore", "iron_ore", "gold_ore", "diamond_ore", "emerald_ore",
    "copper_ore", "lapis_ore", "redstone_ore",
    "deepslate_coal_ore", "deepslate_iron_ore", "deepslate_gold_ore",
    "deepslate_diamond_ore", "deepslate_emerald_ore", "deepslate_copper_ore",
    "deepslate_lapis_ore", "deepslate_redstone_ore",
    "nether_gold_ore", "nether_quartz_ore", "ancient_debris",

    # Stone variants
    "stone", "cobblestone", "deepslate", "cobbled_deepslate",
    "granite", "diorite", "andesite", "tuff", "calcite",
    "blackstone", "basalt", "netherrack", "end_stone",

    # Dirt/Sand variants
    "dirt", "grass_block", "sand", "gravel", "clay",
    "red_sand", "soul_sand", "soul_soil",

    # Wood logs (from ITEM_GROUPS)
    *ITEM_GROUPS["logs"],

    # Wood planks (from ITEM_GROUPS)
    *ITEM_GROUPS["planks"],

    # Crafting blocks
    "crafting_table", "furnace", "blast_furnace", "smoker",
    "anvil", "enchanting_table", "brewing_stand",
    "smithing_table", "stonecutter", "loom", "cartography_table",

    # Storage
    "chest", "barrel", "ender_chest", "shulker_box",

    # Misc
    "obsidian", "crying_obsidian", "glowstone", "sea_lantern",
}

# Common items (subset for fallback validation)
COMMON_ITEMS: Set[str] = {
    # Ores/Minerals
    "coal", "charcoal", "diamond", "emerald", "lapis_lazuli",
    "raw_iron", "raw_gold", "raw_copper",
    "iron_ingot", "gold_ingot", "copper_ingot", "netherite_ingot",
    "iron_nugget", "gold_nugget",
    "redstone", "glowstone_dust", "quartz",

    # Tools (from ITEM_GROUPS)
    *ITEM_GROUPS["pickaxes"],
    *ITEM_GROUPS["axes"],
    *ITEM_GROUPS["swords"],
    *ITEM_GROUPS["shovels"],
    *ITEM_GROUPS["hoes"],

    # Crafting materials
    "stick", "string", "leather", "feather", "flint",
    "bone", "bone_meal", "slime_ball", "ender_pearl", "blaze_rod",

    # Food (from ITEM_GROUPS)
    *ITEM_GROUPS["cooked_food"],
    "apple", "golden_apple", "enchanted_golden_apple",
    "carrot", "potato", "beetroot", "melon_slice", "sweet_berries",
    "raw_beef", "raw_porkchop", "raw_chicken", "raw_mutton",
    "raw_cod", "raw_salmon",

    # Misc
    "torch", "bucket", "water_bucket", "lava_bucket",
    "arrow", "bow", "crossbow", "shield",
    "book", "paper", "map", "compass", "clock",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_group_items(group_name: str) -> List[str]:
    """Get all items in a group, or empty list if group doesn't exist."""
    return ITEM_GROUPS.get(group_name, [])


def get_item_group(item_name: str) -> str | None:
    """Get the group name for an item, or None if item is not in any group."""
    return ITEM_TO_GROUP.get(item_name)


def is_in_group(item_name: str, group_name: str) -> bool:
    """Check if an item belongs to a specific group."""
    return item_name in ITEM_GROUPS.get(group_name, [])


def get_resource_aliases(resource_key: str) -> List[str]:
    """Get all possible resource names for a key, or [key] if no aliases."""
    return RESOURCE_ALIASES.get(resource_key, [resource_key])


def get_block_for_item(item_name: str) -> str | None:
    """Get the corresponding block name for an item, or None if no mapping."""
    return ITEM_TO_BLOCK_MAP.get(item_name)


def is_valid_block(block_name: str) -> bool:
    """Check if a block name is in the common blocks set."""
    return block_name in COMMON_BLOCKS


def is_valid_item(item_name: str) -> bool:
    """Check if an item name is in the common items set."""
    return item_name in COMMON_ITEMS


def get_all_known_blocks() -> Set[str]:
    """Get all known block names."""
    return COMMON_BLOCKS.copy()


def get_all_known_items() -> Set[str]:
    """Get all known item names."""
    return COMMON_ITEMS.copy()


# =============================================================================
# ARMOR TIER SYSTEM - armor tiers (armor only, not tools)
# =============================================================================

# Armor material tiers (higher index is better)
ARMOR_TIER_ORDER = ["leather", "golden", "chainmail", "iron", "diamond", "netherite"]

# Armor slot → suffix mapping
ARMOR_SLOTS = {"head": "helmet", "torso": "chestplate", "legs": "leggings", "feet": "boots"}

# Auto-built: item_name → tier_index (armor only)
ARMOR_TIER: Dict[str, int] = {}
for _idx, _mat in enumerate(ARMOR_TIER_ORDER):
    for _suffix in ARMOR_SLOTS.values():
        ARMOR_TIER[f"{_mat}_{_suffix}"] = _idx

# Auto-built: item_name → equip slot (armor only)
ARMOR_TO_SLOT: Dict[str, str] = {}
for _slot, _suffix in ARMOR_SLOTS.items():
    for _mat in ARMOR_TIER_ORDER:
        ARMOR_TO_SLOT[f"{_mat}_{_suffix}"] = _slot


def get_armor_tier(item_name: str) -> int:
    """Get the tier of an armor item; returns -1 for non-armor."""
    return ARMOR_TIER.get(item_name, -1)


def is_armor_item(item_name: str) -> bool:
    """Check whether this is an armor item."""
    return item_name in ARMOR_TIER


def get_armor_slot(item_name: str) -> Optional[str]:
    """Get the slot for an armor item; returns None for non-armor."""
    return ARMOR_TO_SLOT.get(item_name)
