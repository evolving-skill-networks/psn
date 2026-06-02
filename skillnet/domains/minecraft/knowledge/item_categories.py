"""
Minecraft Item Categories — domain data.

Provides item taxonomy (categories, keywords, items) and safe interchange
categories for the effect matcher subsystem.

Moved from effect_matcher/_utils.py to domain layer.
"""

MINECRAFT_ITEM_CATEGORIES = {
    "log": {
        "keywords": ["log"],
        "items": ["oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log",
                  "dark_oak_log", "mangrove_log", "cherry_log"],
    },
    "plank": {
        "keywords": ["plank"],
        "items": ["oak_planks", "birch_planks", "spruce_planks", "jungle_planks",
                  "acacia_planks", "dark_oak_planks", "mangrove_planks", "cherry_planks"],
    },
    "pickaxe": {
        "keywords": ["pickaxe"],
        "items": ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "golden_pickaxe",
                  "diamond_pickaxe", "netherite_pickaxe"],
    },
    "axe": {
        "keywords": ["axe"],
        "items": ["wooden_axe", "stone_axe", "iron_axe", "golden_axe", "diamond_axe",
                  "netherite_axe"],
    },
    "sword": {
        "keywords": ["sword"],
        "items": ["wooden_sword", "stone_sword", "iron_sword", "golden_sword",
                  "diamond_sword", "netherite_sword"],
    },
    "ore": {
        "keywords": ["ore", "raw"],
        "items": [
            "iron_ore", "gold_ore", "copper_ore", "coal_ore",
            "diamond_ore", "emerald_ore", "lapis_ore", "redstone_ore",
            "deepslate_iron_ore", "deepslate_gold_ore", "deepslate_copper_ore",
            "deepslate_coal_ore", "deepslate_diamond_ore", "deepslate_emerald_ore",
            "deepslate_lapis_ore", "deepslate_redstone_ore",
            "raw_iron", "raw_gold", "raw_copper"
        ],
    },
    "gem": {
        "keywords": ["diamond", "emerald", "lapis", "quartz", "netherite"],
        "items": ["diamond", "emerald", "lapis_lazuli", "netherite_ingot", "quartz"],
    },
    "ingot": {
        "keywords": ["ingot"],
        "items": ["iron_ingot", "gold_ingot", "copper_ingot", "netherite_ingot"],
    },
    "tool": {
        "keywords": ["pickaxe", "axe", "sword", "shovel", "hoe"],
        "items": [],
    },
    "armor": {
        "keywords": ["helmet", "chestplate", "leggings", "boots"],
        "items": [],
    },
}

# Safe category whitelist: items in these categories are interchangeable
# E.g.: oak_log and birch_log are both "logs", collected the same way
# Note: ore categories are NOT in this list because different ores require different tool tiers
MINECRAFT_SAFE_INTERCHANGE_CATEGORIES = {
    "logs", "log", "wood_log", "woodlog",  # All log types
    "planks",                               # All plank types
    "wood",                                 # Generic wood items
}
