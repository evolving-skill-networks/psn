"""
Game Knowledge - Blocks
Block property knowledge: hardness, tool requirements, tool tiers

Migrated from minecraft_env.py and api_behaviors.py
All solution_hint fields removed
"""

from typing import Dict, List, Tuple, Optional
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# Block Hardness Data
# ============================================================

BLOCK_HARDNESS: Dict[str, float] = {
    # Stone variants
    "stone": 1.5,
    "cobblestone": 2.0,
    "deepslate": 3.0,
    "obsidian": 50.0,
    "bedrock": -1.0,  # Unbreakable

    # Wood variants - Overworld
    "oak_log": 2.0,
    "birch_log": 2.0,
    "spruce_log": 2.0,
    "jungle_log": 2.0,
    "acacia_log": 2.0,
    "dark_oak_log": 2.0,
    "oak_planks": 2.0,
    # 1.19+ Mangrove wood
    "mangrove_log": 2.0,
    "mangrove_wood": 2.0,
    "stripped_mangrove_log": 2.0,
    "stripped_mangrove_wood": 2.0,
    "mangrove_planks": 2.0,
    # Nether wood (1.16+) - fireproof
    "crimson_stem": 2.0,
    "warped_stem": 2.0,
    "stripped_crimson_stem": 2.0,
    "stripped_warped_stem": 2.0,
    "crimson_planks": 2.0,
    "warped_planks": 2.0,
    "crimson_hyphae": 2.0,
    "warped_hyphae": 2.0,
    # 1.20 Bamboo and Cherry
    "bamboo_block": 2.0,
    "cherry_log": 2.0,
    "cherry_planks": 2.0,

    # Soil variants
    "dirt": 0.5,
    "grass_block": 0.6,
    "sand": 0.5,
    "gravel": 0.6,
    "clay": 0.6,
    # 1.19 Mud blocks
    "mud": 0.5,
    "packed_mud": 1.0,
    "mud_bricks": 1.5,
    "muddy_mangrove_roots": 0.7,

    # Ores
    "coal_ore": 3.0,
    "iron_ore": 3.0,
    "copper_ore": 3.0,
    "gold_ore": 3.0,
    "diamond_ore": 3.0,
    "lapis_ore": 3.0,
    "redstone_ore": 3.0,
    "emerald_ore": 3.0,
    "deepslate_coal_ore": 4.5,
    "deepslate_iron_ore": 4.5,
    "deepslate_copper_ore": 4.5,
    "deepslate_gold_ore": 4.5,
    "deepslate_diamond_ore": 4.5,
    "deepslate_lapis_ore": 4.5,
    "deepslate_redstone_ore": 4.5,
    "deepslate_emerald_ore": 4.5,

    # Other workstations
    "crafting_table": 2.5,
    "furnace": 3.5,
    "chest": 2.5,

    # Deepslate variants (1.17+)
    "cobbled_deepslate": 3.5,
    "deepslate_bricks": 3.5,
    "deepslate_tiles": 3.5,
    "chiseled_deepslate": 3.5,
    "polished_deepslate": 3.5,
    "tuff": 1.5,
    "calcite": 0.75,

    # 1.19 Wild Update - Sculk blocks (Deep Dark)
    "sculk": 0.2,
    "sculk_vein": 0.2,
    "sculk_sensor": 1.5,
    "sculk_shrieker": 3.0,
    "sculk_catalyst": 3.0,
    "reinforced_deepslate": -1.0,  # Unbreakable in survival (ancient city)

    # 1.19 Wild Update - Froglight (light level 15)
    "ochre_froglight": 0.3,
    "verdant_froglight": 0.3,
    "pearlescent_froglight": 0.3,
}


# ============================================================
# Tool Effectiveness Data
# ============================================================

TOOL_EFFECTIVENESS: Dict[str, List[str]] = {
    # Pickaxes - tier determines what drops
    "wooden_pickaxe": ["stone", "cobblestone", "coal_ore"],
    "stone_pickaxe": ["stone", "cobblestone", "coal_ore", "iron_ore", "copper_ore", "lapis_ore"],
    "iron_pickaxe": [
        "stone", "cobblestone", "coal_ore", "iron_ore", "copper_ore", "lapis_ore",
        "diamond_ore", "gold_ore", "redstone_ore", "emerald_ore"
    ],
    "diamond_pickaxe": [
        "stone", "cobblestone", "coal_ore", "iron_ore", "copper_ore", "lapis_ore",
        "diamond_ore", "gold_ore", "redstone_ore", "emerald_ore", "obsidian", "ancient_debris"
    ],
    "netherite_pickaxe": [
        "stone", "cobblestone", "coal_ore", "iron_ore", "copper_ore", "lapis_ore",
        "diamond_ore", "gold_ore", "redstone_ore", "emerald_ore", "obsidian", "ancient_debris"
    ],

    # Axes - all wood types (includes 1.19+ mangrove, nether stems)
    "wooden_axe": [
        "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log",
        "mangrove_log", "mangrove_wood", "stripped_mangrove_log", "stripped_mangrove_wood",
        "crimson_stem", "warped_stem", "stripped_crimson_stem", "stripped_warped_stem",
        "bamboo_block", "cherry_log",
    ],
    "stone_axe": [
        "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log",
        "mangrove_log", "mangrove_wood", "stripped_mangrove_log", "stripped_mangrove_wood",
        "crimson_stem", "warped_stem", "stripped_crimson_stem", "stripped_warped_stem",
        "bamboo_block", "cherry_log",
    ],
    "iron_axe": [
        "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log",
        "mangrove_log", "mangrove_wood", "stripped_mangrove_log", "stripped_mangrove_wood",
        "crimson_stem", "warped_stem", "stripped_crimson_stem", "stripped_warped_stem",
        "bamboo_block", "cherry_log",
    ],
    "diamond_axe": [
        "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log",
        "mangrove_log", "mangrove_wood", "stripped_mangrove_log", "stripped_mangrove_wood",
        "crimson_stem", "warped_stem", "stripped_crimson_stem", "stripped_warped_stem",
        "bamboo_block", "cherry_log",
    ],

    # Shovels - includes 1.19 mud blocks
    "wooden_shovel": ["dirt", "grass_block", "sand", "gravel", "clay", "mud"],
    "stone_shovel": ["dirt", "grass_block", "sand", "gravel", "clay", "mud"],
    "iron_shovel": ["dirt", "grass_block", "sand", "gravel", "clay", "mud"],
    "diamond_shovel": ["dirt", "grass_block", "sand", "gravel", "clay", "mud"],

    # Hoes - best tool for sculk blocks (1.19+)
    "wooden_hoe": ["sculk", "sculk_vein", "sculk_sensor", "sculk_shrieker", "sculk_catalyst"],
    "stone_hoe": ["sculk", "sculk_vein", "sculk_sensor", "sculk_shrieker", "sculk_catalyst"],
    "iron_hoe": ["sculk", "sculk_vein", "sculk_sensor", "sculk_shrieker", "sculk_catalyst"],
    "diamond_hoe": ["sculk", "sculk_vein", "sculk_sensor", "sculk_shrieker", "sculk_catalyst"],
}


# ============================================================
# Tool Tier System
# ============================================================

TOOL_TIERS: Dict[str, int] = {
    "wooden_pickaxe": 0,
    "golden_pickaxe": 0,
    "stone_pickaxe": 1,
    "iron_pickaxe": 2,
    "diamond_pickaxe": 3,
    "netherite_pickaxe": 4,
}

BLOCK_HARVEST_TIER: Dict[str, int] = {
    # Tier 0: Any pickaxe
    "coal_ore": 0,
    "deepslate_coal_ore": 0,
    "nether_quartz_ore": 0,

    # Tier 1: Stone pickaxe or better
    "iron_ore": 1,
    "deepslate_iron_ore": 1,
    "copper_ore": 1,
    "deepslate_copper_ore": 1,
    "lapis_ore": 1,
    "deepslate_lapis_ore": 1,

    # Tier 2: Iron pickaxe or better
    "diamond_ore": 2,
    "deepslate_diamond_ore": 2,
    "gold_ore": 2,
    "deepslate_gold_ore": 2,
    "emerald_ore": 2,
    "deepslate_emerald_ore": 2,
    "redstone_ore": 2,
    "deepslate_redstone_ore": 2,

    # Tier 3: Diamond pickaxe or better
    "obsidian": 3,
    "ancient_debris": 3,
}


# ============================================================
# Knowledge Items
# ============================================================

BLOCK_KNOWLEDGE: List[KnowledgeItem] = [
    # Tool tier system knowledge
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="tool_tier_hierarchy",
        fact=(
            "Minecraft pickaxes have tiers: Tier 0 (wooden, golden), Tier 1 (stone), "
            "Tier 2 (iron), Tier 3 (diamond), Tier 4 (netherite). "
            "Each block requires a minimum tier to drop items when mined."
        ),
        keywords=["tool", "tier", "pickaxe", "mining", "level"],
        conditions=["when mining any ore or hard block"],
        implications=[
            "using lower tier than required destroys block without drops",
            "block breaks normally but no items are obtained",
            "mining animation completes but inventory unchanged",
        ],
    ),

    # Specific block requirements
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="diamond_ore_harvest",
        fact=(
            "Diamond ore requires iron pickaxe (Tier 2) or better to drop diamonds. "
            "Stone pickaxe or lower will break the block but drop nothing. "
            "Applies to both diamond_ore and deepslate_diamond_ore."
        ),
        keywords=["diamond", "ore", "pickaxe", "iron", "harvest", "tier"],
        conditions=["mining diamond_ore", "mining deepslate_diamond_ore"],
        implications=[
            "stone pickaxe breaks block with no drops",
            "wooden pickaxe breaks block with no drops",
            "need iron pickaxe minimum for diamonds",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="gold_ore_harvest",
        fact=(
            "Gold ore requires iron pickaxe (Tier 2) or better to drop gold. "
            "Stone pickaxe or lower will break the block but drop nothing."
        ),
        keywords=["gold", "ore", "pickaxe", "iron", "harvest"],
        conditions=["mining gold_ore", "mining deepslate_gold_ore"],
        implications=[
            "stone pickaxe destroys gold ore without drops",
            "need iron pickaxe minimum",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="iron_ore_harvest",
        fact=(
            "Iron ore requires stone pickaxe (Tier 1) or better to drop raw iron. "
            "Wooden pickaxe will break the block but drop nothing."
        ),
        keywords=["iron", "ore", "pickaxe", "stone", "harvest"],
        conditions=["mining iron_ore", "mining deepslate_iron_ore"],
        implications=[
            "wooden pickaxe destroys iron ore without drops",
            "need stone pickaxe minimum",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="obsidian_harvest",
        fact=(
            "Obsidian requires diamond pickaxe (Tier 3) or better to drop. "
            "Iron pickaxe or lower will break the block but drop nothing. "
            "Mining takes 9.4 seconds with diamond pickaxe, 50 seconds with lower tiers."
        ),
        keywords=["obsidian", "pickaxe", "diamond", "harvest", "slow"],
        conditions=["mining obsidian"],
        implications=[
            "iron pickaxe destroys obsidian without drops",
            "very slow mining even with diamond pickaxe",
            "must be created from lava + water, rarely found naturally",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="coal_ore_harvest",
        fact=(
            "Coal ore can be mined with any pickaxe (Tier 0+). "
            "This includes wooden and golden pickaxes."
        ),
        keywords=["coal", "ore", "pickaxe", "wooden", "any"],
        conditions=["mining coal_ore", "mining deepslate_coal_ore"],
        implications=[
            "easiest ore to mine",
            "no special tool required beyond basic pickaxe",
        ],
    ),

    # Wood requires axe for efficiency
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="wood_harvest",
        fact=(
            "Wood logs can be broken by hand but axes are 2-3x faster. "
            "All wood types (oak, birch, spruce, jungle, acacia, dark_oak) have same behavior. "
            "No tier requirement - any axe works equally for drops."
        ),
        keywords=["wood", "log", "axe", "tree", "oak", "birch", "spruce"],
        conditions=["mining any log type"],
        implications=[
            "can mine without tools but very slow",
            "axe significantly speeds up harvesting",
            "all wood types behave the same",
        ],
    ),

    # Soil requires shovel
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="soil_harvest",
        fact=(
            "Dirt, sand, gravel, and clay break fastest with shovels. "
            "Can be broken by hand but shovels are 2-4x faster. "
            "No tier requirement for drops."
        ),
        keywords=["dirt", "sand", "gravel", "clay", "shovel", "soil"],
        conditions=["mining dirt", "mining sand", "mining gravel"],
        implications=[
            "shovel speeds up harvesting",
            "any shovel tier works",
            "gravel has chance to drop flint",
        ],
    ),

    # Special blocks
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="bedrock_unbreakable",
        fact=(
            "Bedrock cannot be broken in survival mode by any means. "
            "In the Overworld, bedrock generates as a LAYER from Y=-64 to Y=-60: "
            "100% bedrock at Y=-64, decreasing randomly to ~0% at Y=-60. "
            "Between Y=-63 and Y=-61, bedrock and deepslate coexist randomly. "
            "Also found as Nether ceiling (Y=127) and floor (Y=0-4)."
        ),
        keywords=["bedrock", "unbreakable", "indestructible", "Y=-64", "Y=-60",
                  "world bottom", "layer", "boundary"],
        conditions=["attempting to mine bedrock", "mining near world bottom",
                    "exploring below Y=-60", "strip mining deep underground"],
        logical_implications=[
            "Any block at Y=-64 is guaranteed bedrock",
            "Between Y=-63 and Y=-61, some blocks are bedrock and some are not",
            "No bedrock exists above Y=-60 in the Overworld",
            "Mining or strip mining below Y=-60 risks hitting unbreakable bedrock",
        ],
        constraints=[
            "bot.dig() on bedrock does nothing (times out)",
            "Pathfinder cannot path through bedrock",
            "No tool or enchantment can break bedrock in survival",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="silk_touch_blocks",
        fact=(
            "Ice, glass, glowstone, and certain other blocks require Silk Touch enchantment to collect. "
            "Without Silk Touch: ice turns to water, glass drops nothing, glowstone drops 2-4 dust."
        ),
        keywords=["silk", "touch", "ice", "glass", "glowstone", "enchantment"],
        conditions=["mining ice", "mining glass", "mining glowstone"],
        implications=[
            "ice disappears without silk touch",
            "glass shatters without silk touch",
            "need enchanted tool to collect",
        ],
    ),

    # 1.19 Wild Update - Mangrove Wood
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="mangrove_wood_types",
        fact=(
            "Mangrove trees in mangrove swamps drop mangrove_log. "
            "Can be stripped with axe to stripped_mangrove_log. "
            "Crafts to mangrove_planks (4 planks per log). Same hardness as other logs."
        ),
        keywords=["mangrove", "log", "wood", "swamp", "planks", "strip"],
        conditions=["harvesting in mangrove_swamp"],
        implications=[
            "same hardness as other logs",
            "requires axe for efficiency",
            "4 planks per log like all wood types",
        ],
    ),

    # Nether Wood
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="nether_wood_types",
        fact=(
            "Crimson and warped stems are found in the Nether (crimson_forest and warped_forest biomes). "
            "They are FIREPROOF and do not burn. Craft to crimson_planks/warped_planks. "
            "Functionally identical to overworld wood except fire resistance."
        ),
        keywords=["crimson", "warped", "stem", "nether", "fireproof", "planks", "hyphae"],
        conditions=["harvesting in nether", "need fireproof wood"],
        implications=[
            "does not burn - ideal for nether builds",
            "same tool tier as overworld wood",
            "same crafting recipes as regular wood",
        ],
    ),

    # Deepslate Properties
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="deepslate_properties",
        fact=(
            "Deepslate generates below Y=8, replacing stone. "
            "Harder than stone (3.0 vs 1.5 hardness). Requires pickaxe to harvest. "
            "Deepslate ore variants have same drops as regular ores but take longer to mine."
        ),
        keywords=["deepslate", "underground", "deep", "hard", "pickaxe", "Y=8"],
        conditions=["mining below Y=8", "mining deepslate"],
        implications=[
            "slower to mine than stone",
            "same drops as regular ores",
            "cobbled_deepslate is the broken form",
        ],
    ),

    # 1.19 Wild Update - Sculk Blocks
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="sculk_blocks",
        fact=(
            "Sculk blocks generate in Deep Dark biome below Y=0. "
            "Sculk sensors detect vibrations within 8 blocks. "
            "Sculk shriekers summon Warden after 4 activations. "
            "Best harvested with hoe. Use Silk Touch to prevent shrieker activation."
        ),
        keywords=["sculk", "sensor", "shrieker", "deep dark", "vibration", "warden", "hoe"],
        conditions=["exploring deep dark", "approaching ancient city"],
        implications=[
            "avoid making noise near sensors",
            "silk touch prevents shrieker activation",
            "4 shrieks summon warden",
            "hoe is fastest tool for sculk",
        ],
    ),

    # 1.19 Wild Update - Froglight
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="froglight_acquisition",
        fact=(
            "Froglights are light blocks (level 15) obtained when frogs eat small magma cubes. "
            "Three variants: ochre (temperate frog), verdant (cold frog), pearlescent (warm frog). "
            "Cannot be crafted - must be obtained from frog + magma cube interaction."
        ),
        keywords=["froglight", "frog", "magma cube", "light", "ochre", "verdant", "pearlescent"],
        conditions=["obtaining froglight", "need light source"],
        implications=[
            "need frog + magma cube",
            "frog variant determines color",
            "non-renewable without breeding frogs",
        ],
    ),

    # 1.19 Wild Update - Mud
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="mud_creation",
        fact=(
            "Mud is created by using water bottle on dirt block. "
            "Found naturally in mangrove swamps. "
            "Can be dried on pointed dripstone to make clay (renewable clay source). "
            "Crafts into packed_mud, then mud_bricks for building."
        ),
        keywords=["mud", "dirt", "water", "bottle", "clay", "dripstone", "renewable"],
        conditions=["need mud", "need renewable clay"],
        implications=[
            "water bottle + dirt = mud",
            "dripstone dries mud to clay",
            "only renewable clay method in game",
        ],
    ),

    # Reinforced Deepslate
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="reinforced_deepslate_unbreakable",
        fact=(
            "Reinforced deepslate is found in ancient cities and cannot be broken in survival mode. "
            "Similar to bedrock in that it blocks all mining attempts. "
            "Used to protect ancient city structures."
        ),
        keywords=["reinforced", "deepslate", "unbreakable", "ancient city"],
        conditions=["attempting to mine reinforced_deepslate"],
        implications=[
            "cannot be mined in survival",
            "marks ancient city boundaries",
            "similar to bedrock",
        ],
    ),

    # Block drops vs block names
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="block_drops_vs_block_names",
        fact=(
            "Many blocks drop items with DIFFERENT names than the block itself: "
            "stone → cobblestone, grass_block → dirt, "
            "diamond_ore → diamond, coal_ore → coal, lapis_ore → lapis_lazuli, redstone_ore → redstone, "
            "iron_ore → raw_iron (1.17+), gold_ore → raw_gold (1.17+), copper_ore → raw_copper (1.17+), "
            "leaves → saplings (rarely) and sticks. "
            "Silk Touch enchantment causes blocks to drop themselves instead."
        ),
        keywords=["drop", "drops", "item", "block", "name", "different", "cobblestone",
                  "raw_iron", "raw_gold", "raw_copper", "diamond", "coal", "lapis",
                  "inventory", "count", "check", "mined"],
        conditions=["checking inventory after mining", "item count not increasing",
                    "wrong item name in inventory check"],
        logical_implications=[
            "Inventory count check must use the DROPPED item name, not the block name",
            "After mining stone, check for cobblestone in inventory",
            "After mining iron_ore, check for raw_iron in inventory (1.17+)",
            "After mining diamond_ore, check for diamond in inventory",
            "Silk Touch changes drops to the block itself",
        ],
    ),
]


# ============================================================
# Helper Functions
# ============================================================

def get_required_tool_tier(block_name: str) -> Optional[int]:
    """
    Get the minimum tool tier required to harvest a block.

    Returns:
        int: Minimum tier (0-4), or None if no pickaxe required
    """
    # Check deepslate variants
    if block_name.startswith("deepslate_"):
        base_name = block_name.replace("deepslate_", "")
        if base_name in BLOCK_HARVEST_TIER:
            return BLOCK_HARVEST_TIER[block_name]

    return BLOCK_HARVEST_TIER.get(block_name)


def get_tool_for_block(block_name: str) -> Optional[str]:
    """
    Get the minimum tier tool for mining a block.

    Args:
        block_name: Name of the block

    Returns:
        Tool name, or None if no tool required
    """
    tier = get_required_tool_tier(block_name)
    if tier is None:
        return None

    tier_to_tool = {
        0: "wooden_pickaxe",
        1: "stone_pickaxe",
        2: "iron_pickaxe",
        3: "diamond_pickaxe",
        4: "netherite_pickaxe",
    }

    return tier_to_tool.get(tier)


def can_tool_harvest(tool_name: str, block_name: str) -> bool:
    """
    Check if a tool can harvest a block (get drops).

    Args:
        tool_name: Name of the tool (e.g., "iron_pickaxe")
        block_name: Name of the block

    Returns:
        True if the tool tier is sufficient for drops
    """
    required_tier = get_required_tool_tier(block_name)
    if required_tier is None:
        return True  # No tier requirement

    tool_tier = TOOL_TIERS.get(tool_name, -1)
    return tool_tier >= required_tier


def get_all_block_knowledge() -> List[KnowledgeItem]:
    """Get all block-related knowledge items."""
    return BLOCK_KNOWLEDGE.copy()
