"""
Game Knowledge - Resources
Resource spawning knowledge: ore Y-axis distribution, biome resources

Migrated from minecraft_env.py and api_behaviors.py
Merged duplicate data, all solution_hint fields removed
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# Ore Spawn Data (Merged from both sources)
# ============================================================

@dataclass
class OreSpawnRange:
    """Ore spawning range - pure data structure"""
    min_y: int
    max_y: int
    optimal_y: int
    biome_restrictions: List[str] = field(default_factory=list)


ORE_SPAWN_RANGES: Dict[str, OreSpawnRange] = {
    # Diamond
    "diamond_ore": OreSpawnRange(min_y=-64, max_y=16, optimal_y=-59),
    "deepslate_diamond_ore": OreSpawnRange(min_y=-64, max_y=0, optimal_y=-59),

    # Iron
    "iron_ore": OreSpawnRange(min_y=-64, max_y=72, optimal_y=16),
    "deepslate_iron_ore": OreSpawnRange(min_y=-64, max_y=0, optimal_y=-32),

    # Gold
    "gold_ore": OreSpawnRange(min_y=-64, max_y=32, optimal_y=-16),
    "deepslate_gold_ore": OreSpawnRange(min_y=-64, max_y=0, optimal_y=-16),

    # Coal
    "coal_ore": OreSpawnRange(min_y=0, max_y=256, optimal_y=96),
    "deepslate_coal_ore": OreSpawnRange(min_y=-64, max_y=0, optimal_y=-32),

    # Copper
    "copper_ore": OreSpawnRange(min_y=-16, max_y=112, optimal_y=48),
    "deepslate_copper_ore": OreSpawnRange(min_y=-16, max_y=0, optimal_y=-8),

    # Lapis
    "lapis_ore": OreSpawnRange(min_y=-64, max_y=64, optimal_y=0),
    "deepslate_lapis_ore": OreSpawnRange(min_y=-64, max_y=0, optimal_y=-32),

    # Redstone
    "redstone_ore": OreSpawnRange(min_y=-64, max_y=16, optimal_y=-59),
    "deepslate_redstone_ore": OreSpawnRange(min_y=-64, max_y=0, optimal_y=-59),

    # Emerald (mountains only)
    "emerald_ore": OreSpawnRange(
        min_y=-16, max_y=320, optimal_y=236,
        biome_restrictions=["mountains", "windswept_hills", "stony_peaks"]
    ),
    "deepslate_emerald_ore": OreSpawnRange(
        min_y=-16, max_y=0, optimal_y=-8,
        biome_restrictions=["mountains", "windswept_hills", "stony_peaks"]
    ),

    # Nether
    "nether_gold_ore": OreSpawnRange(min_y=10, max_y=117, optimal_y=15),
    "nether_quartz_ore": OreSpawnRange(min_y=10, max_y=117, optimal_y=15),
    "ancient_debris": OreSpawnRange(
        min_y=8, max_y=119, optimal_y=15,
        biome_restrictions=["nether"]
    ),
}


# ============================================================
# Biome Resources
# ============================================================

BIOME_RESOURCES: Dict[str, List[str]] = {
    "forest": ["oak_log", "birch_log", "apple", "mushroom"],
    "birch_forest": ["birch_log", "mushroom"],
    "dark_forest": ["dark_oak_log", "mushroom", "huge_mushroom"],
    "plains": ["wheat", "grass", "flowers"],
    "sunflower_plains": ["wheat", "grass", "sunflower"],
    "desert": ["sand", "cactus", "dead_bush", "sandstone"],
    "taiga": ["spruce_log", "sweet_berries", "fern"],
    "snowy_taiga": ["spruce_log", "sweet_berries", "snow"],
    "jungle": ["jungle_log", "cocoa_beans", "bamboo", "melon"],
    "bamboo_jungle": ["bamboo", "jungle_log"],
    "savanna": ["acacia_log", "grass", "tall_grass"],
    "mountains": ["stone", "emerald_ore", "coal_ore", "snow"],
    "windswept_hills": ["stone", "emerald_ore", "gravel"],
    "swamp": ["oak_log", "lily_pad", "slime", "witch"],
    # 1.19 Wild Update - Mangrove Swamp
    "mangrove_swamp": [
        "mangrove_log", "mangrove_leaves", "mangrove_propagule",
        "mud", "lily_pad", "blue_orchid", "frog", "tropical_fish",
    ],
    "beach": ["sand", "clay"],
    "ocean": ["kelp", "seagrass", "fish"],
    "river": ["clay", "sand", "fish"],
    "badlands": ["terracotta", "gold_ore", "dead_bush"],
    "mushroom_fields": ["mycelium", "huge_mushroom", "mooshroom"],

    # 1.19 Wild Update - Deep Dark (below Y=0)
    "deep_dark": [
        "sculk", "sculk_vein", "sculk_sensor", "sculk_shrieker", "sculk_catalyst",
        "deepslate", "reinforced_deepslate", "warden",  # warden is a dangerous mob
    ],

    # Nether biomes
    "crimson_forest": ["crimson_stem", "crimson_fungus", "shroomlight", "weeping_vines", "hoglin"],
    "warped_forest": ["warped_stem", "warped_fungus", "shroomlight", "enderman"],
    "nether_wastes": ["netherrack", "nether_quartz_ore", "nether_gold_ore", "glowstone"],
    "soul_sand_valley": ["soul_sand", "soul_soil", "bone_block", "basalt"],
    "basalt_deltas": ["basalt", "blackstone", "magma_block", "magma_cube"],
}


# ============================================================
# Knowledge Items
# ============================================================

RESOURCE_KNOWLEDGE: List[KnowledgeItem] = [
    # Diamond mining depth
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="diamond_y_level",
        fact=(
            "Diamond ore spawns between Y=-64 and Y=16, with highest concentration at Y=-59. "
            "Below Y=0, diamonds appear as deepslate_diamond_ore."
        ),
        keywords=["diamond", "ore", "y", "level", "depth", "spawn", "deepslate",
                  "not found", "search", "findBlock", "deepslate_diamond_ore"],
        conditions=["searching for diamonds", "mining diamonds",
                    "diamond ore not found", "no diamond nearby"],
        logical_implications=[
            "No diamonds exist above Y=16",
            "No diamonds exist below Y=-64",
            "Deepslate variant appears below Y=0",
        ],
        constraints=[
            "bot.findBlock() search radius ~32 blocks",
            "Blocks must be in loaded chunks to be found",
        ],
        available_options=[
            "Search at Y=16 (upper spawn limit)",
            "Search at Y=-59 (highest concentration)",
            "Search at any level between -64 and 16",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="iron_y_level",
        fact=(
            "Iron ore has complex distribution: spawns Y=-64 to Y=72. "
            "Main peak at Y=16, secondary peak at Y=232 (mountains). "
            "Below Y=0, iron appears as deepslate_iron_ore."
        ),
        keywords=["iron", "ore", "y", "level", "depth", "spawn"],
        conditions=["searching for iron", "mining iron"],
        logical_implications=[
            "Iron can be found at many Y levels",
            "Y=16 has highest concentration in normal terrain",
            "Mountains have extra iron at high elevation",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="coal_y_level",
        fact=(
            "Coal ore spawns between Y=0 and Y=256, with highest concentration at Y=96. "
            "Most common ore, found easily in hills and mountains."
        ),
        keywords=["coal", "ore", "y", "level", "spawn"],
        conditions=["searching for coal", "mining coal"],
        logical_implications=[
            "No coal below Y=0",
            "Coal available at all heights above sea level",
            "Mountains have abundant coal",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="gold_y_level",
        fact=(
            "Gold ore spawns between Y=-64 and Y=32, with optimal level at Y=-16. "
            "Below Y=0, gold appears as deepslate_gold_ore. "
            "Badlands biome has extra gold ore above Y=32."
        ),
        keywords=["gold", "ore", "y", "level", "depth", "spawn", "badlands"],
        conditions=["searching for gold", "mining gold"],
        logical_implications=[
            "No gold above Y=32 (except in badlands)",
            "Deepslate variant appears below Y=0",
            "Badlands is an exception with surface gold",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="emerald_biome_restriction",
        fact=(
            "Emerald ore ONLY spawns in mountain biomes: mountains, windswept_hills, stony_peaks. "
            "Y range is -16 to 320, with highest concentration at Y=236. "
            "Does not spawn in other biomes regardless of Y level."
        ),
        keywords=["emerald", "ore", "mountain", "biome", "spawn", "windswept"],
        conditions=["searching for emeralds"],
        logical_implications=[
            "Zero emeralds exist outside mountain biomes",
            "Biome check is required before Y-level check",
            "Very rare even within valid biomes",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="ancient_debris_nether",
        fact=(
            "Ancient debris ONLY spawns in the Nether, between Y=8 and Y=119, "
            "with optimal concentration at Y=15. "
            "Blast resistant, often found by TNT mining. "
            "Smelts into netherite scrap."
        ),
        keywords=["ancient", "debris", "nether", "netherite", "tnt"],
        conditions=["searching for ancient debris", "getting netherite"],
        logical_implications=[
            "Zero ancient debris exists in Overworld",
            "Nether portal required first",
            "Blast-resistant: survives explosions",
        ],
    ),

    # Special resources
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="obsidian_creation",
        fact=(
            "Obsidian does NOT spawn naturally in significant quantities. "
            "It is CREATED by pouring water onto lava source blocks (level=0). "
            "Water on flowing lava (level>0) creates cobblestone, not obsidian."
        ),
        keywords=["obsidian", "lava", "water", "create", "portal", "source"],
        conditions=["searching for obsidian", "need obsidian"],
        logical_implications=[
            "Exploration/findBlock will not find obsidian in quantity",
            "Obsidian must be created, not found",
            "Only lava source blocks (level=0) work",
            "Flowing lava (level>0) produces cobblestone",
        ],
        constraints=[
            "Need water bucket",
            "Need access to lava source blocks",
            "Mining obsidian requires diamond_pickaxe or better",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="lava_location",
        fact=(
            "Lava pools naturally generate in caves below Y=11. "
            "In Minecraft 1.18+, underground lava lakes generate at Y=-64 to Y=-54. "
            "Surface lava lakes are rare. Nether has lava oceans at Y=31."
        ),
        keywords=["lava", "pool", "cave", "nether", "source", "underground"],
        conditions=["searching for lava", "need lava for obsidian"],
        logical_implications=[
            "Lava more common at deeper Y levels",
            "Y=-64 to Y=-54 has lava lakes in 1.18+",
            "Surface lava is rare",
            "Nether has abundant lava at Y=31",
        ],
        constraints=[
            "bot.findBlock() only finds exposed lava",
            "Lava embedded in stone is not visible until excavated",
            "Search radius ~32 blocks",
        ],
        available_options=[
            "Search at current Y level",
            "Descend to Y<11 for cave lava",
            "Descend to Y=-54 to Y=-64 for lava lakes",
            "Enter Nether for guaranteed lava",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="flint_from_gravel",
        fact=(
            "Flint is obtained by mining gravel, with 10% base drop chance. "
            "Fortune enchantment increases chance (Fortune III = 100%). "
            "Gravel found in caves, underwater, and beaches."
        ),
        keywords=["flint", "gravel", "steel", "fortune"],
        conditions=["need flint", "making flint and steel"],
        logical_implications=[
            "~10 gravel blocks needed on average for 1 flint",
            "Fortune III guarantees flint drop",
            "Gravel is the only source of flint",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="clay_location",
        fact=(
            "Clay found underwater in rivers, lakes, swamps, and ocean shores. "
            "Appears as grayish blocks below shallow water. "
            "Drops 4 clay balls, can be smelted into bricks."
        ),
        keywords=["clay", "river", "water", "brick", "underwater"],
        conditions=["searching for clay"],
        logical_implications=[
            "Clay only generates in water",
            "Rivers and swamps have highest concentration",
            "Must search underwater",
        ],
    ),

    # 1.19 Wild Update - Mangrove Swamp
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="mangrove_swamp_resources",
        fact=(
            "Mangrove swamps contain mangrove trees (mangrove_log), mud blocks, frogs, and tropical fish. "
            "Mangrove propagules can be collected and planted to grow mangrove trees. "
            "Only biome where mud generates naturally."
        ),
        keywords=["mangrove", "swamp", "mud", "frog", "tropical", "propagule"],
        conditions=["exploring mangrove_swamp biome", "need mud"],
        logical_implications=[
            "Mud only natural in mangrove swamp",
            "Propagules are renewable source of mangrove wood",
            "Frogs are breedable here",
        ],
    ),

    # 1.19 Wild Update - Deep Dark
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="deep_dark_biome",
        fact=(
            "Deep Dark biome generates below Y=0, containing sculk blocks and ancient cities. "
            "Features sculk sensors (detect vibrations) and sculk shriekers (summon Warden). "
            "Ancient cities contain unique loot: echo shards, disc fragments, swift sneak enchantment."
        ),
        keywords=["deep dark", "sculk", "ancient city", "warden", "echo shard", "swift sneak"],
        conditions=["exploring below Y=0", "searching for ancient city"],
        logical_implications=[
            "Deep Dark only below Y=0",
            "Sculk shriekers summon Warden (extremely dangerous)",
            "Sneaking prevents sculk sensor detection",
            "Unique loot only in ancient city chests",
        ],
    ),

    # Deepslate ore distribution
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="deepslate_ore_distribution",
        fact=(
            "Below Y=0, all ores generate as deepslate variants (deepslate_iron_ore, deepslate_diamond_ore, etc). "
            "Same drop rates as regular ores but harder to mine (4.5 vs 3.0 hardness). "
            "Deepslate itself replaces stone below Y=8. "
            "Deepslate ore variants have DIFFERENT block IDs from their regular counterparts: "
            "e.g., mcData.blocksByName['diamond_ore'].id !== mcData.blocksByName['deepslate_diamond_ore'].id."
        ),
        keywords=["deepslate", "ore", "underground", "deep", "Y=0", "block id",
                  "findBlock", "matching", "diamond_ore", "iron_ore", "variant"],
        conditions=["mining below Y=0", "searching for ores underground",
                    "ore not found at expected Y level"],
        logical_implications=[
            "Same Fortune bonus applies to deepslate ores",
            "Mining takes ~50% longer than regular ores",
            "Tool tier requirements unchanged",
            "bot.findBlock matching only regular ore ID will miss deepslate variants below Y=0",
            "bot.findBlock matching only deepslate ore ID will miss regular variants near Y=0",
            "Both block IDs must be included in findBlock/findBlocks to find all instances of an ore type",
        ],
        constraints=[
            "diamond_ore and deepslate_diamond_ore are separate block types with different IDs",
            "At Y=-59 (optimal diamond level), virtually all diamonds are deepslate_diamond_ore",
            "Transition zone near Y=0 may contain both regular and deepslate variants",
        ],
    ),

    # Overworld vertical structure
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="overworld_vertical_structure",
        fact=(
            "Overworld vertical structure (1.18+): "
            "Y=-64 to Y=-60: Bedrock layer (100% bedrock at -64, random mix above). "
            "Y=-60 to Y=-54: Deepslate with lava lakes (underground lava ocean). "
            "Y=-54 to Y=0: Deepslate zone (all ores are deepslate variants). "
            "Y=0 to Y=8: Transition zone (mix of deepslate and stone). "
            "Y=8 to Y=62: Stone zone (regular stone, regular ore variants). "
            "Y=63: Sea level. "
            "Y=63 to Y=320: Surface and sky."
        ),
        keywords=["overworld", "vertical", "structure", "layer", "Y level", "deepslate",
                  "stone", "bedrock", "sea level", "world", "height", "depth", "underground",
                  "zone", "Y=-64", "Y=-60", "Y=0", "Y=63"],
        conditions=["exploring underground", "choosing mining depth",
                    "navigating vertically", "understanding world structure"],
        logical_implications=[
            "Below Y=-60, some positions are impassable bedrock",
            "Below Y=0, stone is replaced by deepslate",
            "Below Y=-54, lava lakes are common",
            "Sea level is Y=63",
            "World bottom is Y=-64 (all bedrock)",
        ],
        constraints=[
            "Cannot mine below Y=-64",
            "Bedrock layer makes Y=-60 to Y=-64 partially or fully impassable",
            "Lava at Y=-54 to Y=-64 poses burning risk during downward exploration",
        ],
    ),

    # Ore vein density
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="ore_vein_density",
        fact=(
            "Ore vein density per chunk (16x16x384 blocks): "
            "Coal: ~20 veins/chunk (very common). Iron: ~20 veins/chunk (common). "
            "Copper: ~6 veins/chunk. Redstone: ~8 veins/chunk. "
            "Gold: ~4 veins/chunk. Lapis: ~2 veins/chunk. "
            "Diamond: ~1 vein/chunk (rare, 1-10 blocks per vein, avg ~3.7). "
            "Emerald: ~1-2 single blocks/chunk in mountains only (very rare). "
            "A 32-block findBlock radius covers ~4x4 chunk area horizontally."
        ),
        keywords=["ore", "vein", "density", "rarity", "rare", "common", "chunk",
                  "search", "radius", "distance", "diamond", "not found", "nearby",
                  "findBlock", "maxDistance"],
        conditions=["ore not found nearby", "search radius too small",
                    "diamond search fails", "expanding search area"],
        logical_implications=[
            "Common ores (coal, iron) are almost always found within 32 blocks at correct Y level",
            "Rare ores (diamond) may not exist within 32 blocks even at optimal Y level",
            "Diamond at ~1 vein per chunk means ~3.7 diamond blocks in a 16x16 column",
            "Finding zero diamonds in one 32-block search does not mean the Y level is wrong",
        ],
        constraints=[
            "bot.findBlock maxDistance is limited by loaded chunks",
            "Larger search radius increases computation time",
        ],
    ),

    # Nether resources
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.RESOURCE_SPAWN,
        name="nether_wood_locations",
        fact=(
            "Crimson and warped stems (nether wood) found in crimson_forest and warped_forest biomes. "
            "Crimson forests have hoglins (dangerous). Warped forests have endermen. "
            "Both wood types are fireproof - ideal for nether construction."
        ),
        keywords=["crimson", "warped", "nether", "forest", "stem", "fireproof"],
        conditions=["need fireproof wood", "exploring nether"],
        logical_implications=[
            "Nether portal required to access",
            "Crimson forest has hostile hoglins",
            "Warped forest has neutral endermen",
            "Both wood types resist fire",
        ],
    ),
]


# ============================================================
# Helper Functions
# ============================================================

def get_ore_y_range(ore_name: str) -> Optional[Dict[str, int]]:
    """
    Get the Y-axis spawn range for an ore.

    Args:
        ore_name: Name of the ore (e.g., "diamond_ore")

    Returns:
        Dict with min, max, optimal Y values, or None if not found
    """
    # Try exact match first
    if ore_name in ORE_SPAWN_RANGES:
        ore = ORE_SPAWN_RANGES[ore_name]
        return {
            "min": ore.min_y,
            "max": ore.max_y,
            "optimal": ore.optimal_y,
        }

    # Try adding _ore suffix
    ore_with_suffix = f"{ore_name}_ore"
    if ore_with_suffix in ORE_SPAWN_RANGES:
        ore = ORE_SPAWN_RANGES[ore_with_suffix]
        return {
            "min": ore.min_y,
            "max": ore.max_y,
            "optimal": ore.optimal_y,
        }

    return None


def get_biome_resources(biome_name: str) -> List[str]:
    """
    Get resources available in a specific biome.

    Args:
        biome_name: Name of the biome

    Returns:
        List of resource names found in that biome
    """
    # Try exact match
    if biome_name in BIOME_RESOURCES:
        return BIOME_RESOURCES[biome_name].copy()

    # Try partial match
    for biome, resources in BIOME_RESOURCES.items():
        if biome_name in biome or biome in biome_name:
            return resources.copy()

    return []


def get_ore_biome_restriction(ore_name: str) -> Optional[List[str]]:
    """
    Get biome restrictions for an ore.

    Args:
        ore_name: Name of the ore

    Returns:
        List of allowed biomes, or None if no restriction
    """
    if ore_name in ORE_SPAWN_RANGES:
        restrictions = ORE_SPAWN_RANGES[ore_name].biome_restrictions
        return restrictions if restrictions else None
    return None


def get_all_resource_knowledge() -> List[KnowledgeItem]:
    """Get all resource-related knowledge items."""
    return RESOURCE_KNOWLEDGE.copy()
