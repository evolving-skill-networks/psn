"""
Game Knowledge Package - Minecraft game knowledge

Contains pure game knowledge: rules, block properties, resource distributions, etc.
These are rules of Minecraft itself, unrelated to the API.

Submodules:
- blocks.py: block properties, tool requirements, tool tier system
- resources.py: ore distribution, biome resources
- mechanics.py: game mechanics (crafting, placement, combat, etc.)
- ores.py: ore drop entity knowledge (per-entity ore→item mapping)

All knowledge is pure factual description, no solution hints.
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem

# Blocks
from .blocks import (
    BLOCK_HARDNESS,
    TOOL_EFFECTIVENESS,
    TOOL_TIERS,
    BLOCK_HARVEST_TIER,
    BLOCK_KNOWLEDGE,
    get_required_tool_tier,
    get_tool_for_block,
    can_tool_harvest,
    get_all_block_knowledge,
)

# Resources
from .resources import (
    OreSpawnRange,
    ORE_SPAWN_RANGES,
    BIOME_RESOURCES,
    RESOURCE_KNOWLEDGE,
    get_ore_y_range,
    get_biome_resources,
    get_ore_biome_restriction,
    get_all_resource_knowledge,
)

# Mechanics
from .mechanics import (
    PLACEMENT_KNOWLEDGE,
    PATHFINDING_KNOWLEDGE,
    CRAFTING_KNOWLEDGE,
    WATER_MECHANICS_KNOWLEDGE,
    COMBAT_KNOWLEDGE,
    INVENTORY_KNOWLEDGE,
    SPECIAL_ACQUISITION_KNOWLEDGE,
    MECHANICS_KNOWLEDGE,
    get_all_mechanics_knowledge,
    get_relevant_mechanics_knowledge,  # used for error-related knowledge retrieval
)

# Ores (per-entity ore→item knowledge)
from .ores import (
    ORE_DROP_KNOWLEDGE,
    get_all_ore_drop_knowledge,
)


def get_all_game_knowledge() -> List[KnowledgeItem]:
    """Get all game knowledge items"""
    return (
        get_all_block_knowledge() +
        get_all_resource_knowledge() +
        get_all_mechanics_knowledge() +
        get_all_ore_drop_knowledge()
    )


__all__ = [
    # Base
    "get_all_game_knowledge",

    # Blocks
    "BLOCK_HARDNESS",
    "TOOL_EFFECTIVENESS",
    "TOOL_TIERS",
    "BLOCK_HARVEST_TIER",
    "BLOCK_KNOWLEDGE",
    "get_required_tool_tier",
    "get_tool_for_block",
    "can_tool_harvest",
    "get_all_block_knowledge",

    # Resources
    "OreSpawnRange",
    "ORE_SPAWN_RANGES",
    "BIOME_RESOURCES",
    "RESOURCE_KNOWLEDGE",
    "get_ore_y_range",
    "get_biome_resources",
    "get_ore_biome_restriction",
    "get_all_resource_knowledge",

    # Mechanics
    "PLACEMENT_KNOWLEDGE",
    "PATHFINDING_KNOWLEDGE",
    "CRAFTING_KNOWLEDGE",
    "WATER_MECHANICS_KNOWLEDGE",
    "COMBAT_KNOWLEDGE",
    "INVENTORY_KNOWLEDGE",
    "SPECIAL_ACQUISITION_KNOWLEDGE",
    "MECHANICS_KNOWLEDGE",
    "get_all_mechanics_knowledge",
    "get_relevant_mechanics_knowledge",
    # Ores
    "ORE_DROP_KNOWLEDGE",
    "get_all_ore_drop_knowledge",
]
