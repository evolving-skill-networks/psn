"""
Primitives Knowledge Package - Primitive function knowledge

Contains knowledge of our wrapped Primitive functions:
- mining.py: mineBlock knowledge
- crafting.py: craftItem knowledge + RecipeDependencyAnalyzer
- placement.py: placeItem knowledge
- smelting.py: smeltItem knowledge
- combat.py: killMob knowledge
- movement.py: gotoWithTimeout knowledge

All knowledge is pure factual description, no fix_hint/fix_strategy.
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem

# Mining
from .mining import (
    MINING_KNOWLEDGE,
    get_all_mining_knowledge,
)

# Crafting (includes RecipeDependencyAnalyzer)
from .crafting import (
    CRAFTING_KNOWLEDGE,
    RecipeInfo,
    RECIPE_KNOWLEDGE,
    MaterialCalculation,
    RecipeDependencyAnalyzer,
    create_recipe_analyzer,
    get_all_crafting_knowledge,
)

# Placement
from .placement import (
    PLACEMENT_KNOWLEDGE,
    get_all_placement_knowledge,
)

# Smelting
from .smelting import (
    SMELTING_KNOWLEDGE,
    get_all_smelting_knowledge,
)

# Combat
from .combat import (
    COMBAT_KNOWLEDGE,
    get_all_combat_knowledge,
)

# Movement
from .movement import (
    MOVEMENT_KNOWLEDGE,
    get_all_movement_knowledge,
)


def get_all_primitive_knowledge() -> List[KnowledgeItem]:
    """Get all Primitive knowledge"""
    return (
        get_all_mining_knowledge() +
        get_all_crafting_knowledge() +
        get_all_placement_knowledge() +
        get_all_smelting_knowledge() +
        get_all_combat_knowledge() +
        get_all_movement_knowledge()
    )


__all__ = [
    # Base
    "get_all_primitive_knowledge",

    # Mining
    "MINING_KNOWLEDGE",
    "get_all_mining_knowledge",

    # Crafting
    "CRAFTING_KNOWLEDGE",
    "RecipeInfo",
    "RECIPE_KNOWLEDGE",
    "MaterialCalculation",
    "RecipeDependencyAnalyzer",
    "create_recipe_analyzer",
    "get_all_crafting_knowledge",

    # Placement
    "PLACEMENT_KNOWLEDGE",
    "get_all_placement_knowledge",

    # Smelting
    "SMELTING_KNOWLEDGE",
    "get_all_smelting_knowledge",

    # Combat
    "COMBAT_KNOWLEDGE",
    "get_all_combat_knowledge",

    # Movement
    "MOVEMENT_KNOWLEDGE",
    "get_all_movement_knowledge",
]
