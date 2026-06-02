"""
Primitive Knowledge - Movement
gotoWithTimeout primitive knowledge

Migrated from primitive_knowledge.py
All fix_hint, fix_strategy, check_hint fields removed
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# gotoWithTimeout Primitive Knowledge
# ============================================================

MOVEMENT_KNOWLEDGE: List[KnowledgeItem] = [
    # Preconditions
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="goto_destination_reachable",
        fact=(
            "gotoWithTimeout requires the destination to be reachable via pathfinding. "
            "Path must not be completely blocked by walls, water, or impassable terrain. "
            "Very long paths may timeout before completion."
        ),
        keywords=["goto", "destination", "reachable", "path", "blocked"],
        conditions=["calling gotoWithTimeout"],
        implications=[
            "needs valid path",
            "blocked paths fail",
            "long paths may timeout",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="goto_chunk_loaded",
        fact=(
            "gotoWithTimeout works best when destination chunk is loaded. "
            "Pathfinding to unloaded chunks may fail or be unreliable. "
            "Explore toward destination first for very distant goals."
        ),
        keywords=["goto", "chunk", "loaded", "distance", "explore"],
        conditions=["moving to distant locations"],
        implications=[
            "nearby destinations more reliable",
            "far destinations may need steps",
            "explore to load chunks",
        ],
    ),

    # Failure patterns
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="goto_path_timeout",
        fact=(
            "gotoWithTimeout fails with 'Path timeout' when pathfinding takes too long. "
            "Default timeout is typically 60 seconds. "
            "Causes: very long path, complex terrain, or calculation stuck."
        ),
        keywords=["goto", "timeout", "path", "long", "stuck"],
        conditions=["pathfinding times out"],
        implications=[
            "increase timeout value",
            "use intermediate waypoints",
            "simplify path",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="goto_goal_unreachable",
        fact=(
            "gotoWithTimeout fails with 'Goal unreachable' or 'Cannot reach' when no path exists. "
            "Destination completely blocked by terrain, walls, or water. "
            "May need to create a path by digging or building."
        ),
        keywords=["goto", "unreachable", "blocked", "cannot reach", "no path"],
        conditions=["destination blocked"],
        implications=[
            "path is completely blocked",
            "dig through obstacles",
            "bridge over gaps",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="goto_stuck",
        fact=(
            "Bot may get stuck during movement due to complex terrain or edge cases. "
            "Signs: repeated position, no progress, spinning in place. "
            "May need to break nearby blocks or try different approach."
        ),
        keywords=["goto", "stuck", "position", "progress", "spinning"],
        conditions=["bot gets stuck"],
        implications=[
            "bot not making progress",
            "may need manual intervention",
            "try different path",
        ],
    ),

    # Effects
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="goto_movement_behavior",
        fact=(
            "gotoWithTimeout uses A* pathfinding algorithm. "
            "Path recalculates if blocked during movement. "
            "Bot will jump, sprint, and avoid hazards automatically."
        ),
        keywords=["goto", "pathfinding", "astar", "movement", "recalculate"],
        conditions=["during movement"],
        implications=[
            "intelligent pathfinding",
            "adapts to obstacles",
            "avoids hazards",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="goto_fall_damage",
        fact=(
            "Pathfinder avoids high drops by default to prevent fall damage. "
            "maxDropDown setting controls maximum drop height (default ~3 blocks). "
            "May take longer paths to avoid falls."
        ),
        keywords=["goto", "fall", "damage", "drop", "height", "safe"],
        conditions=["paths with height differences"],
        implications=[
            "avoids dangerous drops",
            "may take longer route",
            "configurable max drop",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="goto_water_handling",
        fact=(
            "Pathfinder handles water by swimming. "
            "Deep water may be avoided as swimming is slow. "
            "Boats can be used for faster water travel."
        ),
        keywords=["goto", "water", "swim", "boat", "travel"],
        conditions=["paths crossing water"],
        implications=[
            "can swim through water",
            "swimming is slow",
            "boats are faster",
        ],
    ),
]


def get_all_movement_knowledge() -> List[KnowledgeItem]:
    """Get all movement primitive knowledge items."""
    return MOVEMENT_KNOWLEDGE.copy()
