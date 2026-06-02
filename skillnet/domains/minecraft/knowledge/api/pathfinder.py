"""
API Knowledge - Pathfinder
Mineflayer Pathfinder API knowledge

Migrated from api_behaviors.py
All code examples removed, only behavior descriptions retained
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# Pathfinder Goals Knowledge
# ============================================================

PATHFINDER_KNOWLEDGE: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="GoalNear",
        fact=(
            "GoalNear(x, y, z, range) moves bot within 'range' blocks of target position. "
            "Goal is reached when bot enters the range sphere, which may include the target position itself. "
            "Bot stops at the first position that satisfies the range requirement."
        ),
        keywords=["goal", "near", "range", "pathfinder", "position", "move"],
        conditions=["moving to a general area", "approaching a target"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="GoalPlaceBlock",
        fact=(
            "GoalPlaceBlock(pos, world, options) finds a position ADJACENT to target where bot can place a block. "
            "Bot will never stand at the target position, ensuring it's free for placement. "
            "Considers world geometry to find valid placement positions."
        ),
        keywords=["goal", "place", "block", "pathfinder", "adjacent", "placement"],
        conditions=["placing a block at specific position"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="GoalBlock",
        fact=(
            "GoalBlock(x, y, z) moves bot to stand at exactly the specified position. "
            "Used when precise positioning is required. "
            "Fails if the target position is occupied by a solid block."
        ),
        keywords=["goal", "block", "exact", "position", "pathfinder", "stand"],
        conditions=["need exact position", "precise movement"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="GoalGetToBlock",
        fact=(
            "GoalGetToBlock(x, y, z) moves bot to a position where it can interact with the block at target. "
            "Bot tries to stand ON the target block to be adjacent for interaction. "
            "FAILS if target block is lava, water, air, or other non-solid/dangerous terrain."
        ),
        keywords=["goal", "get", "block", "interact", "pathfinder", "adjacent", "lava", "water", "hazard"],
        conditions=["interacting with a block", "mining", "opening container"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="GoalFollow",
        fact=(
            "GoalFollow(entity, range) continuously follows a moving entity, staying within range. "
            "Goal updates dynamically as target moves. "
            "Higher CPU usage due to constant path recalculation."
        ),
        keywords=["goal", "follow", "entity", "moving", "pathfinder", "track"],
        conditions=["following a mob", "tracking moving target"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="pathfinder_goto",
        fact=(
            "bot.pathfinder.goto(goal) calculates A* path and moves bot toward the goal. "
            "Async operation that resolves when goal reached or rejects on failure. "
            "Path recalculates if blocked; may timeout on very long paths."
        ),
        keywords=["pathfinder", "goto", "move", "path", "async", "navigate"],
        conditions=["any movement operation"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="pathfinder_movements",
        fact=(
            "Pathfinder movements can be configured: canDig, canOpenDoors, maxDropDown, etc. "
            "Default settings avoid dangerous paths (high drops, lava). "
            "May need adjustment for specific tasks like mining through obstacles."
        ),
        keywords=["pathfinder", "movements", "config", "dig", "drop", "settings"],
        conditions=["customizing pathfinder behavior"],
    ),

    # Additional Goals
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="GoalXZ",
        fact=(
            "GoalXZ(x, z) moves bot to specific X,Z coordinates, ignoring Y level. "
            "Bot finds any Y level that reaches the horizontal target. "
            "Useful for horizontal navigation without specifying exact height."
        ),
        keywords=["goal", "xz", "horizontal", "coordinate", "navigation", "pathfinder"],
        conditions=["moving to horizontal position", "ignoring height"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="GoalY",
        fact=(
            "GoalY(y) moves bot to a specific Y level, regardless of X/Z position. "
            "Used for reaching a height, like going down to diamond level. "
            "May dig down or climb up as needed."
        ),
        keywords=["goal", "y", "height", "level", "vertical", "pathfinder"],
        conditions=["reaching specific height", "mining to specific Y level"],
    ),

    # Cross-dimension limitation
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="cross_dimension_limitation",
        fact=(
            "Pathfinder CANNOT path across dimension boundaries (Overworld/Nether/End). "
            "Must use portals manually, then create new path after teleporting. "
            "Bot's coordinates change drastically after dimension change."
        ),
        keywords=["dimension", "nether", "portal", "end", "cross", "teleport", "pathfinder"],
        conditions=["traveling between dimensions"],
    ),

    # Sprint-jumping behavior
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="pathfinder_sprint_jump",
        fact=(
            "Pathfinder uses sprint-jumping for faster movement when path is clear and long. "
            "Can jump 4-block gaps with sprint-jump. "
            "May fall off edges if terrain changes unexpectedly."
        ),
        keywords=["sprint", "jump", "gap", "speed", "fast", "pathfinder"],
        conditions=["long distance travel", "crossing gaps"],
    ),

    # Pathfinder timeout
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.PATHFINDER,
        name="pathfinder_timeout",
        fact=(
            "Pathfinder can timeout on very long or complex paths (default varies). "
            "Timeout occurs during path calculation, not during movement. "
            "For distant goals, use intermediate waypoints to reduce complexity."
        ),
        keywords=["timeout", "pathfinder", "long", "complex", "waypoint"],
        conditions=["path calculation timeout"],
    ),
]


def get_all_pathfinder_knowledge() -> List[KnowledgeItem]:
    """Get all pathfinder-related knowledge items."""
    return PATHFINDER_KNOWLEDGE.copy()
