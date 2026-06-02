"""
Reasoning Guides for the LLM.

Design principles:
- Show the reasoning process, not code solutions.
- reasoning_approach describes how to analyze, not what to write.
- No copyable code is included; the LLM derives the fix from facts.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ReasoningGuide:
    """Teach the LLM how to reason about a failure, not what code to write."""
    scenario: str
    error_pattern: str
    diagnostic_facts: Dict[str, Any]
    relevant_knowledge: List[str]
    reasoning_approach: List[str]
    conclusion: str
    keywords: List[str] = field(default_factory=list)

    def to_prompt_text(self, pure_reasoning: bool = False) -> str:
        """Convert to prompt text format - no code!

        Args:
            pure_reasoning: If True, omit conclusion (let LLM derive it)
        """
        lines = [
            f"### Reasoning Guide: {self.scenario}",
            "",
            "**Diagnostic Facts:**",
            "```json",
            str(self.diagnostic_facts).replace("'", '"'),
            "```",
            "",
            "**Relevant Knowledge:**",
        ]
        for knowledge in self.relevant_knowledge:
            lines.append(f"- {knowledge}")
        lines.append("")

        lines.append("**How to Analyze:**")
        for i, step in enumerate(self.reasoning_approach, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        if not pure_reasoning:
            lines.append(f"**Conclusion:** {self.conclusion}")
            lines.append("")

        return chr(10).join(lines)


# ============================================================
# blockUpdate timeout reasoning guides
# ============================================================

BLOCKUPDATE_POSITION_CONFLICT = ReasoningGuide(
    scenario="blockUpdate timeout due to bot standing at placement target",
    error_pattern="blockUpdate.*did not fire.*timeout",
    diagnostic_facts={
        "bot_position": {"x": -5, "y": 69, "z": 2},
        "target_position": {"x": -5, "y": 69, "z": 2},
        "positions_equal": True,
        "block_at_target": "air",
        "target_occupied": False,
        "distance": "0.0"
    },
    relevant_knowledge=[
        "goalplaceblock_vs_goalnear",
        "placeblock_entity_blocking",
    ],
    reasoning_approach=[
        "Check if bot_position equals target_position",
        "If positions_equal is True, bot is at the target position",
        "Recall: bot is an entity, entities block placement",
        "Recall: blockUpdate will not fire if target is occupied by entity",
        "Identify which Goal type allows bot to stand at target",
        "GoalNear(x,y,z,0) allows standing at target; GoalPlaceBlock does not",
        "Consider which Goal type guarantees bot stands ADJACENT to target",
    ],
    conclusion="Bot is blocking its own placement. Need a Goal type that ensures adjacency, not overlap.",
    keywords=["blockupdate", "timeout", "positions_equal", "placement"]
)

BLOCKUPDATE_TARGET_OCCUPIED = ReasoningGuide(
    scenario="blockUpdate timeout due to target position already occupied",
    error_pattern="blockUpdate.*did not fire.*timeout",
    diagnostic_facts={
        "bot_position": {"x": 10, "y": 65, "z": 20},
        "target_position": {"x": 11, "y": 65, "z": 20},
        "positions_equal": False,
        "block_at_target": "cobblestone",
        "target_occupied": True,
        "distance": "1.0"
    },
    relevant_knowledge=[
        "placeblock_preconditions",
        "bot_blockat_method",
    ],
    reasoning_approach=[
        "Check if positions_equal is False (bot is not at target)",
        "Check if target_occupied is True",
        "Check what block exists at target (block_at_target)",
        "Recall: placeBlock requires target to be air",
        "If target has a block, that block must be removed or a different position must be chosen",
    ],
    conclusion="Target position already has a block. Cannot place on top of existing block.",
    keywords=["blockupdate", "timeout", "target_occupied", "occupied"]
)

# ============================================================
# Pathfinding reasoning guides
# ============================================================

PATHFINDING_TIMEOUT = ReasoningGuide(
    scenario="Pathfinding timeout due to long distance",
    error_pattern="timeout|path.*blocked|cannot reach",
    diagnostic_facts={
        "bot_position": {"x": 0, "y": 64, "z": 0},
        "target_position": {"x": 500, "y": 64, "z": 500},
        "distance": "707.1",
        "path_status": "timeout"
    },
    relevant_knowledge=[
        "pathfinder_timeout_limits",
        "pathfinder_segment_strategy",
    ],
    reasoning_approach=[
        "Calculate distance between bot and target",
        "Compare distance to typical pathfinder limits (~200-300 blocks)",
        "If distance > limit, single pathfinder call will timeout",
        "Consider breaking journey into smaller segments",
        "Each segment should be within pathfinder's reliable range",
    ],
    conclusion="Distance too far for single pathfinding operation. Break into segments.",
    keywords=["pathfinding", "timeout", "distance", "cannot reach"]
)

# ============================================================
# Mining reasoning guides
# ============================================================

MINING_WRONG_Y_LEVEL = ReasoningGuide(
    scenario="Cannot find diamond ore due to wrong Y level",
    error_pattern="cannot find|not found|no.*diamond",
    diagnostic_facts={
        "bot_position": {"x": 100, "y": 64, "z": 200},
        "search_target": "diamond_ore",
        "search_radius": 32,
        "blocks_found": 0,
        "current_y": 64
    },
    relevant_knowledge=[
        "diamond_y_level",
        "ore_spawn_ranges",
    ],
    reasoning_approach=[
        "Check current bot Y level",
        "Recall diamond ore spawn range (Y=-64 to Y=16)",
        "Compare current Y to spawn range",
        "If current_y > max_spawn_y, ore cannot exist at this level",
        "Calculate how far to descend to enter spawn range",
        "Consider optimal Y level for highest concentration",
    ],
    conclusion="Bot is too high to find diamonds. Ore only spawns below Y=16.",
    keywords=["diamond", "ore", "cannot find", "not found", "y level"]
)

# ============================================================
# Crafting reasoning guides
# ============================================================

CRAFTING_TABLE_OUT_OF_RANGE = ReasoningGuide(
    scenario="Crafting fails because crafting table is out of range",
    error_pattern="crafting.*table|cannot craft|craft.*fail",
    diagnostic_facts={
        "bot_position": {"x": 10, "y": 65, "z": 20},
        "crafting_table_position": {"x": 20, "y": 65, "z": 20},
        "distance_to_table": "10.0",
        "recipe_requires_3x3": True
    },
    relevant_knowledge=[
        "crafting_range_requirements",
        "interaction_range_limits",
    ],
    reasoning_approach=[
        "Check if recipe requires 3x3 grid (distance_to_table matters)",
        "Check distance_to_table value",
        "Recall interaction range is ~4 blocks",
        "If distance > interaction range, crafting will fail",
        "Must move closer before attempting craft",
    ],
    conclusion="Bot is too far from crafting table. Must move within interaction range (~4 blocks).",
    keywords=["craft", "crafting_table", "range", "distance"]
)

# ============================================================
# Tool Tier reasoning guides
# ============================================================

MINING_WRONG_TOOL_TIER = ReasoningGuide(
    scenario="Mining diamond ore with wrong tool tier - block breaks but no drops",
    error_pattern="After digging.*now have 0|0/\\d+ diamonds",
    diagnostic_facts={
        "tool_used": "stone_pickaxe",
        "block_mined": "deepslate_diamond_ore",
        "items_obtained": 0,
        "expected_items": 3,
        "tool_tier": 1,
        "required_tier": 2,
    },
    relevant_knowledge=[
        "minecraft_tool_rules",
        "tool_tier_requirements",
    ],
    reasoning_approach=[
        "Check what tool was used (tool_used)",
        "Check what block was mined (block_mined)",
        "Look up required tier for that block type",
        "Compare tool tier vs required tier",
        "If tool_tier < required_tier, block breaks but no drops",
        "Identify minimum required tool for this ore",
    ],
    conclusion="Tool tier insufficient. Stone pickaxe cannot harvest diamond ore - need iron or better.",
    keywords=["diamond", "pickaxe", "mining", "no drops", "tier", "stone_pickaxe", "0/", "now have 0"]
)

MINEBLOCK_NOCHESTS_ERROR = ReasoningGuide(
    scenario="mineBlock fails repeatedly with NoChests error due to full inventory",
    error_pattern="nochests.*no defined chest|mineblock failed.*nochests",
    diagnostic_facts={
        "error_type": "NoChests",
        "inventory_status": "36/36 (full)",
        "mining_attempts": 8,
        "ore_found": True,
        "pickaxe_equipped": True,
        "chest_exists_in_world": True,
        "chest_locations_configured": False,
    },
    relevant_knowledge=[
        "mineblock_collectblock_chain",
        "inventory_management",
    ],
    reasoning_approach=[
        "Check error type (NoChests indicates inventory management issue)",
        "Check inventory status (36/36 means completely full)",
        "Recall: mineBlock uses collectBlock internally",
        "Recall: collectBlock tries to empty to chests when full",
        "If chestLocations not configured, NoChests error thrown",
        "This is NOT a mining failure - it is a collection failure",
        "Consider making inventory space before mining",
    ],
    conclusion="Inventory is full and no chest configured. Mining succeeds but collection fails.",
    keywords=["nochests", "no defined chest", "mineblock failed", "inventory", "36/36", "full", "chest"]
)

CONFINED_SPACE_PLACEMENT = ReasoningGuide(
    scenario="Cannot place block in 1x1 vertical shaft - No valid position found",
    error_pattern=r"no valid.*position|cannot.*place|could not find.*position",
    diagnostic_facts={
        "bot_position": {"x": 100, "y": 45, "z": 200},
        "error_message": "No valid position found nearby to place a crafting table",
        "environment_type": "1x1 vertical shaft",
        "surrounding_blocks": {
            "north": "stone (wall)",
            "south": "stone (wall)",
            "east": "stone (wall)",
            "west": "stone (wall)",
            "above_y+2": "air (shaft continues up)",
            "below": "stone (floor)",
        },
        "search_radius": 4,
    },
    relevant_knowledge=[
        "block_placement_rules",
        "adjacent_reference_blocks",
    ],
    reasoning_approach=[
        "Analyze surrounding blocks for valid placement targets",
        "Recall: placement needs (1) target is air, (2) any adjacent block is solid",
        "Key insight: adjacent solid can be wall, ceiling, or floor - not just below!",
        "Check position y+2 above bot: is it air? Are walls adjacent?",
        "If walls are solid at (x+1,y+2,z), then (x,y+2,z) is valid placement",
        "Consider if search code only checks for solid block below (too restrictive)",
        "Consider if search code only searches y=0 level (missing y+1, y+2 levels)",
    ],
    conclusion="Valid placement exists above bot head with wall as reference. Code search logic is too restrictive.",
    keywords=["no valid position", "cannot place", "could not find", "placement", "shaft", "confined", "1x1", "narrow", "tunnel", "cave"]
)

# ============================================================
# Reference block not found reasoning guide
# ============================================================

REFERENCE_BLOCK_NOT_FOUND = ReasoningGuide(
    scenario="Failed to place block — no valid reference block found (unvalidated position)",
    error_pattern=r"reference block|floating block|no block to place.*on",
    diagnostic_facts={
        "error_message": "Failed to place crafting_table: no valid reference block found. Cannot place a floating block.",
        "what_placeItem_does": (
            "placeItem checks all 6 face-adjacent blocks of the target BLOCK position (±x, ±y, ±z) "
            "for a solid reference. If none found, it tries adaptive fallback (move to open area, dig). "
            "If ALL strategies fail, it throws this error."
        ),
        "position_analysis": {
            "example_target_pos": "bot.entity.position.offset(2, 0, 0)",
            "block_at_target": "air (correct — placement target should be air)",
            "adjacent_blocks": "not checked by the SKILL code before calling placeItem",
            "result": "target and fallback positions all have no adjacent solid block → placement fails",
        },
    },
    relevant_knowledge=[
        "placement_requirements_summary: 3 conditions — (1) target is air, (2) adjacent solid block exists, (3) within range",
        "placeItem_adjacent_block: placeItem requires at least one face-adjacent solid block as reference",
        "placeItem_internal_capabilities: placeItem has internal fallback but caller should provide a valid position",
    ],
    reasoning_approach=[
        "Identify the position argument passed to placeItem — how is it computed?",
        "Check whether the SKILL code validates that the position meets ALL placement requirements before calling placeItem",
        "Recall placement requirements: (1) block at target must be 'air', (2) at least one of its 6 face-adjacent blocks (±x, ±y, ±z) must be solid",
        "An arbitrary hardcoded offset (e.g., offset(N,0,0)) provides NO guarantee about what blocks exist at that position",
        "Determine: does the error occur because the chosen position has no adjacent solid blocks?",
    ],
    conclusion="The placement position is not validated against placement requirements before calling placeItem. "
               "The code must ensure the target position satisfies all placement conditions (air at target + adjacent solid block).",
    keywords=["reference block", "floating block", "no block to place", "cannot place", "failed to place",
              "placeitem", "crafting_table"],
)

# ============================================================
# Crafting table not placed reasoning guide
# ============================================================

CRAFTING_TABLE_NOT_PLACED = ReasoningGuide(
    scenario="Crafting 3x3 recipe fails because no crafting table block placed nearby",
    error_pattern="no crafting table|no available recipes|3x3 recipe|Cannot craft.*crafting table",
    diagnostic_facts={
        "bot_position": {"x": 10, "y": 27, "z": 20},
        "crafting_table_in_inventory": True,
        "crafting_table_placed_nearby": False,
        "recipe_requires_3x3": True,
        "error_message": "Cannot craft furnace: this is a 3x3 recipe that requires a crafting table",
    },
    relevant_knowledge=[
        "craftItem_no_table",
        "craftItem_internal_table_search",
        "crafting_table_item_vs_block",
        "placeItem_places_inventory_as_block",
    ],
    reasoning_approach=[
        "Identify: error says '3x3 recipe that requires a crafting table'",
        "Distinguish: crafting_table as inventory Item vs. placed crafting_table Block in world",
        "Recall: craftItem uses bot.findBlock to search for placed table blocks within 32 blocks",
        "Recall: bot.checkRecipe with craftingTable=null only returns 2x2 recipes",
        "Recall: bot.placeBlock(ref, face) requires Block object as first arg, not inventory Item",
        "Recall: placeItem(bot, name, pos) is a control primitive that places inventory items as world blocks",
    ],
    conclusion="3x3 recipe fails because no crafting table block is placed within range. "
               "Crafting table exists in inventory as an Item but has not been placed as a Block in the world.",
    keywords=["no crafting table", "3x3", "crafting table", "requires a crafting table",
              "furnace", "pickaxe", "no available recipes", "place"]
)

# ============================================================
# Craft silent failure reasoning guide
# ============================================================

CRAFT_SILENT_FAILURE = ReasoningGuide(
    scenario="Craft call executes but inventory unchanged — wrong API or wrong argument type",
    error_pattern=r"need \d+ planks|no planks|have none|inventory unchanged|failed to craft",
    diagnostic_facts={
        "observation": "Code calls a crafting API, but target item did not appear in inventory after execution",
        "inventory_before": {"oak_planks": 8},
        "inventory_after": {"oak_planks": 8},
        "error_raised": False,
        "items_crafted": 0,
    },
    relevant_knowledge=[
        "bot.craft: recipe argument must be Recipe object from bot.recipesFor() or bot.checkRecipe()",
        "itemsByName_vs_blocksByName: inventory methods need itemsByName IDs, not blocksByName",
    ],
    reasoning_approach=[
        "Identify all crafting-related API calls in the code (bot.craft, craftItem, bot.recipesFor, bot.checkRecipe)",
        "For each bot.craft() call: check what the FIRST argument is — is it a Recipe object or something else (item ID, string)?",
        "For each inventory lookup: check whether mcData.itemsByName or mcData.blocksByName is used",
        "Determine: does the code use the correct API and correct argument types for each call?",
    ],
    conclusion="Craft call completed without error but produced no items. "
               "The API was called with wrong argument type or wrong mcData registry lookup.",
    keywords=["craft", "planks", "no planks", "have none", "need", "failed to craft",
              "inventory unchanged", "silent", "bot.craft", "craftitem"],
)

# ============================================================
# All reasoning guides collection
# ============================================================

ALL_REASONING_GUIDES = [
    BLOCKUPDATE_POSITION_CONFLICT,
    BLOCKUPDATE_TARGET_OCCUPIED,
    PATHFINDING_TIMEOUT,
    MINING_WRONG_Y_LEVEL,
    CRAFTING_TABLE_OUT_OF_RANGE,
    CRAFTING_TABLE_NOT_PLACED,
    MINING_WRONG_TOOL_TIER,
    MINEBLOCK_NOCHESTS_ERROR,
    CONFINED_SPACE_PLACEMENT,
    REFERENCE_BLOCK_NOT_FOUND,
    CRAFT_SILENT_FAILURE,
]

# Backwards compatibility alias
ALL_REASONING_EXAMPLES = ALL_REASONING_GUIDES


def get_relevant_examples(
    error_message: str, max_examples: int = 2,
    pure_reasoning: bool = False,
) -> str:
    """
    Return relevant reasoning guides based on error message

    Args:
        error_message: Error message
        max_examples: Maximum number of guides to return
        pure_reasoning: If True, omit conclusion from guides

    Returns:
        Formatted reasoning guide text
    """
    error_lower = error_message.lower()
    scored_guides = []

    for guide in ALL_REASONING_GUIDES:
        score = 0

        for keyword in guide.keywords:
            if keyword in error_lower:
                score += 2

        import re
        if re.search(guide.error_pattern, error_lower):
            score += 3

        facts_str = str(guide.diagnostic_facts).lower()
        if "positions_equal" in error_lower and "positions_equal" in facts_str:
            score += 5
        if "target_occupied" in error_lower and "target_occupied" in facts_str:
            score += 5

        if score > 0:
            scored_guides.append((score, guide))

    scored_guides.sort(key=lambda x: x[0], reverse=True)
    selected = [guide for _, guide in scored_guides[:max_examples]]

    if not selected:
        return ""

    lines = ["## Reasoning Guides", ""]
    lines.append("Learn from these guides how to analyze similar errors:")
    lines.append("")
    for guide in selected:
        lines.append(guide.to_prompt_text(pure_reasoning=pure_reasoning))

    return chr(10).join(lines)


def get_example_for_pattern(pattern: str) -> Optional[ReasoningGuide]:
    """
    Get reasoning guide for a specific pattern

    Args:
        pattern: Pattern string

    Returns:
        Matching reasoning guide or None
    """
    pattern_lower = pattern.lower()

    if "positions_equal" in pattern_lower or "blockupdate" in pattern_lower:
        return BLOCKUPDATE_POSITION_CONFLICT

    if "target_occupied" in pattern_lower:
        return BLOCKUPDATE_TARGET_OCCUPIED

    if "pathfind" in pattern_lower or "timeout" in pattern_lower:
        return PATHFINDING_TIMEOUT

    if "diamond" in pattern_lower or "ore" in pattern_lower:
        return MINING_WRONG_Y_LEVEL

    if "craft" in pattern_lower:
        return CRAFTING_TABLE_OUT_OF_RANGE

    return None
