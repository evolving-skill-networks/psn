"""
Primitive Knowledge - Placement
placeItem primitive knowledge

Migrated from primitive_knowledge.py
All fix_hint, fix_strategy, check_hint fields removed
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# placeItem Primitive Knowledge
# ============================================================

PLACEMENT_KNOWLEDGE: List[KnowledgeItem] = [
    # Preconditions
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="placeItem_in_inventory",
        fact=(
            "placeItem requires the target item to be in inventory. "
            "Item name must match exactly (underscores, singular/plural). "
            "Check inventory count before calling placeItem."
        ),
        keywords=["placeitem", "inventory", "item", "have", "exists"],
        conditions=["calling placeItem"],
        implications=[
            "fails if item not in inventory",
            "exact name match required",
            "craft or gather item first",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="placeItem_adjacent_block",
        fact=(
            "placeItem requires at least one face-adjacent block (±x, ±y, ±z) that is NOT air "
            "as a placement reference. The reference block does not need to be solid — "
            "any non-air block (glass, fence, water, etc.) works. "
            "Cannot place blocks floating in air with all 6 neighbors being air."
        ),
        keywords=["placeitem", "adjacent", "solid", "reference", "face"],
        conditions=["placing blocks"],
        implications=[
            "no floating placement",
            "need existing block nearby",
            "may need to build platform",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="placeItem_position_clear",
        fact=(
            "placeItem requires target position to be air or replaceable (water, tall_grass, snow_layer). "
            "Cannot place where a non-replaceable block already exists. "
            "Cannot place where entities (including bot) are occupying the space."
        ),
        keywords=["placeitem", "position", "clear", "air", "occupied", "entity"],
        conditions=["choosing placement position"],
        implications=[
            "target must be empty",
            "entities block placement",
            "bot cannot be at target",
        ],
    ),

    # Failure patterns
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="placeItem_no_valid_position",
        fact=(
            "placeItem fails with 'No valid position' or 'Cannot place' when no suitable spot found. "
            "All nearby positions are occupied or have no adjacent solid block. "
            "May need to move to different location or create platform."
        ),
        keywords=["placeitem", "no valid", "cannot place", "position", "failed", "reference block", "floating block"],
        conditions=["placement fails"],
        implications=[
            "target block position has no adjacent solid block in any of 6 face-directions (±x, ±y, ±z), and placeItem's internal fallback also failed",
            "common cause: caller passes an arbitrary offset (e.g., offset(2,0,0)) without verifying placement requirements at that position",
            "placeItem does not validate whether the target position meets placement requirements before attempting placement",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="placeItem_blockUpdate_timeout",
        fact=(
            "placeItem fails with 'blockUpdate timeout' when placement couldn't be confirmed. "
            "Causes: position occupied by entity, already has a block, or network issue. "
            "Block may or may not have actually been placed."
        ),
        keywords=["placeitem", "blockupdate", "timeout", "did not fire", "confirm"],
        conditions=["blockUpdate timeout"],
        implications=[
            "check if entity at position",
            "check if block exists there",
            "bot may be standing at target",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="placeItem_bot_at_target",
        fact=(
            "placeItem fails with blockUpdate timeout when bot is standing at or near the "
            "target placement position. The bot's hitbox (approximately 0.6 x 1.8 x 0.6 blocks "
            "centered on bot.entity.position) occupies the space, preventing block placement. "
            "blockUpdate event only fires when a block is SUCCESSFULLY placed."
        ),
        keywords=["placeitem", "bot", "standing", "target", "goalnear", "position",
                  "blockupdate", "timeout", "placement"],
        conditions=["bot at placement target", "blockupdate timeout", "placement timeout"],
        implications=[
            "bot.entity.position.floored() is the foot cell; foot.offset(0,1,0) is the head cell. Both are bot-occupied.",
            "GoalPlaceBlock positions bot ADJACENT to target (not at target); GoalNear only guarantees distance within range",
            "a position is valid for placement if: block is 'air' AND ANY of its 6 face-adjacent neighbors has boundingBox === 'block'",
            "digging a solid block creates an 'air' block suitable for placement at that position",
            "bot.findBlock() can verify whether a placed block exists at expected position",
        ],
    ),

    # API contract: prevents the silent .offset(Vec3) → NaN bug observed in
    # 0.4% of production skill files and 20% of LLM regenerations of placement-style
    # multi-direction enumeration code (Phase 11/12 reproducer).
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="vec3_api_contract",
        fact=(
            "Vec3.offset(dx, dy, dz) takes 3 numeric arguments. To add one Vec3 to "
            "another, use pos.plus(vec) — NOT pos.offset(vec). Writing pos.offset(vec) "
            "where vec is a Vec3 instance silently produces NaN coordinates (a common "
            "bug when iterating over a list of direction vectors)."
        ),
        keywords=["vec3", "offset", "plus", "nan", "direction", "vector", "iterate", "placement"],
        conditions=["constructing positions from direction vectors", "calling Vec3.offset"],
        implications=[
            "for arrays of direction vectors, prefer [dx, dy, dz] tuples then call .offset(dx, dy, dz)",
            "pos.plus(vec) is the correct primitive for Vec3 + Vec3 addition",
            "pos.offset(vec) silently fails: result has NaN x/y/z coordinates",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="placeItem_item_missing",
        fact=(
            "placeItem fails with 'Don't have' or 'Not in inventory' when item not present. "
            "Item name must match exactly what's in inventory. "
            "Check inventory.count() before attempting placement."
        ),
        keywords=["placeitem", "don't have", "not in inventory", "missing"],
        conditions=["item not in inventory"],
        implications=[
            "verify item exists in inventory",
            "craft or gather the item first",
            "check exact item name",
        ],
    ),

    # Effects
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="placeItem_equip_first",
        fact=(
            "placeItem automatically equips the item to hand before placing. "
            "Item is removed from inventory slot and placed as block. "
            "One item is consumed per placement."
        ),
        keywords=["placeitem", "equip", "hand", "consume", "inventory"],
        conditions=["successful placement"],
        implications=[
            "item auto-equipped",
            "one item consumed",
            "block appears at target",
        ],
    ),

    # Placing inventory items as world blocks
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="placeItem_places_inventory_as_block",
        fact=(
            "placeItem(bot, name, position) is a control primitive that takes an item name string "
            "and a target position, then places the inventory item as a world block. "
            "It handles equipping the item, finding adjacent reference blocks, "
            "and face vector calculation automatically. "
            "After successful placement, the item becomes a Block in the world "
            "that can be found by bot.findBlock."
        ),
        keywords=["placeitem", "inventory", "block", "place", "world", "equip",
                  "reference", "automatic", "control primitive"],
        conditions=["placing an inventory item into the world"],
        logical_implications=[
            "converts inventory Item to world Block",
            "placed block becomes findable by bot.findBlock",
            "handles equipping and positioning automatically",
        ],
    ),

    # Capability overview (what placeItem handles vs what caller must handle)
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="placeItem_internal_capabilities",
        fact=(
            "placeItem(bot, name, position) is a high-level placement primitive that internally handles: "
            "1) equipping the item to hand, 2) searching all 6 adjacent directions for a solid reference block, "
            "3) calculating face vector for placement. "
            "Compared to raw bot.placeBlock(referenceBlock, faceVector), placeItem removes the need to "
            "manually find reference blocks and calculate face vectors. "
            "The caller must provide a target POSITION that is air and not occupied by entities. "
            "placeItem checks 6 face-adjacent blocks of the target position for a solid reference, "
            "and has internal adaptive fallback (move, dig) if none found. "
            "If all strategies fail, placeItem throws "
            "'no valid reference block found. Cannot place a floating block.' "
            "The CALLER should provide a position with adjacent solid blocks to avoid relying on fallback."
        ),
        keywords=["placeitem", "place", "capability", "reference", "collision", "face",
                  "bot.placeblock", "placeblock", "crafting_table"],
        logical_implications=[
            "placeItem handles reference block search — caller only needs to provide target position",
            "placeItem handles face vector calculation — caller doesn't need to compute adjacency",
            "placeItem does NOT create valid positions — if no reachable air block with adjacent solid exists, caller must first modify the environment (e.g., mine blocks to create air space)",
            "manual bot.placeBlock requires the caller to find reference block and face vector themselves",
        ],
    ),
]


# ============================================================
# Control Primitive Calling Convention Knowledge
# ============================================================

PRIMITIVE_CALLING_CONVENTION: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="control_primitives_are_standalone_functions",
        fact=(
            "Control primitives (mineBlock, craftItem, smeltItem, placeItem, killMob, "
            "exploreUntil, useChest, shoot, givePlacedItemBack, waitForMobRemoved) are "
            "STANDALONE functions, NOT methods on the bot object. "
            "CORRECT usage: await placeItem(bot, 'crafting_table', position). "
            "WRONG usage: await bot.placeItem('crafting_table', position) — "
            "bot has NO such method and this will cause 'bot.xxx is not a function' error. "
            "Always pass 'bot' as the FIRST argument to these standalone functions."
        ),
        keywords=["placeitem", "mineblock", "craftitem", "smeltitem", "killmob",
                  "exploreuntil", "usechest", "bot method", "standalone", "function",
                  "not a function", "is not a function", "control primitive"],
        conditions=["calling any control primitive"],
        implications=[
            "use await placeItem(bot, ...) not await bot.placeItem(...)",
            "use await mineBlock(bot, ...) not await bot.mineBlock(...)",
            "bot.xxx is not a function means a primitive was called as bot method",
        ],
    ),
]


def get_all_placement_knowledge() -> List[KnowledgeItem]:
    """Get all placement primitive knowledge items."""
    return PLACEMENT_KNOWLEDGE.copy() + PRIMITIVE_CALLING_CONVENTION.copy()
