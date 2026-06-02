"""
Primitive Knowledge - Mining
mineBlock primitive knowledge

Migrated from primitive_knowledge.py
All fix_hint, fix_strategy, check_hint fields removed
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# mineBlock Primitive Knowledge
# ============================================================

MINING_KNOWLEDGE: List[KnowledgeItem] = [
    # Preconditions
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="mineBlock_block_exists",
        fact=(
            "mineBlock requires the target block to exist and be reachable within search range. "
            "Uses bot.findBlock() to locate blocks within maxDistance (typically 32 blocks). "
            "Block must be in a loaded chunk to be found."
        ),
        keywords=["mineblock", "find", "block", "exists", "reachable", "search",
                  "mine", "dig", "stone"],  # generic mining tokens
        conditions=["calling mineBlock"],
        implications=[
            "fails if block not in range",
            "chunk must be loaded",
            "search range is limited",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="mineBlock_tool_requirement",
        fact=(
            "mineBlock auto-equips appropriate tool for the block type. "
            "Pickaxe for stone/ores, axe for wood, shovel for dirt/sand. "
            "Without correct tool, mining is very slow or drops nothing."
        ),
        keywords=["mineblock", "tool", "pickaxe", "axe", "shovel", "equip"],
        conditions=["mining any block"],
        implications=[
            "tool is auto-equipped",
            "wrong tool = slow mining",
            "wrong tier = no drops for ores",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="mineBlock_interaction_range",
        fact=(
            "mineBlock requires being within 4 blocks of the target to start mining. "
            "Pathfinder moves bot to the block before mining. "
            "If path is blocked, mining cannot start."
        ),
        keywords=["mineblock", "range", "distance", "reach", "path",
                  "mine", "dig", "stone", "block"],  # generic mining tokens
        conditions=["mining blocks"],
        implications=[
            "pathfinding required first",
            "blocked paths prevent mining",
            "4 block interaction range",
        ],
    ),

    # Failure patterns
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="mineBlock_not_found",
        fact=(
            "mineBlock fails with 'No X found nearby' when target block doesn't exist in search range. "
            "Common causes: wrong Y level for ores, biome doesn't have the resource, chunk not loaded. "
            "Diamond ore only below Y=16, emerald only in mountains, etc."
        ),
        keywords=["mineblock", "not found", "nearby", "search", "range", "ore"],
        conditions=["block search fails"],
        implications=[
            "check Y level for ores",
            "check biome restrictions",
            "explore to load new chunks",
            "increase search radius",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="mineBlock_wrong_tool",
        fact=(
            "mineBlock fails or produces no drops when using wrong tool tier. "
            "Iron ore needs stone pickaxe+, diamond ore needs iron pickaxe+, obsidian needs diamond pickaxe. "
            "Block breaks but drops nothing with insufficient tool tier."
        ),
        keywords=["mineblock", "tool", "tier", "drops", "pickaxe", "wrong"],
        conditions=["mining with incorrect tool"],
        implications=[
            "check tool tier requirements",
            "upgrade pickaxe before mining",
            "block is destroyed without drops",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="mineBlock_path_blocked",
        fact=(
            "mineBlock fails with 'Path blocked' or 'Cannot reach' when pathfinding fails. "
            "Obstacles include walls, water, lava, or terrain features. "
            "Bot cannot start mining without reaching the block."
        ),
        keywords=["mineblock", "path", "blocked", "reach", "timeout", "obstacle",
                  "mine", "dig"],  # generic mining tokens
        conditions=["pathfinding to block fails"],
        implications=[
            "obstacles block access",
            "may need to dig through",
            "try different approach angle",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="mineBlock_inventory_full",
        fact=(
            "mineBlock fails with 'NoChests' error when inventory is full. "
            "collectBlock plugin tries to empty inventory but no chest configured. "
            "Block IS mined but dropped items cannot be collected."
        ),
        keywords=["mineblock", "inventory", "full", "nochests", "collect", "space"],
        conditions=["mining with full inventory"],
        implications=[
            "make space before mining",
            "drop low-value items",
            "use bot.dig() as fallback",
        ],
    ),

    # Name matching behavior
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="mineBlock_strict_name_matching",
        fact=(
            "mineBlock(bot, name, count) internally searches for blocks using "
            "strict equality: block.name === name. It does NOT match variants or prefixes. "
            "For example, mineBlock(bot, 'diamond_ore', 3) finds ONLY blocks named exactly "
            "'diamond_ore', NOT 'deepslate_diamond_ore'. "
            "Below Y=0, virtually all ores are deepslate variants (deepslate_diamond_ore, "
            "deepslate_iron_ore, etc). mineBlock(bot, 'diamond_ore', 3) at Y=-59 will find "
            "zero blocks because only deepslate_diamond_ore exists at that depth. "
            "When zero blocks are found, mineBlock produces an opaque error: "
            "'Cannot read properties of undefined (reading y)'."
        ),
        keywords=["mineblock", "name", "strict", "matching", "deepslate", "diamond_ore",
                  "deepslate_diamond_ore", "variant", "not found", "undefined", "y",
                  "reading 'y'", "Cannot read properties"],
        conditions=[
            "mining ores below Y=0",
            "mineBlock finds zero blocks",
            "Cannot read properties of undefined",
        ],
        logical_implications=[
            "mineBlock(bot, 'diamond_ore', 3) at Y=-59 finds nothing — only deepslate_diamond_ore exists there",
            "mineBlock(bot, 'iron_ore', 5) at Y=-20 finds nothing — only deepslate_iron_ore exists there",
            "To mine deepslate ores, the deepslate_ prefix variant name must be used",
            "Near Y=0, both regular and deepslate variants may coexist",
            "'Cannot read properties of undefined (reading y)' from mineBlock usually means zero blocks matched the name",
        ],
    ),

    # Tool equip behavior
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="mineBlock_tool_equip_no_harvest_check",
        fact=(
            "mineBlock auto-equips tools using requireHarvest=false, which selects the FASTEST "
            "available tool for the block type, not necessarily one that can harvest drops. "
            "For blocks requiring specific tool tiers (diamond ore → iron+ pickaxe, "
            "gold ore → iron+ pickaxe, obsidian → diamond pickaxe), mineBlock may equip a "
            "lower-tier tool that breaks the block but yields zero drops. "
            "The block is destroyed regardless of tool tier — only drop rates are affected."
        ),
        keywords=["mineblock", "tool", "equip", "harvest", "requireHarvest", "drops",
                  "tier", "diamond", "iron", "pickaxe", "zero", "no drops"],
        conditions=[
            "mining tier-sensitive ores",
            "mined blocks but got zero drops",
            "tool tier mismatch",
        ],
        logical_implications=[
            "mineBlock may equip stone_pickaxe for diamond ore — block breaks, zero diamonds dropped",
            "diamond ore requires iron_pickaxe or better to yield diamond drops",
            "gold ore requires iron_pickaxe or better to yield gold drops",
            "the block is always destroyed, even if no drops are obtained",
        ],
    ),

    # General behavior
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="mineBlock_auto_collect",
        fact=(
            "mineBlock automatically collects dropped items after breaking a block. "
            "Uses bot.collectBlock.collect() to pick up items. "
            "Different from bot.dig() which only breaks blocks without collection."
        ),
        keywords=["mineblock", "collect", "pickup", "drops", "items", "auto",
                  "mine", "dig", "stone"],  # generic mining tokens
        conditions=["after successful mining"],
        implications=[
            "items go to inventory",
            "fails if inventory full",
            "waits for inventory sync",
        ],
    ),

    # Capability overview (what mineBlock handles vs what caller must handle)
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="mineBlock_internal_capabilities",
        fact=(
            "mineBlock(bot, name, count) is a high-level mining primitive that internally handles: "
            "1) searching for the named block within range, 2) auto-equipping the best available tool, "
            "3) navigating to the block via pathfinder, 4) mining it with bot.dig, "
            "5) collecting dropped items. "
            "Compared to raw bot.dig(block), mineBlock removes the need to manually find blocks, "
            "equip tools, navigate, and collect drops. "
            "The caller only needs to provide the block NAME string and desired COUNT."
        ),
        keywords=["mineblock", "mine", "capability", "tool", "equip", "navigate", "collect",
                  "bot.dig", "auto",
                  "stone", "cobblestone", "dig", "block"],  # generic mining tokens
        logical_implications=[
            "mineBlock handles tool equipping — caller doesn't need to manually equip",
            "mineBlock handles block searching — caller only provides block name string",
            "mineBlock handles navigation — will pathfind to the block",
            "manual bot.dig requires the caller to find the block, equip tool, and navigate themselves",
        ],
    ),
]


def get_all_mining_knowledge() -> List[KnowledgeItem]:
    """Get all mining primitive knowledge items."""
    return MINING_KNOWLEDGE.copy()
