"""
Game Knowledge - Ore Drops (Per-Entity)
Ore drop entity knowledge: type, drop source, tool requirements, and mcData location for each ore/item

Design principles:
- Entity-based: each KnowledgeItem describes one ore drop item
- Clearly distinguish block vs item to prevent LLM confusion between mcData.blocksByName and itemsByName
- Pure factual description, no solution hints
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# Ore Drop Entity Knowledge
# ============================================================

ORE_DROP_KNOWLEDGE: List[KnowledgeItem] = [

    # ===== raw_iron =====
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="entity_raw_iron",
        fact=(
            "raw_iron is an ITEM, not a block. It does NOT exist in mcData.blocksByName "
            "(mcData.blocksByName['raw_iron'] returns undefined). "
            "raw_iron is obtained by mining iron_ore or deepslate_iron_ore with "
            "stone_pickaxe or better — it drops directly, no smelting needed. "
            "raw_iron can be smelted in a furnace to produce iron_ingot."
        ),
        keywords=["raw_iron", "iron_ore", "deepslate_iron_ore", "blocksByName",
                  "mcData", "undefined", "item", "block", "drop",
                  "stone_pickaxe", "iron_ingot", "smelt",
                  "Cannot read properties"],
        conditions=[
            "mcData.blocksByName.raw_iron returns undefined",
            "needing raw_iron for smelting or crafting",
        ],
        logical_implications=[
            "mcData.blocksByName['raw_iron'] is undefined — raw_iron is not a block",
            "mcData.itemsByName['raw_iron'] is defined — use this for inventory checks",
            "To obtain raw_iron: mine iron_ore or deepslate_iron_ore blocks",
            "iron_ore needs stone_pickaxe or better (NOT iron_pickaxe)",
            "Mining iron_ore directly drops raw_iron — no furnace/smelting step needed",
            "Smelting raw_iron in furnace yields iron_ingot",
        ],
    ),

    # ===== raw_gold =====
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="entity_raw_gold",
        fact=(
            "raw_gold is an ITEM, not a block. It does NOT exist in mcData.blocksByName. "
            "raw_gold is obtained by mining gold_ore or deepslate_gold_ore with "
            "iron_pickaxe or better — it drops directly. "
            "raw_gold can be smelted in a furnace to produce gold_ingot."
        ),
        keywords=["raw_gold", "gold_ore", "deepslate_gold_ore", "blocksByName",
                  "mcData", "undefined", "item", "block", "drop",
                  "iron_pickaxe", "gold_ingot", "smelt"],
        conditions=[
            "mcData.blocksByName.raw_gold returns undefined",
            "needing raw_gold",
        ],
        logical_implications=[
            "mcData.blocksByName['raw_gold'] is undefined — raw_gold is not a block",
            "To obtain raw_gold: mine gold_ore or deepslate_gold_ore",
            "gold_ore needs iron_pickaxe or better",
            "Mining gold_ore directly drops raw_gold — no smelting needed",
        ],
    ),

    # ===== raw_copper =====
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="entity_raw_copper",
        fact=(
            "raw_copper is an ITEM, not a block. It does NOT exist in mcData.blocksByName. "
            "raw_copper is obtained by mining copper_ore or deepslate_copper_ore with "
            "stone_pickaxe or better — it drops directly. "
            "raw_copper can be smelted in a furnace to produce copper_ingot."
        ),
        keywords=["raw_copper", "copper_ore", "deepslate_copper_ore", "blocksByName",
                  "mcData", "undefined", "item", "block", "drop",
                  "stone_pickaxe", "copper_ingot", "smelt"],
        conditions=[
            "mcData.blocksByName.raw_copper returns undefined",
            "needing raw_copper",
        ],
        logical_implications=[
            "mcData.blocksByName['raw_copper'] is undefined — raw_copper is not a block",
            "To obtain raw_copper: mine copper_ore or deepslate_copper_ore",
            "copper_ore needs stone_pickaxe or better",
            "Mining copper_ore directly drops raw_copper — no smelting needed",
        ],
    ),

    # ===== diamond (item) =====
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="entity_diamond_item",
        fact=(
            "diamond is an ITEM obtained by mining diamond_ore or deepslate_diamond_ore "
            "with iron_pickaxe or better. It drops directly — no smelting needed. "
            "diamond_block is a DIFFERENT entity (a storage block crafted from 9 diamonds). "
            "diamond (item) does not exist in mcData.blocksByName."
        ),
        keywords=["diamond", "diamond_ore", "deepslate_diamond_ore", "blocksByName",
                  "item", "block", "drop", "iron_pickaxe", "diamond_block"],
        conditions=[
            "needing diamonds for crafting",
            "mining diamond ore",
        ],
        logical_implications=[
            "diamond (item) is not the same as diamond_block",
            "To obtain diamond: mine diamond_ore/deepslate_diamond_ore with iron_pickaxe+",
            "Mining diamond ore directly drops diamond — no smelting needed",
        ],
    ),

    # ===== coal (item) =====
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="entity_coal_item",
        fact=(
            "coal is an ITEM obtained by mining coal_ore or deepslate_coal_ore "
            "with any pickaxe (wooden or better). It drops directly. "
            "coal_block is a DIFFERENT entity (crafted from 9 coal, used as fuel). "
            "coal (item) is the primary fuel source (smelts 8 items per coal)."
        ),
        keywords=["coal", "coal_ore", "deepslate_coal_ore", "fuel",
                  "item", "block", "drop", "pickaxe", "coal_block", "smelt"],
        conditions=[
            "needing coal for fuel or crafting",
        ],
        logical_implications=[
            "coal (item) is not the same as coal_block",
            "Any pickaxe (wooden+) can mine coal ore",
            "Mining coal ore directly drops coal — no smelting needed",
        ],
    ),
]


def get_all_ore_drop_knowledge() -> List[KnowledgeItem]:
    """Get all ore drop entity knowledge items."""
    return ORE_DROP_KNOWLEDGE.copy()
