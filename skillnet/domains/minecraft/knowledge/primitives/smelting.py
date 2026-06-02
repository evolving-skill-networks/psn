"""
Primitive Knowledge - Smelting
smeltItem primitive knowledge

Migrated from primitive_knowledge.py
All fix_hint, fix_strategy, check_hint fields removed
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# smeltItem Primitive Knowledge
# ============================================================

SMELTING_KNOWLEDGE: List[KnowledgeItem] = [
    # Preconditions
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="smeltItem_furnace_required",
        fact=(
            "smeltItem requires a furnace to be placed and accessible nearby. "
            "Bot must be within 32 blocks of furnace. "
            "Blast furnace for ores only, smoker for food only."
        ),
        keywords=["smeltitem", "furnace", "nearby", "placed", "accessible"],
        conditions=["calling smeltItem"],
        implications=[
            "need furnace first",
            "place and approach furnace",
            "within interaction range",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="smeltItem_items_to_smelt",
        fact=(
            "smeltItem requires smeltable items in inventory. "
            "Raw ores (raw_iron, raw_gold, raw_copper) smelt to ingots. "
            "Food items can be cooked. Sand smelts to glass."
        ),
        keywords=["smeltitem", "items", "raw", "smelt", "cook", "inventory"],
        conditions=["smelting items"],
        implications=[
            "have items to smelt",
            "item must be smeltable",
            "check inventory count",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="smeltItem_fuel_required",
        fact=(
            "smeltItem requires fuel to power the furnace. "
            "Coal smelts 8 items, charcoal smelts 8, log smelts 1.5, plank smelts 1.5. "
            "Fuel is consumed during smelting process."
        ),
        keywords=["smeltitem", "fuel", "coal", "charcoal", "wood", "consume"],
        conditions=["smelting requires fuel"],
        implications=[
            "must have fuel",
            "coal is most efficient",
            "fuel is consumed",
        ],
    ),

    # Failure patterns
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="smeltItem_no_furnace",
        fact=(
            "smeltItem fails with 'No furnace' or 'Furnace not found' when no furnace nearby. "
            "Furnace must be placed in the world, not just in inventory. "
            "Must be within search range (typically 32 blocks)."
        ),
        keywords=["smeltitem", "no furnace", "not found", "place"],
        conditions=["no furnace available"],
        implications=[
            "craft 8 cobblestone into furnace",
            "place furnace in world",
            "move closer to furnace",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="smeltItem_no_fuel",
        fact=(
            "smeltItem fails with 'No fuel' or 'Out of fuel' when no fuel available. "
            "Check for coal, charcoal, wood planks, or logs in inventory. "
            "Coal is most efficient: 1 coal = 8 items smelted."
        ),
        keywords=["smeltitem", "no fuel", "out of fuel", "coal"],
        conditions=["smelting without fuel"],
        implications=[
            "gather fuel first",
            "coal from mining",
            "charcoal from wood",
        ],
    ),

    # Effects
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="smeltItem_output",
        fact=(
            "smeltItem produces smelted items in furnace output slot. "
            "raw_iron -> iron_ingot, raw_gold -> gold_ingot, raw_copper -> copper_ingot. "
            "IMPORTANT: In Minecraft 1.17+, you smelt raw_iron (NOT iron_ore). "
            "iron_ore is a block that drops raw_iron when mined. "
            "oak_log/birch_log/spruce_log -> charcoal. "
            "Sand -> glass, cobblestone -> stone, food -> cooked food."
        ),
        keywords=["smeltitem", "output", "ingot", "cooked", "glass", "charcoal",
                  "raw_iron", "iron_ore", "iron_ingot"],
        conditions=["successful smelting"],
        implications=[
            "items appear in furnace",
            "need to collect output",
            "wait for smelting to complete",
        ],
    ),

    # Furnace slot semantics
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="smeltItem_furnace_slots",
        fact=(
            "Furnace has two input methods: "
            "putInput(itemId, null, count) places item in the TOP slot (item to be smelted). "
            "putFuel(fuelId, null, count) places fuel in the BOTTOM slot (powers smelting). "
            "Common mistake: putting the item-to-smelt in the fuel slot via putFuel(). "
            "Example: to make charcoal, oak_log goes in putInput() (top), "
            "oak_planks/coal goes in putFuel() (bottom). "
            "To make iron_ingot, raw_iron goes in putInput(), fuel goes in putFuel(). "
            "Prefer using smeltItem(bot, itemName, fuelName, count) which handles slots correctly."
        ),
        keywords=["furnace", "putfuel", "putinput", "slot", "charcoal", "smelt",
                  "openFurnace", "raw_iron", "oak_log"],
        conditions=["direct furnace API usage", "furnace slot confusion"],
        implications=[
            "smeltItem handles slot assignment automatically",
            "putInput for item to smelt (top slot)",
            "putFuel for fuel only (bottom slot)",
        ],
    ),

    # Smelting loop and fuel matching
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="smeltItem_loop_pattern",
        fact=(
            "When smelting multiple items, fuel must match the number of items. "
            "1 coal/charcoal smelts 8 items. 1 oak_planks smelts 1.5 items. 1 oak_log smelts 1.5 items. "
            "The smeltItem() primitive handles this correctly with a per-item loop: "
            "for each item: putFuel(1) -> putInput(1) -> waitForTicks(240) -> takeOutput(). "
            "Common mistake: adding 1 fuel but 3 items — only ~1 item gets smelted. "
            "If not using smeltItem(), ensure fuel count matches: "
            "e.g., 3 raw_iron with oak_planks fuel needs at least 2 planks (not 1)."
        ),
        keywords=["smeltitem", "fuel", "count", "loop", "multiple", "putfuel", "putinput",
                  "oak_planks", "coal", "partial"],
        conditions=["smelting multiple items", "partial smelting"],
        implications=[
            "match fuel count to item count",
            "use per-item loop pattern",
            "smeltItem handles this automatically",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="smeltItem_timing",
        fact=(
            "Each item takes 10 seconds to smelt (200 ticks). "
            "Blast furnace smelts ores in 5 seconds. "
            "Smoker cooks food in 5 seconds."
        ),
        keywords=["smeltitem", "time", "seconds", "ticks", "wait"],
        conditions=["smelting process"],
        implications=[
            "takes time per item",
            "10 seconds per item normally",
            "specialized furnaces faster",
        ],
    ),
]


def get_all_smelting_knowledge() -> List[KnowledgeItem]:
    """Get all smelting primitive knowledge items."""
    return SMELTING_KNOWLEDGE.copy()
