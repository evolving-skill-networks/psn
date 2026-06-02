"""
API Knowledge - Inventory
Mineflayer inventory API knowledge

Migrated from api_behaviors.py
All code examples removed, only behavior descriptions retained
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# Inventory Methods Knowledge
# ============================================================

INVENTORY_KNOWLEDGE: List[KnowledgeItem] = [
    # bot.inventory.items.logical_implications encodes the .reduce() pattern
    # for counting stacked items — NOT captured in the fact field. Removing
    # this re-introduces a stacked-items counting bug where filter().length
    # silently undercounts a single multi-count slot.
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="bot.inventory.items",
        fact=(
            "bot.inventory.items() returns array of Item objects in inventory. "
            "Each Item object has properties: name (string), count (stack quantity), type (numeric ID), slot (index). "
            "Same-type items are stacked into ONE Item object — e.g., 8 birch_log in one slot produces "
            "one Item with count=8. The array .length is the number of distinct item stacks (slots occupied), "
            "NOT the total number of items. Returns empty array if inventory is empty."
        ),
        keywords=["inventory", "items", "list", "array", "all", "length", "count",
                  "stack", "slot", "only have", "need"],
        conditions=["listing inventory contents", "counting items in inventory",
                    "inventory count mismatch"],
        logical_implications=[
            "returns Item objects — each has .name (string) and .count (quantity in that stack)",
            ".filter(...).length counts the number of matching item STACKS, not total item quantity",
            "to get total quantity: .filter(...).reduce((sum, item) => sum + item.count, 0)",
            "8 birch_log in 1 stack → .filter(log).length is 1, but total count is 8",
            "does not include armor slots",
        ],
        source="study04_retained_r12_production_fix",
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="bot.inventory.count",
        fact=(
            "bot.inventory.count(itemType, metadata) returns total count of items matching the type. "
            "itemType is the numeric item ID, not string name. "
            "Returns 0 if item not found, not null or undefined."
        ),
        keywords=["inventory", "count", "amount", "quantity", "id"],
        conditions=["counting specific items"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="bot.inventory.findInventoryItem",
        fact=(
            "bot.inventory.findInventoryItem(itemType, metadata) returns first Item matching criteria. "
            "Returns null if not found. itemType is numeric item ID. "
            "Only returns first match; use items() to find all."
        ),
        keywords=["inventory", "find", "item", "first", "null"],
        conditions=["finding a specific item"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="bot.inventory.emptySlotCount",
        fact=(
            "bot.inventory.emptySlotCount() returns number of empty slots in inventory. "
            "Does not include armor slots or off-hand. "
            "Does not account for stackable items that could fit in existing stacks."
        ),
        keywords=["inventory", "empty", "slots", "count", "space", "full"],
        conditions=["checking available space"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="bot.inventory.slots",
        fact=(
            "bot.inventory.slots is raw array of all inventory slots including armor. "
            "Empty slots are null. Slot indices are version-dependent. "
            "Low-level access; prefer items() for most use cases."
        ),
        keywords=["inventory", "slots", "raw", "array", "armor", "null"],
        conditions=["low-level inventory access"],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="inventory_item_id",
        fact=(
            "Inventory methods use numeric item IDs, not string names. "
            "Convert name to ID: mcData.itemsByName['item_name'].id "
            "Example: mcData.itemsByName['diamond'].id returns the numeric ID for diamonds."
        ),
        keywords=["inventory", "id", "name", "mcdata", "convert", "numeric"],
        conditions=["using inventory methods"],
    ),

    # Armor slot knowledge
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="armor_slot_indices",
        fact=(
            "Armor slots in player inventory (1.19+): helmet=5, chestplate=6, leggings=7, boots=8. "
            "Offhand slot is index 45. Hotbar is slots 36-44, main inventory is 9-35. "
            "Use bot.equip(item, 'head'/'torso'/'legs'/'feet'/'off-hand') for equipping."
        ),
        keywords=["armor", "slot", "helmet", "chestplate", "leggings", "boots", "offhand", "index"],
        conditions=["equipping armor", "checking armor slots", "accessing equipment"],
    ),

    # Stack size rules
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="stack_size_rules",
        fact=(
            "Most items stack to 64. Eggs, snowballs, ender pearls, signs stack to 16. "
            "Tools, weapons, armor, potions, beds, boats DO NOT stack (stack size 1). "
            "Unstackable items fill inventory slots quickly."
        ),
        keywords=["stack", "size", "64", "16", "unstackable", "limit"],
        conditions=["managing inventory space", "collecting items"],
    ),

    # Window IDs
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="inventory_window_ids",
        fact=(
            "Container window IDs: player inventory is 0, opened containers start at 1. "
            "Chests, crafting tables, furnaces, etc. each open as separate windows. "
            "Wait for 'windowOpen' event before interacting with container contents."
        ),
        keywords=["window", "id", "chest", "crafting table", "furnace", "container", "open"],
        conditions=["interacting with containers", "opening chests"],
    ),

    # Equip methods
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="bot.equip",
        fact=(
            "bot.equip(item, destination) moves item to specified slot. "
            "Destinations: 'hand' (mainhand), 'off-hand', 'head', 'torso', 'legs', 'feet'. "
            "Returns promise that resolves when equipped or rejects if item not found."
        ),
        keywords=["equip", "hand", "armor", "hold", "wear", "destination"],
        conditions=["equipping items", "holding tools"],
    ),

    # blocksByName vs itemsByName confusion
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="itemsByName_vs_blocksByName",
        fact=(
            "minecraft-data has TWO separate registries: mcData.itemsByName (for items/inventory) "
            "and mcData.blocksByName (for world blocks). "
            "Some names exist in both (e.g., oak_log, birch_planks, crafting_table) but their numeric IDs may differ. "
            "bot.inventory methods (findInventoryItem, count) use ITEM IDs from mcData.itemsByName. "
            "bot.findBlock uses BLOCK IDs from mcData.blocksByName. "
            "Using blocksByName ID for inventory lookup may return wrong item or null."
        ),
        keywords=["itemsByName", "blocksByName", "inventory", "findInventoryItem", "count",
                  "id", "mcdata", "wrong", "null", "item", "block"],
        conditions=["inventory lookup returns null unexpectedly", "wrong item count"],
    ),

    # Toss/Drop methods
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.INVENTORY,
        name="bot.toss",
        fact=(
            "bot.toss(itemType, metadata, count) drops items from inventory. "
            "Items appear as dropped entities in world. "
            "Others can pick up dropped items. Count is optional (all by default)."
        ),
        keywords=["toss", "drop", "throw", "discard", "inventory"],
        conditions=["dropping items", "clearing inventory space"],
    ),
]


def get_all_inventory_knowledge() -> List[KnowledgeItem]:
    """Get all inventory-related knowledge items."""
    return INVENTORY_KNOWLEDGE.copy()
