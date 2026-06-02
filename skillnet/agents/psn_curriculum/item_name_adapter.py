"""
Item Name Adapter - Handles naming conventions across different system layers.

This module provides conversion between three naming conventions used in PSN:
1. Inventory Layer: singular + type (e.g., "oak_log", "birch_planks", "stick")
2. Task Layer: plural/natural forms (e.g., "logs", "planks", "sticks")
3. Knowledge Base Layer: singular + type (same as inventory)

The adapter helps PSN Curriculum correctly:
- Query inventory for task-layer item names
- Look up recipes using correct keys
- Convert between different naming conventions
"""

from typing import List, Dict, Optional

from skillnet.core.dk_registry import get_domain_knowledge, register_on_change


def _get_item_groups():
    dk = get_domain_knowledge()
    return dk.get_item_groups() if dk else {}


def _get_item_to_group():
    dk = get_domain_knowledge()
    return dk.get_item_to_group_mapping() if dk else {}


class ItemNameAdapter:
    """
    Converts item names between different system layers.

    Layer 1 (Inventory): "oak_log", "birch_planks", "stick"
    Layer 2 (Task): "logs", "planks", "sticks", "wood logs"
    Layer 3 (Knowledge Base): Same as inventory layer
    """

    # Task layer natural language → ITEM_GROUPS key mapping
    # Populated from DK via on-change hook
    TASK_TO_GROUP_KEY: Dict[str, str] = {}

    # Plural → singular for recipe lookup — populated from DK via on-change hook
    PLURAL_TO_SINGULAR: Dict[str, str] = {}

    # Singular → plural for task description generation — populated from DK via on-change hook
    SINGULAR_TO_PLURAL: Dict[str, str] = {}

    @classmethod
    def get_inventory_items(cls, task_item: str) -> List[str]:
        """
        Get all inventory keys that match a task-layer item name.

        Args:
            task_item: Item name from task layer (e.g., "logs", "planks", "sticks")

        Returns:
            List of inventory keys (e.g., ["oak_log", "birch_log", ...])
        """
        task_item_lower = task_item.lower().strip()

        # Check if it maps to an ITEM_GROUPS key
        group_key = cls.TASK_TO_GROUP_KEY.get(task_item_lower)
        if group_key and group_key in _get_item_groups():
            return _get_item_groups()[group_key]

        # Check if it's already an ITEM_GROUPS key
        if task_item_lower in _get_item_groups():
            return _get_item_groups()[task_item_lower]

        # Convert plural to singular for direct inventory lookup
        singular = cls.PLURAL_TO_SINGULAR.get(task_item_lower, task_item_lower)

        # Check if singular form maps to a group
        if singular in _get_item_to_group():
            group_key = _get_item_to_group()[singular]
            return _get_item_groups()[group_key]

        # Return as single-item list (already in inventory format)
        return [singular]

    @classmethod
    def get_recipe_key(cls, task_item: str) -> str:
        """
        Get the recipe key for a task-layer item name.

        Args:
            task_item: Item name from task layer (e.g., "sticks", "planks")

        Returns:
            Recipe key for knowledge base lookup (e.g., "stick", "oak_planks")
        """
        task_item_lower = task_item.lower().strip()

        # Handle group names - for groups, we need to pick a specific item
        group_key = cls.TASK_TO_GROUP_KEY.get(task_item_lower)
        if group_key and group_key in _get_item_groups():
            # For groups like "planks", return the first variant (e.g., "oak_planks")
            items = _get_item_groups()[group_key]
            return items[0] if items else task_item_lower

        if task_item_lower in _get_item_groups():
            items = _get_item_groups()[task_item_lower]
            return items[0] if items else task_item_lower

        # Convert plural to singular
        return cls.PLURAL_TO_SINGULAR.get(task_item_lower, task_item_lower)

    @classmethod
    def get_inventory_count(cls, task_item: str, inventory: Dict[str, int]) -> int:
        """
        Get the total inventory count for a task-layer item name.

        Args:
            task_item: Item name from task layer (e.g., "logs", "sticks")
            inventory: Current inventory dict

        Returns:
            Total count of matching items in inventory
        """
        items = cls.get_inventory_items(task_item)
        return sum(inventory.get(item, 0) for item in items)

    @classmethod
    def normalize_for_task(cls, item: str, count: int) -> str:
        """
        Create natural language task description for an item.

        Args:
            item: Item name (any format)
            count: Number of items

        Returns:
            Natural language item description (e.g., "4 oak logs", "8 sticks")
        """
        item_lower = item.lower().strip()

        # Check if it's a known group
        if item_lower in _get_item_groups() or item_lower in cls.TASK_TO_GROUP_KEY.values():
            # Use the group name directly (already plural-ish)
            return f"{count} {item_lower.replace('_', ' ')}"

        # For specific items, use singular/plural appropriately
        if count == 1:
            singular = cls.PLURAL_TO_SINGULAR.get(item_lower, item_lower)
            return f"1 {singular.replace('_', ' ')}"
        else:
            plural = cls.SINGULAR_TO_PLURAL.get(item_lower, item_lower + "s")
            return f"{count} {plural.replace('_', ' ')}"

    @classmethod
    def is_group_name(cls, item: str) -> bool:
        """
        Check if an item name refers to a group.

        Args:
            item: Item name to check

        Returns:
            True if the item is a group name
        """
        item_lower = item.lower().strip()
        return (item_lower in _get_item_groups() or
                item_lower in cls.TASK_TO_GROUP_KEY)

    @classmethod
    def get_group_name(cls, item: str) -> Optional[str]:
        """
        Get the group name for an item if it belongs to a group.

        Args:
            item: Item name to look up

        Returns:
            Group name or None if not in a group
        """
        item_lower = item.lower().strip()

        # Direct group name
        if item_lower in _get_item_groups():
            return item_lower

        # Task to group key mapping
        if item_lower in cls.TASK_TO_GROUP_KEY:
            return cls.TASK_TO_GROUP_KEY[item_lower]

        # Singular item to group
        singular = cls.PLURAL_TO_SINGULAR.get(item_lower, item_lower)
        if singular in _get_item_to_group():
            return _get_item_to_group()[singular]

        return None


# ── On-change hook: populate ItemNameAdapter class dicts when DK changes ──

def _on_dk_changed(dk):
    """Populate ItemNameAdapter class-level dicts from new domain knowledge."""
    if dk:
        p2s = dk.get_plural_to_singular()
        ItemNameAdapter.PLURAL_TO_SINGULAR = p2s
        ItemNameAdapter.SINGULAR_TO_PLURAL = {v: k for k, v in p2s.items()}
        ItemNameAdapter.TASK_TO_GROUP_KEY = dk.get_task_to_group_mapping()
    else:
        ItemNameAdapter.PLURAL_TO_SINGULAR = {}
        ItemNameAdapter.SINGULAR_TO_PLURAL = {}
        ItemNameAdapter.TASK_TO_GROUP_KEY = {}


register_on_change(_on_dk_changed)
