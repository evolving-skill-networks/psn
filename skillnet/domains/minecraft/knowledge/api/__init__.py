"""
API Knowledge Package - Mineflayer API knowledge

Contains Mineflayer API behavior knowledge:
- pathfinder.py: pathfinding API behavior (Goals, movements)
- bot_methods.py: Bot method behavior (dig, place, craft, etc.)
- inventory.py: inventory API behavior (items, count, slots)

All knowledge is pure factual description, no code examples.
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem

# Pathfinder
from .pathfinder import (
    PATHFINDER_KNOWLEDGE,
    get_all_pathfinder_knowledge,
)

# Bot Methods
from .bot_methods import (
    BOT_METHODS_KNOWLEDGE,
    SPECIAL_API_KNOWLEDGE,
    BOT_KNOWLEDGE,
    get_all_bot_methods_knowledge,
)

# Inventory
from .inventory import (
    INVENTORY_KNOWLEDGE,
    get_all_inventory_knowledge,
)


def get_all_api_knowledge() -> List[KnowledgeItem]:
    """Get all API knowledge"""
    return (
        get_all_pathfinder_knowledge() +
        get_all_bot_methods_knowledge() +
        get_all_inventory_knowledge()
    )


__all__ = [
    # Base
    "get_all_api_knowledge",

    # Pathfinder
    "PATHFINDER_KNOWLEDGE",
    "get_all_pathfinder_knowledge",

    # Bot Methods
    "BOT_METHODS_KNOWLEDGE",
    "SPECIAL_API_KNOWLEDGE",
    "BOT_KNOWLEDGE",
    "get_all_bot_methods_knowledge",

    # Inventory
    "INVENTORY_KNOWLEDGE",
    "get_all_inventory_knowledge",
]
