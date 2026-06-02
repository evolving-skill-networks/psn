"""
ResourceTracker - Tracks resource state and generates alerts for low inventory

Responsibilities:
- Track current inventory state
- Maintain resource thresholds (minimum and target levels)
- Generate alerts when resources fall below thresholds
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from skillnet.core.dk_registry import get_domain_knowledge
from skillnet.utils.event_utils import unpack_event



class AlertUrgency(Enum):
    """Urgency levels for resource alerts"""
    CRITICAL = "critical"  # Below minimum, immediate action needed
    WARNING = "warning"    # Below target but above minimum
    INFO = "info"          # Informational, may want to stockpile


@dataclass
class ResourceThreshold:
    """Threshold configuration for a resource or resource group"""
    item: str             # Item name or group name (e.g., "logs", "cooked_food")
    min_count: int        # Below this triggers CRITICAL alert
    target_count: int     # Below this triggers WARNING alert
    priority: int = 5     # 1-10, higher = more important
    is_consumable: bool = False  # Consumables (food, fuel) need special handling
    is_group: bool = False  # True if this threshold tracks a group of items

    def __post_init__(self):
        """Auto-detect if item is a group name"""
        dk = get_domain_knowledge()
        item_groups = dk.get_item_groups() if dk else {}
        if self.item in item_groups:
            self.is_group = True

    @property
    def group_items(self) -> List[str]:
        """Get list of items in the group (or single item if not a group)"""
        if self.is_group:
            dk = get_domain_knowledge()
            item_groups = dk.get_item_groups() if dk else {}
            if self.item in item_groups:
                return item_groups[self.item]
        return [self.item]

    def to_dict(self) -> Dict:
        return {
            "item": self.item,
            "min_count": self.min_count,
            "target_count": self.target_count,
            "priority": self.priority,
            "is_consumable": self.is_consumable,
            "is_group": self.is_group
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ResourceThreshold':
        return cls(**data)


@dataclass
class ResourceAlert:
    """An alert about a resource that needs attention"""
    item: str
    current_count: int
    threshold: ResourceThreshold
    urgency: AlertUrgency
    suggested_action: str

    @property
    def deficit(self) -> int:
        """How many items needed to reach target"""
        return max(0, self.threshold.target_count - self.current_count)

    def to_dict(self) -> Dict:
        return {
            "item": self.item,
            "current_count": self.current_count,
            "deficit": self.deficit,
            "urgency": self.urgency.value,
            "priority": self.threshold.priority,
            "suggested_action": self.suggested_action
        }


_FALLBACK_THRESHOLDS: Dict[str, ResourceThreshold] = {}


def _build_thresholds_from_domain(domain_knowledge) -> Dict[str, ResourceThreshold]:
    """Build ResourceThreshold dict from domain knowledge dict format."""
    raw = domain_knowledge.get_resource_thresholds()
    if not raw:
        return {}
    result = {}
    for item, config in raw.items():
        result[item] = ResourceThreshold(
            item=item,
            min_count=config.get("min_count", 0),
            target_count=config.get("target_count", 0),
            priority=config.get("priority", 5),
            is_consumable=config.get("is_consumable", False),
        )
    return result


class ResourceTracker:
    """
    Tracks resource state and generates alerts.

    Features:
    - Maintains current inventory snapshot
    - Checks resources against thresholds
    - Generates prioritized alerts
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, ResourceThreshold]] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the resource tracker.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
            config_path: Path to JSON config file to load/save thresholds
        """
        if thresholds is not None:
            self.thresholds = thresholds
        elif get_domain_knowledge():
            self.thresholds = _build_thresholds_from_domain(get_domain_knowledge())
        else:
            self.thresholds = _FALLBACK_THRESHOLDS.copy()
        self.config_path = config_path

        # Current state
        self.current_inventory: Dict[str, int] = {}

        # Load config if path provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, path: str) -> None:
        """Load thresholds from JSON config file"""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            for item, data in config.get("thresholds", {}).items():
                self.thresholds[item] = ResourceThreshold.from_dict(data)
        except Exception as e:
            print(f"Warning: Failed to load resource config: {e}")

    def update_inventory(self, events: List[Any]) -> None:
        """
        Update current inventory from events.

        Args:
            events: List of (event_type, event_data) tuples from PSN
        """
        if not events:
            return

        # Get the most recent inventory state. Events may be tuple-shaped
        # (Minecraft) or dict-shaped (a dict-event domain); route through ``unpack_event``
        # so both are tolerated rather than crashing on a raw tuple-unpack.
        for ev in reversed(events):
            event_type, event_data = unpack_event(ev)
            if event_type is None:
                continue
            if isinstance(event_data, dict) and "inventory" in event_data:
                new_inventory = event_data["inventory"]
                if isinstance(new_inventory, dict):
                    self.current_inventory = new_inventory.copy()
                    break

    def get_count(self, item: str) -> int:
        """Get current count of an item"""
        return self.current_inventory.get(item, 0)

    def has_enough(self, item: str, count: int) -> bool:
        """Check if inventory has at least `count` of an item."""
        return self.get_count(item) >= count

    def has_any(self, items: List[str]) -> bool:
        """Check if inventory has at least 1 of any item in the list."""
        return any(self.get_count(item) > 0 for item in items)

    def add_threshold(
        self,
        item: str,
        min_count: int,
        target_count: int,
        priority: int = 5,
        is_consumable: bool = False,
    ) -> None:
        """Add or update a resource threshold."""
        self.thresholds[item] = ResourceThreshold(
            item=item,
            min_count=min_count,
            target_count=target_count,
            priority=priority,
            is_consumable=is_consumable,
        )

    def remove_threshold(self, item: str) -> None:
        """Remove a threshold from tracking."""
        self.thresholds.pop(item, None)

    def get_group_count(self, group_or_item: str) -> int:
        """
        Get total count of items in a group, or single item count.

        Supports group aliases: e.g., when ENABLE_WOOD_AS_FUEL is True,
        counting 'fuel' also includes items from the 'logs' group.

        Args:
            group_or_item: Group name (e.g., "logs") or item name (e.g., "oak_log")

        Returns:
            Total count of all items in the group, or single item count
        """
        dk = get_domain_knowledge()
        item_groups = dk.get_item_groups() if dk else {}
        if group_or_item in item_groups:
            # Sum all items in the group
            total = sum(
                self.current_inventory.get(item, 0)
                for item in item_groups[group_or_item]
            )

            # Check group aliases (e.g., fuel -> also count logs)
            group_aliases = dk.get_group_aliases() if dk else {}
            for alias_group in group_aliases.get(group_or_item, []):
                if alias_group in item_groups:
                    total += sum(self.current_inventory.get(item, 0) for item in item_groups[alias_group])

            return total
        else:
            return self.current_inventory.get(group_or_item, 0)

    def check_resource_levels(self) -> List[ResourceAlert]:
        """
        Check all tracked resources against thresholds.

        Returns:
            List of ResourceAlerts, sorted by priority (highest first)
        """
        alerts: List[ResourceAlert] = []

        for item, threshold in self.thresholds.items():
            # Use group count if this is a group threshold
            current = self.get_group_count(item)

            if current < threshold.min_count:
                # Critical - below minimum
                # Use target_count as the goal rather than delta (target_count - current)
                # This ensures the task uses TARGET semantics: "Ensure you have 26 logs" rather than "Ensure you have 11 logs"
                # After completion inventory will be >= target_count >= min_count, avoiding repeated CRITICAL alerts
                alerts.append(ResourceAlert(
                    item=item,
                    current_count=current,
                    threshold=threshold,
                    urgency=AlertUrgency.CRITICAL,
                    suggested_action=self._get_gather_action(item, threshold.target_count)
                ))
            elif current < threshold.target_count:
                # Warning - below target
                # Likewise use target_count as the goal
                alerts.append(ResourceAlert(
                    item=item,
                    current_count=current,
                    threshold=threshold,
                    urgency=AlertUrgency.WARNING,
                    suggested_action=self._get_gather_action(item, threshold.target_count)
                ))

        # Sort by priority (highest first) and urgency (critical first)
        alerts.sort(key=lambda a: (-a.threshold.priority, a.urgency.value))
        return alerts

    def get_critical_alerts(self) -> List[ResourceAlert]:
        """Get only critical alerts (below minimum threshold)"""
        return [a for a in self.check_resource_levels() if a.urgency == AlertUrgency.CRITICAL]

    def _get_gather_action(self, item: str, count: int) -> str:
        """Generate a suggested action to gather an item or item group.

        PSN Curriculum semantic design:
        =========================
        All tasks use TARGET semantics ("Ensure you have X" means "make sure you end up with X").

        This is consistent with the ResourceTracker threshold-based system:
        - target_count=26, current=5 → emits "Ensure you have 26 logs"
        - Evaluation checks final inventory >= 26, not the delta

        Benefits of using "Ensure you have":
        1. Explicit and natural semantics: both Graph Planner and Critic understand directly
        2. Consistent with skill naming: ensureLogs, ensurePlanks, etc.
        3. No ambiguity: no longer need backend semantic inference
        """
        # Handle item groups with generic actions
        if item == "logs":
            return f"Ensure you have {count} wood logs"
        elif item == "planks":
            return f"Ensure you have {count} planks"
        elif item == "cooked_food":
            return f"Ensure you have {count} cooked food"
        elif item == "fuel":
            return f"Ensure you have {count} fuel (coal or charcoal)"
        elif item == "pickaxes":
            return f"Ensure you have a pickaxe"
        elif item == "axes":
            return f"Ensure you have an axe"
        elif item == "swords":
            return f"Ensure you have a sword"

        # Handle specific items
        if "log" in item:
            return f"Ensure you have {count} {item.replace('_', ' ')}"
        elif item in ["cobblestone", "stone"]:
            return f"Ensure you have {count} {item.replace('_', ' ')}"
        elif "ore" in item or item in ["coal", "charcoal", "iron_ingot", "gold_ingot", "diamond"]:
            return f"Ensure you have {count} {item.replace('_', ' ')}"
        elif "cooked" in item:
            return f"Ensure you have {count} {item.replace('_', ' ')}"
        elif item == "torch":
            return f"Ensure you have {count} torch"
        elif item == "stick":
            return f"Ensure you have {count} stick"
        elif "planks" in item:
            return f"Ensure you have {count} {item.replace('_', ' ')}"
        else:
            return f"Ensure you have {count} {item.replace('_', ' ')}"

    def get_inventory_summary(self) -> str:
        """Get a human-readable summary of tracked resources"""
        lines = ["Current Inventory Status:"]
        for item, threshold in sorted(self.thresholds.items(), key=lambda x: -x[1].priority):
            current = self.get_count(item)
            status = ""
            if current < threshold.min_count:
                status = " [CRITICAL]"
            elif current < threshold.target_count:
                status = " [LOW]"
            lines.append(f"  {item}: {current}/{threshold.target_count}{status}")
        return "\n".join(lines)
