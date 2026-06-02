"""
PlanningState: Simulated world state during backward-chaining planning.

Replaces the ad-hoc `expected_inventory = {}` dict with a structured
representation that tracks inventory, equipment, and position through
the skill chain. Each skill's effects are applied to update the state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanningState:
    """Simulated world state accumulated during skill chain construction.

    As the planner resolves skills in sequence, each skill's expected effects
    are applied to this state. Downstream skills can then use the accumulated
    state for parameter resolution (e.g., checking what materials will be
    available from upstream skills).
    """
    inventory: Dict[str, int] = field(default_factory=dict)
    equipment: Dict[str, str] = field(default_factory=dict)
    position: Optional[Dict[str, float]] = None

    @classmethod
    def from_current_state(cls, current_state: Dict[str, Any]) -> "PlanningState":
        """Initialize from a current game state snapshot.

        Args:
            current_state: Dict with optional keys 'inventory', 'equipment', 'position'
        """
        return cls(
            inventory=dict(current_state.get("inventory", {})),
            equipment=dict(current_state.get("equipment", {})),
            position=current_state.get("position"),
        )

    def apply_effects(self, effects: List[Dict[str, Any]]) -> "PlanningState":
        """Apply a list of effect state_representations to update the state.

        Each effect dict should have 'type', 'operation', 'item', 'count'.
        Returns self for chaining.
        """
        for effect in effects:
            effect_type = effect.get("type", "inventory")
            operation = effect.get("operation", "add")
            item = effect.get("item")
            count = effect.get("count", 1)

            if not item:
                continue

            # Normalize count
            if count is None:
                count = 1
            elif isinstance(count, str):
                try:
                    count = int(count)
                except (ValueError, TypeError):
                    count = 1

            if effect_type == "inventory":
                if operation in ("add", "ensure"):
                    self.inventory[item] = self.inventory.get(item, 0) + count
                elif operation == "remove":
                    current = self.inventory.get(item, 0)
                    self.inventory[item] = max(0, current - count)
            elif effect_type == "equipment":
                if operation in ("equip", "add"):
                    slot = effect.get("slot", "mainhand")
                    self.equipment[slot] = item

        return self

    def get_effective_inventory(self) -> Dict[str, int]:
        """Get the current simulated inventory."""
        return dict(self.inventory)

    def has_item(self, item: str, count: int = 1) -> bool:
        """Check if the simulated inventory has at least `count` of `item`."""
        return self.inventory.get(item, 0) >= count
