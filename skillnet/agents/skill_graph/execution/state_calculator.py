"""
State Change Calculator Module

Compute state changes before and after execution.

v5.0 architecture reorganization - migrated from graph_manager_impl.py
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from skillnet.agents.skill_graph.models import ActualEffect

logger = logging.getLogger(__name__)


class StateChangeCalculator:
    """
    State-change calculator.

    Responsibilities:
    1. Compute inventory changes between pre- and post-execution
    2. Compute position changes
    3. Compute equipment changes
    4. Extract block changes from events
    5. Produce an ActualEffect object
    """

    def __init__(self, custom_logger: Optional[logging.Logger] = None):
        """
        Initialize the state-change calculator.

        Args:
            custom_logger: optional custom logger
        """
        self.logger = custom_logger or logger

    def calculate(
        self,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
        environment_events: Optional[List[Any]] = None,
    ) -> ActualEffect:
        """
        Compute state changes before and after execution.

        Args:
            pre_state: pre-execution state
            post_state: post-execution state
            environment_events: list of environment events

        Returns:
            ActualEffect: the computed actual effect
        """
        inventory_changes = {}
        position_changes = {}
        equipment_changes = {}
        block_changes = []
        state_changes = []

        # 1. Compute inventory changes
        inventory_changes, inv_state_changes = self._calculate_inventory_changes(
            pre_state.get("inventory", {}),
            post_state.get("inventory", {}),
        )
        state_changes.extend(inv_state_changes)

        # 2. Compute position changes
        position_changes = self._calculate_position_changes(
            pre_state.get("position", {}),
            post_state.get("position", {}),
        )

        # 3. Compute equipment changes
        equipment_changes = self._calculate_equipment_changes(
            pre_state.get("equipment", {}),
            post_state.get("equipment", {}),
        )

        # 4. Extract block changes from events
        if environment_events:
            block_changes = self._extract_block_changes(environment_events)
            # Add block changes to state_changes
            for block_change in block_changes:
                if block_change.get("type") == "place":
                    state_changes.append({
                        "type": "block",
                        "item": block_change.get("block", {}).get("name", "unknown"),
                        "count": 1,
                        "operation": "place"
                    })
                elif block_change.get("type") == "break":
                    state_changes.append({
                        "type": "block",
                        "item": block_change.get("block", {}).get("name", "unknown"),
                        "count": 1,
                        "operation": "remove"
                    })

        # 5. Generate a description
        description = self._generate_description(
            inventory_changes, position_changes, equipment_changes, block_changes
        )

        return ActualEffect(
            execution_id=f"effect_{datetime.now().timestamp()}",
            timestamp=datetime.now().isoformat(),
            description=description,
            state_changes=state_changes,
            inventory_changes=inventory_changes,
            position_changes=position_changes,
            equipment_changes=equipment_changes,
            block_changes=block_changes,
            pre_state=pre_state,
            post_state=post_state,
        )

    def _calculate_inventory_changes(
        self,
        pre_inventory: Dict[str, int],
        post_inventory: Dict[str, int],
    ) -> tuple:
        """
        Compute inventory changes

        Returns:
            Tuple[Dict[str, int], List[Dict]]: (inventory-change dict, state-change list)
        """
        inventory_changes = {}
        state_changes = []

        # Combine all item names
        all_items = set()
        if isinstance(pre_inventory, dict):
            all_items.update(pre_inventory.keys())
        if isinstance(post_inventory, dict):
            all_items.update(post_inventory.keys())

        for item_name in all_items:
            pre_count = pre_inventory.get(item_name, 0) if isinstance(pre_inventory, dict) else 0
            post_count = post_inventory.get(item_name, 0) if isinstance(post_inventory, dict) else 0
            change = post_count - pre_count

            if change != 0:
                inventory_changes[item_name] = change
                if change > 0:
                    state_changes.append({
                        "type": "inventory",
                        "item": item_name,
                        "count": change,
                        "operation": "add"
                    })
                else:
                    state_changes.append({
                        "type": "inventory",
                        "item": item_name,
                        "count": abs(change),
                        "operation": "remove"
                    })

        return inventory_changes, state_changes

    def _calculate_position_changes(
        self,
        pre_pos: Dict[str, float],
        post_pos: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute position changes"""
        position_changes = {}

        if isinstance(pre_pos, dict) and isinstance(post_pos, dict):
            for axis in ["x", "y", "z"]:
                pre_val = pre_pos.get(axis, 0)
                post_val = post_pos.get(axis, 0)
                change = post_val - pre_val
                if abs(change) > 0.01:  # ignore very small changes
                    position_changes[axis] = change

        return position_changes

    def _calculate_equipment_changes(
        self,
        pre_equipment: Dict[str, Any],
        post_equipment: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute equipment changes"""
        equipment_changes = {}

        if isinstance(pre_equipment, dict) and isinstance(post_equipment, dict):
            from skillnet.core.dk_registry import get_domain_knowledge
            _dk = get_domain_knowledge()
            _slot_names = _dk.get_equipment_slot_names() if _dk else []
            for slot in _slot_names:
                pre_item = pre_equipment.get(slot)
                post_item = post_equipment.get(slot)
                if pre_item != post_item:
                    equipment_changes[slot] = {"before": pre_item, "after": post_item}

        return equipment_changes

    def _extract_block_changes(
        self,
        environment_events: List[Any],
    ) -> List[Dict[str, Any]]:
        """Extract block changes from events"""
        block_changes = []

        for event in environment_events:
            if isinstance(event, dict):
                event_type = event.get("type", "")
                event_data = event.get("data", {})

                if event_type == "onBlockBreak":
                    block_changes.append({
                        "type": "break",
                        "position": event_data.get("position"),
                        "block": event_data.get("block"),
                    })
                elif event_type == "onBlockPlace":
                    block_changes.append({
                        "type": "place",
                        "position": event_data.get("position"),
                        "block": event_data.get("block"),
                    })

        return block_changes

    def _generate_description(
        self,
        inventory_changes: Dict[str, int],
        position_changes: Dict[str, float],
        equipment_changes: Dict[str, Dict[str, Any]],
        block_changes: List[Dict[str, Any]],
    ) -> str:
        """Generate the description text"""
        description_parts = []

        if inventory_changes:
            consumed = [f"{abs(v)} {k}" for k, v in inventory_changes.items() if v < 0]
            produced = [f"{v} {k}" for k, v in inventory_changes.items() if v > 0]
            if consumed:
                description_parts.append(f"Consumes {', '.join(consumed)}")
            if produced:
                description_parts.append(f"Produces {', '.join(produced)}")

        if position_changes:
            pos_str = ", ".join([f"{k}: {v:.2f}" for k, v in position_changes.items()])
            description_parts.append(f"Position change: {pos_str}")

        if equipment_changes:
            eq_str = ", ".join([
                f"{k}: {v['before']} -> {v['after']}"
                for k, v in equipment_changes.items()
            ])
            description_parts.append(f"Equipment change: {eq_str}")

        if block_changes:
            breaks = len([b for b in block_changes if b.get("type") == "break"])
            places = len([b for b in block_changes if b.get("type") == "place"])
            if breaks > 0:
                description_parts.append(f"Breaks {breaks} block(s)")
            if places > 0:
                description_parts.append(f"Places {places} block(s)")

        return ". ".join(description_parts) if description_parts else "No significant state changes detected"


# ============================================================================
# Pure Function (for backward compatibility)
# ============================================================================

def calculate_state_changes(
    pre_state: Dict[str, Any],
    post_state: Dict[str, Any],
    environment_events: Optional[List[Any]] = None,
) -> ActualEffect:
    """
    Compute state changes before and after execution (pure-function version)

    Args:
        pre_state: pre-execution state
        post_state: post-execution state
        environment_events: list of environment events

    Returns:
        ActualEffect: the computed actual effect
    """
    calculator = StateChangeCalculator()
    return calculator.calculate(pre_state, post_state, environment_events)
