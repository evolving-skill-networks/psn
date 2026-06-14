"""
EventProcessingMixin for PSNAgent

Methods for extracting state from events, calculating state changes,
diagnosing event structure, and building synthetic error events.
"""

import copy
import json
import re
from typing import TYPE_CHECKING

from skillnet._psn_impl.event_helpers import (
    find_last_observe,
    iter_events,
    unpack_event,
)

if TYPE_CHECKING:
    from ..psn import PSNAgent


class EventProcessingMixin:
    """Event processing and state extraction helpers."""

    def _extract_current_state(self, events):
        """
        Extract current state info from the events list.

        Args:
            events: list of events, in [(event_type, event_data), ...] form

        Returns:
            dict: dict with inventory, position, biome, and equipment, or None if extraction fails
                - inventory: {item_name: count}
                - position: {x, y, z}
                - biome: str
                - equipment: {slot_name: item_name} form, aligned with _calculate_state_changes
        """
        if not events or len(events) == 0:
            return None

        observe_data = find_last_observe(events)
        if observe_data is not None:
            equipment_list = observe_data.get("status", {}).get("equipment", [])
            state = {
                "inventory": observe_data.get("inventory", {}),
                "position": observe_data.get("status", {}).get("position", {}),
                "biome": observe_data.get("status", {}).get("biome", ""),
                "equipment": self._convert_equipment_to_dict(equipment_list),
                "nearby_blocks": observe_data.get("voxels", []),
            }
            # Include the dimension only when the payload carries it, so
            # events recorded before status.js emitted it parse unchanged.
            dimension = observe_data.get("status", {}).get("dimension")
            if dimension:
                state["dimension"] = dimension
            return state

        return None

    def _calculate_task_state_changes(self, pre_events, post_events):
        """
        Compute state changes between pre- and post-task execution.

        Args:
            pre_events: pre-execution event list
            post_events: post-execution event list

        Returns:
            dict: state-change info, in the form:
                {
                    "inventory_changes": {item: {"before": int, "after": int, "delta": int}},
                    "equipment_changes": {slot: {"before": str, "after": str}},
                    "position_changes": {"x": float, "y": float, "z": float}
                }
        """
        state_changes = {
            "inventory_changes": {},
            "equipment_changes": {},
            "position_changes": {},
            "nearby_blocks_changes": {},
        }

        # Extract pre- and post-execution state
        pre_state = self._extract_current_state(pre_events) if pre_events else None
        post_state = self._extract_current_state(post_events) if post_events else None

        if not pre_state or not post_state:
            return state_changes

        # Compute inventory changes
        pre_inv = pre_state.get("inventory", {}) or {}
        post_inv = post_state.get("inventory", {}) or {}

        all_items = set(pre_inv.keys()) | set(post_inv.keys())
        for item in all_items:
            pre_count = pre_inv.get(item, 0)
            post_count = post_inv.get(item, 0)
            if pre_count != post_count:
                state_changes["inventory_changes"][item] = {
                    "before": pre_count,
                    "after": post_count,
                    "delta": post_count - pre_count
                }

        # Compute equipment changes
        pre_equip = pre_state.get("equipment", {}) or {}
        post_equip = post_state.get("equipment", {}) or {}

        all_slots = set(pre_equip.keys()) | set(post_equip.keys())
        for slot in all_slots:
            pre_item = pre_equip.get(slot)
            post_item = post_equip.get(slot)
            if pre_item != post_item:
                state_changes["equipment_changes"][slot] = {
                    "before": pre_item,
                    "after": post_item
                }

        # Compute nearby_blocks changes (additions/removals in the block-name set)
        pre_nearby = set(pre_state.get("nearby_blocks", []))
        post_nearby = set(post_state.get("nearby_blocks", []))
        added_blocks = sorted(post_nearby - pre_nearby)
        removed_blocks = sorted(pre_nearby - post_nearby)
        if added_blocks or removed_blocks:
            state_changes["nearby_blocks_changes"] = {
                "added": added_blocks,
                "removed": removed_blocks,
            }

        # Record the dimension transition whenever both sides observed it.
        # An unchanged value is recorded too: explicit "still in X" evidence
        # lets the critic reject skill-name-based success claims for
        # dimension-entry tasks.
        pre_dim = pre_state.get("dimension")
        post_dim = post_state.get("dimension")
        if pre_dim and post_dim:
            state_changes["dimension_change"] = {
                "before": pre_dim,
                "after": post_dim,
            }

        # Extract placed blocks evidence.
        # Primary: returnItems() in index.js records blocks before destroying
        # them via /setblock air destroy + /give — authoritative even if the
        # skill never called bot.save(). Fallback: onSave events.
        placed_blocks = []

        # Primary: returnItems() recorded blocks
        for event_type, event_data in iter_events(post_events):
            if (event_type == "placedBlocks"
                    and isinstance(event_data, dict)):
                placed_blocks.extend(event_data.get("placedBlocks", []))

        # Fallback: onSave events (legacy, kept for compatibility)
        if not placed_blocks:
            for event_type, event_data in iter_events(post_events):
                if (event_type == "onSave"
                        and isinstance(event_data, dict)
                        and event_data.get("onSave", "").endswith("_placed")):
                    block = event_data["onSave"].split("_placed")[0]
                    placed_blocks.append(block)

        if placed_blocks:
            state_changes["placed_blocks"] = placed_blocks

        return state_changes

    def _get_executed_skills(self, events) -> list:
        """Extract executed skill names from events.

        Used by the PSN Critic for semantic detection, so it can judge the task
        semantic type based on the actually-executed skill (e.g. ensureLogs).

        Args:
            events: event list

        Returns:
            list: list of executed skill names (in execution order)
        """
        skills = []
        for event_type, event_data in iter_events(events):
            # Look for skillStart events
            if event_type == "skillStart" and isinstance(event_data, dict):
                skill_name = event_data.get("skillName", "")
                if skill_name and skill_name not in skills:
                    skills.append(skill_name)
        return skills

    def _diagnose_events_structure(self, events):
        """
        Diagnose events structure to help locate issues.

        Args:
            events: event list

        Returns:
            str: diagnostic info
        """
        if not events:
            return "  [WARNING] events is None or empty"

        diagnosis = []
        diagnosis.append(f"  Total events: {len(events)}")

        # Check the last event
        last_event = events[-1]
        unpacked = unpack_event(last_event)
        if unpacked is not None:
            event_type, event_data = unpacked
            diagnosis.append(f"  Last event type: {event_type}")

            if isinstance(event_data, dict):
                diagnosis.append(f"  Last event keys: {list(event_data.keys())}")

                # Check key fields
                if "inventory" not in event_data:
                    diagnosis.append(f"  [MISSING] 'inventory' field not found in last event!")
                    diagnosis.append(f"    Available fields: {list(event_data.keys())}")
                else:
                    inventory = event_data.get("inventory", {})
                    diagnosis.append(f"  inventory type: {type(inventory)}, value: {inventory}")

                if "status" not in event_data:
                    diagnosis.append(f"  [MISSING] 'status' field not found in last event!")
                else:
                    status = event_data.get("status", {})
                    diagnosis.append(f"  status type: {type(status)}, keys: {list(status.keys()) if isinstance(status, dict) else 'N/A'}")
            else:
                diagnosis.append(f"  [WARNING] Last event data is not a dict, type: {type(event_data)}")
        else:
            diagnosis.append(f"  [WARNING] Last event structure invalid: {last_event}")

        # Check all observe events
        observe_events = [(et, ed) for et, ed in iter_events(events) if et == "observe"]
        diagnosis.append(f"  Total observe events: {len(observe_events)}")

        # Check which observe events lack inventory
        missing_inventory_count = 0
        for i, (event_type, event_data) in enumerate(observe_events):
            if isinstance(event_data, dict) and "inventory" not in event_data:
                missing_inventory_count += 1
                if i == len(observe_events) - 1:
                    diagnosis.append(f"  [CRITICAL] Last observe event (index {i}) is missing 'inventory' field!")

        if missing_inventory_count > 0:
            diagnosis.append(f"  [WARNING] {missing_inventory_count} observe event(s) missing 'inventory' field")

        return "\n".join(diagnosis)

    def _ensure_events_list(self, events):
        """Ensure events is in list form; handle possible JSON strings or invalid types."""
        if events is None:
            return []

        # If it is a string, try parsing as JSON
        if isinstance(events, str):
            try:
                events = json.loads(events)
            except (json.JSONDecodeError, TypeError):
                print(f"\033[33m[Warning] events is an invalid JSON string; returning empty list\033[0m")
                return []

        # Ensure it is a list
        if not isinstance(events, list):
            print(f"\033[33m[Warning] events has unexpected type {type(events)}; returning empty list\033[0m")
            return []

        return events

    def _build_synthetic_error_events(self, error_message: str):
        """Inject an onError event before the final observe event so the next prompt sees the error."""
        # Use _ensure_events_list to handle possible string types
        raw_events = copy.deepcopy(self.last_events) if getattr(self, "last_events", None) else []
        base = self._ensure_events_list(raw_events)

        if not base:
            base = [("observe", {"status": {}, "inventory": {}, "voxels": []})]
        last_unpacked = unpack_event(base[-1])
        if not (last_unpacked and last_unpacked[0] == "observe"):
            base.append(("observe", {"status": {}, "inventory": {}, "voxels": []}))
        base.insert(len(base) - 1, ("onError", {"onError": str(error_message)}))
        return base

    def _convert_equipment_to_dict(self, equipment_list):
        """
        Convert the equipment array to dict form.

        Per the getEquipment() method in status.js, the array order is:
        [head, torso, legs, feet, hand, off-hand]

        Args:
            equipment_list: equipment array, in [item_name, ...] form or None

        Returns:
            dict: equipment dict, in {"head": item_name, "torso": item_name, ...} form
        """
        if not equipment_list or not isinstance(equipment_list, list):
            return {}

        # Equipment slot names from domain knowledge
        dk = getattr(self, '_domain_knowledge', None)
        slot_names = dk.get_equipment_slot_names() if dk else []
        if not slot_names:
            return {}

        equipment_dict = {}
        for i, slot_name in enumerate(slot_names):
            if i < len(equipment_list):
                item = equipment_list[i]
                # item may be None (if the slot is empty)
                equipment_dict[slot_name] = item if item else None
            else:
                equipment_dict[slot_name] = None

        return equipment_dict

    @staticmethod
    def _find_nearby_blocks_from_events(events, current_index):
        """Find the voxels data from the most recent observe event in events.

        Skill events (skillStart/skillEnd/skillError) have state_data without voxels;
        we supplement them from the most recent observe event.

        Args:
            events: event list
            current_index: current event index

        Returns:
            List[str]: list of nearby block names
        """
        for evt in reversed(events[:current_index]):
            unpacked = unpack_event(evt)
            if (unpacked and unpacked[0] == "observe"
                    and isinstance(unpacked[1], dict)
                    and "voxels" in unpacked[1]):
                return unpacked[1]["voxels"]
        return []
