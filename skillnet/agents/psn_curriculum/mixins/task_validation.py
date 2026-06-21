"""
TaskValidationMixin - Task format validation, environment checks, and special case handling.

Methods:
    _validate_task(task) -> Tuple[bool, Optional[str]]
    _check_special_cases(events, chest_observation) -> Optional[str]
    _check_environment_requirements(task, chest_observation) -> Optional[str]
    _check_armor_upgrade(events) -> Optional[str]
    _handle_full_inventory(inventory, chest_observation) -> str
    _has_sufficient_materials_for_milestone(inventory) -> bool
    _should_skip_special_task(task, threshold) -> bool

Self attributes used:
    resource_tracker, knowledge_base, failed_tasks, progress,
    milestone_task_failure_threshold, skill_manager
Cross-mixin calls (MRO):
    _has_sufficient_materials_for_milestone -> Facade.has_learned_basic_tools
    _should_skip_special_task -> Facade._count_task_failures
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

def _get_armor_funcs_from_dk(dk):
    """Get armor functions from domain knowledge or no-op defaults."""
    if dk:
        return dk.is_armor_item, dk.get_armor_tier, dk.get_armor_slot
    return (lambda x: False), (lambda x: -1), (lambda x: None)


class TaskValidationMixin:
    """Mixin for task validation, environment checks, and special case handling."""

    def _validate_task(self, task: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a task is well-formed and targets known items.

        Args:
            task: The task description to validate

        Returns:
            (is_valid, error_message) - True with None if valid,
            False with error description if invalid
        """
        if not task or not isinstance(task, str):
            return False, "Task is empty or not a string"

        task = task.strip()
        if len(task) < 5:
            return False, "Task is too short"

        task_lower = task.lower()

        # Check for common invalid patterns
        invalid_patterns = [
            r"^reasoning:",  # LLM accidentally included reasoning
            r"^task:",       # LLM included the label
            r"^\d+\.",       # Numbered list item
            r"^-\s",         # Bullet point
        ]
        for pattern in invalid_patterns:
            if re.match(pattern, task_lower):
                return False, f"Task has invalid format (matched pattern: {pattern})"

        # Reject compound tasks with ";" separator
        # LLM sometimes generates "Craft X; Ensure Y; Craft Z" which breaks
        # semantic evaluation (single semantic applied to multiple sub-tasks)
        if ";" in task:
            return False, "Compound task with ';' separator - propose one task at a time"

        # Validate task structure based on action type
        if "craft" in task_lower:
            match = re.match(r"craft\s+(\d+)?\s*(.+)", task_lower)
            if not match:
                return False, "Craft task has invalid format"
            item_name = match.group(2).strip().replace(" ", "_")

            # Check if item exists in knowledge base
            recipe = self.knowledge_base.get_recipe(item_name)
            if not recipe:
                # Also check common variations
                variations = [
                    item_name,
                    item_name.replace("_", ""),
                    f"oak_{item_name}",  # e.g., "planks" -> "oak_planks"
                ]
                found = False
                for var in variations:
                    if self.knowledge_base.get_recipe(var):
                        found = True
                        break
                if not found:
                    # It might be a valid item we just don't have a recipe for
                    # Log a warning but don't reject
                    print(f"[PSN] Warning: No recipe found for '{item_name}'")

        elif "mine" in task_lower or "collect" in task_lower or "gather" in task_lower:
            match = re.match(r"(?:mine|collect|gather)\s+(\d+)?\s*(.+)", task_lower)
            if not match:
                return False, "Mining task has invalid format"
            # Mining tasks are generally valid if they have a target

        elif task_lower.startswith("smelt") or task_lower.startswith("cook"):
            # Only check if task STARTS with smelt/cook (not just contains, e.g. "cooked food")
            match = re.match(r"(?:smelt|cook)\s+(\d+)?\s*(.+)", task_lower)
            if not match:
                return False, "Smelting task has invalid format"

        elif "ensure" in task_lower:
            # "Ensure you have N X" — extract X and reject placeholder/invalid
            # targets. Root cause of the iter-92 r4 failure: task_decomposer
            # emitted "unknown" as a fallback when LLM response missed the
            # "target" field (task_decomposer.py:260), producing tasks like
            # "Ensure you have 1 unknown" that passed the original lenient
            # pass-through and wasted an iteration.
            m = re.match(r"ensure\s+(?:you\s+have\s+)?(\d+)?\s*(.+)", task_lower)
            if not m:
                return False, "Ensure task has invalid format"
            target = m.group(2).strip().rstrip(".").rstrip("s").replace(" ", "_")
            placeholder_targets = {
                "unknown", "item", "items", "thing", "things",
                "something", "object", "objects", "resource", "resources",
                "stuff", "",
            }
            if target in placeholder_targets:
                return False, (
                    f"Ensure task has placeholder target '{target}' — "
                    "propose a specific Minecraft item with exact name and count"
                )

        elif "explore" in task_lower or "find" in task_lower:
            # Exploration tasks are generally valid
            pass

        elif "deposit" in task_lower or "store" in task_lower:
            # Storage tasks are generally valid
            pass

        elif "place" in task_lower:
            # Placement tasks are generally valid
            pass

        else:
            # Unknown task type - log warning but allow
            print(f"[PSN] Warning: Unknown task type: '{task}'")

        return True, None

    def _check_special_cases(
        self,
        events: List,
        chest_observation: str
    ) -> Optional[str]:
        """
        Check for special cases that override normal task selection.

        Special cases are checked in priority order:
        1. First task: Mine initial logs
        2. Progressive skill learning: Prioritize crafting tasks for unlearned skills
        3. Underground without pickaxe: Craft pickaxe urgently
        4. Full inventory: Handle inventory management

        This method implements the "learn skills first, stockpile later" strategy.
        """
        # First task: Ensure we have enough wood for tool crafting
        # Changed from 8 to 4 logs for faster progression to first crafting task
        # 4 logs = 16 planks, enough for:
        # - crafting_table (4 planks)
        # - wooden_pickaxe (3 planks + 2 for sticks = 5 planks)
        # - leftover: 7 planks for wooden_axe or sticks
        # Using TARGET semantic: "Ensure you have X" means ensure inventory has X total
        if self.progress == 0:
            dk = getattr(self, '_domain_knowledge', None)
            first_task = dk.get_initial_task() if dk else ""
            if first_task:
                # === P0 fix: inventory check ===
                inventory = self.resource_tracker.current_inventory
                logs_count = sum(v for k, v in inventory.items() if k.endswith("_log"))
                if logs_count >= 4:
                    print(f"\033[32m[PSN Special] First task skipped - have {logs_count} logs\033[0m")
                    # Skip the first task; continue checking other special cases
                elif not self._should_skip_special_task(first_task):
                    return first_task
                # === end of fix ===
            # If first task satisfied or failed too many times, fall through to normal selection

        # Get current state via the active DomainKnowledge
        try:
            dk = getattr(self, '_domain_knowledge', None)
            obs = dk.extract_observation(events) if dk is not None else None
            if obs is not None and obs.position and len(obs.position) >= 2:
                current_y = obs.position[1]
            else:
                current_y = 100
            inventory = self.resource_tracker.current_inventory

            # Calculate inventory slots used via domain config
            inv_cfg = dk.get_inventory_config() if dk else {}
            stack_size = inv_cfg.get("stack_size", 64)
            full_threshold = inv_cfg.get("full_threshold", 27)
            inventory_slots_used = sum(
                -(-count // stack_size) for count in inventory.values()  # ceiling division
            )

            # Emergency task (e.g., underground without pickaxe)
            if dk:
                emergency = dk.get_emergency_task(current_y, inventory, [])
                if emergency and not self._should_skip_special_task(emergency):
                    return emergency

            # PRIORITY 4: Armor upgrade check
            equip_task = self._check_armor_upgrade(events)
            if equip_task and not self._should_skip_special_task(equip_task):
                print(f"\033[35m[PSN Equipment] Armor upgrade: {equip_task}\033[0m")
                return equip_task

            # Inventory full — threshold from domain config. Only short-circuit to
            # a deposit task when there is actually something worth depositing;
            # otherwise fall through to the normal milestone/exploration flow so a
            # full-of-keepers inventory cannot loop on the deposit task.
            if inventory_slots_used >= full_threshold:
                worn = []
                try:
                    _obs = dk.extract_observation(events) if dk else None
                    worn = list(_obs.extra.get("equipment", [])) if (_obs and _obs.extra) else []
                except Exception:
                    worn = []
                full_task = self._handle_full_inventory(inventory, chest_observation, equipment=worn)
                if full_task:
                    return full_task

        except Exception as e:
            print(f"[PSN] Warning in special case check: {e}")

        return None

    def _check_armor_upgrade(self, events) -> Optional[str]:
        """Check whether there is armor in inventory better than what is currently worn.

        Only checks armor slots (head/torso/legs/feet); does not handle the hand slot.
        Reason: tool choice is context-dependent (mining needs a pickaxe, combat needs a sword),
        whereas armor upgrades are unconditionally beneficial (a higher tier is strictly better than a lower tier).
        """
        try:
            _dk = getattr(self, '_domain_knowledge', None)
            equipment_list: list = []
            if _dk is not None:
                _obs = _dk.extract_observation(events)
                equipment_list = list(_obs.extra.get("equipment", [])) if _obs.extra else []
            if len(equipment_list) < 4:
                return None
        except (IndexError, TypeError, AttributeError):
            return None

        inventory = self.resource_tracker.current_inventory

        _dk = getattr(self, '_domain_knowledge', None)
        slot_names = _dk.get_equipment_slot_names() if _dk else []
        if not slot_names:
            return None
        current_equipment = {
            slot_names[i]: equipment_list[i]
            for i in range(min(len(slot_names), len(equipment_list)))
        }

        is_armor_item, get_armor_tier, get_armor_slot = _get_armor_funcs_from_dk(_dk)

        best_upgrade = None  # (item_name, tier_diff)
        for item_name in inventory:
            if not is_armor_item(item_name):
                continue
            slot = get_armor_slot(item_name)
            if slot is None:
                continue
            current_item = current_equipment.get(slot)
            current_tier = get_armor_tier(current_item) if current_item else -1
            new_tier = get_armor_tier(item_name)
            tier_diff = new_tier - current_tier
            if tier_diff > 0 and (best_upgrade is None or tier_diff > best_upgrade[1]):
                best_upgrade = (item_name, tier_diff)

        result = f"Equip {best_upgrade[0]}" if best_upgrade else None
        # Diagnostic — track armor upgrade check behavior for smoke test analysis
        if result:
            print(f"\033[35m[Curriculum P5] Armor upgrade detected: {result} (tier_diff={best_upgrade[1]})\033[0m")
        return result

    def _handle_full_inventory(
        self,
        inventory: Dict[str, int],
        chest_observation: str,
        equipment: list = None,
    ) -> Optional[str]:
        """Handle a full inventory by delegating to domain knowledge.

        Depositability-aware: returns a deposit/chest task ONLY when the canonical
        classification finds something worth depositing (>=1 slot freeable). If the
        inventory is full of items that should all be KEPT (tools/ingots/valuables),
        there is nothing to deposit, so we return None and let the normal flow
        proceed — this is what makes the deposit loop escapable without a separate
        failure-count guard.
        """
        dk = getattr(self, '_domain_knowledge', None)
        if not dk:
            return None
        try:
            from skillnet.domains.minecraft.knowledge.inventory_classification import (
                classify_deposit,
            )
            if classify_deposit(inventory, equipment=equipment)["freed_slots"] < 1:
                return None  # nothing useful to deposit — do not issue a deposit task
        except Exception:
            pass  # classification unavailable: fall back to legacy behavior
        return dk.get_full_inventory_task(inventory, chest_observation)

    def _has_sufficient_materials_for_milestone(self, inventory: Dict[str, int]) -> bool:
        """Check if we have sufficient raw materials for early-game progression.

        Delegates to domain knowledge. Before basic tools are learned,
        returns True if gathering should be skipped.
        """
        # After basic tools are learned, defer to resource threshold system
        if self.has_learned_basic_tools():
            return False

        dk = getattr(self, '_domain_knowledge', None)
        if dk:
            return dk.has_sufficient_materials_for_progression(inventory)
        return False

    def _should_skip_special_task(self, task: str, threshold: int = None) -> bool:
        """
        Check if a special case task should be skipped due to too many failures.

        Args:
            task: The special task to check
            threshold: Max failures before skipping (default: self.milestone_task_failure_threshold)

        Returns:
            True if task should be skipped
        """
        threshold = threshold if threshold is not None else self.milestone_task_failure_threshold
        # Normalize task key
        normalized = task.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        failure_counts = self._count_task_failures()
        count = failure_counts.get(normalized, 0)
        if count >= threshold:
            print(
                f"\033[33m[Special Case] '{task}' failed {count} times "
                f"(threshold={threshold}), skipping\033[0m"
            )
            return True
        return False

    def _check_environment_requirements(
        self,
        task: str,
        chest_observation: str
    ) -> Optional[str]:
        """
        Check if a task requires environment conditions that are not available.

        Returns:
            None if task is feasible, or a string describing the issue if not.
        """
        task_lower = task.lower()

        # Tasks that require a chest
        chest_keywords = ["deposit", "store items", "put items", "into the chest", "into chest"]
        needs_chest = any(kw in task_lower for kw in chest_keywords)

        if needs_chest:
            # Check if there's actually a chest available
            # chest_observation format: "Chests: None\n\n" or "Chests:\n(x,y,z): {...}\n\n"
            has_chest = (
                chest_observation
                and not chest_observation.startswith("Chests: None")
                and "\n(" in chest_observation
            )
            if not has_chest:
                return "No chest available in the environment"

        # Future: Add more environment checks here
        # - Tasks requiring furnace: "smelt" without furnace nearby
        # - Tasks requiring water: "fish" without water nearby
        # - etc.

        return None
