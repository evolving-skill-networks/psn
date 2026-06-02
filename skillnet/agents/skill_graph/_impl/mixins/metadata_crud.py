"""
Metadata CRUD Mixin for SkillGraphManager

CRUD operations and getters for preconditions, effects, and actual effects.

Extracted from metadata_mgmt.py for modularity.
Contains 11 methods: add/remove/get for preconditions and effects,
plus effect filtering, serialization, and change logging helpers.
"""

import copy
import os
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from skillnet.agents.skill_graph.models import (
    SkillPrecondition,
    SkillEffect,
    ActualEffect,
)
from skillnet.agents.skill_graph.utils import serialize_effects
import skillnet.utils as U

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class MetadataCrudMixin:
    """Metadata CRUD Mixin - Precondition and effect add/remove/get operations.

    Methods:
        add_precondition: Add precondition to a skill
        remove_precondition: Remove precondition from a skill
        add_effect: Add effect to a skill
        remove_effect: Remove effect from a skill
        remove_effect_by_item: Remove effect by item name (with validation and rollback)
        get_preconditions: Get skill preconditions
        get_effects: Get skill expected effects
        get_actual_effects: Get skill actual effects

    Attributes (from SkillGraphManager):
        graph: SkillGraph instance
        ckpt_dir: Checkpoint directory
        logger: Logger instance
    """

    # =========================================================================
    # CRUD Operations - Preconditions and Effects
    # =========================================================================

    def add_precondition(
        self: "SkillGraphManager",
        skill_name: str,
        description: str,
        code: str = "",
    ) -> bool:
        """
        Add a precondition to a skill.

        Args:
            skill_name: Skill name
            description: Precondition description
            code: Executable check code (optional)

        Returns:
            bool: Whether addition succeeded
        """
        if skill_name not in self.graph.nodes:
            print(f"\033[31mError: Skill '{skill_name}' not found.\033[0m")
            return False

        node = self.graph.get_node(skill_name)
        precondition = SkillPrecondition(description=description, code=code)
        node.add_precondition(precondition)

        # Save to checkpoint
        self._save_to_checkpoint()

        print(f"\033[32mAdded precondition to '{skill_name}': {description}\033[0m")
        return True

    def add_effect(
        self: "SkillGraphManager",
        skill_name: str,
        description: str,
        code: str = "",
    ) -> bool:
        """
        Add an expected effect to a skill.

        Args:
            skill_name: Skill name
            description: Effect description
            code: Executable validation code (optional)

        Returns:
            bool: Whether addition succeeded
        """
        if skill_name not in self.graph.nodes:
            print(f"\033[31mError: Skill '{skill_name}' not found.\033[0m")
            return False

        node = self.graph.get_node(skill_name)
        effect = SkillEffect(description=description, code=code)
        node.add_effect(effect)

        # Save to checkpoint
        self._save_to_checkpoint()

        print(f"\033[32mAdded effect to '{skill_name}': {description}\033[0m")
        return True

    def _filter_out_item_from_effects(
        self: "SkillGraphManager",
        effects: List[SkillEffect],
        item_to_remove: str
    ) -> Tuple[List[SkillEffect], int]:
        """
        Filter out effect declarations containing a specific item.

        Handles OR logic effects:
        - If OR condition contains target item, remove only that item, keep others
        - If removal leaves only one condition, downgrade to normal effect
        - If removal leaves no conditions, remove entire effect

        Args:
            effects: Original effects list
            item_to_remove: Item name to remove

        Returns:
            Tuple[List[SkillEffect], int]: (New effects list, removed condition count)
        """
        new_effects = []
        removed_count = 0
        item_lower = item_to_remove.lower()

        for effect in effects:
            state_repr = getattr(effect, 'state_representation', None)

            if state_repr and isinstance(state_repr, dict):
                if state_repr.get("logic", "").upper() == "OR":
                    # Handle OR logic
                    conditions = state_repr.get("conditions", [])
                    new_conditions = []

                    for condition in conditions:
                        condition_item = condition.get("item", "").lower()
                        if condition_item != item_lower:
                            new_conditions.append(condition)
                        else:
                            removed_count += 1

                    if len(new_conditions) == 0:
                        # All OR conditions removed, skip this effect
                        continue
                    elif len(new_conditions) == 1:
                        # Only one condition left, downgrade to normal effect
                        effect.state_representation = new_conditions[0]
                        new_effects.append(effect)
                    else:
                        # Keep OR structure
                        effect.state_representation["conditions"] = new_conditions
                        new_effects.append(effect)
                else:
                    # Single condition
                    effect_item = state_repr.get("item", "").lower()
                    if effect_item != item_lower:
                        new_effects.append(effect)
                    else:
                        removed_count += 1
            else:
                # No state_representation, keep
                new_effects.append(effect)

        return new_effects, removed_count

    def _serialize_effects(self: "SkillGraphManager", effects: List[SkillEffect]) -> List[Dict[str, Any]]:
        """Delegate to code_analysis module (v4.0 refactor)"""
        return serialize_effects(effects)

    def _log_effects_change(
        self: "SkillGraphManager",
        skill_name: str,
        item: str,
        action: str,
        reason: str
    ):
        """
        Log effects change.

        Args:
            skill_name: Skill name
            item: Affected item
            action: Action type ("added", "removed", "modified")
            reason: Change reason
        """
        log_dir = os.path.join(self.ckpt_dir, "skill_graph", "effects_changelog")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{skill_name}.log")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {action.upper()}: '{item}' - {reason}\n"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except (IOError, OSError) as e:
            self.logger.warning(f"[Effects] Cannot write change log: {e}")

    # =========================================================================
    # Getters - Preconditions and Effects
    # =========================================================================

    def update_parameter_semantic(
        self: "SkillGraphManager",
        skill_name: str,
        param_name: str,
        semantic_update: Dict[str, Any],
    ) -> bool:
        """Update a parameter's semantic metadata.

        Called by the optimizer feedback loop to fix incorrect state_mappings
        at the root cause (metadata) rather than ephemeral wrapper code.

        Args:
            skill_name: Skill whose parameter to update
            param_name: Parameter name to update
            semantic_update: Dict with fields to merge into semantic
                (e.g., {"state_mapping": {"source": "effect", ...}})

        Returns:
            True if update succeeded, False if skill/param not found
        """
        if skill_name not in self.graph.nodes:
            return False

        node = self.graph.get_node(skill_name)
        if not node.parameters or param_name not in node.parameters:
            return False

        param_info = node.parameters[param_name]
        existing_semantic = param_info.get("semantic", {})

        # Normalize: if semantic was stored as a string (legacy), convert to dict
        if isinstance(existing_semantic, str):
            existing_semantic = {"quantity_semantic": existing_semantic}

        existing_semantic.update(semantic_update)
        param_info["semantic"] = existing_semantic

        # Persist changes
        self._save_to_checkpoint()
        return True

    # =========================================================================
    # Task-Scoped Parameter Corrections (in-memory only)
    # =========================================================================

    def set_task_parameter_correction(
        self: "SkillGraphManager",
        skill_name: str,
        correction: Dict[str, Any],
    ) -> None:
        """Store a task-scoped parameter correction (in-memory only).

        Called by the optimizer when Phase 1 RCA identifies that a parameter
        value contributed to execution failure. Corrections are consumed by
        PureLLM on retry and cleared on task completion.

        Args:
            skill_name: Skill whose parameter was wrong
            correction: Dict with param_name, passed_value, suggested_value, reason, confidence
        """
        if not hasattr(self, '_task_parameter_corrections'):
            self._task_parameter_corrections: Dict[str, List[Dict[str, Any]]] = {}
        key = f"{skill_name}.{correction['param_name']}"
        if key not in self._task_parameter_corrections:
            self._task_parameter_corrections[key] = []
        self._task_parameter_corrections[key].append(correction)
        # Keep last 3 per key
        self._task_parameter_corrections[key] = self._task_parameter_corrections[key][-3:]

    def get_task_parameter_corrections(
        self: "SkillGraphManager",
        skill_name: str,
        param_name: str,
    ) -> List[Dict[str, Any]]:
        """Get task-scoped corrections for a specific skill+param.

        Args:
            skill_name: Skill name
            param_name: Parameter name

        Returns:
            List of correction dicts, empty if none exist
        """
        if not hasattr(self, '_task_parameter_corrections'):
            return []
        return self._task_parameter_corrections.get(f"{skill_name}.{param_name}", [])

    def clear_task_parameter_corrections(self: "SkillGraphManager") -> None:
        """Clear all task-scoped corrections (call on task completion)."""
        self._task_parameter_corrections = {}

    # =========================================================================
    # Getters - Actual Effects
    # =========================================================================
