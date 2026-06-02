"""
SkillLearningMixin - Skill lookup, cache management, and fuzzy matching.

Methods:
    is_skill_learned(task_type) -> bool
    _get_learned_skill_names() -> List[str]
    clear_skill_cache()
    _extract_keywords_from_task(task_type) -> List[str]
    _extract_words_from_skill_name(name) -> set

Self attributes used:
    skill_manager, _learned_skills_cache, TASK_TO_SKILL_MAPPING, _domain_knowledge
"""

from __future__ import annotations

import re
from typing import List


class SkillLearningMixin:
    """Mixin for skill learning status checks and cache management."""

    def _get_task_skill_mapping(self) -> dict:
        """Return the task-to-skill mapping, preferring domain knowledge.

        Checks ``self._domain_knowledge`` (set by ``set_domain_knowledge``)
        first, then falls back to the class-level ``TASK_TO_SKILL_MAPPING``.
        """
        dk = getattr(self, '_domain_knowledge', None)
        if dk:
            mapping = dk.get_task_skill_mapping()
            if mapping:
                return mapping
        return getattr(self, 'TASK_TO_SKILL_MAPPING', {})

    def is_skill_learned(self, task_type: str) -> bool:
        """
        Check if a skill for the given task type has been learned.

        This method checks the skill graph to see if any skill matching
        the task type exists and has been successfully executed.

        Args:
            task_type: Task type like "craft planks", "mine logs", etc.

        Returns:
            True if at least one skill for this task type exists in the graph
        """
        # Normalize task type
        task_type_lower = task_type.lower().strip()

        # Check cache first
        if task_type_lower in self._learned_skills_cache:
            return self._learned_skills_cache[task_type_lower]

        # If no skill manager, assume not learned
        if self.skill_manager is None:
            return False

        # Check if any matching skill exists in the graph
        skill_names = self._get_task_skill_mapping().get(task_type_lower, [])
        for skill_name in skill_names:
            if self.skill_manager.has_node(skill_name):
                # Check if skill is verified (successfully executed at least once)
                node = self.skill_manager.get_node(skill_name)
                # Use statistics.successful_executions (not success_count which doesn't exist)
                if node and hasattr(node, 'statistics') and node.statistics.successful_executions > 0:
                    self._learned_skills_cache[task_type_lower] = True
                    print(f"\033[32m[PSN] Skill '{skill_name}' is learned (success={node.statistics.successful_executions})\033[0m")
                    return True

        # Also check for skills with similar names using fuzzy matching
        # Uses (item_keywords, verb_keywords) tuple.
        # Matching rule: ALL item_keywords must be in skill words
        # AND ANY verb_keyword must be in skill words.
        # This prevents "setup crafting_table" from matching "craftCraftingTable"
        # (craft ≠ place/setup) while allowing "placeCraftingTable" to match.
        item_keywords, verb_keywords = self._extract_keywords_from_task(task_type_lower)
        if item_keywords:
            # Include task-specific skills: skills with 0 user params
            # (craftOakCraftingTable, placeCraftingTable) are marked task-specific
            # but ARE functionally correct and reusable.
            all_skills = self.skill_manager.get_all_skill_names(
                include_task_specific=True
            )
            for node_name in all_skills:
                skill_words = self._extract_words_from_skill_name(node_name)
                # ALL item keywords must match as whole words
                if not all(keyword in skill_words for keyword in item_keywords):
                    continue
                # ANY verb keyword must match (OR logic)
                if verb_keywords and not any(v in skill_words for v in verb_keywords):
                    continue
                node = self.skill_manager.get_node(node_name)
                if node and hasattr(node, 'statistics') and node.statistics.successful_executions > 0:
                    self._learned_skills_cache[task_type_lower] = True
                    print(f"\033[32m[PSN] Skill '{node_name}' matches task '{task_type}' "
                          f"(fuzzy match, item={item_keywords}, verb={verb_keywords}, "
                          f"skill_words={skill_words})\033[0m")
                    return True

        self._learned_skills_cache[task_type_lower] = False
        return False

    def _extract_keywords_from_task(self, task_type: str):
        """Extract keywords from task type for fuzzy skill matching.

        Returns:
            Tuple[List[str], List[str]]: (item_keywords, verb_keywords).
            Matching rule: ALL item_keywords must match AND ANY verb_keyword must match.
            This prevents "setup crafting_table" from matching "craftCraftingTable"
            (craft ≠ place/setup) while allowing it to match "placeCraftingTable".
        """
        item_keywords = []
        verb_keywords = []
        task_lower = task_type.lower()

        # Use startswith to avoid substring matching
        # ("craft" in "setup crafting_table" = True because of "crafting")
        if task_lower.startswith("setup") or task_lower.startswith("place"):
            item_part = re.sub(r'^(?:setup|place)[_\s]+', '', task_lower).strip()
            parts = item_part.replace("_", " ").split()
            item_keywords.extend(parts)
            verb_keywords.extend(["place", "setup"])
        elif task_lower.startswith("craft"):
            item_part = re.sub(r'^craft[_\s]+', '', task_lower).strip()
            parts = item_part.replace("_", " ").split()
            item_keywords.extend(parts)
            verb_keywords.extend(["craft"])
        elif task_lower.startswith("mine"):
            item_part = re.sub(r'^mine[_\s]+', '', task_lower).strip()
            parts = item_part.replace("_", " ").split()
            item_keywords.extend(parts)
            for part in parts:
                if part.endswith("s"):
                    item_keywords.append(part[:-1])
            verb_keywords.extend(["mine"])

        return item_keywords, verb_keywords

    def _extract_words_from_skill_name(self, name: str) -> set:
        """
        Extract individual words from skill name for precise matching.

        This handles both camelCase and snake_case naming conventions:
        - "craftWoodenPickaxe" -> {"craft", "wooden", "pickaxe"}
        - "ensure_wooden_axe" -> {"ensure", "wooden", "axe"}

        IMPORTANT: This is used to prevent false matches like:
        - "axe" should NOT match "pickaxe" (axe != pickaxe)
        - "sword" should NOT match "swordfish"
        """
        # Split camelCase: "craftWoodenPickaxe" -> ["craft", "Wooden", "Pickaxe"]
        # Then split snake_case and lowercase
        # Pattern: split before uppercase letters that follow lowercase letters
        words = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)  # camelCase to snake_case
        words = words.lower().replace('_', ' ').split()
        return set(words)

    def clear_skill_cache(self) -> None:
        """Clear the learned skills cache. Call after skill graph changes."""
        self._learned_skills_cache.clear()

    def _get_learned_skill_names(self) -> List[str]:
        """
        Get list of learned GENERAL skill names from skill graph.

        Excludes task-specific wrappers (e.g., ensure_you_have_1_raw_iron) which
        should not participate in task matching. Only general skills with
        parameters (e.g., ensureRawIron) should be matched and have parameters
        inferred by the planner.

        Returns:
            List of general skill names that have been successfully executed
        """
        if not self.skill_manager or not hasattr(self.skill_manager, 'has_node'):
            return []

        # Use get_all_skill_names to exclude task-specific skills
        all_general_skills = self.skill_manager.get_all_skill_names(include_task_specific=False)

        learned = []
        for skill_name in all_general_skills:
            node = self.skill_manager.get_node(skill_name)
            if node and hasattr(node, 'statistics'):
                if node.statistics.successful_executions > 0:
                    learned.append(skill_name)

        return learned

    def get_next_unlearned_crafting_task(self, inventory: dict):
        """Get next unlearned crafting skill in priority order.

        Priority: crafting_table → setup_crafting_table → wooden_pickaxe → wooden_axe.
        The setup step teaches the agent to place a crafting_table as a world block,
        which is required by all subsequent 3x3 recipes.
        Returns task string or None if all learned or materials insufficient.
        """
        if not hasattr(self, '_learned_skills_cache'):
            return None
        logs = sum(v for k, v in inventory.items() if k.endswith("_log"))
        planks = sum(v for k, v in inventory.items() if k.endswith("_planks"))
        sticks = inventory.get("stick", 0)
        potential_planks = planks + (logs * 4)

        # Phase 1: Craft a crafting_table item
        if not self.is_skill_learned("craft crafting_table"):
            if potential_planks >= 4:
                print(f"\033[35m[PSN Skill Learning] Next unlearned skill: craft crafting_table\033[0m")
                return "Craft 1 crafting table"
            return None

        # Phase 2: Place the crafting_table as a world block
        # This teaches a reusable setupCraftingTable skill for all 3x3 recipes.
        has_table_item = inventory.get("crafting_table", 0) > 0
        if not self.is_skill_learned("setup crafting_table"):
            if has_table_item:
                print(f"\033[35m[PSN Skill Learning] Next unlearned skill: setup crafting_table\033[0m")
                return "Place a crafting table nearby"
            # Need to craft one first — but craft is already learned, so just need materials
            if potential_planks >= 4:
                return "Craft 1 crafting table"
            return None

        # Phase 3: Tools (require placed crafting_table)
        tool_order = [
            ("craft wooden_pickaxe", "Craft 1 wooden pickaxe", 5),
            ("craft wooden_axe", "Craft 1 wooden axe", 5),
        ]

        for task_type, task, min_planks in tool_order:
            if self.is_skill_learned(task_type):
                continue

            # Ensure crafting_table is available
            if not has_table_item and not self.is_skill_learned("craft crafting_table"):
                if potential_planks >= 4:
                    return "Craft 1 crafting table"
                return None

            planks_for_sticks = 0 if sticks >= 2 else 2
            if potential_planks >= min_planks + planks_for_sticks:
                print(f"\033[35m[PSN Skill Learning] Next unlearned skill: {task_type}\033[0m")
                return task
            return None

        return None
