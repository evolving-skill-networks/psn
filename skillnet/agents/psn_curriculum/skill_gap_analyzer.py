"""
SkillGapAnalyzer - skill-gap analyzer

Analyzes whether each subtask has a corresponding skill and identifies the skills that must be learned.

Core responsibilities:
- Check whether each subtask has a corresponding skill
- Evaluate skill reliability (success rate)
- Decide learning priority
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Any, TYPE_CHECKING

from skillnet.core.dk_registry import get_domain_knowledge

if TYPE_CHECKING:
    from skillnet.agents.psn_curriculum.task_decomposer import SubTask, DecomposedTask


def _get_ore_to_drop_map():
    """Return ore-to-drop mapping from domain or empty default."""
    dk = get_domain_knowledge()
    return dk.get_ore_to_drop_mapping() if dk else {}


@dataclass
class SkillGap:
    """Skill gap"""
    subtask: SubTask
    gap_type: str  # "no_skill" | "low_success_rate" | "unverified"
    matching_skills: List[str]  # matching skills (may be empty)
    suggested_learning_task: str  # suggested learning task


class SkillGapAnalyzer:
    """
    Skill-gap analyzer

    Core responsibilities:
    - Check whether each subtask has a corresponding skill
    - Evaluate skill reliability (success rate)
    - Decide learning priority
    """

    def __init__(
        self,
        skill_graph_manager,
        effect_matcher=None,
        min_success_rate: float = 0.5,
    ):
        """
        Initialize the skill-gap analyzer

        Args:
            skill_graph_manager: SkillGraphManager instance
            effect_matcher: EffectMatcher instance (optional, for precise matching)
            min_success_rate: minimum required success rate
        """
        self.skill_graph_manager = skill_graph_manager
        self.effect_matcher = effect_matcher
        self.min_success_rate = min_success_rate

    def analyze(self, decomposed_task: DecomposedTask) -> List[SkillGap]:
        """
        Analyze every skill gap for a decomposed task

        Args:
            decomposed_task: the decomposed task

        Returns:
            List[SkillGap]: SkillGap list sorted by learning priority
        """
        gaps = []

        for subtask in decomposed_task.subtasks:
            gap = self._analyze_subtask(subtask)
            if gap:
                gaps.append(gap)

        # Sort by priority: learn the ones with fewer dependencies first
        return self._sort_by_priority(gaps)

    def _analyze_subtask(self, subtask: SubTask) -> Optional[SkillGap]:
        """
        Analyze the skill gap for a single subtask

        Args:
            subtask: subtask

        Returns:
            SkillGap if a gap exists, otherwise None
        """
        # Find matching skills
        matching_skills = self._find_matching_skills(subtask)

        if not matching_skills:
            # No matching skill
            return SkillGap(
                subtask=subtask,
                gap_type="no_skill",
                matching_skills=[],
                suggested_learning_task=self._simplify_for_learning(subtask)
            )

        # Check the reliability of the best matching skill
        best_skill = matching_skills[0]
        if not self._is_skill_reliable(best_skill):
            return SkillGap(
                subtask=subtask,
                gap_type="low_success_rate",
                matching_skills=[best_skill],
                suggested_learning_task=self._simplify_for_learning(subtask)
            )

        # Skill exists and is reliable; no gap
        return None

    def _find_matching_skills(self, subtask: SubTask) -> List[str]:
        """
        Find skills matching the subtask

        Args:
            subtask: subtask

        Returns:
            List of matching skill names
        """
        matching = []

        # Prefer EffectMatcher when available
        if self.effect_matcher is not None:
            try:
                target_effects = self.effect_matcher.extract_target_effects(
                    subtask.task_description
                )
                if target_effects:
                    matching = self.effect_matcher.find_skills_by_effects(target_effects)
                    if matching:
                        return matching
            except Exception:
                pass

        # Fallback: use simple keyword matching
        return self._simple_skill_match(subtask)

    def _simple_skill_match(self, subtask: SubTask) -> List[str]:
        """
        Simple skill matching

        Match based on the task's target item and keywords
        """
        if not self.skill_graph_manager:
            return []

        # Check whether a graph attribute exists
        if not hasattr(self.skill_graph_manager, 'has_node'):
            return []

        matching = []
        target_item = subtask.target_item.lower().replace("_", "")
        task_type = subtask.task_type.lower()

        # Exclude task-specific skills - only general skills should be matched for reuse
        all_general_skills = self.skill_graph_manager.get_all_skill_names(include_task_specific=False)
        for node_name in all_general_skills:
            node_lower = node_name.lower()

            # Check whether the skill name contains the target item
            if target_item in node_lower.replace("_", ""):
                # Further check the task type
                if task_type == "craft" and ("craft" in node_lower or "ensure" in node_lower):
                    matching.append(node_name)
                elif task_type == "mine" and ("mine" in node_lower or "collect" in node_lower or "ensure" in node_lower):
                    matching.append(node_name)
                elif task_type == "ensure":
                    matching.append(node_name)
                elif task_type in ["find", "place"]:
                    matching.append(node_name)

        return matching

    def _is_skill_reliable(self, skill_name: str) -> bool:
        """
        Check whether a skill is reliable

        Args:
            skill_name: skill name

        Returns:
            True if the skill is reliable (success rate >= min_success_rate)
        """
        if not self.skill_graph_manager:
            return False

        if not hasattr(self.skill_graph_manager, 'has_node'):
            return False

        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return False

        # Check statistics
        if not hasattr(node, 'statistics'):
            return False

        stats = node.statistics
        if stats.total_executions == 0:
            # Never executed; treat as unreliable
            return False

        return stats.success_rate >= self.min_success_rate

    def _simplify_for_learning(self, subtask: SubTask) -> str:
        """
        Convert a task into a learning version, accounting for task type and Minecraft item rules.

        For example:
        - "Find 3 diamond_ore" -> "Find 3 diamond_ore" (keep the original description)
        - "Mine 3 coal_ore" -> "Ensure you have 3 coal" (ore-to-drop mapping)
        - "Craft 4 oak_planks" -> "Ensure you have 4 oak planks" (regular item)
        """
        target_item = subtask.target_item
        target_count = subtask.target_count
        task_type = subtask.task_type.lower()

        # 1. For "find"-type tasks, keep the original description (treated as a navigation/search skill)
        if task_type == "find":
            return subtask.task_description

        # 2. For ore-class items, convert to the actual drop
        ore_drops = _get_ore_to_drop_map()
        if target_item in ore_drops:
            actual_drop = ore_drops[target_item]
            return f"Ensure you have {target_count} {actual_drop.replace('_', ' ')}"

        # 3. Otherwise, use the default format
        return f"Ensure you have {target_count} {target_item.replace('_', ' ')}"

    def _sort_by_priority(self, gaps: List[SkillGap]) -> List[SkillGap]:
        """
        Sort by learning priority.

        Priority rules:
        1. Fewer dependencies are learned first
        2. no_skill takes priority over low_success_rate
        3. Simpler task types come first (mine > craft > place)
        """
        def priority_key(gap: SkillGap):
            # Dependency count
            prereq_count = len(gap.subtask.prerequisites)

            # gap-type priority
            type_priority = {
                "no_skill": 0,
                "low_success_rate": 1,
                "unverified": 2,
            }.get(gap.gap_type, 3)

            # Task-type priority
            task_type_priority = {
                "mine": 0,
                "craft": 1,
                "ensure": 2,
                "find": 3,
                "place": 4,
            }.get(gap.subtask.task_type, 5)

            return (prereq_count, type_priority, task_type_priority)

        return sorted(gaps, key=priority_key)

    def get_skill_statistics(self, skill_name: str) -> Optional[dict]:
        """
        Get the skill's statistics

        Args:
            skill_name: skill name

        Returns:
            Statistics dict, or None if not found
        """
        if not self.skill_graph_manager:
            return None

        if not hasattr(self.skill_graph_manager, 'has_node'):
            return None

        node = self.skill_graph_manager.get_node(skill_name)
        if not node or not hasattr(node, 'statistics'):
            return None

        stats = node.statistics
        return {
            "total_executions": stats.total_executions,
            "successful_executions": stats.successful_executions,
            "failed_executions": stats.failed_executions,
            "success_rate": stats.success_rate,
        }
