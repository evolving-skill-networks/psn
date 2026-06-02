"""
DecompositionFilter - layered filter

Decides whether a task needs LLM decomposition, avoiding unnecessary overhead on simple tasks.

Filter flow:
1. Existing matching skill → no decomposition
2. check_feasibility can handle it → no decomposition
3. Simple task type (with no prior failures) → no decomposition
4. Passes all filters → needs LLM decomposition
"""

from __future__ import annotations

import re
from typing import List, Set, Optional, TYPE_CHECKING

from skillnet.agents.constants.task_semantics import TaskWithSemantic
from skillnet.core.dk_registry import get_domain_knowledge

if TYPE_CHECKING:
    from skillnet.agents.psn_curriculum.goal_planner import GoalPlanner


class DecompositionFilter:
    """
    Decomposition filter.

    Decides whether a task needs LLM decomposition, avoiding unnecessary overhead on simple tasks.
    """

    _FALLBACK_SIMPLE_TASK_PATTERNS = []

    _FALLBACK_COMPLEX_TASK_KEYWORDS = []

    def __init__(
        self,
        goal_planner: GoalPlanner,
        effect_matcher=None,
        skill_graph_manager=None,
        failed_tasks_history: Optional[Set[str]] = None,
        min_success_rate: float = 0.5,
    ):
        """
        Initialize the decomposition filter.

        Args:
            goal_planner: GoalPlanner instance, used for check_feasibility
            effect_matcher: EffectMatcher instance, used for skill matching (optional)
            skill_graph_manager: SkillGraphManager instance, used to check skill success rates (optional)
            failed_tasks_history: set of historically failed tasks
            min_success_rate: minimum success rate requirement; skills below this are considered unreliable
        """
        self.goal_planner = goal_planner
        self.effect_matcher = effect_matcher
        self.skill_graph_manager = skill_graph_manager
        self.failed_tasks_history = failed_tasks_history or set()
        self.min_success_rate = min_success_rate

        # Resolve simple task patterns from domain knowledge or fallback
        _dk = get_domain_knowledge()
        if _dk and hasattr(_dk, 'get_simple_task_patterns'):
            patterns = _dk.get_simple_task_patterns()
            if not patterns:
                patterns = self._FALLBACK_SIMPLE_TASK_PATTERNS
        else:
            patterns = self._FALLBACK_SIMPLE_TASK_PATTERNS

        # Compile regex patterns for performance
        self._simple_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def should_decompose(self, task: TaskWithSemantic, skill_names: Optional[List[str]] = None) -> bool:
        """
        Determine whether a task needs LLM decomposition.

        Args:
            task: TaskWithSemantic object
            skill_names: list of learned skill names (optional)

        Returns:
            True: needs LLM decomposition
            False: no decomposition needed; use existing mechanism

        Filter priority (high to low):
        1. Has a reliable matching skill → no decomposition (highest priority)
        2. Complex-task keyword → force decomposition
        3. check_feasibility can handle it → no decomposition
        4. Simple task type (with no prior failures) → no decomposition
        5. Passes all filters → needs LLM decomposition
        """
        task_str = task.task
        task_lower = task_str.lower().strip()
        is_complex = self._is_complex_task(task_lower)

        # Tier 1: has a reliable matching skill → no decomposition (highest priority)
        # For complex tasks use stricter reliability criteria (require actual success records)
        if skill_names and self._has_reliable_matching_skill(task_str, skill_names, is_complex=is_complex):
            print(f"\033[36m[DecompositionFilter] Has reliable matching skill, skip decomposition: {task_str}\033[0m")
            return False

        # Tier 2: if a complex-task keyword is present, force decomposition
        # Triggered only when no reliable skill is found
        if is_complex:
            print(f"\033[33m[DecompositionFilter] Complex task keyword detected, need decomposition: {task_str}\033[0m")
            return True

        # Tier 3: check_feasibility can handle it → no decomposition
        feasibility = self.goal_planner.check_feasibility(task)
        if feasibility.suggested_prerequisites:
            # An expansion result means the existing system can handle it
            return False

        # Tier 4: simple task type (with no prior failures) → no decomposition
        if self._is_simple_task(task_lower) and not self._has_failed_before(task_str):
            return False

        # Passes all filters; needs LLM decomposition
        return True

    def _is_complex_task(self, task_lower: str) -> bool:
        """
        Check whether this is a complex task (contains complex keywords).

        These tasks always need decomposition.
        Prefer injected domain knowledge; otherwise fall back to the built-in list.
        """
        _dk = get_domain_knowledge()
        if _dk and hasattr(_dk, 'get_complex_task_keywords'):
            keywords = _dk.get_complex_task_keywords()
        else:
            keywords = self._FALLBACK_COMPLEX_TASK_KEYWORDS
        return any(keyword in task_lower for keyword in keywords)

    def _has_reliable_matching_skill(self, task: str, skill_names: List[str], is_complex: bool = False) -> bool:
        """
        Check whether there is a reliable matching skill (success rate >= min_success_rate).

        Args:
            task: task description
            skill_names: list of learned skill names
            is_complex: whether this is a complex task (uses a higher reliability threshold)

        Returns:
            True if a reliable matching skill is found
        """
        if not skill_names:
            return False

        # If an EffectMatcher is available, use it for precise matching
        if self.effect_matcher is not None:
            try:
                target_effects = self.effect_matcher.extract_target_effects(task)
                if target_effects:
                    matching = self.effect_matcher.find_skills_by_effects(target_effects)
                    if matching:
                        # Check whether the matched skill is reliable
                        for skill_name in matching:
                            if self._is_skill_reliable(skill_name, is_complex=is_complex):
                                return True
                        # Found matches but none are reliable
                        print(f"\033[33m[DecompositionFilter] Found matching skills but none are reliable: {matching}\033[0m")
                        return False
            except Exception:
                pass

        # Fallback: simple keyword match + reliability check
        return self._simple_skill_match_with_reliability(task, skill_names, is_complex=is_complex)

    def _is_skill_reliable(self, skill_name: str, is_complex: bool = False) -> bool:
        """
        Check whether a skill is reliable.

        Args:
            skill_name: skill name
            is_complex: whether this is a complex task. Complex tasks have higher thresholds:
                        - must have at least 1 successful execution record (do not trust unexecuted skills)
                        - success-rate threshold 0.7 (vs 0.5 for simple tasks)

        Returns:
            True if the skill is reliable
        """
        if not self.skill_graph_manager:
            return True

        if not hasattr(self.skill_graph_manager, 'has_node'):
            return True

        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return False

        if not hasattr(node, 'statistics'):
            if is_complex:
                return False  # complex task: no statistics → untrustworthy
            return True

        stats = node.statistics
        if stats.total_executions == 0:
            if is_complex:
                return False  # complex task: never executed → untrustworthy
            return True       # simple task: give new skill a chance

        threshold = 0.7 if is_complex else self.min_success_rate
        is_reliable = stats.success_rate >= threshold
        if not is_reliable:
            print(f"\033[33m[DecompositionFilter] Skill '{skill_name}' has low success rate: {stats.success_rate:.2%} < {threshold:.2%} ({'complex' if is_complex else 'simple'})\033[0m")
        return is_reliable

    def _simple_skill_match_with_reliability(self, task: str, skill_names: List[str], is_complex: bool = False) -> bool:
        """
        Simple skill match + reliability check.

        Extract keywords from the task and check whether a reliable matching skill exists.
        """
        task_lower = task.lower()

        # Extract the task target item
        item = self._extract_item(task_lower)
        if not item:
            return False

        # Normalize item name
        item_normalized = item.replace(" ", "_").replace("-", "_")

        # Check whether the skill name contains the target item and validate reliability
        for skill_name in skill_names:
            skill_lower = skill_name.lower()
            # Check exact match or partial match
            if item_normalized in skill_lower or item.replace("_", "") in skill_lower:
                if self._is_skill_reliable(skill_name, is_complex=is_complex):
                    return True

        return False

    def _simple_skill_match(self, task: str, skill_names: List[str]) -> bool:
        """
        Simple skill match (no dependency on EffectMatcher).

        Extract keywords from the task and check whether a matching skill exists.
        """
        task_lower = task.lower()

        # Extract the task target item
        item = self._extract_item(task_lower)
        if not item:
            return False

        # Normalize item name
        item_normalized = item.replace(" ", "_").replace("-", "_")

        # Check whether the skill name contains the target item
        for skill_name in skill_names:
            skill_lower = skill_name.lower()
            # Check exact match or partial match
            if item_normalized in skill_lower or item.replace("_", "") in skill_lower:
                return True

        return False

    def _is_simple_task(self, task_lower: str) -> bool:
        """
        Determine whether this is a simple task.

        Simple tasks can be executed directly without decomposition.
        """
        return any(pattern.match(task_lower) for pattern in self._simple_patterns)

    def _has_failed_before(self, task: str) -> bool:
        """
        Check whether the task has a historical failure record.

        Use fuzzy matching to check whether the task or its item has failed before.
        """
        if not self.failed_tasks_history:
            return False

        # Exact match
        if task in self.failed_tasks_history:
            return True

        # Fuzzy match: extract the item name from the task
        item = self._extract_item(task.lower())
        if item:
            return any(item in failed.lower() for failed in self.failed_tasks_history)

        return False

    def _extract_item(self, task_lower: str) -> Optional[str]:
        """
        Extract the target item name from a task description.

        Args:
            task_lower: lowercase task description

        Returns:
            item name, or None if not extractable
        """
        # Match "Ensure you have X <item>"
        match = re.search(r"ensure you have\s+\d+\s+(.+?)(?:\s*$|\s*\()", task_lower)
        if match:
            return match.group(1).strip()

        # Match "Craft X <item>"
        match = re.search(r"craft\s+\d+\s+(.+?)(?:\s*$|\s*\()", task_lower)
        if match:
            return match.group(1).strip()

        # Match "Mine X <item>"
        match = re.search(r"mine\s+\d+\s+(.+?)(?:\s*$|\s*\()", task_lower)
        if match:
            return match.group(1).strip()

        # Match "Smelt X <item>"
        match = re.search(r"smelt\s+\d+\s+(.+?)(?:\s*$|\s*\()", task_lower)
        if match:
            return match.group(1).strip()

        return None

    def add_failed_task(self, task: str) -> None:
        """
        Add a failed task to the history.

        Args:
            task: failed task description
        """
        self.failed_tasks_history.add(task)

    def clear_failed_history(self) -> None:
        """Clear the failure history."""
        self.failed_tasks_history.clear()
