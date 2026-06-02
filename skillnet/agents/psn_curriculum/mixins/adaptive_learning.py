"""
AdaptiveLearningMixin - Adaptive learning path management and task decomposition.

Methods:
    _init_adaptive_learning() -> None
    _check_adaptive_learning(milestone_tasks, inventory) -> Optional[Tuple[str, str]]
    update_task_result(task, success, error_msg) -> None

Self attributes used:
    skill_manager, llm, goal_planner, knowledge_base,
    _failed_tasks_history, adaptive_planner,
    decomposition_filter, task_decomposer, skill_gap_analyzer
Cross-mixin calls (MRO):
    _check_adaptive_learning -> PromptBuilding._generate_learning_context
    _check_adaptive_learning -> SkillLearning._get_learned_skill_names
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from skillnet.agents.psn_curriculum.decomposition_filter import DecompositionFilter
from skillnet.agents.psn_curriculum.task_decomposer import TaskDecomposer
from skillnet.agents.psn_curriculum.skill_gap_analyzer import SkillGapAnalyzer
from skillnet.agents.psn_curriculum.adaptive_planner import AdaptiveLearningPlanner


class AdaptiveLearningMixin:
    """Mixin for adaptive learning path management and task decomposition."""

    def _init_adaptive_learning(self) -> None:
        """Initialize adaptive learning components"""
        # Get EffectMatcher if available (from planning module)
        effect_matcher = None
        try:
            from skillnet.agents.planning.effect_matcher import EffectMatcher
            if self.skill_manager and hasattr(self.skill_manager, 'has_node'):
                effect_matcher = EffectMatcher(
                    skill_graph_manager=self.skill_manager,
                    llm=self.llm,
                    use_llm_for_extraction=False,  # Use rules for efficiency
                )
        except ImportError:
            pass

        # Decomposition Filter
        self.decomposition_filter = DecompositionFilter(
            goal_planner=self.goal_planner,
            effect_matcher=effect_matcher,
            skill_graph_manager=self.skill_manager,
            failed_tasks_history=self._failed_tasks_history,
            min_success_rate=0.5,
        )

        # Task Decomposer
        self.task_decomposer = TaskDecomposer(
            llm=self.llm,
            knowledge_base=self.knowledge_base,
            max_subtasks=10,
            cache_enabled=True,
        )

        # Skill Gap Analyzer
        self.skill_gap_analyzer = SkillGapAnalyzer(
            skill_graph_manager=self.skill_manager,
            effect_matcher=effect_matcher,
            min_success_rate=0.5,
        )

        # Adaptive Learning Planner
        self.adaptive_planner = AdaptiveLearningPlanner(
            decomposition_filter=self.decomposition_filter,
            task_decomposer=self.task_decomposer,
            skill_gap_analyzer=self.skill_gap_analyzer,
            max_attempts_per_task=2,
            continue_on_partial_success=True,
        )

        print(f"\033[35m[PSN Curriculum] Adaptive learning components initialized\033[0m")

    def _check_adaptive_learning(
        self,
        milestone_tasks: List[str],
        inventory: Dict[str, int]
    ) -> Optional[Tuple[str, str]]:
        """
        Check if any milestone task needs adaptive learning (decomposition).

        This method checks if any task in the milestone list requires learning
        sub-skills first. If so, it returns the next learning task.

        Args:
            milestone_tasks: List of milestone tasks to check
            inventory: Current inventory

        Returns:
            (task, context) tuple if adaptive learning is needed, None otherwise
        """
        # diagnostic logging for tracing milestone_tasks
        print(f"\033[36m[PSN Debug] _check_adaptive_learning: milestone_tasks={milestone_tasks[:5]}\033[0m")

        if not self.adaptive_planner:
            return None

        # Check if we have an ongoing learning path
        current_path = self.adaptive_planner.get_current_path()
        if current_path and current_path.path_type == "learning_first":
            # Check if the path is blocked
            if self.adaptive_planner.is_path_blocked():
                print(f"\033[31m[PSN Adaptive] Learning path blocked, clearing...\033[0m")
                self.adaptive_planner.clear_path()
            elif not self.adaptive_planner.is_path_completed():
                # Continue with current learning path
                next_task = self.adaptive_planner.get_next_task(inventory=inventory)
                if next_task:
                    context = self._generate_learning_context(next_task, current_path)
                    print(f"\033[35m[PSN Adaptive] Continuing learning: {next_task}\033[0m")
                    return next_task, context

        # Check if any milestone task needs decomposition
        skill_names = self._get_learned_skill_names()

        for task in milestone_tasks:
            # Plan learning path for this task
            learning_path = self.adaptive_planner.plan_learning_path(
                task=task,
                inventory=inventory,
                skill_names=skill_names,
            )

            if learning_path.path_type == "learning_first":
                # This task needs learning first
                next_task = self.adaptive_planner.get_next_task()
                if next_task:
                    context = self._generate_learning_context(next_task, learning_path)
                    print(f"\033[35m[PSN Adaptive] Starting learning path for '{task}': {next_task}\033[0m")
                    return next_task, context

        # No task needs adaptive learning
        return None

    def update_task_result(self, task: str, success: bool, error_msg: Optional[str] = None) -> None:
        """
        Update task execution result for adaptive learning.

        This method should be called after each task execution to update
        the adaptive learning progress.

        Args:
            task: The task that was executed
            success: Whether the task succeeded
            error_msg: Error message if failed
        """
        # Update adaptive learning progress
        if self.adaptive_planner:
            self.adaptive_planner.update_progress(task, success, error_msg)

        # Record failure history
        if not success:
            self._failed_tasks_history.add(task)
