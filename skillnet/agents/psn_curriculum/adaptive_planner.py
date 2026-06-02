"""
AdaptiveLearningPlanner - adaptive learning-path planner

Integrates decomposition and skill-gap analysis to generate a learning path.

Core responsibilities:
- Decide whether a task needs prior learning
- Generate the learning-task sequence
- Track learning progress
- Handle failures and retries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from skillnet.agents.constants.task_semantics import TaskWithSemantic

if TYPE_CHECKING:
    from skillnet.agents.psn_curriculum.decomposition_filter import DecompositionFilter
    from skillnet.agents.psn_curriculum.task_decomposer import TaskDecomposer
    from skillnet.agents.psn_curriculum.skill_gap_analyzer import SkillGapAnalyzer, SkillGap


@dataclass
class LearningPath:
    """Learning path"""
    original_task: str
    path_type: str  # "direct_execute" | "learning_first"
    learning_tasks: List[str]  # tasks that must be learned first
    execution_tasks: List[str]  # tasks to ultimately execute
    skill_gaps: List[SkillGap] = field(default_factory=list)


@dataclass
class TaskProgress:
    """Task-progress tracking"""
    task: str
    status: str  # "pending" | "in_progress" | "completed" | "failed" | "blocked"
    attempts: int = 0
    max_attempts: int = 2
    failure_reason: Optional[str] = None


class AdaptiveLearningPlanner:
    """
    Adaptive learning-path planner

    Core responsibilities:
    - Decide whether a task needs prior learning
    - Generate the learning-task sequence
    - Track learning progress
    - Handle failures and retries
    """

    def __init__(
        self,
        decomposition_filter: DecompositionFilter,
        task_decomposer: TaskDecomposer,
        skill_gap_analyzer: SkillGapAnalyzer,
        max_attempts_per_task: int = 2,
        continue_on_partial_success: bool = True,
    ):
        """
        Initialize the adaptive learning planner

        Args:
            decomposition_filter: decomposition filter
            task_decomposer: task decomposer
            skill_gap_analyzer: skill-gap analyzer
            max_attempts_per_task: maximum number of retries per task
            continue_on_partial_success: whether to keep attempting the original task on partial success
        """
        self.decomposition_filter = decomposition_filter
        self.task_decomposer = task_decomposer
        self.skill_gap_analyzer = skill_gap_analyzer
        self.max_attempts_per_task = max_attempts_per_task
        self.continue_on_partial_success = continue_on_partial_success

        # State
        self._current_path: Optional[LearningPath] = None
        self._task_progress: Dict[str, TaskProgress] = {}
        self._failed_tasks: Set[str] = set()

    def plan_learning_path(
        self,
        task: str,
        inventory: Optional[Dict[str, int]] = None,
        skill_names: Optional[List[str]] = None,
    ) -> LearningPath:
        """
        Plan a learning path

        Args:
            task: target task
            inventory: current inventory
            skill_names: list of already-learned skill names

        Returns:
            LearningPath: the learning path
        """
        # Guard: don't overwrite an active, non-blocked path for a different task
        if (self._current_path
                and self._current_path.original_task != task
                and not self.is_path_completed()
                and not self.is_path_blocked()):
            return LearningPath(
                original_task=task,
                path_type="direct_execute",
                learning_tasks=[],
                execution_tasks=[task],
            )

        # wrap the string as TaskWithSemantic
        task_obj = TaskWithSemantic.from_legacy_string(task)

        # Layered filtering: check whether LLM decomposition is needed
        if not self.decomposition_filter.should_decompose(task_obj, skill_names):
            print(f"\033[36m[AdaptivePlanner] Task doesn't need decomposition: {task}\033[0m")
            return LearningPath(
                original_task=task,
                path_type="direct_execute",
                learning_tasks=[],
                execution_tasks=[task],
                skill_gaps=[]
            )

        print(f"\033[36m[AdaptivePlanner] Planning learning path for: {task}\033[0m")

        # Decompose the task
        decomposed = self.task_decomposer.decompose(task_obj, inventory, skill_names)

        # Guard: task_decomposer may return None when LLM call fails (vLLM
        # connection error) or when target extraction aborts. Without this
        # guard, skill_gap_analyzer.analyze(None) crashes with AttributeError
        # ('NoneType' has no attribute 'subtasks'). Fall back to direct
        # execution so the curriculum agent doesn't bring the whole run down
        # over a transient LLM outage.
        if decomposed is None:
            print(f"\033[33m[AdaptivePlanner] decomposition unavailable (LLM error or no target); falling back to direct execute: {task}\033[0m")
            return LearningPath(
                original_task=task,
                path_type="direct_execute",
                learning_tasks=[],
                execution_tasks=[task],
                skill_gaps=[]
            )

        # Analyze skill gaps
        gaps = self.skill_gap_analyzer.analyze(decomposed)

        if not gaps:
            # All skills are available; execute directly
            print(f"\033[32m[AdaptivePlanner] All skills available, direct execution\033[0m")
            return LearningPath(
                original_task=task,
                path_type="direct_execute",
                learning_tasks=[],
                execution_tasks=[task],
                skill_gaps=[]
            )

        # Generate learning tasks (using the simplified version)
        learning_tasks = [gap.suggested_learning_task for gap in gaps]

        print(f"\033[33m[AdaptivePlanner] Need to learn {len(learning_tasks)} skills first:\033[0m")
        for i, lt in enumerate(learning_tasks):
            print(f"  {i+1}. {lt}")

        self._current_path = LearningPath(
            original_task=task,
            path_type="learning_first",
            learning_tasks=learning_tasks,
            execution_tasks=[task],
            skill_gaps=gaps
        )

        return self._current_path

    def get_next_task(self, inventory: Optional[Dict[str, int]] = None) -> Optional[str]:
        """
        Get the next task to execute

        Priority:
        1. Outstanding learning tasks (in dependency order), skipping "ensure" tasks already satisfied by inventory
        2. The original task (once all learning tasks are completed or blocked)

        Args:
            inventory: Current inventory for skipping already-satisfied "ensure" tasks

        Returns:
            The next task, or None if there is none
        """
        if not self._current_path:
            return None

        # Check for outstanding learning tasks
        for learning_task in self._current_path.learning_tasks:
            # Auto-complete "ensure" tasks already satisfied by inventory
            if inventory and self._is_ensure_task_satisfied(learning_task, inventory):
                self._task_progress[learning_task] = TaskProgress(
                    task=learning_task,
                    status="completed",
                    max_attempts=self.max_attempts_per_task,
                )
                print(f"\033[32m[AdaptivePlanner] Auto-completed (inventory satisfied): {learning_task}\033[0m")
                continue

            progress = self._task_progress.get(learning_task)

            if progress is None:
                # Not yet started
                self._task_progress[learning_task] = TaskProgress(
                    task=learning_task,
                    status="in_progress",
                    max_attempts=self.max_attempts_per_task
                )
                return learning_task

            elif progress.status == "failed":
                # Failed but retryable
                progress.status = "in_progress"
                return learning_task

            elif progress.status in ["pending", "in_progress"]:
                return learning_task

            # "completed" or "blocked" -> skip

        # All learning tasks handled; check whether the original task can run
        return self._get_original_task_if_ready()

    def _get_original_task_if_ready(self) -> Optional[str]:
        """
        Check whether the original task can be executed

        Strategy:
        - If all learning tasks are completed → run the original task
        - If some tasks are blocked but enough sub-skills exist → still attempt the original task
        - If a critical dependency is blocked → return None (give up)
        """
        if not self._current_path:
            return None

        original_task = self._current_path.original_task

        # Check whether the original task has already been processed
        original_progress = self._task_progress.get(original_task)
        if original_progress and original_progress.status in ["completed", "blocked"]:
            return None

        # Aggregate learning-task statuses
        completed = 0
        blocked = 0
        for learning_task in self._current_path.learning_tasks:
            progress = self._task_progress.get(learning_task)
            if progress:
                if progress.status == "completed":
                    completed += 1
                elif progress.status == "blocked":
                    blocked += 1

        total = len(self._current_path.learning_tasks)

        # Decision logic
        if completed == total:
            # All learning tasks completed; execute the original task
            print(f"\033[32m[AdaptivePlanner] All learning tasks completed, executing original task\033[0m")
            return original_task
        elif completed + blocked == total and completed > 0 and self.continue_on_partial_success:
            # Partial success; try the original task (some sub-skills may be enough)
            print(f"\033[33m[AdaptivePlanner] Partial success ({completed}/{total}), attempting original task\033[0m")
            return original_task
        else:
            # Some tasks are still in progress
            return None

    def update_progress(self, task: str, success: bool, failure_reason: Optional[str] = None):
        """
        Update task progress

        Args:
            task: completed task
            success: whether it succeeded
            failure_reason: failure reason (if it failed)
        """
        if task not in self._task_progress:
            self._task_progress[task] = TaskProgress(
                task=task,
                status="pending",
                max_attempts=self.max_attempts_per_task
            )

        progress = self._task_progress[task]
        progress.attempts += 1

        if success:
            progress.status = "completed"
            print(f"\033[32m[AdaptivePlanner] Task completed: {task}\033[0m")
        else:
            progress.failure_reason = failure_reason
            if progress.attempts >= progress.max_attempts:
                progress.status = "blocked"
                self._failed_tasks.add(task)
                print(f"\033[31m[AdaptivePlanner] Task blocked after {progress.attempts} attempts: {task}\033[0m")

                # Also record in the filter's failure history
                if self.decomposition_filter:
                    self.decomposition_filter.add_failed_task(task)
            else:
                progress.status = "failed"  # retryable
                print(f"\033[33m[AdaptivePlanner] Task failed (attempt {progress.attempts}): {task}\033[0m")

    def is_path_completed(self) -> bool:
        """Check whether the current learning path is complete"""
        if not self._current_path:
            return True

        original_progress = self._task_progress.get(self._current_path.original_task)
        return original_progress is not None and original_progress.status == "completed"

    def is_path_blocked(self) -> bool:
        """Check whether the current learning path is blocked (cannot proceed)"""
        if not self._current_path:
            return False

        # If the original task has been marked as blocked
        original_progress = self._task_progress.get(self._current_path.original_task)
        if original_progress and original_progress.status == "blocked":
            return True

        # If all learning tasks are blocked
        if self._current_path.learning_tasks:
            all_blocked = all(
                self._task_progress.get(t, TaskProgress(t, "pending")).status == "blocked"
                for t in self._current_path.learning_tasks
            )
            return all_blocked

        return False

    def get_current_path(self) -> Optional[LearningPath]:
        """Get the current learning path"""
        return self._current_path

    def get_progress_summary(self) -> Dict:
        """
        Get a progress summary

        Returns:
            Progress-summary dict
        """
        if not self._current_path:
            return {"status": "no_path", "learning_tasks": 0, "completed": 0, "blocked": 0}

        completed = sum(
            1 for t in self._current_path.learning_tasks
            if self._task_progress.get(t, TaskProgress(t, "pending")).status == "completed"
        )
        blocked = sum(
            1 for t in self._current_path.learning_tasks
            if self._task_progress.get(t, TaskProgress(t, "pending")).status == "blocked"
        )
        total = len(self._current_path.learning_tasks)

        original_status = "pending"
        original_progress = self._task_progress.get(self._current_path.original_task)
        if original_progress:
            original_status = original_progress.status

        return {
            "original_task": self._current_path.original_task,
            "path_type": self._current_path.path_type,
            "learning_tasks": total,
            "completed": completed,
            "blocked": blocked,
            "in_progress": total - completed - blocked,
            "original_status": original_status,
        }

    @staticmethod
    def _is_ensure_task_satisfied(task: str, inventory: Dict[str, int]) -> bool:
        """Check if an 'Ensure you have X item' task is already satisfied by inventory."""
        import re
        match = re.match(
            r"ensure\s+you\s+have\s+(\d+)?\s*(.+)", task.lower()
        )
        if not match:
            return False
        count = int(match.group(1)) if match.group(1) else 1
        item = match.group(2).strip().replace(" ", "_")

        # Exact match
        if inventory.get(item, 0) >= count:
            return True

        # Group matching via _get_group_count (handles "logs", "planks", etc.)
        from skillnet.agents.psn_curriculum.goal_planner import _get_group_count
        if _get_group_count(inventory, item) >= count:
            return True

        # Fuzzy: "wood_logs" → try "logs" group
        if "log" in item:
            if _get_group_count(inventory, "logs") >= count:
                return True

        return False

    def clear_path(self):
        """Clear the current learning path (called when starting a new task)"""
        self._current_path = None
        # Note: do NOT clear _task_progress and _failed_tasks; they are accumulated historical records

    def reset(self):
        """Fully reset the planner's state"""
        self._current_path = None
        self._task_progress.clear()
        self._failed_tasks.clear()
