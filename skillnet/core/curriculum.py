"""
Curriculum Strategy Abstract Base Class

Defines the interface for task proposal strategies that drive the
PSN learning loop.  Different domains implement different curriculum
strategies (e.g., Minecraft milestones).

Usage:
    class MyCurriculum(CurriculumStrategy):
        def propose_next_task(self, *, events, **kwargs): ...
        def task_count(self): ...
        ...
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple


class CurriculumStrategy(ABC):
    """
    Abstract interface for curriculum / task proposal.

    The PSN learning loop calls propose_next_task() each iteration
    to get the next task to attempt.

    Subclasses MUST implement the 4 abstract methods.
    The remaining methods have sensible defaults and can be overridden
    as needed by domain-specific implementations.
    """

    # ---- Abstract (required) ----

    @abstractmethod
    def propose_next_task(
        self,
        *,
        events: Any,
        chest_observation: str = "",
        max_retries: int = 5,
        **kwargs,
    ) -> Tuple[Any, str]:
        """
        Propose the next task for the agent to attempt.

        Args:
            events: Current environment observations.
            chest_observation: Inventory/storage description (if applicable).
            max_retries: Maximum LLM retries for task proposal.

        Returns:
            (task, context) — the task object and a context string.
        """

    @abstractmethod
    def task_count(self) -> int:
        """Return the number of completed tasks so far."""

    @abstractmethod
    def add_completed_tasks(self, tasks: List[str]) -> None:
        """Record tasks as completed."""

    @abstractmethod
    def add_failed_tasks(self, tasks: List[str]) -> None:
        """Record tasks as failed."""

    # ---- Optional (with defaults) ----

    def set_skill_manager(self, skill_manager) -> None:
        """Inject skill manager for skill-aware task selection."""
        pass

    def get_completed_tasks(self) -> List[str]:
        """Return list of completed task names."""
        return []

    def get_failed_tasks(self) -> List[str]:
        """Return list of failed task names."""
        return []

    def count_task_failures(self) -> Dict[str, int]:
        """Return failure counts per task name."""
        return {}

    def remove_matching_failed_tasks(self, predicate: Callable) -> int:
        """Remove failed tasks matching predicate. Return count removed."""
        return 0

    def update_exploration_progress(self, info: Dict[str, Any]) -> None:
        """Update progress tracking after an iteration."""
        pass

    def update_task_result(self, task: str, success: bool) -> None:
        """Record outcome of a task attempt."""
        pass

    def decompose_task(self, task: str, events: Any) -> Any:
        """Decompose a complex task into subtasks. Return None if not supported."""
        return None

    def reset_task_lists(self) -> None:
        """Reset completed/failed task lists."""
        pass

    @property
    def progress(self) -> int:
        """Current progress counter (default = task_count)."""
        return self.task_count()

    def get_task_context(self, task: str) -> str:
        """Return context string for a task."""
        return ""
