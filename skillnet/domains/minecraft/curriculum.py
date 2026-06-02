"""
Minecraft Curriculum Strategy

Wraps PSNCurriculumAgent to implement the CurriculumStrategy ABC.
"""

from typing import Any, Callable, Dict, List, Tuple

from skillnet.core.curriculum import CurriculumStrategy


class MinecraftCurriculum(CurriculumStrategy):
    """
    CurriculumStrategy implementation for Minecraft.

    Thin wrapper around PSNCurriculumAgent — all logic is delegated.
    The underlying agent must be set via constructor or set_agent().
    """

    def __init__(self, agent=None):
        self._agent = agent

    def set_agent(self, agent) -> None:
        """Set the underlying PSNCurriculumAgent (for late binding)."""
        self._agent = agent

    @property
    def unwrapped(self):
        """Access the underlying PSNCurriculumAgent."""
        return self._agent

    # ---- Abstract method implementations ----

    def propose_next_task(
        self,
        *,
        events: Any,
        chest_observation: str = "",
        max_retries: int = 5,
        **kwargs,
    ) -> Tuple[Any, str]:
        if self._agent is None:
            raise RuntimeError("MinecraftCurriculum: agent not set")
        return self._agent.propose_next_task(
            events=events,
            chest_observation=chest_observation,
            max_retries=max_retries,
        )

    def task_count(self) -> int:
        if self._agent is None:
            return 0
        return self._agent.task_count()

    def add_completed_tasks(self, tasks: List[str]) -> None:
        if self._agent is None:
            raise RuntimeError("MinecraftCurriculum: agent not set")
        # PSNCurriculumAgent stores completed tasks directly
        self._agent.completed_tasks.extend(tasks)

    def add_failed_tasks(self, tasks: List[str]) -> None:
        if self._agent is None:
            raise RuntimeError("MinecraftCurriculum: agent not set")
        self._agent.add_failed_tasks(tasks)

    # ---- Optional method overrides (delegate to agent) ----

    def set_skill_manager(self, skill_manager) -> None:
        if self._agent:
            self._agent.set_skill_manager(skill_manager)

    def get_completed_tasks(self) -> List[str]:
        return self._agent.get_completed_tasks() if self._agent else []

    def get_failed_tasks(self) -> List[str]:
        return self._agent.get_failed_tasks() if self._agent else []

    def count_task_failures(self) -> Dict[str, int]:
        return self._agent.count_task_failures() if self._agent else {}

    def remove_matching_failed_tasks(self, predicate: Callable) -> int:
        return self._agent.remove_matching_failed_tasks(predicate) if self._agent else 0

    def update_exploration_progress(self, info: Dict[str, Any]) -> None:
        if self._agent:
            self._agent.update_exploration_progress(info)

    def update_task_result(self, task: str, success: bool) -> None:
        if self._agent:
            self._agent.update_task_result(task, success)

    def decompose_task(self, task: str, events: Any) -> Any:
        return self._agent.decompose_task(task, events) if self._agent else None

    def reset_task_lists(self) -> None:
        if self._agent:
            self._agent.reset_task_lists()

    @property
    def progress(self) -> int:
        return self._agent.progress if self._agent else 0

    def get_task_context(self, task: str) -> str:
        return self._agent.get_task_context(task) if self._agent else ""
