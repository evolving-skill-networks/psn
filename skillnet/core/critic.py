"""
Critic Strategy Abstract Base Class

Defines the interface for task success evaluation.  Different domains
have different notions of success (e.g. inventory changes in Minecraft).

Usage:
    class MyCritic(CriticStrategy):
        def check_task_success(self, *, task, events, **kwargs): ...
        def render_human_message(self, *, events, task, context, **kwargs): ...
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class CriticStrategy(ABC):
    """
    Abstract interface for evaluating whether a task was completed.

    The PSN learning loop calls check_task_success() after each execution
    attempt to decide whether to record success or retry/fail.
    """

    @abstractmethod
    def check_task_success(
        self,
        *,
        task: Any,
        events: Any,
        context: str = "",
        chest_observation: str = "",
        max_retries: int = 5,
        **kwargs,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluate whether the task was completed successfully.

        Args:
            task: The task being evaluated.
            events: Environment observations after execution.
            context: Task context string.
            chest_observation: Inventory/storage description.
            max_retries: Maximum LLM retries for evaluation.

        Returns:
            (success, critique, quality_metrics) — boolean result,
            explanation string, and optional quality metrics dict.
        """

    @abstractmethod
    def render_human_message(
        self,
        *,
        events: Any,
        task: Any,
        context: str = "",
        chest_observation: str = "",
        **kwargs,
    ) -> Any:
        """
        Build the evaluation message for the LLM critic.

        Args:
            events: Environment observations.
            task: The task being evaluated.
            context: Task context string.
            chest_observation: Inventory/storage description.

        Returns:
            Message object suitable for the underlying LLM.
        """
