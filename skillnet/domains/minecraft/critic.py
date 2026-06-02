"""
Minecraft Critic Strategy

Wraps PSNCriticAgent to implement the CriticStrategy ABC.
"""

from typing import Any, Dict, Tuple

from skillnet.core.critic import CriticStrategy


class MinecraftCritic(CriticStrategy):
    """
    CriticStrategy implementation for Minecraft.

    Thin wrapper around PSNCriticAgent — all logic is delegated.
    The underlying agent must be set via constructor or set_agent().
    """

    def __init__(self, agent=None):
        self._agent = agent

    def set_agent(self, agent) -> None:
        """Set the underlying PSNCriticAgent (for late binding)."""
        self._agent = agent

    @property
    def unwrapped(self):
        """Access the underlying PSNCriticAgent."""
        return self._agent

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
        if self._agent is None:
            raise RuntimeError("MinecraftCritic: agent not set")
        return self._agent.check_task_success(
            events=events,
            task=task,
            context=context,
            chest_observation=chest_observation,
            max_retries=max_retries,
            **kwargs,
        )

    def render_human_message(
        self,
        *,
        events: Any,
        task: Any,
        context: str = "",
        chest_observation: str = "",
        **kwargs,
    ) -> Any:
        if self._agent is None:
            raise RuntimeError("MinecraftCritic: agent not set")
        return self._agent.render_human_message(
            events=events,
            task=task,
            context=context,
            chest_observation=chest_observation,
            **kwargs,
        )
