"""
Environment Abstract Base Class

Defines the interface that all PSN environments must implement.
Decouples the core PSN learning loop from any specific execution backend
(e.g., Minecraft/Mineflayer).

Usage:
    class MyEnv(Environment):
        def reset(self, *, seed=None, options=None): ...
        def step(self, code, programs="", skill_names=None): ...
        def close(self): ...
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class Environment(ABC):
    """
    Abstract environment interface for PSN agents.

    Mirrors the Gymnasium-style reset/step/close lifecycle but with
    PSN-specific step semantics (code execution rather than discrete actions).
    """

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Backend-specific reset options.

        Returns:
            (observation, info) tuple.
        """

    @abstractmethod
    def step(
        self,
        code: str,
        programs: str = "",
        skill_names: Optional[List[str]] = None,
        is_iteration: bool = True,
    ) -> Any:
        """
        Execute code in the environment and return observations.

        Args:
            code: Skill code to execute.
            programs: Library code to make available during execution.
            skill_names: Names of skills being executed (for logging/tracking).
            is_iteration: Whether this counts as a real task-execution
                iteration of the agent loop. Non-iteration callers
                (revert-placed-blocks, reset peeks, observe refreshes)
                should pass False so they are excluded from iteration
                counters and per-iteration logs.

        Returns:
            Execution result (format is backend-specific).
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by the environment."""

    # ------------------------------------------------------------------
    # Optional hooks — concrete implementations may override these.
    # ------------------------------------------------------------------

    def pause(self) -> bool:
        """Pause the environment (e.g., freeze game ticks). Returns success."""
        return False

    def unpause(self) -> bool:
        """Unpause a previously paused environment. Returns success."""
        return False
