"""
Optimization Escalation Protocol.

Tracks per-task failure patterns and triggers escalation when optimization
repeatedly fails to improve skill performance.

Levels:
  0 (default): Normal optimizer Phase 1 + Phase 2
  1 (regenerate): Send old code + failure history as negative example to Action Agent
                  for fresh code generation
  2 (curriculum): Notify Curriculum for strategic decision (deferred to Option 3)
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass
class TaskEscalationState:
    """Per-task escalation tracking state."""

    level: int = 0
    iteration_count: int = 0
    v_history: List[float] = field(default_factory=list)
    failure_codes: List[str] = field(default_factory=list)
    failure_errors: List[str] = field(default_factory=list)
    level1_triggered: bool = False


class EscalationTracker:
    """Tracks per-task optimization failures and triggers escalation.

    Escalation from Level 0 → Level 1 triggers when:
    1. At least `min_attempts` consecutive failures have occurred, AND
    2. V(s) has not improved by more than `improvement_threshold` across those attempts.

    Level 1 action: regenerate code with failure history as negative example.
    On task success: immediately reset to Level 0.
    """

    def __init__(
        self,
        min_attempts: int = 3,
        improvement_threshold: float = 0.05,
    ):
        self.min_attempts = min_attempts
        self.improvement_threshold = improvement_threshold
        self._states: Dict[str, TaskEscalationState] = {}

    @staticmethod
    def _normalize_task(task: str) -> str:
        """Normalize task string for consistent tracking."""
        return task.strip().lower()

    def get_state(self, task: str) -> TaskEscalationState:
        """Get or create escalation state for a task."""
        key = self._normalize_task(task)
        if key not in self._states:
            self._states[key] = TaskEscalationState()
        return self._states[key]

    def iter_task_states(self) -> Iterable[Tuple[str, TaskEscalationState]]:
        """Iterate over all tracked task states (public API for SEM)."""
        return self._states.items()

    def record_step_result(
        self,
        task: str,
        success: bool,
        v_s: float = 0.0,
        error: str = "",
        code: str = "",
    ):
        """Record one step iteration's result for escalation tracking.

        Should be called after each failed step (optimization phase complete).
        """
        if success:
            self.on_task_success(task)
            return

        state = self.get_state(task)

        state.iteration_count += 1
        state.v_history.append(v_s)

        if code:
            state.failure_codes.append(code)
        if error:
            state.failure_errors.append(error)

        # Keep last max_history entries to avoid unbounded growth
        max_history = self.min_attempts + 2
        if len(state.v_history) > max_history:
            state.v_history = state.v_history[-max_history:]
        if len(state.failure_codes) > max_history:
            state.failure_codes = state.failure_codes[-max_history:]
        if len(state.failure_errors) > max_history:
            state.failure_errors = state.failure_errors[-max_history:]

    def should_escalate(self, task: str) -> bool:
        """Check if task should escalate from Level 0 → Level 1.

        Returns True when:
        - At least min_attempts consecutive failures have been recorded
        - Level 1 has not already been triggered for this task cycle

        Uses pure failure counting (no V(s) check) because V(s) can be
        contaminated by sub-skill optimizations that don't fix the root cause.
        """
        state = self.get_state(task)

        if state.level >= 1:
            return False

        if state.level1_triggered:
            return False

        if state.iteration_count < self.min_attempts:
            return False

        return True

    def mark_escalated(self, task: str):
        """Mark that Level 1 escalation has been triggered for this task."""
        state = self.get_state(task)
        state.level = 1
        state.level1_triggered = True

    def get_failure_summary(self, task: str) -> str:
        """Build failure history summary for Level 1 regeneration.

        Returns a formatted string describing past failed approaches and errors,
        suitable for injection as critique/context in the Action Agent prompt.
        """
        state = self.get_state(task)

        parts = [
            "ESCALATION: Previous approaches have repeatedly failed. "
            "You MUST try a completely different strategy.\n"
        ]

        # Include recent errors
        unique_errors = []
        seen = set()
        for err in state.failure_errors:
            err_key = err[:100]  # Deduplicate by prefix
            if err_key not in seen and err:
                seen.add(err_key)
                unique_errors.append(err)

        if unique_errors:
            parts.append("Previous errors encountered:")
            for i, err in enumerate(unique_errors[-3:], 1):  # Last 3 unique errors
                err_display = err[:300] + "..." if len(err) > 300 else err
                parts.append(f"  {i}. {err_display}")

        # V(s) trend
        if state.v_history:
            recent_v = state.v_history[-self.min_attempts:]
            if recent_v:
                parts.append(
                    f"\nSkill value over last {len(recent_v)} attempts: "
                    f"{', '.join(f'{v:.3f}' for v in recent_v)} "
                    f"(no improvement detected)"
                )

        parts.append(
            "\nDo NOT repeat the same approach. Consider a fundamentally "
            "different algorithm or strategy."
        )

        return "\n".join(parts)

    def on_task_success(self, task: str):
        """Reset escalation state on task success (de-escalation)."""
        key = self._normalize_task(task)
        if key in self._states:
            self._states[key] = TaskEscalationState()
