"""
Dataclasses related to execution tracing and statistics.

Includes SkillExecutionTrace and SkillStatistics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional

from .precondition import ActualEffect


class SkillExecutionStatus(Enum):
    """Skill execution status enum.

    Used to precisely distinguish skill execution states:
    - SUCCESS: execution completed successfully.
    - FAILED: execution failed (runtime or logic error).
    - NOT_EXECUTED: called in code but not triggered at runtime (e.g. early return, conditional skip).
    - INTERRUPTED: execution was interrupted (e.g. user cancel, timeout).
    """
    SUCCESS = "success"
    FAILED = "failed"
    NOT_EXECUTED = "not_executed"
    INTERRUPTED = "interrupted"


@dataclass
class SkillExecutionTrace:
    """Skill execution trace info."""
    execution_id: str  # Execution ID
    timestamp: str  # Execution timestamp
    trajectory_segment_id: Optional[str] = None  # Corresponding trajectory segment ID
    trajectory_file: Optional[str] = None  # Corresponding trajectory file path
    task: Optional[str] = None  # Corresponding task
    context: Optional[str] = None  # Execution context
    environment_events: Optional[List[Any]] = None  # Environment events (runtime observations)
    environment_state: Optional[Dict[str, Any]] = None  # Environment state (post-execution)
    pre_state: Optional[Dict[str, Any]] = None  # State before execution
    post_state: Optional[Dict[str, Any]] = None  # State after execution (same as environment_state, but explicit)
    action_info: Optional[Dict[str, Any]] = None  # Action info (includes code, program_name, etc.)
    success: bool = False  # Whether succeeded
    error_message: Optional[str] = None  # Error message (when failed)
    error_stack: Optional[str] = None  # Error stack trace (stored separately for precise error attribution)
    critique: Optional[str] = None  # Critic-agent feedback
    actual_effects: Optional[List[ActualEffect]] = None  # List of actually produced effects
    call_args: Optional[Dict[str, Any]] = None  # Call arguments (option 6: parameterized effects)
    # Format: {"count": 3, "logTypes": ["oak_log", "birch_log"]}
    call_stack: Optional[List[str]] = None  # Call stack (used to determine top-level vs nested call)
    call_depth: int = 1  # Call depth: 1 = top-level call, >1 = nested call (default 1 for backward compatibility)
    execution_status: str = "success"  # Execution status: success/failed/not_executed/interrupted

    @property
    def was_executed(self) -> bool:
        """Whether actually executed (not_executed returns False)."""
        return self.execution_status != "not_executed"

    def __init__(
        self,
        execution_id: str = None,
        timestamp: str = None,
        trajectory_segment_id: str = None,
        trajectory_file: str = None,
        task: str = None,
        context: str = None,
        environment_events: List[Any] = None,
        environment_state: Dict[str, Any] = None,
        pre_state: Dict[str, Any] = None,
        post_state: Dict[str, Any] = None,
        action_info: Dict[str, Any] = None,
        success: bool = False,
        error_message: str = None,
        error_stack: str = None,
        critique: str = None,
        actual_effects: List[ActualEffect] = None,
        call_args: Dict[str, Any] = None,
        call_stack: List[str] = None,
        call_depth: int = 1,
        execution_status: str = None,
    ):
        self.execution_id = execution_id or f"exec_{datetime.now().timestamp()}"
        self.timestamp = timestamp or datetime.now().isoformat()
        self.trajectory_segment_id = trajectory_segment_id
        self.trajectory_file = trajectory_file
        self.task = task
        self.context = context
        self.environment_events = environment_events or []
        self.environment_state = environment_state or {}
        self.pre_state = pre_state or {}
        self.post_state = post_state or (self.environment_state or {})
        self.action_info = action_info or {}
        self.success = success
        self.error_message = error_message
        self.error_stack = error_stack
        self.critique = critique
        self.actual_effects = actual_effects or []
        self.call_args = call_args or {}
        self.call_stack = call_stack or []
        self.call_depth = call_depth
        # execution_status: if unspecified, infer from success
        if execution_status is not None:
            self.execution_status = execution_status
        else:
            self.execution_status = "success" if success else "failed"


@dataclass
class SkillStatistics:
    """Skill execution statistics."""
    total_executions: int = 0  # Total executions
    successful_executions: int = 0  # Successful executions
    failed_executions: int = 0  # Failed executions
    execution_traces: List[SkillExecutionTrace] = field(default_factory=list)  # Trace list

    @property
    def success_rate(self) -> float:
        """success rate"""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    def add_execution(self, trace: SkillExecutionTrace):
        """Add an execution record.

        Note: NOT_EXECUTED statuses do not count toward total_executions but
        the trace is still recorded for traceability.
        """
        # NOT_EXECUTED does not count toward execution stats (no actual execution)
        if trace.execution_status == "not_executed":
            # Still record the trace but do not affect statistics
            pass
        else:
            self.total_executions += 1
            if trace.success:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
        self.execution_traces.append(trace)
        # Keep only the most recent N records (avoid memory bloat)
        if len(self.execution_traces) > 100:
            self.execution_traces = self.execution_traces[-100:]

    def to_dict(self) -> dict:
        """Convert to dict (excluding the full execution_traces)."""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.success_rate,
        }
