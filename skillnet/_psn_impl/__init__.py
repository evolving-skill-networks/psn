"""
PSNAgent implementation mixins.

Extracted from psn.py for better modularity.

Inheritance order (MUST be preserved in PSNAgent class definition):
    class PSNAgent(
        EventProcessingMixin,        # Layer 0: event parsing utilities
        SkillRecordingMixin,         # Layer 1: depends on EventProcessing
        EffectVerificationMixin,     # Layer 1: depends on EventProcessing
        StepExecutionMixin,          # Layer 2: depends on SkillRecording + EffectVerification
        TaskManagementMixin,         # Layer 3: standalone
    )

StepExecutionMixin internal composition:
    StepExecutionMixin(StepPlanMixin, StepExecutePhaseMixin, StepOptimizeMixin)
      step_plan.py         -> Phases 1-3: planning, code extraction, recursive error handling
      step_execute_phase.py -> Phases 4-5: env execution, diagnostics
      step_optimize.py      -> Phase 6: two-phase optimization
      step_execution.py     -> step() orchestrator, Phase 7 (message rebuild), preflight

Cross-mixin call graph:
    StepExecutionMixin
      -> SkillRecordingMixin._auto_record_skill_executions()
      -> SkillRecordingMixin._find_reuse_skill_hint()
    SkillRecordingMixin
      -> EventProcessingMixin._find_nearby_blocks_from_events()
      -> EffectVerificationMixin._check_skill_effect_achieved()

PSNAgent.__init__ must initialize these attributes before mixin methods are called:
    Required: skill_manager, action_agent, planner, env, curriculum_agent,
              task, context, messages, conversations, last_events,
              last_skill_execution_results, planner_mode,
              enable_optimizer, _task_semantic, _task_executed_skills
    Optional: optimizer
"""

from .event_processing import EventProcessingMixin
from .skill_recording import SkillRecordingMixin
from .effect_verification import EffectVerificationMixin
from .step_context import StepContext
from .step_execution import StepExecutionMixin
from .task_management import TaskManagementMixin
from .event_helpers import (
    Event,
    Events,
    unpack_event,
    get_event_data_dict,
    iter_events,
    find_last_observe,
)

__all__ = [
    "EventProcessingMixin",
    "SkillRecordingMixin",
    "EffectVerificationMixin",
    "StepExecutionMixin",
    "StepContext",
    "TaskManagementMixin",
    "Event",
    "Events",
    "unpack_event",
    "get_event_data_dict",
    "iter_events",
    "find_last_observe",
]
