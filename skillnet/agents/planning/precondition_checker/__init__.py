"""
Precondition Checker Package

Precondition checker: validates preconditions, finds skills that satisfy
conditions, and collects environmental feedback.

refactored from the single-file precondition_checker.py into an internal layered package.
- _utils.py: pure functions and constants (tool tiers, generic types, condition evaluation)
- _environmental.py: EnvironmentalFeedback data class + error inference functions
- _checking.py: CheckingMixin (precondition validation)
- _skill_finding.py: SkillFindingMixin (skill lookup)
- _checker.py: PreconditionChecker main class (facade)
"""

from ._checker import PreconditionChecker
from ._utils import (
    get_tool_tier,
    extract_tool_type,
    check_tool_requirement,
    condition_matches,
    extract_conditions_from_state_repr,
    normalize_shorthand_condition,
    extract_generic_type,
    is_generic_item_type,
    get_tool_tiers,
    get_material_prefixes,
    get_generic_patterns,
)
from ._environmental import (
    EnvironmentalFeedback,
    infer_environmental_feedback_from_error,
)

__all__ = [
    "PreconditionChecker",
    # Pure functions
    "condition_matches",
    "extract_generic_type",
    "is_generic_item_type",
    "extract_conditions_from_state_repr",
    "check_tool_requirement",
    "get_tool_tier",
    "extract_tool_type",
    "normalize_shorthand_condition",
    # Domain-aware accessors
    "get_tool_tiers",
    "get_material_prefixes",
    "get_generic_patterns",
    # Environmental feedback
    "EnvironmentalFeedback",
    "infer_environmental_feedback_from_error",
]
