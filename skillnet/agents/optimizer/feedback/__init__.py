"""
Optimizer Feedback Package

Feedback system module, including:
- Feedback type definitions
- Forward propagation (opt_forward_feedback)
- Feedback parser
"""

from .types import (
    SkillFeedback,
    OptimizationRecord,
    CodeEdit,
    ModificationAnalysis,
)

from .forward_propagation import (
    InterfaceChange,
    EffectChange,
    OptimizationForwardFeedback,
)

from .parser import (
    extract_code_example_from_feedback,
    extract_issue_type_from_content,
    format_feedbacks_for_llm,
    extract_constraint_feedback_from_critique,
)

__all__ = [
    # Types
    "SkillFeedback",
    "OptimizationRecord",
    "CodeEdit",
    "ModificationAnalysis",

    # Forward Propagation
    "InterfaceChange",
    "EffectChange",
    "OptimizationForwardFeedback",

    # Parser
    "extract_code_example_from_feedback",
    "extract_issue_type_from_content",
    "format_feedbacks_for_llm",
    "extract_constraint_feedback_from_critique",
]
