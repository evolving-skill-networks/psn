"""
Skill Graph Package

Skill graph module, containing data structures and graph management functionality.
"""

# Export core dataclasses from the models submodule
from .models import (
    # Base config
    FailureCategory,
    PreconditionValueFunctionConfig,
    EffectValueFunctionConfig,

    # Preconditions and effects
    SkillPrecondition,
    SkillEffect,
    ActualEffect,

    # Version and execution
    SkillVersion,
    SkillExecutionTrace,
    SkillStatistics,

    # Gradients
    SkillGradientItem,
    SkillGradients,

    # Nodes and graph
    SkillNode,
    SkillGraph,
    GraphVersion,
    should_skip_covered_skill,
)

# Export the graph manager and helper functions from manager
from .manager import SkillGraphManager, _strip_comments_and_strings

# Utility functions
from .utils import (
    validate_code_brackets,
    validate_code_syntax,
    check_line_length,
    is_control_primitive_name,
    resolve_primitive_name_conflict,
)

__all__ = [
    # Config classes
    "FailureCategory",
    "PreconditionValueFunctionConfig",
    "EffectValueFunctionConfig",

    # Data classes
    "SkillPrecondition",
    "SkillEffect",
    "ActualEffect",
    "SkillVersion",
    "SkillExecutionTrace",
    "SkillStatistics",
    "SkillGradientItem",
    "SkillGradients",
    "SkillNode",
    "SkillGraph",
    "GraphVersion",
    "should_skip_covered_skill",

    # Manager
    "SkillGraphManager",

    # Utility functions
    "validate_code_brackets",
    "validate_code_syntax",
    "check_line_length",
    "is_control_primitive_name",
    "resolve_primitive_name_conflict",
]
