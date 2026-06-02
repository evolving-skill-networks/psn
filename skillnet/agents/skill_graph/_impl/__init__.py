"""
Skill Graph Internal Implementation

This is the internal implementation module, containing the complete implementation of SkillGraphManager.
External code should access these classes through the public API of the skill_graph package.

Note: the API of this module may change in future versions; do not import directly.
"""

# Import the core manager from graph_manager_impl
from .graph_manager_impl import SkillGraphManager

# Import all dataclasses from models/ (the sole source after modular refactor)
from skillnet.agents.skill_graph.models import (
    FailureCategory,
    PreconditionValueFunctionConfig,
    EffectValueFunctionConfig,
    SkillPrecondition,
    SkillEffect,
    ActualEffect,
    SkillVersion,
    SkillExecutionTrace,
    SkillStatistics,
    SkillGradientItem,
    SkillGradients,
    SkillNode,
    GraphVersion,
    SkillGraph,
    should_skip_covered_skill,
    CoverageType,
)

# Import utility functions from utils/
from skillnet.agents.skill_graph.utils import (
    _strip_comments_and_strings,
    validate_code_brackets,
    validate_code_syntax,
    check_line_length,
    is_control_primitive_name,
    resolve_primitive_name_conflict,
)

__all__ = [
    # Core manager
    "SkillGraphManager",

    # Dataclasses (re-exported from models/)
    "FailureCategory",
    "PreconditionValueFunctionConfig",
    "EffectValueFunctionConfig",
    "SkillPrecondition",
    "SkillEffect",
    "ActualEffect",
    "SkillVersion",
    "SkillExecutionTrace",
    "SkillStatistics",
    "SkillGradientItem",
    "SkillGradients",
    "SkillNode",
    "GraphVersion",
    "SkillGraph",
    "should_skip_covered_skill",
    "CoverageType",

    # Utility functions (re-exported from utils/)
    "_strip_comments_and_strings",
    "validate_code_brackets",
    "validate_code_syntax",
    "check_line_length",
    "is_control_primitive_name",
    "resolve_primitive_name_conflict",
]
