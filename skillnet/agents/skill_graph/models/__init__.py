"""
Skill Graph Models

Data model definitions; contains all core dataclasses.
"""

from .config import (
    FailureCategory,
    PreconditionValueFunctionConfig,
    EffectValueFunctionConfig,
)

from .precondition import (
    SkillPrecondition,
    SkillEffect,
    ActualEffect,
)

from .gradients import (
    SkillGradientItem,
    SkillGradients,
)

from .execution import (
    SkillExecutionTrace,
    SkillStatistics,
)

from .version import SkillVersion

from .node import (
    SkillNode,
    CodeSource,
    UpdateResult,
    ConfirmResult,
)

from .graph import (
    SkillGraph,
    GraphVersion,
    should_skip_covered_skill,
)

from .coverage import CoverageType

__all__ = [
    # Config
    "FailureCategory",
    "PreconditionValueFunctionConfig",
    "EffectValueFunctionConfig",

    # Preconditions and effects
    "SkillPrecondition",
    "SkillEffect",
    "ActualEffect",

    # Gradients
    "SkillGradientItem",
    "SkillGradients",

    # Execution
    "SkillExecutionTrace",
    "SkillStatistics",

    # Version
    "SkillVersion",

    # Nodes and graph
    "SkillNode",
    "SkillGraph",
    "GraphVersion",
    "should_skip_covered_skill",

    # Code update interface types
    "CodeSource",
    "UpdateResult",
    "ConfirmResult",

    # Coverage type
    "CoverageType",
]
