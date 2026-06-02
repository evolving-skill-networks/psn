"""
Optimizer Internal Implementation

This is the internal implementation module containing the full SkillGraphOptimizer.
External code should access these classes via the public API of the optimizer package.

Note: this module's API may change in future versions, please do not import from it directly.
"""

# Import core classes from optimizer_impl (the only ones defined here)
from .optimizer_impl import (
    # Core optimizer
    SkillGraphOptimizer,

    # Tracker
    OptimizationTracker,
)

# CodeBloatTracker has been split out into the tracking module
from skillnet.agents.optimizer.tracking import CodeBloatTracker

# Import error classification and diagnostic types from analysis/
# (the single source of truth after the modular refactor)
from skillnet.agents.optimizer.analysis import (
    ErrorCategory,
    FixTargetType,
)

# Import feedback types from feedback/
# (the single source of truth after the modular refactor)
from skillnet.agents.optimizer.feedback import (
    SkillFeedback,
    OptimizationRecord,
    CodeEdit,
    ModificationAnalysis,
)

__all__ = [
    # Core optimizer
    "SkillGraphOptimizer",

    # Tracker (defined only in optimizer_impl.py)
    "OptimizationTracker",

    # Error classification (re-exported from analysis/)
    "ErrorCategory",
    "FixTargetType",

    # Feedback types (re-exported from feedback/)
    "SkillFeedback",
    "OptimizationRecord",
    "CodeEdit",
    "ModificationAnalysis",

    # Code bloat control
    "CodeBloatTracker",
]
