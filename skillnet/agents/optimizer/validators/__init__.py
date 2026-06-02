"""
Optimizer validators — semantic checks, growth checks, naming/typing checks.
"""

from .bloat_checker import (
    BloatChecker,
    FixPriority,
    FixType,
)

from .skip_checker import SkipChecker

from .effects_consistency_checker import EffectsConsistencyValidator

from .code_validator import (
    validate_tdz_issues,
    find_bracket_mismatch_line,
    validate_code_completeness,
    fix_tdz_issues,
    validate_function_implementation,
)

from .overclaim_detector import (
    OverclaimDetector,
    OverclaimResult,
)

from .naming_conflict_checker import check_naming_conflicts

from .type_adapter import TypeMismatchDetector

from .semantic_compatibility import SemanticEquivalenceValidator

__all__ = [
    # Bloat checking
    "BloatChecker",
    "FixPriority",
    "FixType",
    # Skip / consistency / overclaim / naming / type / semantic compat
    "SkipChecker",
    "EffectsConsistencyValidator",
    "OverclaimDetector",
    "OverclaimResult",
    "check_naming_conflicts",
    "TypeMismatchDetector",
    "SemanticEquivalenceValidator",
    # Code validation primitives
    "validate_tdz_issues",
    "find_bracket_mismatch_line",
    "validate_code_completeness",
    "fix_tdz_issues",
    "validate_function_implementation",
]
