"""
Optimizer analysis — error classification and interface/semantic change detection.
"""

from .error_classifier import (
    ErrorCategory,
    FixTargetType,
)

from .interface_analyzer import (
    parse_function_params,
    analyze_param_changes,
    classify_interface_change,
    detect_options_object_change,
    determine_impact_and_strategy,
)

from .semantic_change_detector import (
    SemanticChangeResult,
    detect_semantic_changes,
)

__all__ = [
    "ErrorCategory",
    "FixTargetType",
    "parse_function_params",
    "analyze_param_changes",
    "classify_interface_change",
    "detect_options_object_change",
    "determine_impact_and_strategy",
    "SemanticChangeResult",
    "detect_semantic_changes",
]
