"""
Optimizer Helper Modules

Module splitting for better maintainability.
Contains pure functions and utility methods extracted from optimizer_impl.py.
"""

from .feedback_persistence import (
    save_feedback_history,
    load_feedback_history,
    mark_feedbacks_resolved,
)
from .semantic_analysis import (
    detect_parameter_semantic_mismatch,
    analyze_semantic_from_context,
    detect_semantic_backprop_candidate,
)
from .error_stack_parser import (
    normalize_error_pattern,
)
from .code_context import (
    extract_focus_lines,
    extract_code_with_context,
    extract_function_signature,
    extract_code_with_signature,
)

__all__ = [
    # Feedback persistence
    "save_feedback_history",
    "load_feedback_history",
    "mark_feedbacks_resolved",
    # Semantic analysis
    "detect_parameter_semantic_mismatch",
    "analyze_semantic_from_context",
    "detect_semantic_backprop_candidate",
    # Error pattern normalization
    "normalize_error_pattern",
    # Code context
    "extract_focus_lines",
    "extract_code_with_context",
    "extract_function_signature",
    "extract_code_with_signature",
]
