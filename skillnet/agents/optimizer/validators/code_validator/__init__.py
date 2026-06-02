"""
Code Validator Package

Code validators: provides pure-function JavaScript code validation utilities.
Includes syntax checking, TDZ detection, semantic validation, function reference
validation, etc.

refactored from the single-file code_validator.py into an internal layered package.
- _syntax.py: basic syntax utilities (comment/string stripping, bracket matching)
- _quality.py: code quality validation (TDZ, completeness, duplication, requirement verification, minimal change)
- _semantic.py: semantic validation (function name vs. implementation matching, delegate detection, control-primitive diagnostics)
- _references.py: reference and data validation (undefined functions, bot method misuse, error propagation, item names)
"""

from ._syntax import (
    _strip_comments_and_strings,
    find_bracket_mismatch_line,
)

from ._quality import (
    validate_tdz_issues,
    validate_code_completeness,
    fix_tdz_issues,
    detect_code_duplication,
    validate_minimal_change,
)

from ._semantic import (
    VALIDATION_WHITELIST,
    _detect_delegate_only,
    validate_function_implementation,
    validate_control_primitive_usage,
)

from ._references import (
    BUILTIN_FUNCTIONS,
    extract_function_call_names,
    _extract_function_calls,
    _extract_local_definitions,
    _find_similar_names,
    validate_function_references,
    validate_bot_method_calls,
    validate_data_only_object_calls,
    validate_error_propagation,
    validate_item_names,
)

__all__ = [
    # Syntax utilities
    "_strip_comments_and_strings",
    "find_bracket_mismatch_line",
    # Quality validators
    "validate_tdz_issues",
    "validate_code_completeness",
    "fix_tdz_issues",
    "detect_code_duplication",
    "validate_minimal_change",
    # Semantic validators
    "VALIDATION_WHITELIST",
    "_detect_delegate_only",
    "validate_function_implementation",
    "validate_control_primitive_usage",
    # Reference & data validators
    "BUILTIN_FUNCTIONS",
    "extract_function_call_names",
    "_extract_function_calls",
    "_extract_local_definitions",
    "_find_similar_names",
    "validate_function_references",
    "validate_bot_method_calls",
    "validate_data_only_object_calls",
    "validate_error_propagation",
    "validate_item_names",
]
