"""
Skill Graph Utils Package

Utility function module: reusable pure functions for parameter parsing, code analysis, etc.

v4.0 modular refactor
"""

from .parameter_parser import (
    extract_destructured_params,
    infer_type_from_node,
    extract_value_from_node,
    infer_param_type_from_babel,
    extract_default_value_from_babel,
    extract_value_from_babel_node,
    generate_param_description,
)

from .code_verifier import (
    verify_precondition_in_code,
    code_has_precondition_check,
    code_has_effect_implementation,
    classify_effect_importance,
    validate_naming_effect_consistency,
)

from .code_analysis import (
    camel_to_snake,
    sanitize_python_to_js,
    extract_function_calls,
    extract_all_function_definitions,
    serialize_effects,
    filter_preconditions,
    is_task_specific_skill,
    is_general_skill,
)

from .execution_analysis import (
    calculate_correlation,
    categorize_failure,
    parse_js_value,
    parse_js_arguments,
)

# Code validation and name conflict detection (migrated from utils.py)
from .code_validation import (
    _strip_comments_and_strings,
    validate_code_brackets,
    validate_code_syntax,
    check_line_length,
    get_control_primitives,
    is_control_primitive_name,
    resolve_primitive_name_conflict,
    _find_validated_string_vars,
)

__all__ = [
    # Parameter parsing
    "extract_destructured_params",
    "infer_type_from_node",
    "extract_value_from_node",
    "infer_param_type_from_babel",
    "extract_default_value_from_babel",
    "extract_value_from_babel_node",
    "generate_param_description",
    # Code validation
    "verify_precondition_in_code",
    "code_has_precondition_check",
    "code_has_effect_implementation",
    "classify_effect_importance",
    "validate_naming_effect_consistency",
    # Code analysis
    "camel_to_snake",
    "sanitize_python_to_js",
    "extract_function_calls",
    "extract_all_function_definitions",
    "serialize_effects",
    "filter_preconditions",
    "is_task_specific_skill",
    "is_general_skill",
    # Execution analysis
    "calculate_correlation",
    "categorize_failure",
    "parse_js_value",
    "parse_js_arguments",
    # Code validation
    "_strip_comments_and_strings",
    "validate_code_brackets",
    "validate_code_syntax",
    "check_line_length",
    "get_control_primitives",
    "is_control_primitive_name",
    "resolve_primitive_name_conflict",
    "_find_validated_string_vars",
]
