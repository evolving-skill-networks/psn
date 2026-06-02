"""
Planning Module for PSN

v5.1 architectural reorganization - modules with separated responsibilities extracted from planner.py.

Main components:
- EffectMatcher: effect matcher for matching task targets against Skill effects
- PreconditionChecker: precondition checker that validates and finds skills satisfying preconditions
- ParameterInferenceEngine: parameter inference engine (added in v5.1, replaces ParameterResolver)
- CodeGenerator: collection of pure functions for code generation

v5.1 changes:
- ParameterResolver has been moved to the inference module
- Pure functions moved to the inference.config_validator module
"""

from .effect_matcher import (
    EffectMatcher,
    # operation-type support
    SUPPORTED_OPERATIONS,
    OPERATION_ALIASES,
    normalize_operation,
)
from .precondition_checker import (
    PreconditionChecker,
    # Pure functions
    condition_matches,
    extract_generic_type,
    is_generic_item_type,
    extract_conditions_from_state_repr,
    check_tool_requirement,
    get_tool_tier,
)

# import from the new inference module (replaces parameter_resolver)
from .inference import (
    ParameterInferenceEngine,
    InferenceContext,
    InferenceResult,
    ParameterSemantic,
    QuantitySemantic,
    DirectionSemantic,
    TransformHint,
)
from .inference.config_validator import (
    # Pure functions
    extract_base_type,
    parse_function_parameters,
    is_destructured_parameter,
    item_matches_config_context,
    map_output_to_input_material,
    is_config_parameter,
    is_input_material_param,
    get_fuel_items_from_inventory,
    # Constants
    TRANSFORM_TO_PATTERN,
    CONFIG_INDICATORS,
)
from .code_generator import (
    # Pure functions - code processing
    strip_js_comments,
    validate_param_name,
    sanitize_python_to_js,
    # Pure functions - name normalization
    normalize_task_name,
    generate_function_name,
    generate_plan_description,
    # Helper functions - variable declaration extraction
    extract_main_function_body,
    is_module_level_declaration,
    extract_require_declarations,
    check_variable_usage,
    # Domain-aware accessors
    get_standard_declarations,
    get_global_deps_vars,
    # Constants
    STANDARD_VARIABLE_DECLARATIONS,
    REQUIRE_PATTERNS,
    GENERIC_REQUIRE_PATTERN,
    GLOBAL_DEPS_VARS,
)

__all__ = [
    "EffectMatcher",
    # operation-type support
    "SUPPORTED_OPERATIONS",
    "OPERATION_ALIASES",
    "normalize_operation",
    "PreconditionChecker",
    # new inference engine
    "ParameterInferenceEngine",
    "InferenceContext",
    "InferenceResult",
    "ParameterSemantic",
    "QuantitySemantic",
    "DirectionSemantic",
    "TransformHint",
    # precondition_checker pure functions
    "condition_matches",
    "extract_generic_type",
    "is_generic_item_type",
    "extract_conditions_from_state_repr",
    "check_tool_requirement",
    "get_tool_tier",
    # inference.config_validator pure functions (formerly parameter_resolver)
    "extract_base_type",
    "parse_function_parameters",
    "is_destructured_parameter",
    "item_matches_config_context",
    "map_output_to_input_material",
    "is_config_parameter",
    "is_input_material_param",
    "get_fuel_items_from_inventory",
    # inference.config_validator constants
    "TRANSFORM_TO_PATTERN",
    "CONFIG_INDICATORS",
    # code_generator pure functions
    "strip_js_comments",
    "validate_param_name",
    "sanitize_python_to_js",
    "normalize_task_name",
    "generate_function_name",
    "generate_plan_description",
    # code_generator helper functions
    "extract_main_function_body",
    "is_module_level_declaration",
    "extract_require_declarations",
    "check_variable_usage",
    # code_generator domain-aware accessors
    "get_standard_declarations",
    "get_global_deps_vars",
    # code_generator constants
    "STANDARD_VARIABLE_DECLARATIONS",
    "REQUIRE_PATTERNS",
    "GENERIC_REQUIRE_PATTERN",
    "GLOBAL_DEPS_VARS",
]
