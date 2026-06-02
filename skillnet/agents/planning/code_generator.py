"""
Code Generator Module

v5.1 architecture reorganization — extracted code-generation related
functionality from planner.py.

Responsibilities:
- JavaScript code generation and processing
- Function-name normalization
- Code-syntax conversion

Main components:
- strip_js_comments: remove JS comments
- validate_param_name: validate a JS parameter name
- sanitize_python_to_js: convert Python syntax to JS syntax
- normalize_task_name: convert a task name to a function name
- generate_function_name: generate a function name
- generate_plan_description: generate a plan description
"""

import re
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models.graph import SkillCallContext


# =============================================================================
# Pure functions - JavaScript code processing
# =============================================================================

def strip_js_comments(code: str) -> str:
    """
    Remove JavaScript comments while preserving string contents.

    Handles:
    - Single-line comments //
    - Multi-line comments /* */
    - // and /* inside strings are NOT removed

    Args:
        code: JavaScript code string

    Returns:
        Code with comments stripped.

    Examples:
        >>> strip_js_comments('const x = 1; // comment')
        'const x = 1; \\n'
        >>> strip_js_comments('const url = "http://example.com";')
        'const url = "http://example.com";'
    """
    result = []
    i = 0
    in_string = False
    string_char = None

    while i < len(code):
        # Handle string literals
        if code[i] in ['"', "'", '`'] and (i == 0 or code[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = code[i]
            elif code[i] == string_char:
                in_string = False
            result.append(code[i])
        elif in_string:
            result.append(code[i])
        # Strip single-line comments
        elif i < len(code) - 1 and code[i:i+2] == '//':
            # Skip to end of line
            while i < len(code) and code[i] != '\n':
                i += 1
            # Preserve the newline
            if i < len(code):
                result.append('\n')
            continue
        # Strip multi-line comments
        elif i < len(code) - 1 and code[i:i+2] == '/*':
            i += 2
            while i < len(code) - 1 and code[i:i+2] != '*/':
                i += 1
            i += 2  # Skip past */
            continue
        else:
            result.append(code[i])
        i += 1

    return ''.join(result)


def validate_param_name(param_name: str) -> Tuple[bool, str, str]:
    """
    Validate whether a parameter name is a valid JavaScript identifier.

    Args:
        param_name: parameter-name string

    Returns:
        (is_valid, cleaned_name, error_reason)
        - is_valid: whether it is valid
        - cleaned_name: cleaned parameter name (empty string when invalid)
        - error_reason: reason for invalidity (empty string when valid)

    Examples:
        >>> validate_param_name('count')
        (True, 'count', '')
        >>> validate_param_name('')
        (False, '', 'empty_param_name')
        >>> validate_param_name('my // comment')
        (False, '', 'contains_comment_syntax')
    """
    if not param_name:
        return False, "", "empty_param_name"

    # Check for comment syntax
    if '//' in param_name or '/*' in param_name:
        return False, "", "contains_comment_syntax"

    # Check for newline characters
    if '\n' in param_name or '\r' in param_name:
        return False, "", "contains_newline"

    cleaned = param_name.strip()

    # Check whether it's a valid JavaScript identifier
    js_identifier_pattern = r'^[a-zA-Z_$][a-zA-Z0-9_$]*$'
    if not re.match(js_identifier_pattern, cleaned):
        return False, cleaned, "invalid_identifier"

    return True, cleaned, ""


def sanitize_python_to_js(code: str, logger=None) -> str:
    """
    Convert Python syntax to JavaScript syntax.

    Fixes cases where an LLM may confuse Python and JavaScript syntax:
    - Python: False, True, None
    - JavaScript: false, true, null

    Args:
        code: code possibly containing Python syntax
        logger: optional logger

    Returns:
        str: converted JavaScript code.

    Examples:
        >>> sanitize_python_to_js('const x = True;')
        'const x = true;'
        >>> sanitize_python_to_js('const y = None;')
        'const y = null;'
    """
    original_code = code

    # Convert boolean/None values (use word boundaries so variable names like "isFalse" are unaffected)
    # False -> false
    code = re.sub(r'\bFalse\b', 'false', code)
    # True -> true
    code = re.sub(r'\bTrue\b', 'true', code)
    # None -> null
    code = re.sub(r'\bNone\b', 'null', code)

    # Log when a conversion occurred and a logger is provided
    if code != original_code and logger:
        logger.warning(f"\033[33m[CodeGenerator] detected Python syntax and converted to JavaScript:\033[0m")
        # Count conversions
        false_count = len(re.findall(r'\bFalse\b', original_code))
        true_count = len(re.findall(r'\bTrue\b', original_code))
        none_count = len(re.findall(r'\bNone\b', original_code))
        if false_count:
            logger.warning(f"\033[33m[CodeGenerator]   - False -> false: {false_count} occurrence(s)\033[0m")
        if true_count:
            logger.warning(f"\033[33m[CodeGenerator]   - True -> true: {true_count} occurrence(s)\033[0m")
        if none_count:
            logger.warning(f"\033[33m[CodeGenerator]   - None -> null: {none_count} occurrence(s)\033[0m")

    return code


# =============================================================================
# Pure functions - name normalization
# =============================================================================

def normalize_task_name(task: str) -> str:
    """
    Normalize a task name into a function name.

    Args:
        task: task description, e.g. "Mine 1 oak_log" or "Craft 4 oak planks"

    Returns:
        str: normalized function name, e.g. "mine_1_oak_log" or "craft_4_oak_planks"

    Examples:
        >>> normalize_task_name("Mine 1 oak_log")
        'mine_1_oak_log'
        >>> normalize_task_name("Craft 4 oak planks")
        'craft_4_oak_planks'
        >>> normalize_task_name("123 start")
        'task_123_start'
    """
    if not task:
        return ""

    # Lowercase
    normalized = task.lower().strip()

    # Remove special characters, keep only letters, digits, spaces and underscores
    normalized = re.sub(r'[^a-z0-9_\s]', '', normalized)

    # Replace runs of whitespace with a single underscore
    normalized = re.sub(r'\s+', '_', normalized)

    # Trim leading/trailing underscores
    normalized = normalized.strip('_')

    # Ensure function name starts with a letter (add prefix if not)
    if normalized and not normalized[0].isalpha():
        normalized = "task_" + normalized

    # Cap length (JavaScript function names should not be too long)
    if len(normalized) > 50:
        normalized = normalized[:50]

    # If empty after normalization, return empty string
    if not normalized:
        return ""

    return normalized


def generate_function_name(
    task: Optional[str],
    skill_sequence: List["SkillCallContext"]
) -> str:
    """
    Generate a function name.

    Prefers the task name; falls back to the skill sequence when unavailable.

    Args:
        task: task description
        skill_sequence: skill execution sequence (list of SkillCallContext)

    Returns:
        str: function name.

    Examples:
        >>> # Assuming skill_sequence = [SkillCallContext(skill_name='mineOak'), ...]
        >>> generate_function_name("Mine 1 oak_log", [])
        'mine_1_oak_log'
    """
    if task:
        # Extract function name from task
        # e.g. "Mine 1 oak_log" -> "mine_1_oak_log"
        func_name = normalize_task_name(task)
        if func_name:
            return func_name

    # Fall back to using the skill sequence
    skill_names = [ctx.skill_name for ctx in skill_sequence]
    if skill_names:
        return f"execute_{'_'.join(skill_names)}"

    return "execute_task"


def generate_plan_description(
    skill_sequence: List["SkillCallContext"],
    target_effects: List[Dict[str, Any]]
) -> str:
    """
    Generate a plan description.

    Args:
        skill_sequence: skill execution sequence
        target_effects: list of target effects

    Returns:
        str: plan description text.

    Examples:
        >>> generate_plan_description([], [{"item": "oak_log", "count": 1}])
        'Graph-based planning:\\nTarget: [{\\'item\\': \\'oak_log\\', \\'count\\': 1}]\\nSkill sequence: '
    """
    plan_parts = []
    plan_parts.append("Graph-based planning:")
    plan_parts.append(f"Target: {target_effects}")
    skill_names = [ctx.skill_name for ctx in skill_sequence]
    plan_parts.append(f"Skill sequence: {' -> '.join(skill_names)}")
    return "\n".join(plan_parts)


# =============================================================================
# Helper functions - JavaScript variable-declaration extraction
# =============================================================================

def extract_main_function_body(code: str) -> str:
    """
    Extract the main function body's code, ignoring module-level code.

    Args:
        code: full JavaScript code

    Returns:
        str: the main function body's code, or the entire code if no main
        function is found.
    """
    # Find the main function definition (async function xxx(bot) { ... })
    main_func_pattern = r'(async\s+function\s+\w+\s*\([^)]*\)\s*\{)'
    match = re.search(main_func_pattern, code)
    if not match:
        return code  # No main function found — return the whole code

    func_start = match.end()
    # Find the end of the main function (match braces)
    brace_count = 1
    func_end = func_start
    for i in range(func_start, len(code)):
        if code[i] == '{':
            brace_count += 1
        elif code[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                func_end = i
                break

    return code[func_start:func_end]


def is_module_level_declaration(code: str, var_pattern: str) -> bool:
    """
    Check whether a variable declaration is at module level (outside any function).

    Args:
        code: JavaScript code
        var_pattern: regex pattern for the variable declaration

    Returns:
        bool: whether the declaration is module-level.
    """
    match = re.search(var_pattern, code, re.MULTILINE)
    if not match:
        return False

    # Get the position of the declaration
    decl_pos = match.start()

    # Check whether any function definition begins before the declaration.
    # If no function definition precedes it, it is module-level.
    func_pattern = r'(async\s+)?function\s+\w+\s*\([^)]*\)\s*\{'
    func_matches = list(re.finditer(func_pattern, code[:decl_pos]))

    if not func_matches:
        return True  # No function definition before this — module-level

    # Check whether the declaration is inside the last preceding function.
    # Done via brace counting.
    last_func = func_matches[-1]
    func_start = last_func.end()

    if decl_pos < func_start:
        return True  # Declaration is before the function start

    # Compute brace balance up to the declaration position
    brace_count = 1  # The function's opening {
    for i in range(func_start, min(decl_pos, len(code))):
        if code[i] == '{':
            brace_count += 1
        elif code[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                # Function has ended — if the declaration is after this, it's module-level
                return decl_pos > i

    return False  # Declaration is inside the function


# Standard variable-declaration mapping (populated by domain knowledge when available)
STANDARD_VARIABLE_DECLARATIONS = {}

# Domain knowledge injection for require patterns
from skillnet.core.dk_registry import get_domain_knowledge


# require declaration patterns (populated by domain knowledge when available)
_FALLBACK_REQUIRE_PATTERNS = []

# Backward-compat alias
REQUIRE_PATTERNS = _FALLBACK_REQUIRE_PATTERNS

# Generic require pattern
GENERIC_REQUIRE_PATTERN = r'const\s+(\w+)\s*=\s*require\([\'"]([^\'"]+)[\'"]\)[^;]*;'


def get_standard_declarations(domain_knowledge=None) -> Dict[str, str]:
    """Get variable declarations from domain knowledge.

    Returns a mapping of variable name → declaration statement.
    Returns empty dict when no domain is available.
    """
    if domain_knowledge:
        deps = domain_knowledge.get_global_dependencies()
        if deps:
            return deps
    return {}


def get_global_deps_vars(domain_knowledge=None) -> set:
    """Get set of global dependency variable names from domain knowledge.

    These are variables already declared by the environment runtime
    (e.g. via globalDepsCode) and should NOT be re-declared in
    composed wrapper code. Returns empty set when no domain is available.
    """
    if domain_knowledge:
        deps = domain_knowledge.get_global_dependencies()
        if deps:
            return set(deps.keys())
    return set()


def extract_require_declarations(
    skill_code: str,
    seen_variable_names: set,
    module_level_vars: set,
    logger=None
) -> List[str]:
    """
    Extract require declarations from skill code.

    Args:
        skill_code: skill's JavaScript code
        seen_variable_names: set of variable names already seen (will be modified)
        module_level_vars: set of module-level variables (will be modified)
        logger: optional logger

    Returns:
        List[str]: list of variable declarations.
    """
    declared_variables = []
    seen_declarations = set()

    # Get patterns from domain knowledge or fallback
    dk = get_domain_knowledge()
    if dk:
        patterns = dk.get_require_patterns()
        if not patterns:
            patterns = _FALLBACK_REQUIRE_PATTERNS
    else:
        patterns = _FALLBACK_REQUIRE_PATTERNS

    # First match specific common patterns
    for pattern, var_name, default_decl in patterns:
        if re.search(pattern, skill_code, re.MULTILINE):
            # Check whether it's a module-level declaration
            if is_module_level_declaration(skill_code, pattern):
                # Module-level declaration — record but do NOT add to declared_variables
                module_level_vars.add(var_name)
                if logger:
                    logger.debug(f"\033[33m[CodeGenerator] skipping module-level declaration: {var_name} (avoid TDZ issues)\033[0m")
                continue

            # Dedupe by variable name, not by declaration string
            if var_name not in seen_variable_names and var_name not in module_level_vars:
                seen_variable_names.add(var_name)
                seen_declarations.add(default_decl)
                declared_variables.append(default_decl)

    # Then match the generic pattern
    matches = re.finditer(GENERIC_REQUIRE_PATTERN, skill_code, re.MULTILINE)
    for match in matches:
        var_name = match.group(1)
        var_decl = match.group(0).strip()

        # Skip variables already matched by specific patterns
        if var_name in seen_variable_names or var_name in module_level_vars:
            continue

        # Check whether it's a module-level declaration
        if is_module_level_declaration(skill_code, re.escape(var_decl)):
            module_level_vars.add(var_name)
            if logger:
                logger.debug(f"\033[33m[CodeGenerator] skipping module-level declaration: {var_name} (avoid TDZ issues)\033[0m")
            continue

        # Normalize the declaration (collapse whitespace)
        var_decl_normalized = re.sub(r'\s+', ' ', var_decl)

        # Skip declarations we've already seen
        if var_decl_normalized not in seen_declarations:
            seen_variable_names.add(var_name)
            seen_declarations.add(var_decl_normalized)
            declared_variables.append(var_decl_normalized)

    return declared_variables


def check_variable_usage(skill_code: str, domain_knowledge=None) -> Dict[str, bool]:
    """
    Check whether the code uses specific variables.

    When *domain_knowledge* is provided, checks all variables from
    ``get_global_dependencies()`` instead of only the hardcoded Minecraft
    defaults.

    Args:
        skill_code: skill's JavaScript code
        domain_knowledge: Optional DomainKnowledge instance

    Returns:
        Dict[str, bool]: mapping of variable name to whether it is used.
    """
    declarations = get_standard_declarations(domain_knowledge)
    skill_code_lower = skill_code.lower()
    result = {}
    for var_name in declarations:
        result[var_name] = var_name.lower() in skill_code_lower
    return result


# =============================================================================
# Constants - global dependency variables
# =============================================================================

# Variables already declared in globalDepsCode — must not be redeclared in the wrapper
# (populated by domain knowledge when available)
GLOBAL_DEPS_VARS = set()


__all__ = [
    # Pure functions - code processing
    "strip_js_comments",
    "validate_param_name",
    "sanitize_python_to_js",
    # Pure functions - name normalization
    "normalize_task_name",
    "generate_function_name",
    "generate_plan_description",
    # Helper functions - variable-declaration extraction
    "extract_main_function_body",
    "is_module_level_declaration",
    "extract_require_declarations",
    "check_variable_usage",
    # Domain-aware accessors
    "get_standard_declarations",
    "get_global_deps_vars",
    "get_domain_knowledge",
    # Constants
    "STANDARD_VARIABLE_DECLARATIONS",
    "REQUIRE_PATTERNS",
    "_FALLBACK_REQUIRE_PATTERNS",
    "GENERIC_REQUIRE_PATTERN",
    "GLOBAL_DEPS_VARS",
]
