"""
Reference & Data Validators (Layer D)

Function reference validation: detect undefined function calls, misuse of bot
methods, error-propagation detection, and item-name validation.

extracted from code_validator.py.

Dependencies:
- _syntax._strip_comments_and_strings
"""

import re
from typing import Dict, List, Any, Tuple

from ._syntax import _strip_comments_and_strings


# Domain knowledge via central registry
from skillnet.core.dk_registry import get_domain_knowledge


# ============================================================================
# Function reference validation: detect undefined functions called in code
# ============================================================================

# JavaScript builtins — always valid regardless of domain
BUILTIN_FUNCTIONS = {
    # JavaScript global objects and constructors
    "console", "Math", "JSON", "Date", "Array", "Object", "String", "Number",
    "Boolean", "RegExp", "Error", "TypeError", "RangeError", "SyntaxError",
    "Promise", "Map", "Set", "WeakMap", "WeakSet", "Symbol", "Proxy", "Reflect",
    "parseInt", "parseFloat", "isNaN", "isFinite", "encodeURI", "decodeURI",
    "encodeURIComponent", "decodeURIComponent",
    # Timers
    "setTimeout", "setInterval", "clearTimeout", "clearInterval", "setImmediate",
    # Common Node.js
    "require", "Buffer", "process",
}


def _get_domain_globals() -> set:
    """Return domain-specific global variable names."""
    dk = get_domain_knowledge()
    if dk:
        deps = dk.get_global_dependencies()
        env_globals = dk.get_environment_globals()
        return set(deps.keys()) | env_globals if deps else env_globals
    return set()





def extract_function_call_names(code: str) -> set:
    """Extract the set of function names called in JavaScript code (public API).

    Public wrapper for cycle detection in optimizer.
    """
    return {name for name, _, _ in _extract_function_calls(code)}


def _extract_function_calls(code: str) -> List[Tuple[str, int, str]]:
    """
    Extract all function calls from code.

    Args:
        code: JavaScript code

    Returns:
        List of (function_name, line_number, call_context)
    """
    # First strip comments and strings
    clean_code = _strip_comments_and_strings(code)

    calls = []
    lines = clean_code.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Match the `await xxx(` form
        await_matches = re.finditer(r'\bawait\s+(\w+)\s*\(', line)
        for match in await_matches:
            func_name = match.group(1)
            calls.append((func_name, line_num, f"await {func_name}("))

        # Match ordinary function calls of the form `xxx(` (excluding already-matched await calls)
        # Exclude `bot.xxx(` (those are bot method calls)
        # Exclude `function xxx(` (that is a function definition)
        # Exclude `async function xxx(`
        # Exclude `new Xxx(` (constructor calls)
        func_matches = re.finditer(r'(?<!\.)\b([a-z_][a-zA-Z0-9_]*)\s*\(', line)
        for match in func_matches:
            func_name = match.group(1)
            # Check whether the preceding token is await, function, async function, or new
            start = match.start()
            prefix = line[:start].rstrip()
            if prefix.endswith('await'):
                continue  # Already handled in the await match
            if prefix.endswith('function'):
                continue  # This is a function definition
            if re.search(r'\bfunction\s*$', prefix):
                continue  # This is a function definition
            if prefix.endswith('new'):
                continue  # Constructor call

            calls.append((func_name, line_num, f"{func_name}("))

    return calls


def _add_param_names(target_set: set, params_str: str):
    """Extract parameter names from a comma-separated parameter list."""
    for param in params_str.split(','):
        name = param.strip()
        if name and name.isidentifier():
            target_set.add(name)


def _extract_local_definitions(code: str) -> set:
    """
    Extract local function definitions and function parameter names from the code.

    Args:
        code: JavaScript code

    Returns:
        Set of locally defined function/parameter names
    """
    local_functions = set()

    # Match function xxx(
    func_defs = re.finditer(r'\bfunction\s+(\w+)\s*\(', code)
    for match in func_defs:
        local_functions.add(match.group(1))

    # Match async function xxx(
    async_func_defs = re.finditer(r'\basync\s+function\s+(\w+)\s*\(', code)
    for match in async_func_defs:
        local_functions.add(match.group(1))

    # Match const/let/var xxx = function(
    const_func_defs = re.finditer(r'\b(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function\s*\(', code)
    for match in const_func_defs:
        local_functions.add(match.group(1))

    # Match const/let/var xxx = async (  or const/let/var xxx = (  (arrow functions)
    arrow_func_defs = re.finditer(r'\b(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>', code)
    for match in arrow_func_defs:
        local_functions.add(match.group(1))

    # Extract function parameter names (callback/predicate parameters may be invoked as functions)
    # Match function name(param1, param2) and async function name(...)
    for match in re.finditer(r'\bfunction\s+\w+\s*\(([^)]*)\)', code):
        _add_param_names(local_functions, match.group(1))

    # Match const/let/var name = function(param1, ...)
    for match in re.finditer(r'\b(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?function\s*\(([^)]*)\)', code):
        _add_param_names(local_functions, match.group(1))

    # Match arrow-function parameters (param1, param2) =>
    for match in re.finditer(r'\(([^)]*)\)\s*=>', code):
        _add_param_names(local_functions, match.group(1))

    return local_functions


def _find_similar_names(name: str, candidates: set, max_suggestions: int = 3) -> List[str]:
    """
    Find similar names using edit distance.

    Args:
        name: Target name to find similar names for
        candidates: Set of candidate names
        max_suggestions: Maximum number of suggestions

    Returns:
        List of similar names sorted by similarity
    """
    import difflib

    # Use difflib's SequenceMatcher to compute similarity
    similarities = []
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
        if ratio > 0.5:  # Only consider similarities above 50%
            similarities.append((candidate, ratio))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    return [s[0] for s in similarities[:max_suggestions]]


def validate_function_references(
    code: str,
    available_skills: set = None,
    known_primitives: set = None,
    strict_mode: bool = False,
    domain_functions: Dict[str, set] = None,
) -> Dict[str, Any]:
    """
    Validate that function calls in the code refer to existing functions.

    Used to detect non-existent function calls hallucinated by the LLM, such as
    initBotHelpers.

    Args:
        code: JavaScript code
        available_skills: Set of available skill names (from skill_graph)
        known_primitives: Set of known primitives (defaults to KNOWN_PRIMITIVES)
        strict_mode: Strict mode; return invalid as soon as an unknown function is found
        domain_functions: Domain function set from DomainKnowledge.get_known_functions().
            Keys: "bot_methods", "primitives", "helpers".
            Replaces the hard-coded Minecraft constants when provided.

    Returns:
        {
            "valid": bool,
            "undefined_functions": [{"name": str, "line": int, "context": str}],
            "warnings": [str],
            "suggestions": {"func_name": ["similar1", "similar2"]}
        }

    Example:
        >>> result = validate_function_references(
        ...     code="await initBotHelpers(bot);",
        ...     available_skills={"ensureFuel", "craftItem"}
        ... )
        >>> result["valid"]
        False
        >>> result["undefined_functions"]
        [{"name": "initBotHelpers", "line": 1, "context": "await initBotHelpers("}]
        >>> result["suggestions"]
        {"initBotHelpers": ["ensureFuel"]}
    """
    if available_skills is None:
        available_skills = set()

    # Resolve function sets from domain knowledge (empty defaults if not provided)
    if domain_functions:
        bot_methods = domain_functions.get("bot_methods", set())
        primitives = domain_functions.get("primitives", set())
        helpers = domain_functions.get("helpers", set())
    else:
        bot_methods = set()
        primitives = set()
        helpers = set()

    if known_primitives is None:
        known_primitives = primitives.copy()
    else:
        known_primitives = known_primitives | primitives

    # Extract local function definitions from the code
    local_definitions = _extract_local_definitions(code)

    # Build the set of all known functions
    common_helpers = domain_functions.get("common_helpers", set()) if domain_functions else set()
    all_known = (
        BUILTIN_FUNCTIONS |
        _get_domain_globals() |
        bot_methods |
        known_primitives |
        helpers |
        common_helpers |
        available_skills |
        local_definitions
    )

    # Extract all function calls
    function_calls = _extract_function_calls(code)

    undefined_functions = []
    warnings = []
    suggestions = {}

    # Check each function call
    seen_undefined = set()  # Avoid reporting the same function twice
    for func_name, line_num, context in function_calls:
        # Skip known functions
        if func_name in all_known:
            continue

        # Skip JavaScript keywords and common patterns
        if func_name in {'if', 'for', 'while', 'switch', 'catch', 'with', 'return', 'throw', 'typeof', 'instanceof', 'function', 'async', 'await', 'let', 'const', 'var', 'new', 'delete', 'void', 'yield', 'class', 'import', 'export', 'from', 'of', 'in'}:
            continue

        # Avoid duplicate reports
        if func_name in seen_undefined:
            continue
        seen_undefined.add(func_name)

        # Record the undefined function
        undefined_functions.append({
            "name": func_name,
            "line": line_num,
            "context": context,
        })

        # Try to find similar function names as suggestions
        all_candidates = available_skills | known_primitives | common_helpers
        similar = _find_similar_names(func_name, all_candidates)
        if similar:
            suggestions[func_name] = similar

    # Generate warnings
    if undefined_functions:
        for undef in undefined_functions:
            func_name = undef["name"]
            line = undef["line"]
            sugg = suggestions.get(func_name, [])
            if sugg:
                warnings.append(
                    f"Line {line}: function '{func_name}' is undefined (suggestions: {', '.join(sugg)})"
                )
            else:
                warnings.append(
                    f"Line {line}: function '{func_name}' is undefined"
                )

    # Determine validity
    valid = len(undefined_functions) == 0 or not strict_mode

    return {
        "valid": valid,
        "undefined_functions": undefined_functions,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def validate_bot_method_calls(
    code: str,
    known_primitives: set = None,
) -> List[Dict[str, Any]]:
    """
    Check whether any control primitives are misused as bot methods in bot.xxx() calls.

    The LLM easily confuses control primitives with bot methods, e.g. generating
    bot.placeItem() rather than the correct placeItem(bot, ...). These calls
    escape validate_function_references because _extract_function_calls
    explicitly excludes the .xxx() form.

    Args:
        code: JavaScript code
        known_primitives: Set of known primitives. When None, uses the Minecraft default KNOWN_PRIMITIVES.

    Returns:
        List of error info dicts, each containing method, line, message.
    """
    primitives = known_primitives if known_primitives is not None else set()
    errors = []
    clean_code = _strip_comments_and_strings(code)

    for match in re.finditer(r'\bbot\.(\w+)\s*\(', clean_code):
        method = match.group(1)
        if method in primitives:
            line_num = clean_code[:match.start()].count('\n') + 1
            errors.append({
                "method": method,
                "line": line_num,
                "message": (
                    f"bot.{method}() is NOT a valid API method. "
                    f"'{method}' is a standalone control primitive — "
                    f"use `await {method}(bot, ...)` instead."
                ),
            })
    return errors


def validate_data_only_object_calls(
    code: str,
    data_only_objects,
) -> List[Dict[str, Any]]:
    """Flag any ``obj.method(...)`` call where ``obj`` is declared data-only.

    Some bindings exposed to skill code carry only data attributes, no
    methods. When an LLM writes ``obj.something()`` against such a binding,
    the syntax is locally valid but the call resolves to ``undefined`` at
    runtime and throws ``X is not a function``. The pattern is invisible
    to ``validate_function_references`` because its extractor regex
    explicitly skips ``obj.x(`` forms.

    Args:
        code: skill source.
        data_only_objects: iterable of object identifiers (e.g. ``{"mcData"}``)
            whose ``.X(...)`` calls should always be rejected.

    Returns:
        List of ``{object, method, line, message}`` dicts; empty when no
        such call is present.
    """
    objects = set(data_only_objects or ())
    if not objects:
        return []

    clean_code = _strip_comments_and_strings(code)
    errors: List[Dict[str, Any]] = []
    for obj_name in objects:
        pattern = rf'\b{re.escape(obj_name)}\.(\w+)\s*\('
        for match in re.finditer(pattern, clean_code):
            method = match.group(1)
            line_num = clean_code[:match.start()].count('\n') + 1
            errors.append({
                "object": obj_name,
                "method": method,
                "line": line_num,
                "message": (
                    f"{obj_name}.{method}() — {obj_name} carries only data "
                    f"attributes, no methods. Calling .{method}() resolves "
                    f"to undefined at runtime."
                ),
            })
    return errors


# ============================================================================
# Error Swallowing Detection
# ============================================================================

def validate_error_propagation(code: str) -> Tuple[bool, str]:
    """
    Detect whether catch blocks swallow errors without rethrowing.

    Common problematic pattern: catch (e) { bot.chat(e.message); }
    The execution framework treats the skill as successful (no exception thrown),
    but the actual operation failed.

    Args:
        code: JavaScript code

    Returns:
        Tuple[bool, str]: (is_valid, warning_message)
        - is_valid=False indicates error swallowing was detected
    """
    # Match catch blocks — uses simplified nested-brace matching
    # Supports one level of nested {} inside the catch block
    catch_blocks = re.findall(
        r'catch\s*\([^)]*\)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}',
        code, re.DOTALL
    )
    for block in catch_blocks:
        has_chat = 'bot.chat' in block
        has_throw = 'throw' in block
        if has_chat and not has_throw:
            return False, (
                "Error swallowing detected: catch block calls bot.chat() "
                "but does not rethrow the error. Add 'throw' to propagate failures."
            )
    return True, ""


# ============================================================================
# Item Name Validation
# ============================================================================

def validate_item_names(
    code: str,
    invalid_names: Dict[str, str] = None,
    ambiguous_names: set = None,
    primitives: list = None,
) -> Tuple[bool, List[str]]:
    """
    Check whether item names used in control-primitive calls are valid.

    Detects two classes of problems:
    1. KNOWN_INVALID: explicit misspellings (e.g. "sticks" → "stick"), rejected directly
    2. AMBIGUOUS: ambiguous item names (e.g. "planks" needs a prefix), marked as errors

    Args:
        code: JavaScript code
        invalid_names: Map of invalid name → correct name. Skip the check when None.
        ambiguous_names: Set of ambiguous item names. Empty set when None.
        primitives: List of control-primitive names (used to build the matching pattern). Skip the check when None.

    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
        - is_valid=False indicates invalid item names were detected
    """
    inv_names = invalid_names if invalid_names is not None else {}
    amb_names = ambiguous_names if ambiguous_names is not None else set()

    if not primitives:
        return True, []

    errors = []
    # Match item name in control primitive calls: primitive(entry_param, "xxx", ...)
    prims_alt = '|'.join(re.escape(p) for p in primitives)
    pattern = rf'(?:{prims_alt})\s*\(\s*\w+\s*,\s*["\']([^"\']+)["\']'
    for match in re.finditer(pattern, code):
        item_name = match.group(1)
        if item_name in inv_names:
            correct = inv_names[item_name]
            errors.append(f'Wrong item name "{item_name}" → should be "{correct}"')
        elif item_name in amb_names:
            errors.append(
                f'Ambiguous item name "{item_name}" — '
                f'requires specific type variant'
            )
    return len(errors) == 0, errors
