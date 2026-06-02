"""
Semantic Validators (Layer C)

Semantic validation: consistency between function name and implementation,
delegate detection, and control-primitive-usage diagnostics.

extracted from code_validator.py.
"""

import re
from typing import Dict, Any, Tuple, Optional


# ============================================================================
# Semantic validation: consistency between function name and implementation
# ============================================================================




# Whitelist: function-name patterns that skip the check
VALIDATION_WHITELIST = [
    r'^ensure\w*$',      # ensure* has different semantics; delegation is allowed
    r'^get\w*$',         # get* is a generic getter
    r'^setup\w*$',       # setup* may delegate to ensure + place
    r'^check\w*$',       # check* is a verifier
    r'^find\w*$',        # find* is a finder
    r'^count\w*$',       # count* is a counter
    r'^explore\w*$',     # explore* is an explorer
    r'^wait\w*$',        # wait* is a waiter
]


def _detect_delegate_only(code: str, function_name: str) -> Tuple[bool, Optional[str]]:
    """
    Detect whether a function is simply delegating to another function.

    Recognized patterns:
    - The entire function body has only one await call.
    - Or only simple parameter transformation followed by a call.
    - Or only setup/ensure calls with no actual operations.

    Args:
        code: function code
        function_name: function name

    Returns:
        Tuple[bool, Optional[str]]: (is_delegate_only, delegate_target_name)
    """
    # Strip comments
    code_clean = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)

    # Extract function body
    func_body_match = re.search(
        r'async\s+function\s+\w+\s*\([^)]*\)\s*\{([\s\S]*)\}',
        code_clean
    )
    if not func_body_match:
        return False, None

    func_body = func_body_match.group(1).strip()

    # Strip empty lines and lines that are only bot.chat
    lines = [
        line.strip() for line in func_body.split('\n')
        if line.strip()
        and not line.strip().startswith('//')
        and not re.match(r'^\s*await\s+bot\.chat\s*\(', line.strip())
        and not re.match(r'^\s*bot\.chat\s*\(', line.strip())
        and not re.match(r'^\s*return\s*;?\s*$', line.strip())
        and not re.match(r'^\s*\}\s*$', line.strip())
        and not re.match(r'^\s*try\s*\{', line.strip())
        and not re.match(r'^\s*\}\s*catch', line.strip())
        and not re.match(r'^\s*throw\s+', line.strip())
    ]

    # If only 0-2 meaningful lines remain, check whether this is pure delegation
    if len(lines) <= 2:
        # Check whether only one await function was called
        await_calls = re.findall(r'await\s+(\w+)\s*\(', '\n'.join(lines))

        if len(await_calls) == 1:
            delegate_target = await_calls[0]
            # If the delegate target performs the same kind of operation as
            # the current function, this is not pure delegation.
            # E.g. mineOakLogs -> mineBlock is reasonable, but
            # mineLogs -> ensureLogs is pure delegation.
            return True, delegate_target

    return False, None


def validate_function_implementation(
    function_name: str,
    code: str,
    strict: bool = False,
    operation_patterns: Dict = None,
) -> Dict[str, Any]:
    """
    Validate the semantic consistency between the function name and its implementation.

    Checks whether operations implied by the function name actually exist in
    the code. For example, craftWoodenPickaxe should contain real crafting
    logic.

    Args:
        function_name: function name
        code: function code
        strict: in strict mode, validation failure is marked invalid
        operation_patterns: domain-specific operation patterns. None = skip check.

    Returns:
        Dict containing:
        - valid: bool - whether validation passed (always True in non-strict mode)
        - semantic_match: bool - whether semantics match
        - detected_prefix: str - detected function prefix
        - has_required_operation: bool - whether required operations are present
        - delegate_only: bool - whether the function is pure delegation
        - delegate_target: str - delegate-target function name
        - warning: str - warning message
        - error: str - error message
    """
    result = {
        "valid": True,
        "semantic_match": True,
        "detected_prefix": None,
        "has_required_operation": False,
        "delegate_only": False,
        "delegate_target": None,
        "warning": None,
        "error": None,
    }

    # Check whether the name is in the whitelist
    for whitelist_pattern in VALIDATION_WHITELIST:
        if re.match(whitelist_pattern, function_name, re.IGNORECASE):
            result["has_required_operation"] = True  # Whitelisted functions skip the check
            return result

    # Use provided operation patterns or empty (skip check)
    patterns = operation_patterns if operation_patterns is not None else {}

    # Detect function prefix
    detected_prefix = None
    for prefix in patterns:
        if function_name.lower().startswith(prefix):
            detected_prefix = prefix
            break

    if not detected_prefix:
        # No matching prefix — skip the check
        return result

    result["detected_prefix"] = detected_prefix
    pattern_config = patterns[detected_prefix]

    # Check whether required operations are present
    has_required = False
    for pattern in pattern_config["required_patterns"]:
        if re.search(pattern, code):
            has_required = True
            break

    result["has_required_operation"] = has_required

    # Check whether this is pure delegation
    is_delegate, delegate_target = _detect_delegate_only(code, function_name)
    result["delegate_only"] = is_delegate
    result["delegate_target"] = delegate_target

    # Decide whether semantics match
    if has_required:
        # Required operation present — semantics match
        result["semantic_match"] = True
    elif is_delegate and delegate_target:
        # Check whether the delegate target is on the forbidden list
        forbidden_delegate = False
        for forbidden_pattern in pattern_config.get("forbidden_only_delegates", []):
            if re.match(forbidden_pattern, delegate_target, re.IGNORECASE):
                forbidden_delegate = True
                break

        if forbidden_delegate:
            # Pure delegation to a forbidden target
            result["semantic_match"] = False
            result["warning"] = (
                f"function '{function_name}' (prefix '{detected_prefix}') only delegates to '{delegate_target}', "
                f"but contains no actual {detected_prefix} logic. {pattern_config['description']}"
            )
            if strict:
                result["valid"] = False
                result["error"] = result["warning"]
        else:
            # Delegating to a same-category operation — semantics match.
            # E.g. mineOakLogs -> mineBlock.
            result["semantic_match"] = True
    else:
        # No required operation and not delegating to a same-category function
        result["semantic_match"] = False
        result["warning"] = (
            f"function '{function_name}' (prefix '{detected_prefix}') is missing required operations. "
            f"{pattern_config['description']}"
        )
        if strict:
            result["valid"] = False
            result["error"] = result["warning"]

    return result


def validate_control_primitive_usage(
    skill_name: str,
    code: str,
    api_mappings: list = None,
) -> Dict[str, Any]:
    """
    Detect whether the code uses raw APIs instead of higher-level control
    primitives (diagnostic suggestions only, not enforced).

    Args:
        skill_name: skill name (for logging)
        code: code to scan
        api_mappings: domain-specific API -> primitive mappings. None = no diagnostics.

    Returns:
        Dict containing suggestions, detected_raw_apis, recommended_apis.
    """
    suggestions = []

    if not api_mappings:
        return {"suggestions": [], "detected_raw_apis": [], "recommended_apis": []}

    detected_apis = []
    recommended_apis = []  # Record detected recommended APIs
    for mapping in api_mappings:
        matches = re.findall(mapping["pattern"], code)
        if matches:
            detected_apis.append(mapping["api"])
            # If it's a recommended API, record but don't emit a suggestion
            if mapping.get("is_recommended"):
                recommended_apis.append(mapping["api"])
            elif mapping.get("primitive"):
                # Only emit a suggestion for non-recommended APIs that have alternatives
                suggestions.append(
                    f"detected use of {mapping['api']}. If issues arise, consider using {mapping['primitive']}, "
                    f"which {mapping['benefit']}."
                )

    # If recipesFor is used but checkRecipe isn't, suggest checkRecipe
    if "bot.recipesFor()" in detected_apis and "bot.checkRecipe()" not in detected_apis:
        suggestions.append(
            "detected use of bot.recipesFor(). Recommend using bot.checkRecipe() instead — "
            "it distinguishes 'recipe does not exist' from 'insufficient materials' and provides more accurate error messages."
        )

    # Return suggestions instead of issues (diagnostic suggestions, not forced modifications)
    return {"suggestions": suggestions, "detected_raw_apis": detected_apis, "recommended_apis": recommended_apis}
