"""
Code Validation Utilities

Code validation and name-conflict detection utility functions.

Migrated from skill_graph/utils.py.
"""

import re
from typing import Dict, List, Tuple, Any, Optional

from skillnet.core.dk_registry import get_domain_knowledge


def _strip_comments_and_strings(code: str) -> str:
    """
    Strip comments and strings from code, keeping only actual code.

    This allows accurate brace counting without interference from braces inside comments or strings.

    Args:
        code: JavaScript code string

    Returns:
        str: code after removing comments and strings
    """
    result = []
    i = 0
    in_string = None  # None, '"', "'", '`'

    while i < len(code):
        # Check string start/end
        if in_string is None:
            # Check multi-line comment /* ... */
            if code[i:i+2] == '/*':
                # Find comment end
                end = code.find('*/', i + 2)
                if end == -1:
                    break  # Unclosed comment; skip the rest
                i = end + 2
                continue

            # Check single-line comment // ...
            if code[i:i+2] == '//':
                # Find end of line
                end = code.find('\n', i + 2)
                if end == -1:
                    break  # Last line
                i = end + 1
                result.append('\n')  # Preserve newline
                continue

            # Check string start
            if code[i] in ('"', "'", '`'):
                in_string = code[i]
                i += 1
                continue

            # Normal code character
            result.append(code[i])
            i += 1
        else:
            # Inside a string
            # Check for escape characters
            if code[i] == '\\' and i + 1 < len(code):
                i += 2  # Skip escape char and the escaped char
                continue

            # Check for string end
            if code[i] == in_string:
                in_string = None
            i += 1

    return ''.join(result)


def validate_code_brackets(code: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate whether brackets in the code are balanced.

    Call this before saving code to prevent syntactically invalid code from
    being saved. Brackets inside comments and strings are ignored automatically.

    Args:
        code: JavaScript code string

    Returns:
        Tuple[bool, str, Dict]:
            - is_valid: whether brackets are balanced
            - error_message: error message (if any)
            - details: detailed bracket counts
    """
    # Remove comments and strings; keep only real code
    stripped_code = _strip_comments_and_strings(code)

    # Count brackets (on the comment/string-stripped code)
    open_parens = stripped_code.count('(')
    close_parens = stripped_code.count(')')
    open_brackets = stripped_code.count('[')
    close_brackets = stripped_code.count(']')
    open_braces = stripped_code.count('{')
    close_braces = stripped_code.count('}')

    details = {
        "parentheses": {"open": open_parens, "close": close_parens, "diff": open_parens - close_parens},
        "brackets": {"open": open_brackets, "close": close_brackets, "diff": open_brackets - close_brackets},
        "braces": {"open": open_braces, "close": close_braces, "diff": open_braces - close_braces},
    }

    errors = []

    if open_parens != close_parens:
        diff = open_parens - close_parens
        if diff > 0:
            errors.append(f"missing {diff} closing parenthesis/parentheses ')'")
        else:
            errors.append(f"extra {-diff} closing parenthesis/parentheses ')'")

    if open_brackets != close_brackets:
        diff = open_brackets - close_brackets
        if diff > 0:
            errors.append(f"missing {diff} closing bracket(s) ']'")
        else:
            errors.append(f"extra {-diff} closing bracket(s) ']'")

    if open_braces != close_braces:
        diff = open_braces - close_braces
        if diff > 0:
            errors.append(f"missing {diff} closing brace(s) '}}'")
        else:
            errors.append(f"extra {-diff} closing brace(s) '}}'")

    if errors:
        return False, "; ".join(errors), details

    return True, "", details


def _basic_syntax_check(code: str) -> Tuple[bool, str]:
    """Basic syntax check (fallback when Babel is unavailable)."""
    # Reuse validate_code_brackets logic
    is_valid, error_msg, _ = validate_code_brackets(code)
    return is_valid, error_msg


def validate_code_syntax(code: str) -> Tuple[bool, str]:
    """
    Validate skill code syntax via the active DomainKnowledge's SkillLanguage.

    Args:
        code: skill code (language depends on the registered DomainKnowledge)

    Returns:
        Tuple[bool, str]: (valid?, error message).
    """
    if not code or not code.strip():
        return False, "Empty code"

    dk = get_domain_knowledge()
    if dk is None:
        raise RuntimeError("No DomainKnowledge registered; cannot parse skill code")
    lang = dk.get_skill_language_impl()

    try:
        result = lang.validate_syntax(code)
        if result.valid:
            return True, ""
        return False, result.errors[0] if result.errors else ""
    except Exception as e:
        # On unexpected exceptions (e.g. language backend load failure), fall
        # back to a basic bracket check so callers still get a useful signal.
        error_str = str(e)
        print(f"\033[33m[Syntax Check] Language backend failed, using basic validation: {error_str[:100]}\033[0m")
        return _basic_syntax_check(code)


def check_line_length(code: str, max_length: int = 200) -> List[str]:
    """
    Check whether the code has overly long lines.

    Args:
        code: JavaScript code
        max_length: maximum line length (default 200)

    Returns:
        List[str]: list of warning messages for long lines.
    """
    warnings = []
    for i, line in enumerate(code.split('\n'), 1):
        if len(line) > max_length:
            warnings.append(f"Line {i}: {len(line)} chars (max {max_length})")
    return warnings


# ========== Control-primitive name-conflict detection ==========
# List of control-primitive names (module-level constant, lazily loaded)
_CONTROL_PRIMITIVES: Optional[List[str]] = None


def get_control_primitives() -> List[str]:
    """Return the runtime primitive names (lazily loaded).

    Used to prevent LLM-generated skill code from shadowing names already
    defined in the domain's runtime action layer (including helper functions).
    """
    global _CONTROL_PRIMITIVES
    if _CONTROL_PRIMITIVES is None:
        _dk = get_domain_knowledge()
        _CONTROL_PRIMITIVES = _dk.get_runtime_primitive_names() if _dk else []
    return _CONTROL_PRIMITIVES


def is_control_primitive_name(name: str) -> bool:
    """Check whether the name conflicts with a control primitive."""
    return name in get_control_primitives()


def resolve_primitive_name_conflict(name: str, code: str) -> Tuple[str, str, bool]:
    """
    Resolve a name conflict with a control primitive.

    Args:
        name: original skill name
        code: skill code

    Returns:
        (new_name, new_code, was_renamed)
    """
    if not is_control_primitive_name(name):
        return name, code, False

    # Auto-rename strategy: append "Skill" suffix
    new_name = f"{name}Skill"

    # If the renamed version still conflicts, keep adding suffixes
    suffix_count = 2
    while is_control_primitive_name(new_name) or new_name == name:
        new_name = f"{name}Skill{suffix_count}"
        suffix_count += 1

    # Update the function name in the code
    # Use a regex to replace the function definition
    new_code = re.sub(
        rf'\basync\s+function\s+{re.escape(name)}\s*\(',
        f'async function {new_name}(',
        code
    )

    print(f"\033[33m[Name Conflict] Skill '{name}' conflicts with control primitive, renamed to '{new_name}'\033[0m")

    return new_name, new_code, True


def _find_validated_string_vars(code: str) -> set:
    """
    Find variables that have been validated as strings via typeof checks.

    Patterns recognized:
    - typeof varName === 'string'
    - typeof varName !== 'string' (followed by throw/error handling)
    - // @contract-safe: varName
    - varName = obj.name (string extraction from object)
    """
    validated = set()

    # Pattern 1: typeof x === 'string' or typeof x == 'string'
    for m in re.finditer(r"typeof\s+(\w+)\s*[!=]==?\s*['\"]string['\"]", code):
        validated.add(m.group(1))

    # Pattern 2: // @contract-safe: varName1, varName2, ...
    for m in re.finditer(r"//\s*@contract-safe:\s*([\w\s,]+)", code):
        vars_str = m.group(1)
        for var in vars_str.split(","):
            var = var.strip()
            if var and re.match(r"^\w+$", var):
                validated.add(var)

    # Pattern 3: Variable assigned from .name property (e.g., blockNameStr = nearest.name)
    # These are typically string extractions from block objects
    for m in re.finditer(r"(\w+)\s*=\s*\w+\.name\b", code):
        validated.add(m.group(1))

    return validated


