"""
Code Quality Validators (Layer B)

Code quality validation: TDZ detection and fix, code completeness, duplication detection,
requirements validation, minimal-change constraint.

extracted from code_validator.py.

Dependencies:
- _syntax._strip_comments_and_strings
- _syntax.find_bracket_mismatch_line
"""

import re
from typing import Dict, List, Any, Tuple

from ._syntax import _strip_comments_and_strings, find_bracket_mismatch_line


# Domain knowledge via central registry
from skillnet.core.dk_registry import get_domain_knowledge


def _get_require_info():
    """Return (decl_patterns, var_names) from domain knowledge require patterns."""
    dk = get_domain_knowledge()
    if not dk:
        return [], set()
    require_patterns = dk.get_require_patterns()
    if not require_patterns:
        return [], set()
    decl_patterns = [rp[0] for rp in require_patterns]
    var_names = {rp[1] for rp in require_patterns}
    return decl_patterns, var_names


def validate_tdz_issues(code: str) -> Dict[str, Any]:
    """
    Detect Temporal Dead Zone (TDZ) issues in code

    TDZ issues typically occur when:
    1. A helper function is defined before a require declaration but references one of that declaration's variables
    2. const/let declarations inside a function are referenced by other code before declaration

    Args:
        code: code to validate

    Returns:
        dict: {
            "valid": bool,
            "error": str or None,
            "fix_suggestion": str or None,
            "tdz_details": dict or None
        }
    """
    decl_patterns, var_names = _get_require_info()
    if not decl_patterns:
        return {"valid": True, "error": None, "fix_suggestion": None, "tdz_details": None}

    # Remove comments to avoid false positives
    code_clean = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)

    # Find all function definitions
    main_func_pattern = r'async\s+function\s+(\w+)\s*\([^)]*\)\s*\{'
    main_func_match = re.search(main_func_pattern, code_clean)

    if not main_func_match:
        return {"valid": True, "error": None, "fix_suggestion": None, "tdz_details": None}

    main_func_name = main_func_match.group(1)
    main_func_start = main_func_match.end()

    # Find the end position of the main function body
    brace_count = 1
    main_func_end = main_func_start
    for i in range(main_func_start, len(code_clean)):
        if code_clean[i] == '{':
            brace_count += 1
        elif code_clean[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                main_func_end = i
                break

    func_body = code_clean[main_func_start:main_func_end]

    # Find the position of the require declaration within the function body
    combined_decl_pattern = '|'.join(f'(?:{p})' for p in decl_patterns)
    decl_match = re.search(combined_decl_pattern, func_body)

    if not decl_match:
        return {"valid": True, "error": None, "fix_suggestion": None, "tdz_details": None}

    decl_pos = decl_match.start()

    # Find helper function definitions inside the function body
    helper_func_pattern = r'(function\s+(\w+)\s*\([^)]*\)\s*\{)'
    helper_matches = list(re.finditer(helper_func_pattern, func_body))

    tdz_issues = []

    for helper_match in helper_matches:
        helper_start = helper_match.start()
        helper_name = helper_match.group(2)

        # Find the helper function body
        helper_body_start = helper_match.end()
        helper_brace_count = 1
        helper_body_end = helper_body_start
        for i in range(helper_body_start, len(func_body)):
            if func_body[i] == '{':
                helper_brace_count += 1
            elif func_body[i] == '}':
                helper_brace_count -= 1
                if helper_brace_count == 0:
                    helper_body_end = i
                    break

        helper_body = func_body[helper_body_start:helper_body_end]

        # Check whether the helper references any require-declared variable
        referenced_vars = [vn for vn in var_names if vn in helper_body]
        if referenced_vars and helper_start < decl_pos:
            vars_str = ', '.join(referenced_vars)
            tdz_issues.append({
                "helper_name": helper_name,
                "helper_pos": helper_start,
                "mcdata_pos": decl_pos,
                "issue": f"Helper function '{helper_name}' is defined before the declaration of {vars_str} but references {vars_str}"
            })

    if tdz_issues:
        issue_descriptions = [issue["issue"] for issue in tdz_issues]
        vars_mentioned = ', '.join(var_names)
        return {
            "valid": False,
            "error": f"TDZ issue detected: {'; '.join(issue_descriptions)}",
            "fix_suggestion": f"Move the {vars_mentioned} declaration before all helper function definitions",
            "tdz_details": {
                "main_func_name": main_func_name,
                "issues": tdz_issues
            }
        }

    return {"valid": True, "error": None, "fix_suggestion": None, "tdz_details": None}


def validate_code_completeness(
    optimized_code: str,
    original_code: str,
    llm_response: str = None,
) -> Dict[str, Any]:
    """
    Verify that the optimized code is complete (guard against truncation)

    Args:
        optimized_code: optimized code
        original_code: original code
        llm_response: raw LLM response (used to diagnose truncation issues)

    Returns:
        Dict[str, Any]: validation result
        {
            "complete": bool,
            "errors": List[str],
            "warnings": List[str],
            "diagnostics": Dict[str, Any]  # detailed diagnostics
        }
    """
    errors = []
    warnings = []
    diagnostics = {
        "original_code_stats": {},
        "optimized_code_stats": {},
        "llm_response_stats": {},
        "bracket_analysis": {},
        "truncation_indicators": [],
    }

    # === Collect code statistics ===
    original_lines = original_code.split('\n')
    optimized_lines = optimized_code.split('\n') if optimized_code else []

    # Ignore brackets inside comments and strings for counting purposes
    stripped_original = _strip_comments_and_strings(original_code)
    stripped_optimized = _strip_comments_and_strings(optimized_code) if optimized_code else ""

    diagnostics["original_code_stats"] = {
        "lines": len(original_lines),
        "chars": len(original_code),
        "open_brackets": stripped_original.count('(') + stripped_original.count('[') + stripped_original.count('{'),
        "close_brackets": stripped_original.count(')') + stripped_original.count(']') + stripped_original.count('}'),
    }

    diagnostics["optimized_code_stats"] = {
        "lines": len(optimized_lines),
        "chars": len(optimized_code) if optimized_code else 0,
        "open_brackets": stripped_optimized.count('(') + stripped_optimized.count('[') + stripped_optimized.count('{'),
        "close_brackets": stripped_optimized.count(')') + stripped_optimized.count(']') + stripped_optimized.count('}'),
    }

    # === LLM response analysis ===
    if llm_response:
        diagnostics["llm_response_stats"] = {
            "total_chars": len(llm_response),
            "ends_with_code_block": llm_response.rstrip().endswith('```'),
            "contains_json_close": '}' in llm_response[-50:] if len(llm_response) >= 50 else '}' in llm_response,
        }
        # Detect potential truncation
        if not llm_response.rstrip().endswith('```') and '```javascript' in llm_response:
            diagnostics["truncation_indicators"].append("LLM response does not end with a code-block end marker (```)")
        if llm_response.endswith('...') or llm_response.endswith('…'):
            diagnostics["truncation_indicators"].append("LLM response ends with an ellipsis")

    # 1. Check whether the code is empty
    if not optimized_code or not optimized_code.strip():
        errors.append("Optimized code is empty")
        diagnostics["truncation_indicators"].append("Code is empty; extraction may have failed")
        return {"complete": False, "errors": errors, "warnings": warnings, "diagnostics": diagnostics}

    # 2. Check for obvious signs of truncation
    incomplete_patterns = [
        (r'["\']\s*$', "Unfinished string literal"),
        (r'//\s*$', "Unfinished single-line comment"),
        (r'/\*\s*$', "Unfinished multi-line comment"),
        (r'\(\s*$', "Unfinished function call"),
        (r'\[\s*$', "Unfinished array"),
        (r'\{\s*$', "Unfinished object"),
        (r'await\s+$', "Unfinished await statement"),
        (r'return\s+$', "Unfinished return statement"),
        (r'throw\s+$', "Unfinished throw statement"),
        (r'if\s*\(\s*$', "Unfinished if statement"),
        (r'for\s*\(\s*$', "Unfinished for loop"),
        (r'while\s*\(\s*$', "Unfinished while loop"),
        (r'function\s+\w+\s*\(\s*$', "Unfinished function definition"),
        (r'async\s+function\s+\w+\s*\(\s*$', "Unfinished async function definition"),
    ]

    lines = optimized_code.split('\n')
    last_line = lines[-1].strip() if lines else ""

    for pattern, description in incomplete_patterns:
        if re.search(pattern, last_line):
            errors.append(f"Code may be truncated: {description} (last line: {last_line[:50]}...)")
            diagnostics["truncation_indicators"].append(f"Last line matches truncation pattern: {description}")

    # 3. Detailed bracket analysis (ignoring brackets inside comments and strings)
    stripped_code = _strip_comments_and_strings(optimized_code)
    open_parens = stripped_code.count('(')
    close_parens = stripped_code.count(')')
    open_brackets = stripped_code.count('[')
    close_brackets = stripped_code.count(']')
    open_braces = stripped_code.count('{')
    close_braces = stripped_code.count('}')

    diagnostics["bracket_analysis"] = {
        "parentheses": {"open": open_parens, "close": close_parens, "diff": open_parens - close_parens},
        "brackets": {"open": open_brackets, "close": close_brackets, "diff": open_brackets - close_brackets},
        "braces": {"open": open_braces, "close": close_braces, "diff": open_braces - close_braces},
    }

    total_open = open_parens + open_brackets + open_braces
    total_close = close_parens + close_brackets + close_braces

    if total_open != total_close:
        bracket_diff = total_open - total_close

        # Determine which bracket type is mismatched
        mismatch_details = []
        if open_parens != close_parens:
            diff = open_parens - close_parens
            mismatch_details.append(f"parentheses() {'+' if diff > 0 else ''}{diff}")
        if open_brackets != close_brackets:
            diff = open_brackets - close_brackets
            mismatch_details.append(f"brackets[] {'+' if diff > 0 else ''}{diff}")
        if open_braces != close_braces:
            diff = open_braces - close_braces
            mismatch_details.append(f"braces{{}} {'+' if diff > 0 else ''}{diff}")

        # Try to identify the offending line
        problem_line = find_bracket_mismatch_line(optimized_code)

        if bracket_diff > 0:
            errors.append(f"Bracket mismatch: missing {bracket_diff} closing bracket(s) (details: {', '.join(mismatch_details)})")
        else:
            errors.append(f"Bracket mismatch: {abs(bracket_diff)} extra closing bracket(s) (details: {', '.join(mismatch_details)})")

        if problem_line:
            diagnostics["bracket_analysis"]["problem_line"] = problem_line

    # 4. Check for obvious syntax-error patterns
    syntax_error_patterns = [
        (r'[a-zA-Z_$][a-zA-Z0-9_$]*\s*\([^)]*,\s*$', "Function call argument list incomplete"),
        (r'const\s+\w+\s*=\s*$', "const declaration incomplete"),
        (r'let\s+\w+\s*=\s*$', "let declaration incomplete"),
        (r'var\s+\w+\s*=\s*$', "var declaration incomplete"),
    ]

    for pattern, description in syntax_error_patterns:
        if re.search(pattern, last_line):
            warnings.append(f"Possible syntax issue: {description}")

    # 5. Check whether code length is unusually short (may indicate truncation)
    if len(optimized_code) < len(original_code) * 0.3:
        warnings.append(f"Optimized code length is unusually short ({len(optimized_code)/len(original_code)*100:.1f}% of original); may be truncated")
        diagnostics["truncation_indicators"].append(f"Unusually short code length: {len(optimized_code)}/{len(original_code)} chars")

    # 6. Check whether the main function definition is included
    main_function_match = re.search(r'(?:async\s+)?function\s+(\w+)\s*\(', original_code)
    if main_function_match:
        main_function_name = main_function_match.group(1)
        if main_function_name not in optimized_code:
            warnings.append(f"Main function '{main_function_name}' not found in optimized code; may be incomplete")

    # 7. TDZ (Temporal Dead Zone) issue detection
    tdz_result = validate_tdz_issues(optimized_code)
    if not tdz_result["valid"]:
        errors.append(f"TDZ issue: {tdz_result['error']}")

    return {
        "complete": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "tdz_details": tdz_result.get("tdz_details"),
        "can_fix_tdz": not tdz_result["valid"]
    }


def fix_tdz_issues(code: str) -> Tuple[str, bool, str]:
    """
    Automatically fix TDZ issues in code

    Fix strategy:
    1. Extract all require statements and important declarations (patterns from domain knowledge)
    2. Move them to the very beginning of the function body (before all helper functions)

    Args:
        code: code to fix

    Returns:
        Tuple[str, bool, str]: (fixed_code, was_fixed, fix_description)
    """
    # Find the main function
    main_func_pattern = r'(async\s+function\s+\w+\s*\([^)]*\)\s*\{)'
    main_func_match = re.search(main_func_pattern, code)

    if not main_func_match:
        return code, False, "Main function not found; cannot fix"

    main_func_header = main_func_match.group(1)
    main_func_start = main_func_match.end()

    # Find the end of the main function body
    brace_count = 1
    main_func_end = main_func_start
    for i in range(main_func_start, len(code)):
        if code[i] == '{':
            brace_count += 1
        elif code[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                main_func_end = i
                break

    func_body = code[main_func_start:main_func_end]

    # Find declarations that need to be hoisted
    declarations_to_move = []
    remaining_body_lines = []

    # Declaration patterns that should be moved to the top (from domain knowledge)
    decl_patterns, _ = _get_require_info()
    if not decl_patterns:
        return code, False, "No require patterns configured"
    priority_patterns = decl_patterns

    lines = func_body.split('\n')

    for line in lines:
        line_stripped = line.strip()
        is_declaration = False

        for pattern in priority_patterns:
            if re.match(pattern, line_stripped):
                declarations_to_move.append(line)
                is_declaration = True
                break

        if not is_declaration:
            remaining_body_lines.append(line)

    if not declarations_to_move:
        return code, False, "No declarations found to move"

    # Reassemble the function body
    new_func_body = '\n'.join(declarations_to_move) + '\n' + '\n'.join(remaining_body_lines)

    # Rebuild the code
    before_func = code[:main_func_match.start()]
    after_func = code[main_func_end:]

    new_code = before_func + main_func_header + new_func_body + after_func

    return new_code, True, f"Moved {len(declarations_to_move)} declarations to the start of the function body"


def detect_code_duplication(code: str, min_duplicate_lines: int = 4) -> Dict[str, Any]:
    """
    Detect duplicate fragments in code

    Args:
        code: code to analyze
        min_duplicate_lines: minimum duplicate-line count threshold

    Returns:
        {
            "has_duplication": bool,
            "duplicates": List[Dict],  # [{lines: [line_nums], content: str}]
            "duplication_ratio": float,  # duplication ratio
            "suggestion": str
        }
    """
    lines = code.strip().split('\n')
    total_lines = len(lines)

    if total_lines < min_duplicate_lines * 2:
        return {"has_duplication": False, "duplicates": [], "duplication_ratio": 0.0, "suggestion": ""}

    # Normalize lines (strip leading/trailing whitespace)
    normalized_lines = [line.strip() for line in lines]

    duplicates = []
    seen_sequences = {}
    duplicate_line_indices = set()

    # Detect consecutive duplicate line sequences
    for start in range(total_lines - min_duplicate_lines + 1):
        # Build a signature of min_duplicate_lines lines
        sequence = tuple(normalized_lines[start:start + min_duplicate_lines])

        # Skip empty-line sequences or sequences containing only braces
        if all(line in ('', '{', '}', '});', '});') for line in sequence):
            continue

        if sequence in seen_sequences:
            # Duplicate found
            original_start = seen_sequences[sequence]
            duplicates.append({
                "original_lines": list(range(original_start + 1, original_start + min_duplicate_lines + 1)),
                "duplicate_lines": list(range(start + 1, start + min_duplicate_lines + 1)),
                "content": '\n'.join(lines[start:start + min_duplicate_lines][:3]) + "..."  # Show only the first 3 lines
            })
            duplicate_line_indices.update(range(start, start + min_duplicate_lines))
        else:
            seen_sequences[sequence] = start

    # Compute duplication ratio
    duplication_ratio = len(duplicate_line_indices) / total_lines if total_lines > 0 else 0.0
    has_duplication = duplication_ratio > 0.1  # Treat >10% as a duplication issue

    suggestion = ""
    if has_duplication:
        suggestion = f"Detected {len(duplicates)} duplicate code section(s) (duplication ratio {duplication_ratio:.1%}); recommend extracting a common function"

    return {
        "has_duplication": has_duplication,
        "duplicates": duplicates[:5],  # Only return the first 5
        "duplication_ratio": duplication_ratio,
        "suggestion": suggestion
    }


def validate_minimal_change(
    original: str,
    modified: str,
    max_diff_lines: int = 30
) -> Tuple[bool, int, str]:
    """
    Verify that the modification is indeed minimal

    Args:
        original: original code
        modified: modified code
        max_diff_lines: maximum allowed number of changed lines

    Returns:
        Tuple[bool, int, str]: (is_valid, changed_lines, message)
    """
    import difflib

    original_lines = original.strip().split('\n')
    modified_lines = modified.strip().split('\n')

    # Count diff lines
    diff = list(difflib.unified_diff(original_lines, modified_lines, lineterm=''))
    changed_lines = sum(1 for line in diff if line.startswith('+') or line.startswith('-'))
    # Exclude the --- and +++ header lines of the diff
    changed_lines = max(0, changed_lines - 2)

    if changed_lines > max_diff_lines:
        return False, changed_lines, f"Modified {changed_lines} lines, exceeding limit {max_diff_lines}"

    return True, changed_lines, f"Modified {changed_lines} lines, within limit {max_diff_lines}"
