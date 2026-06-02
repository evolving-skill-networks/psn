"""
Syntax Utilities (Layer A)

Basic syntax utility functions: comment/string stripping and bracket-mismatch
detection.  These are stateless pure functions that other sub-modules depend on.

extracted from code_validator.py.
"""

import re
from typing import Dict, Any, Optional


def _strip_comments_and_strings(code: str) -> str:
    """
    Strip comments and strings from the code for accurate bracket counting.

    Args:
        code: JavaScript code

    Returns:
        Code with comments and strings removed
    """
    # Remove single-line comments
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove strings (simplified handling)
    code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
    code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
    code = re.sub(r'`[^`\\]*(?:\\.[^`\\]*)*`', '``', code)
    return code


def find_bracket_mismatch_line(code: str) -> Optional[Dict[str, Any]]:
    """
    Try to find the line where a bracket mismatch starts.

    Args:
        code: JavaScript code

    Returns:
        Dict with line number and context, or None if not found
    """
    lines = code.split('\n')

    # Use a stack to track brackets
    stack = []
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}
    reverse_pairs = {')': '(', ']': '[', '}': '{'}

    for line_num, line in enumerate(lines, 1):
        # Skip brackets inside strings and comments (simplified handling)
        in_string = False
        string_char = None

        for i, char in enumerate(line):
            # Simple string detection
            if char in '"\'`' and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                continue

            if in_string:
                continue

            if char in bracket_pairs:
                stack.append((char, line_num, i))
            elif char in reverse_pairs:
                if stack and stack[-1][0] == reverse_pairs[char]:
                    stack.pop()
                else:
                    # Found an unmatched closing bracket
                    return {
                        "line_num": line_num,
                        "char_pos": i,
                        "issue": f"extra '{char}'",
                        "line_content": line.strip()[:80],
                    }

    # If there are still unclosed brackets on the stack
    if stack:
        unclosed = stack[-1]
        return {
            "line_num": unclosed[1],
            "char_pos": unclosed[2],
            "issue": f"unclosed '{unclosed[0]}'",
            "line_content": lines[unclosed[1]-1].strip()[:80] if unclosed[1] <= len(lines) else "",
        }

    return None
