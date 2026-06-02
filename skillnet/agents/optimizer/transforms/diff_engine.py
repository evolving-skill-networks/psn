"""
Diff and basic-syntax helpers used by the optimizer.

Functions:
- generate_unified_diff   - unified-diff representation between two code strings
- generate_annotated_diff - per-line +/- annotated diff for prompts
- basic_syntax_check      - lightweight Python/JS bracket-balance check
"""

import re
from typing import Tuple





# Convenience function

# ========== diff generation functions (migrated from optimizer_impl.py) ==========

def generate_unified_diff(old_code: str, new_code: str, skill_name: str) -> str:
    """
    Generate a unified diff format displaying code changes

    Args:
        old_code: original code
        new_code: new code
        skill_name: skill name

    Returns:
        str: unified-diff formatted string
    """
    import difflib

    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)

    # Generate the unified diff
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{skill_name}.js (original)",
        tofile=f"{skill_name}.js (optimized)",
        lineterm=""
    )

    diff_text = ''.join(diff)

    # If the diff is too long, keep only the key parts
    if len(diff_text) > 8000:
        # Split into hunks by @@
        hunks = re.split(r'(@@.*?@@)', diff_text)
        truncated_hunks = []
        total_len = 0

        for i, hunk in enumerate(hunks):
            if total_len + len(hunk) > 7500:
                truncated_hunks.append("\n... (diff truncated, more changes follow) ...")
                break
            truncated_hunks.append(hunk)
            total_len += len(hunk)

        diff_text = ''.join(truncated_hunks)

    return diff_text if diff_text else "(no changes detected)"


def generate_annotated_diff(
    old_code: str,
    new_code: str,
    skill_name: str,
    collapse_unchanged: int = 15
) -> str:
    """
    Generate an annotated full-code diff (similar to Cursor's display style)

    Marks modifications in the full code so the LLM can see the complete code structure and context.

    Args:
        old_code: original code
        new_code: new code
        skill_name: skill name
        collapse_unchanged: collapse when this many consecutive unchanged lines (default 15)

    Returns:
        str: annotated full code
    """
    import difflib

    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    # Use SequenceMatcher to get opcodes
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = matcher.get_opcodes()

    result = []
    result.append(f"// === {skill_name}.js (changes marked) ===")
    result.append("// Legend: [-] deleted, [+] added, [ ] unchanged")
    result.append("")

    new_line_num = 1

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Unchanged lines
            unchanged_lines = new_lines[j1:j2]
            num_unchanged = len(unchanged_lines)

            if num_unchanged > collapse_unchanged:
                # Collapse long unchanged runs; keep 3 lines on each side
                keep_lines = 3
                # Show the first few lines
                for idx, line in enumerate(unchanged_lines[:keep_lines]):
                    result.append(f"[ ] {new_line_num + idx:4} | {line}")

                # Collapse the middle section
                collapsed_count = num_unchanged - 2 * keep_lines
                result.append(f"         ... // ({collapsed_count} unchanged lines)")

                # Show the last few lines
                for idx, line in enumerate(unchanged_lines[-keep_lines:]):
                    line_num = new_line_num + num_unchanged - keep_lines + idx
                    result.append(f"[ ] {line_num:4} | {line}")
            else:
                # Show all
                for idx, line in enumerate(unchanged_lines):
                    result.append(f"[ ] {new_line_num + idx:4} | {line}")

            new_line_num += num_unchanged

        elif tag == 'replace':
            # Replace: show deletions first, then additions
            for line in old_lines[i1:i2]:
                result.append(f"[-]      | {line}")
            for idx, line in enumerate(new_lines[j1:j2]):
                result.append(f"[+] {new_line_num + idx:4} | {line}")
            new_line_num += (j2 - j1)

        elif tag == 'delete':
            # Deleted lines (do not take up new line numbers)
            for line in old_lines[i1:i2]:
                result.append(f"[-]      | {line}")

        elif tag == 'insert':
            # Newly added lines
            for idx, line in enumerate(new_lines[j1:j2]):
                result.append(f"[+] {new_line_num + idx:4} | {line}")
            new_line_num += (j2 - j1)

    return '\n'.join(result)


def basic_syntax_check(code: str) -> Tuple[bool, str]:
    """
    Perform basic syntax validation without Babel.
    Checks for common issues like unbalanced brackets.

    Args:
        code: JavaScript code to check

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for balanced brackets
    brackets = {'(': ')', '{': '}', '[': ']'}
    stack = []
    in_string = False
    string_char = None
    in_comment = False
    in_line_comment = False
    prev_char = None

    for i, char in enumerate(code):
        # Handle line comments
        if in_line_comment:
            if char == '\n':
                in_line_comment = False
            prev_char = char
            continue

        # Handle block comments
        if in_comment:
            if prev_char == '*' and char == '/':
                in_comment = False
            prev_char = char
            continue

        # Detect start of comments
        if not in_string:
            if prev_char == '/' and char == '/':
                in_line_comment = True
                prev_char = char
                continue
            if prev_char == '/' and char == '*':
                in_comment = True
                prev_char = char
                continue

        # Handle strings
        if char in ('"', "'", '`') and prev_char != '\\':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None

        # Only check brackets outside strings and comments
        if not in_string:
            if char in brackets:
                stack.append((char, i))
            elif char in brackets.values():
                if not stack:
                    return False, f"Unexpected closing bracket '{char}' at position {i}"
                open_bracket, _ = stack.pop()
                if brackets[open_bracket] != char:
                    return False, f"Mismatched brackets: expected '{brackets[open_bracket]}', got '{char}' at position {i}"

        prev_char = char

    if stack:
        unclosed = stack[-1]
        return False, f"Unclosed bracket '{unclosed[0]}' at position {unclosed[1]}"

    if in_string:
        return False, f"Unterminated string (started with '{string_char}')"

    # Check for invalid catch syntax: catch (/**/) {}
    if re.search(r'catch\s*\(\s*/\*.*?\*/\s*\)', code):
        return False, "Invalid catch syntax: 'catch (/**/)' - use 'catch {}' or 'catch (e) {}' instead"

    return True, ""
