"""
Code Context - Code extraction and context building utilities

Extracted from optimizer_impl.py for better modularity.
Contains pure functions for code context manipulation.
"""

from typing import Dict, List, Set


def extract_focus_lines(
    issues: List[Dict],
    suggested_fixes: List[Dict]
) -> Set[int]:
    """
    Extract line numbers that need attention based on issues and suggested fixes.

    Args:
        issues: List of issue dictionaries with 'line' field
        suggested_fixes: List of fix dictionaries with 'line_range' field

    Returns:
        Set of line numbers that need modification
    """
    lines: Set[int] = set()
    for iss in issues:
        if 'line' in iss and isinstance(iss['line'], int):
            lines.add(iss['line'])
    for fix in suggested_fixes:
        line_range = fix.get('line_range')
        if isinstance(line_range, (list, tuple)) and len(line_range) >= 2:
            start, end = line_range[0], line_range[1]
            if isinstance(start, int) and isinstance(end, int):
                for ln in range(start, end + 1):
                    lines.add(ln)
    return lines


def extract_code_with_context(
    code: str,
    focus_lines: Set[int],
    context: int = 5
) -> str:
    """
    Extract code segments around focus lines with context.
    Shows numbered lines with focus areas highlighted.

    Args:
        code: The full code string
        focus_lines: Set of line numbers to focus on
        context: Number of context lines before and after each focus area

    Returns:
        Formatted code string with line numbers and focus indicators
    """
    if not focus_lines:
        # If no focus lines, return first 50 lines with line numbers
        lines = code.split('\n')
        return '\n'.join(
            f"{i+1:4d}| {line}" for i, line in enumerate(lines[:50])
        )

    lines = code.split('\n')
    total_lines = len(lines)

    # Expand focus areas with context
    expanded_lines: Set[int] = set()
    for ln in focus_lines:
        for offset in range(-context, context + 1):
            expanded_ln = ln + offset
            if 1 <= expanded_ln <= total_lines:
                expanded_lines.add(expanded_ln)

    # Sort and find contiguous ranges
    sorted_lines = sorted(expanded_lines)

    result_parts = []
    prev_ln = 0

    for ln in sorted_lines:
        # Add ellipsis if there's a gap
        if ln > prev_ln + 1 and prev_ln > 0:
            result_parts.append("    ... (lines omitted)")

        # Format the line with focus indicator
        line_idx = ln - 1
        if 0 <= line_idx < total_lines:
            prefix = ">>> " if ln in focus_lines else "    "
            result_parts.append(f"{ln:4d}|{prefix}{lines[line_idx]}")

        prev_ln = ln

    return '\n'.join(result_parts)


def extract_function_signature(code: str) -> str:
    """
    Extract the function signature from the code that must not change.

    Args:
        code: The full code string

    Returns:
        The function signature string
    """
    lines = code.split('\n')
    for line in lines:
        stripped = line.strip()
        # Match async function declarations
        if stripped.startswith('async function ') and '(' in stripped:
            # Extract up to the opening brace or end of parameters
            if '{' in stripped:
                return stripped[:stripped.index('{')] + ' {'
            return stripped
        # Match regular function declarations
        elif stripped.startswith('function ') and '(' in stripped:
            if '{' in stripped:
                return stripped[:stripped.index('{')] + ' {'
            return stripped
    return "// Could not extract function signature"


def extract_code_with_signature(
    code: str,
    skill_name: str,
    max_chars: int = 4000
) -> str:
    """
    Smart code extraction, ensuring main function signature is included.

    Args:
        code: Full code string
        skill_name: Skill name (for matching main function)
        max_chars: Maximum characters

    Returns:
        str: Code snippet containing the main function signature
    """
    import re

    # Find main function definition
    pattern = rf'(async\s+)?function\s+{re.escape(skill_name)}\s*\([^)]*\)'
    match = re.search(pattern, code)

    if match:
        # Found signature, extract from there
        start = match.start()
        # Extract some context before (like comments, max 200 chars)
        context_start = max(0, code.rfind('\n', max(0, start - 200), start))
        if context_start == -1:
            context_start = max(0, start - 200)

        # Extract code snippet
        extracted = code[context_start:context_start + max_chars]

        # Add hints
        if context_start > 0:
            line_num = code[:context_start].count('\n') + 1
            extracted = (
                f"// ... (code before line {line_num} omitted) ...\n\n"
                + extracted
            )
        if context_start + max_chars < len(code):
            extracted = extracted + "\n\n// ... (remaining code omitted) ..."

        return extracted
    else:
        # Signature not found, use original truncation
        if len(code) > max_chars:
            return code[:max_chars] + "\n\n// ... (code truncated) ..."
        return code
