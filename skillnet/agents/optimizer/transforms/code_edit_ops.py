"""
Code Edit Operations

Code edit operations module, providing general-purpose JavaScript code modification utilities.
All methods are pure functions that do not depend on external state.
"""

import re
from typing import Dict, List, Any, Tuple, Optional


def remove_unused_helper_functions(
    code: str,
    main_func_name: str,
) -> Tuple[str, List[str]]:
    """
    Remove unused helper functions from the code.

    Args:
        code: JavaScript code
        main_func_name: Main function name

    Returns:
        Tuple[modified code, list of removed function names]
    """
    # Find all function definitions
    func_pattern = r'(?:async\s+)?function\s+(\w+)\s*\('
    all_funcs = set(re.findall(func_pattern, code))

    if main_func_name in all_funcs:
        all_funcs.remove(main_func_name)

    # Check which functions are called
    called_funcs = set()
    for func_name in all_funcs:
        # Strategy 1: check for direct call funcName()
        call_pattern = rf'(?<!function\s){func_name}\s*\('
        if re.search(call_pattern, code):
            called_funcs.add(func_name)
            continue

        # Strategy 2: check for callback/reference uses (no parentheses)
        # e.g.: .filter(funcName), .map(funcName), callback = funcName
        callback_patterns = [
            rf'\.\w+\s*\(\s*{func_name}\s*[,)]',   # .filter(funcName) or .filter(funcName, ...)
            rf'[=:]\s*{func_name}\s*[,;)\]}}]',    # = funcName; or : funcName,
            rf'\(\s*{func_name}\s*,',               # (funcName, ...)
            rf',\s*{func_name}\s*[,)]',             # (..., funcName, ...) or (..., funcName)
        ]
        for pattern in callback_patterns:
            if re.search(pattern, code):
                called_funcs.add(func_name)
                break

    # Remove functions that are never called
    removed = []
    modified_code = code

    for func_name in all_funcs - called_funcs:
        # Match the complete function definition
        func_def_pattern = rf'(?:async\s+)?function\s+{func_name}\s*\([^)]*\)\s*\{{[^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*\}}'
        match = re.search(func_def_pattern, modified_code, re.DOTALL)
        if match:
            modified_code = modified_code[:match.start()] + modified_code[match.end():]
            removed.append(func_name)

    return modified_code, removed
