"""
Execution Analysis - execution-analysis utility

Pure functions extracted from graph_manager_impl.py for analyzing execution records and parsing JavaScript arguments.

v4.0 modular refactor
"""

import re
import json
import logging
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import FailureCategory

logger = logging.getLogger(__name__)


def calculate_correlation(pairs: List[Tuple[float, float]]) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        pairs: list of (x, y) value pairs

    Returns:
        correlation coefficient in [-1, 1]
    """
    if len(pairs) < 2:
        return 0

    n = len(pairs)
    sum_x = sum(p[0] for p in pairs)
    sum_y = sum(p[1] for p in pairs)
    sum_xy = sum(p[0] * p[1] for p in pairs)
    sum_x2 = sum(p[0] ** 2 for p in pairs)
    sum_y2 = sum(p[1] ** 2 for p in pairs)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

    if denominator == 0:
        return 0

    return numerator / denominator


def categorize_failure(error_message: str) -> str:
    """
    Classify failure reason.

    Not all failures should be used to infer preconditions:
    - CODE_BUG: code did not run or had syntax errors; cannot be used to infer preconditions
    - EXTERNAL_FAILURE: failure caused by external factors; cannot be used to infer
    - EXECUTION_ERROR: runtime error during execution; some can be used for inference
    - PRECONDITION_NOT_MET: can be used to infer preconditions
    - TIMEOUT: timeout
    - UNKNOWN: unknown category

    Args:
        error_message: Error message string

    Returns:
        str: Failure category (CODE_BUG, EXTERNAL_FAILURE, EXECUTION_ERROR,
             PRECONDITION_NOT_MET, TIMEOUT, UNKNOWN)
    """
    error_msg = (error_message or "").lower()

    # Check whether the code never ran (e.g., the previous craftWoodenPickaxe issue)
    if any(pattern in error_msg for pattern in [
        "no skill execution events",
        "skill not executed",
        "code not executed",
        "undefined variable",
        "undefinedvariable",
        "syntax error",
        "syntaxerror",
        "reference error",
        "referenceerror",
        "is not defined",
        "is not a function",
        "unexpected token",
        "babel",  # Babel parse error
    ]):
        return "CODE_BUG"

    # Check for timeout
    if any(pattern in error_msg for pattern in [
        "timeout",
        "timed out",
        "took too long",
    ]):
        return "TIMEOUT"

    # Check for external/network issues
    if any(pattern in error_msg for pattern in [
        "connection",
        "network",
        "server",
        "disconnected",
        "econnreset",
        "socket",
    ]):
        return "EXTERNAL_FAILURE"

    # Check whether a block/item could not be found (runtime error, not a precondition issue)
    if any(pattern in error_msg for pattern in [
        "could not find",
        "couldn't find",
        "not found nearby",
        "no blocks found",
        "no nearby",
        "unable to reach",
        "pathfinding",
    ]):
        return "EXECUTION_ERROR"

    # Check inventory/material-related (possibly a precondition issue)
    if any(pattern in error_msg for pattern in [
        "not enough",
        "need more",
        "missing",
        "require",
        "insufficient",
        "don't have",
        "do not have",
    ]):
        return "PRECONDITION_NOT_MET"

    return "UNKNOWN"


def parse_js_value(value_str: str) -> Any:
    """
    Parse a single JavaScript value.

    Args:
        value_str: Value string

    Returns:
        Parsed Python value
    """
    value_str = value_str.strip()

    if not value_str:
        return None

    # Boolean
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False

    # null/undefined
    if value_str.lower() in ['null', 'undefined']:
        return None

    # Identifier (e.g., bot)
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', value_str):
        return value_str

    # Number
    try:
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # String
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]

    # Array
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            # Try parsing with json (replace single quotes with double quotes)
            json_str = value_str.replace("'", '"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If that fails, parse manually
            inner = value_str[1:-1].strip()
            if not inner:
                return []
            # Recursively parse the array elements
            return parse_js_arguments(inner)

    # Object
    if value_str.startswith('{') and value_str.endswith('}'):
        try:
            json_str = value_str.replace("'", '"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            return value_str  # Unable to parse; return the original string

    return value_str


def parse_js_arguments(args_str: str) -> List[Any]:
    """
    Parse JavaScript function-call arguments.

    Handles:
    - Simple values: numbers, strings, booleans
    - Arrays: ["a", "b"]
    - Objects: {key: value}
    - Nested structures

    Args:
        args_str: Argument string, e.g. 'bot, 3, ["oak_log", "birch_log"]'

    Returns:
        List[Any]: List of parsed argument values
    """
    args = []
    current_arg = ""
    depth = 0  # Nesting depth of parentheses/arrays/objects
    in_string = False
    string_char = None

    i = 0
    while i < len(args_str):
        char = args_str[i]

        # Handle strings
        if char in ['"', "'", '`'] and (i == 0 or args_str[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
            current_arg += char
        elif in_string:
            current_arg += char
        elif char in ['[', '{', '(']:
            depth += 1
            current_arg += char
        elif char in [']', '}', ')']:
            depth -= 1
            current_arg += char
        elif char == ',' and depth == 0:
            # Argument separator
            arg_value = parse_js_value(current_arg.strip())
            args.append(arg_value)
            current_arg = ""
        else:
            current_arg += char

        i += 1

    # Handle the last argument
    if current_arg.strip():
        arg_value = parse_js_value(current_arg.strip())
        args.append(arg_value)

    return args
