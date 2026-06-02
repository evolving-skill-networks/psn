"""
Error pattern normalization.

Pure helper extracted from optimizer_impl.py for use in loop detection.
"""

import re


def normalize_error_pattern(error_message: str) -> str:
    """
    Normalize error message, ignoring line numbers and other variable parts.

    Used for loop detection — the same logical error should produce the same
    pattern even if line numbers or memory addresses differ.

    Args:
        error_message: Raw error message string

    Returns:
        Normalized error pattern (max 200 chars)
    """
    if not error_message:
        return ""
    # Remove line numbers (e.g., :123:45)
    pattern = re.sub(r':\d+:\d+', ':X:X', error_message)
    # Remove memory addresses
    pattern = re.sub(r'0x[0-9a-fA-F]+', '0xXXX', pattern)
    return pattern[:200]  # Truncate long messages
