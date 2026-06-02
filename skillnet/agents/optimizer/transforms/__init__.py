"""
Optimizer transforms — diff helpers and code-edit operations.
"""

from .diff_engine import (
    generate_unified_diff,
    generate_annotated_diff,
    basic_syntax_check,
)

from .code_edit_ops import remove_unused_helper_functions

__all__ = [
    "generate_unified_diff",
    "generate_annotated_diff",
    "basic_syntax_check",
    "remove_unused_helper_functions",
]
