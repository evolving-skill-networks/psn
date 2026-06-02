"""
Optimizer Core Package

Core utility module: shared utilities for LLM invocation, JSON parsing, etc.
"""

from .llm_invoker import LLMInvoker, robust_json_parse
from .optimization_helpers import (
    build_current_state_info,
    extract_issues_from_feedbacks,
    check_requirements_addressed,
)

__all__ = [
    "LLMInvoker",
    "robust_json_parse",
    # P2 Phase A: pure helper functions for optimization
    "build_current_state_info",
    "extract_issues_from_feedbacks",
    "check_requirements_addressed",
]
