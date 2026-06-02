"""
Effect Matcher Package

Effect matcher: matches a task's target effects against a Skill's expected_effects.

refactored from the single-file effect_matcher.py into an internal layered package.
- _utils.py: module-level constants and helper functions
- _extraction.py: ExtractionMixin (target-effect extraction)
- _matching.py: MatchingMixin (core effect-matching logic)
- _matcher.py: EffectMatcher main class (facade + orchestration)
"""

from ._matcher import EffectMatcher
from ._utils import (
    SAFE_INTERCHANGE_CATEGORIES,
    ITEM_CATEGORIES,
    COUNT_PARAM_NAMES,
    SUPPORTED_OPERATIONS,
    OPERATION_ALIASES,
    is_category_name,
    get_category_items,
    normalize_operation,
    get_item_categories,
    get_safe_interchange_categories,
)

__all__ = [
    "EffectMatcher",
    "SAFE_INTERCHANGE_CATEGORIES",
    "ITEM_CATEGORIES",
    "COUNT_PARAM_NAMES",
    "SUPPORTED_OPERATIONS",
    "OPERATION_ALIASES",
    "is_category_name",
    "get_category_items",
    "normalize_operation",
    "get_item_categories",
    "get_safe_interchange_categories",
]
