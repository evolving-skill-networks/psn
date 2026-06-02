"""
Effect Matcher Utilities

Module-level constants and helper functions used by the various EffectMatcher layers.

extracted from effect_matcher.py.
Domain-inject item categories and safe interchange categories.
"""

import logging
from typing import Optional

from skillnet.core.dk_registry import get_domain_knowledge, register_on_change

logger = logging.getLogger(__name__)


# ── Item knowledge (lazy loaded, domain-first) ──

_resource_aliases = None
_common_items = None
_common_blocks = None


def _on_dk_changed(dk):
    """Invalidate lazy caches when domain knowledge changes."""
    global _resource_aliases, _common_items, _common_blocks
    _resource_aliases = None
    _common_items = None
    _common_blocks = None


register_on_change(_on_dk_changed)


def _get_resource_aliases():
    global _resource_aliases
    if _resource_aliases is not None:
        return _resource_aliases
    dk = get_domain_knowledge()
    if dk:
        aliases = dk.get_resource_aliases()
        if aliases:
            _resource_aliases = aliases
            return _resource_aliases
    # Don't cache empty — allows future DI to take effect
    return {}


def _get_common_items():
    global _common_items
    if _common_items is not None:
        return _common_items
    dk = get_domain_knowledge()
    if dk:
        items = dk.get_common_items()
        if items:
            _common_items = items
            return _common_items
    return set()


def _get_common_blocks():
    global _common_blocks
    if _common_blocks is not None:
        return _common_blocks
    dk = get_domain_knowledge()
    if dk:
        blocks = dk.get_common_blocks()
        if blocks:
            _common_blocks = blocks
            return _common_blocks
    return set()


# ============================================================================
# Fallback constants (empty — domain knowledge provides data at runtime)
# ============================================================================

_FALLBACK_SAFE_INTERCHANGE_CATEGORIES = set()

_FALLBACK_ITEM_CATEGORIES = {}

# Quantity parameter names
COUNT_PARAM_NAMES = {'count', 'targettotal', 'amount', 'quantity', 'num', 'total', 'target'}

# supported operation types (extended with place/equip/find support)
SUPPORTED_OPERATIONS = {"add", "place", "equip", "find"}

# operation-type alias normalization (ensure → add)
OPERATION_ALIASES = {"ensure": "add"}


# ============================================================================
# Backward-compatible aliases (read-only references that existing code uses)
# ============================================================================
# Some consumers import ITEM_CATEGORIES / SAFE_INTERCHANGE_CATEGORIES directly.
# Keep module-level names pointing to the fallback dicts so ``from _utils
# import ITEM_CATEGORIES`` still works.  The accessor functions below should
# be used for runtime lookups that respect domain injection.

ITEM_CATEGORIES = _FALLBACK_ITEM_CATEGORIES
SAFE_INTERCHANGE_CATEGORIES = _FALLBACK_SAFE_INTERCHANGE_CATEGORIES


# ============================================================================
# Accessor functions (domain-aware)
# ============================================================================

def get_safe_interchange_categories():
    """Get safe interchange categories from domain knowledge.
    Returns empty set when no domain is available.
    """
    dk = get_domain_knowledge()
    if dk:
        cats = dk.get_safe_interchange_categories()
        if cats:
            return cats
    return set()


# ============================================================================
# Helper functions: category matching
# ============================================================================

def is_category_name(item_name: str) -> bool:
    """Determine whether the value is a category name (e.g. 'log') rather than a concrete item (e.g. 'oak_log').

    Properties of a category name:
    1. Present as a key in resource_aliases
    2. Not a concrete Minecraft item
    """
    item_lower = item_name.lower().strip()
    if item_lower not in _get_resource_aliases():
        return False
    # A category name must not be a concrete Minecraft item
    if item_lower in _get_common_items() or item_lower in _get_common_blocks():
        return False
    return True


def get_category_items(category_name: str) -> list:
    """Get all concrete items under a category.

    Example: get_category_items("log") -> ["oak_log", "birch_log", ...]
    """
    return _get_resource_aliases().get(category_name.lower().strip(), [])


def normalize_operation(operation: str) -> str:
    """Normalize the operation type, handling aliases."""
    return OPERATION_ALIASES.get(operation, operation)


def get_item_categories(domain_knowledge=None):
    """Get item categories from domain knowledge.

    Checks the explicit ``domain_knowledge`` argument first, then
    the central DK registry.  Returns empty dict when no domain is available.

    Args:
        domain_knowledge: Optional DomainKnowledge instance.

    Returns:
        Dict mapping category names to category info (keywords, items, actions).
    """
    dk = domain_knowledge or get_domain_knowledge()
    if dk:
        cats = dk.get_item_categories()
        if cats:
            return cats
    return {}
