"""Central Domain Knowledge Registry.

Single point of truth for the current DomainKnowledge instance.
Replaces 20 module-level ``_domain_knowledge`` singletons.

Usage::

    from skillnet.core.dk_registry import get_domain_knowledge

    dk = get_domain_knowledge()
    if dk:
        result = dk.some_method()

Modules that maintain lazy caches derived from DK should register
an on-change hook::

    from skillnet.core.dk_registry import register_on_change

    def _invalidate_caches(dk):
        global _my_cache
        _my_cache = None

    register_on_change(_invalidate_caches)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from skillnet.core.domain import DomainKnowledge

logger = logging.getLogger(__name__)

_current_dk: Optional["DomainKnowledge"] = None
_on_change_hooks: List[Callable] = []


def get_domain_knowledge() -> Optional["DomainKnowledge"]:
    """Return the current DomainKnowledge instance, or None."""
    return _current_dk


def set_domain_knowledge(dk: Optional["DomainKnowledge"]) -> None:
    """Set the global DomainKnowledge and notify on-change hooks."""
    global _current_dk
    _current_dk = dk
    for hook in _on_change_hooks:
        try:
            hook(dk)
        except Exception:
            logger.warning("DK on-change hook failed", exc_info=True)


def register_on_change(hook: Callable) -> None:
    """Register a callback invoked when DK changes.

    Useful for modules that maintain lazy caches derived from DK.
    The hook receives the new DK (may be None).
    """
    _on_change_hooks.append(hook)
