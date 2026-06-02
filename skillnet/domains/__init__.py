"""
PSN Domain Registry

Each subdirectory provides a DomainModule for a specific environment
(e.g., Minecraft).

Usage:
    from skillnet.domains import get_domain, list_domains

    domain = get_domain("minecraft", mc_port=25565)
    print(list_domains())  # ['minecraft']
"""

from typing import Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from skillnet.core.domain import DomainModule

_DOMAIN_REGISTRY: Dict[str, Callable[..., "DomainModule"]] = {}


def register_domain(name: str, factory: Callable[..., "DomainModule"]):
    """Register a domain factory function."""
    _DOMAIN_REGISTRY[name.lower()] = factory


def get_domain(name: str, **kwargs) -> "DomainModule":
    """Create a DomainModule by registered name."""
    key = name.lower()
    if key not in _DOMAIN_REGISTRY:
        available = list(_DOMAIN_REGISTRY.keys())
        raise KeyError(f"Unknown domain: '{name}'. Available: {available}")
    return _DOMAIN_REGISTRY[key](**kwargs)


def list_domains() -> list:
    """Return names of all registered domains."""
    return list(_DOMAIN_REGISTRY.keys())


# Auto-register built-in domains
try:
    from skillnet.domains.minecraft import MinecraftDomain
    register_domain("minecraft", MinecraftDomain)
except ImportError:
    pass
