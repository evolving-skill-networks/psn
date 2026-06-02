"""Normalized observation produced by DomainKnowledge.extract_observation.

A bridge type between the env-specific event stream (which each domain
emits in its own shape) and PSN's domain-agnostic core agents.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Observation:
    """Universal cross-domain observation.

    All fields are optional or have sensible defaults so domains can populate
    only what they have. PSN core agents read these fields directly; they
    DO NOT index raw event payloads.

    Domain-specific data that doesn't fit the universal fields lives in
    ``extra`` (e.g. Minecraft's voxels/biome/entities, a dict-event domain's raw
    achievement counts).
    """
    inventory: Dict[str, int] = field(default_factory=dict)
    """Item name -> count. Universal."""

    milestones_unlocked: List[str] = field(default_factory=list)
    """Names of unlocked milestones/achievements. Universal."""

    health: Optional[float] = None
    """Bot's current health (None if domain doesn't track health)."""

    food: Optional[float] = None
    """Bot's current food level (None if domain doesn't track food/hunger)."""

    position: Optional[Tuple[float, ...]] = None
    """Bot's current position. Domain-specific shape:
    (x, y, z) for Minecraft, (x, y) for a 2D dict-event domain."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Overflow for domain-specific fields. Minecraft: voxels, biome,
    entities, blockRecords, equipment, inventoryUsed, timeOfDay.
    A dict-event domain: achievements (raw count dict), daylight, drink, energy,
    facing_direction."""
