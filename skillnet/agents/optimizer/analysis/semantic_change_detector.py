"""
Semantic Change Detector

Detects parameter-semantic changes, especially TARGET_TOTAL <-> DELTA
breaking changes.

Breaking changes:
- TARGET_TOTAL -> DELTA: caller expects "ensure there are N" but actually
  gets "add N".
- DELTA -> TARGET_TOTAL: caller expects "add N" but the action may be
  skipped (if N is already met).

These semantic changes are not caught by structural interface checks
(parameter names/count remain the same), but they cause runtime behavior
to diverge from caller expectations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from skillnet.agents.planning.inference.parameter_semantic import (
    QuantitySemantic,
    DirectionSemantic,
    ParameterSemantic,
)


@dataclass
class SemanticChange:
    """Semantic change for a single parameter."""
    param_name: str
    old_quantity: Optional[QuantitySemantic] = None
    new_quantity: Optional[QuantitySemantic] = None
    old_direction: Optional[DirectionSemantic] = None
    new_direction: Optional[DirectionSemantic] = None
    is_breaking: bool = False
    description: str = ""

    def __str__(self) -> str:
        parts = [f"Parameter '{self.param_name}'"]
        if self.old_quantity and self.new_quantity and self.old_quantity != self.new_quantity:
            parts.append(f"quantity: {self.old_quantity.value} -> {self.new_quantity.value}")
        if self.old_direction and self.new_direction and self.old_direction != self.new_direction:
            parts.append(f"direction: {self.old_direction.value} -> {self.new_direction.value}")
        if self.is_breaking:
            parts.append("[BREAKING]")
        return " | ".join(parts)


@dataclass
class SemanticChangeResult:
    """Result of semantic-change detection."""
    changes: List[SemanticChange] = field(default_factory=list)
    has_breaking_change: bool = False

    @property
    def breaking_changes(self) -> List[SemanticChange]:
        """Return all breaking changes."""
        return [c for c in self.changes if c.is_breaking]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict form."""
        return {
            "changes": [
                {
                    "param_name": c.param_name,
                    "old_quantity": c.old_quantity.value if c.old_quantity else None,
                    "new_quantity": c.new_quantity.value if c.new_quantity else None,
                    "old_direction": c.old_direction.value if c.old_direction else None,
                    "new_direction": c.new_direction.value if c.new_direction else None,
                    "is_breaking": c.is_breaking,
                    "description": c.description,
                }
                for c in self.changes
            ],
            "has_breaking_change": self.has_breaking_change,
        }


def detect_semantic_changes(
    old_metadata: Dict[str, Dict[str, Any]],
    new_metadata: Dict[str, Dict[str, Any]],
) -> SemanticChangeResult:
    """
    Detect parameter semantic changes.

    Args:
        old_metadata: old version parameter metadata {param_name: {type, semantic, ...}}
        new_metadata: new version parameter metadata {param_name: {type, semantic, ...}}

    Returns:
        SemanticChangeResult: detection result including all changes and whether any are breaking.
    """
    changes = []

    # Get all parameter names (exclude bot)
    all_params = set(old_metadata.keys()) | set(new_metadata.keys())
    all_params.discard("bot")

    for param_name in all_params:
        old_info = old_metadata.get(param_name, {})
        new_info = new_metadata.get(param_name, {})

        # Parse semantic info (compatible with old and new formats)
        old_sem = _parse_semantic(old_info.get("semantic"))
        new_sem = _parse_semantic(new_info.get("semantic"))

        # Check quantity semantic change
        quantity_changed = (
            old_sem and new_sem and
            old_sem.quantity_semantic != new_sem.quantity_semantic and
            old_sem.quantity_semantic != QuantitySemantic.NONE and
            new_sem.quantity_semantic != QuantitySemantic.NONE
        )

        # Check direction semantic change
        direction_changed = (
            old_sem and new_sem and
            old_sem.direction != new_sem.direction and
            old_sem.direction != DirectionSemantic.CONFIG and
            new_sem.direction != DirectionSemantic.CONFIG
        )

        if quantity_changed or direction_changed:
            is_breaking = False
            description_parts = []

            if quantity_changed:
                is_breaking = _is_quantity_change_breaking(
                    old_sem.quantity_semantic,
                    new_sem.quantity_semantic
                )
                description_parts.append(
                    f"quantity semantic: {old_sem.quantity_semantic.value} -> {new_sem.quantity_semantic.value}"
                )

            if direction_changed:
                # INPUT <-> OUTPUT is a breaking change
                if _is_direction_change_breaking(old_sem.direction, new_sem.direction):
                    is_breaking = True
                description_parts.append(
                    f"direction: {old_sem.direction.value} -> {new_sem.direction.value}"
                )

            changes.append(SemanticChange(
                param_name=param_name,
                old_quantity=old_sem.quantity_semantic if old_sem else None,
                new_quantity=new_sem.quantity_semantic if new_sem else None,
                old_direction=old_sem.direction if old_sem else None,
                new_direction=new_sem.direction if new_sem else None,
                is_breaking=is_breaking,
                description="; ".join(description_parts),
            ))

    has_breaking = any(c.is_breaking for c in changes)

    return SemanticChangeResult(
        changes=changes,
        has_breaking_change=has_breaking,
    )


def _parse_semantic(semantic_data: Any) -> Optional[ParameterSemantic]:
    """
    Parse semantic data, supporting both old and new formats.

    Old format: "target_total" | "delta" | "config" (string)
    New format: {"quantity_semantic": "...", "direction": "...", ...} (dict)
    """
    if semantic_data is None:
        return None

    # String format (legacy compatibility)
    if isinstance(semantic_data, str):
        quantity = QuantitySemantic.from_string(semantic_data)
        return ParameterSemantic(
            quantity_semantic=quantity,
            _direction_fallback=DirectionSemantic.CONFIG,
            inferred_by="legacy",
            confidence=0.5,
        )

    # Dict format (new version)
    if isinstance(semantic_data, dict):
        quantity_str = semantic_data.get("quantity_semantic", "none")
        direction_str = semantic_data.get("direction", "config")

        return ParameterSemantic(
            quantity_semantic=QuantitySemantic.from_string(quantity_str),
            _direction_fallback=DirectionSemantic.from_string(direction_str),
            transform_hint=semantic_data.get("transform_hint"),
            config_context=semantic_data.get("config_context"),
            is_fuel_param=semantic_data.get("is_fuel_param", False),
            flexible_compatible=semantic_data.get("flexible_compatible", False),
            field_directions=semantic_data.get("field_directions"),
            inferred_by=semantic_data.get("inferred_by", "unknown"),
            confidence=semantic_data.get("confidence", 0.5),
        )

    return None


def _is_quantity_change_breaking(
    old: QuantitySemantic,
    new: QuantitySemantic
) -> bool:
    """
    Decide whether a quantity-semantic change is breaking.

    Breaking pairs:
    - TARGET_TOTAL -> DELTA: caller expects "ensure there are N" but gets "add N".
    - DELTA -> TARGET_TOTAL: caller expects "add N" but it may be skipped
      (when there are already enough).
    """
    breaking_pairs = {
        (QuantitySemantic.TARGET_TOTAL, QuantitySemantic.DELTA),
        (QuantitySemantic.DELTA, QuantitySemantic.TARGET_TOTAL),
    }
    return (old, new) in breaking_pairs


def _is_direction_change_breaking(
    old: DirectionSemantic,
    new: DirectionSemantic
) -> bool:
    """
    Decide whether a direction-semantic change is breaking.

    Breaking pairs:
    - INPUT -> OUTPUT: parameter shifts from "input material" to "output product".
    - OUTPUT -> INPUT: parameter shifts from "output product" to "input material".
    """
    breaking_pairs = {
        (DirectionSemantic.INPUT, DirectionSemantic.OUTPUT),
        (DirectionSemantic.OUTPUT, DirectionSemantic.INPUT),
    }
    return (old, new) in breaking_pairs
