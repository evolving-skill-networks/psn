"""
Parameter Semantic Data Classes

This module defines the core data structures for parameter semantic analysis:
- QuantitySemantic: Whether a count parameter is target_total or delta
- DirectionSemantic: Whether a type parameter refers to input or output
- TransformHint: How to transform between input/output item types
- StateType: Types in the structured state space
- StateMapping: Canonical mapping from a parameter to a state-space field
- ParameterSemantic: Complete semantic information for a parameter
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import re
import json


class QuantitySemantic(Enum):
    """
    Semantic type for quantity/count parameters.

    - TARGET_TOTAL: Ensure at least N items exist (e.g., ensureLogs(count=8) means "have at least 8 logs")
    - DELTA: Produce N additional items (e.g., craftPlanks(count=8) means "craft 8 more planks")
    - NONE: Not a quantity parameter
    """
    TARGET_TOTAL = "target_total"
    DELTA = "delta"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> "QuantitySemantic":
        """Convert string to enum, with fallback to NONE."""
        if not value:
            return cls.NONE
        value_lower = value.lower()
        if value_lower in ("target_total", "target", "total", "ensure"):
            return cls.TARGET_TOTAL
        elif value_lower in ("delta", "additional", "craft", "produce"):
            return cls.DELTA
        return cls.NONE


class DirectionSemantic(Enum):
    """
    Semantic type for item type parameters.

    - INPUT: The parameter specifies input/source material (e.g., logType in craftPlanks)
    - OUTPUT: The parameter specifies output/product type (e.g., plankType in ensurePlanks)
    - CONFIG: The parameter is a configuration option, not an item type
    - BIDIRECTIONAL: Context-dependent, could be either input or output
    """
    INPUT = "input"
    OUTPUT = "output"
    CONFIG = "config"
    BIDIRECTIONAL = "bidirectional"

    @classmethod
    def from_string(cls, value: str) -> "DirectionSemantic":
        """Convert string to enum, with fallback to CONFIG."""
        if not value:
            return cls.CONFIG
        value_lower = value.lower()
        if value_lower in ("input", "source", "material", "raw"):
            return cls.INPUT
        elif value_lower in ("output", "product", "result", "target"):
            return cls.OUTPUT
        elif value_lower in ("bidirectional", "both", "either"):
            return cls.BIDIRECTIONAL
        return cls.CONFIG


@dataclass
class TransformHint:
    """
    Hints for transforming between input and output item types.

    Supports multiple transformation strategies:
    - suffix_replace: Replace suffix (e.g., oak_planks -> oak_log)
    - regex: Use regex pattern matching (e.g., deepslate_iron_ore -> raw_iron)
    - lookup: Direct lookup table (e.g., stick -> oak_planks)
    - chain: Apply multiple transforms in sequence
    """
    transform_type: str = "none"  # "suffix_replace" | "regex" | "lookup" | "chain" | "none"

    # For suffix_replace
    source_suffix: Optional[str] = None
    target_suffix: Optional[str] = None

    # For regex (solves the deepslate_* problem)
    pattern: Optional[str] = None
    replacement: Optional[str] = None  # Can use {1}, {2} for groups

    # For lookup (special cases)
    lookup_table: Dict[str, str] = field(default_factory=dict)

    # For chain (multi-step transform)
    chain: Optional[List["TransformHint"]] = None

    def apply(self, item: str) -> Optional[str]:
        """
        Apply the transform to an item name.

        Returns the transformed item name, or None if transform doesn't apply.
        """
        if self.transform_type == "none":
            return None

        elif self.transform_type == "suffix_replace":
            if self.source_suffix and item.endswith(self.source_suffix):
                base = item[:-len(self.source_suffix)]
                return base + (self.target_suffix or "")
            return None

        elif self.transform_type == "regex":
            if self.pattern:
                match = re.match(self.pattern, item)
                if match:
                    if self.replacement:
                        # Support {1}, {2} style group references
                        result = self.replacement
                        for i, group in enumerate(match.groups(), 1):
                            result = result.replace(f"{{{i}}}", group or "")
                        return result
                    return match.group(1) if match.groups() else item
            return None

        elif self.transform_type == "lookup":
            return self.lookup_table.get(item)

        elif self.transform_type == "chain":
            if self.chain:
                result = item
                for hint in self.chain:
                    transformed = hint.apply(result)
                    if transformed:
                        result = transformed
                    else:
                        return None  # Chain broken
                return result
            return None

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        result = {"transform_type": self.transform_type}

        if self.transform_type == "suffix_replace":
            result["source_suffix"] = self.source_suffix
            result["target_suffix"] = self.target_suffix
        elif self.transform_type == "regex":
            result["pattern"] = self.pattern
            result["replacement"] = self.replacement
        elif self.transform_type == "lookup":
            result["lookup_table"] = self.lookup_table
        elif self.transform_type == "chain":
            result["chain"] = [h.to_dict() for h in (self.chain or [])]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformHint":
        """Deserialize from dictionary."""
        if not data:
            return cls()

        transform_type = data.get("transform_type", "none")

        if transform_type == "suffix_replace":
            return cls(
                transform_type=transform_type,
                source_suffix=data.get("source_suffix"),
                target_suffix=data.get("target_suffix"),
            )
        elif transform_type == "regex":
            return cls(
                transform_type=transform_type,
                pattern=data.get("pattern"),
                replacement=data.get("replacement"),
            )
        elif transform_type == "lookup":
            return cls(
                transform_type=transform_type,
                lookup_table=data.get("lookup_table", {}),
            )
        elif transform_type == "chain":
            chain_data = data.get("chain", [])
            return cls(
                transform_type=transform_type,
                chain=[cls.from_dict(h) for h in chain_data],
            )

        return cls(transform_type=transform_type)


class StateType(Enum):
    """State types in the structured state space.

    Defines the categories of world state that skills can read/modify.
    Used by StateMapping to specify what kind of state a parameter maps to.
    """
    INVENTORY = "inventory"
    BLOCK = "block"
    EQUIPMENT = "equipment"
    NEARBY_BLOCK = "nearby_block"
    BIOME_RESOURCE = "biome_resource"
    POSITION = "position"
    ENTITY = "entity"
    ENVIRONMENT = "environment"


@dataclass
class StateMapping:
    """Maps a parameter to a field in the structured state space.

    This is the canonical representation for parameter semantics.
    Direction (INPUT/OUTPUT/CONFIG) is derived from ``source``.

    Attributes:
        source: Where the parameter's value comes from at planning time.
            "effect" = skill output, "precondition" = skill input, "config" = configuration.
        field: Which state field the parameter corresponds to (e.g. "item", "count").
        state_type: What kind of state this maps to (default "inventory").
        transform: Optional named transform from the registry
            (e.g. "infer_input", "extract_ore_base").
    """
    source: str
    field: str
    state_type: str = "inventory"
    transform: Optional[str] = None

    @property
    def derived_direction(self) -> "DirectionSemantic":
        """Derive direction from source."""
        if self.source == "precondition":
            return DirectionSemantic.INPUT
        elif self.source == "effect":
            return DirectionSemantic.OUTPUT
        return DirectionSemantic.CONFIG

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"source": self.source, "field": self.field}
        if self.state_type != "inventory":
            d["state_type"] = self.state_type
        if self.transform:
            d["transform"] = self.transform
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["StateMapping"]:
        if not data or not isinstance(data, dict):
            return None
        return cls(
            source=data.get("source", "config"),
            field=data.get("field", "item"),
            state_type=data.get("state_type", "inventory"),
            transform=data.get("transform"),
        )


@dataclass
class ParameterSemantic:
    """
    Complete semantic information for a parameter.

    ``state_mapping`` is the canonical representation — ``direction`` is a
    derived property (from ``state_mapping.source``).  When ``state_mapping``
    is ``None`` (backward compat), ``_direction_fallback`` is used instead.

    Attributes:
        state_mapping: Canonical mapping to structured state space.
        quantity_semantic: Numeric interpretation (target_total vs delta).
        _direction_fallback: Stored direction for old metadata without state_mapping.
        transform_hint: Legacy transform specification (used when state_mapping is absent).
    """

    # ===== PRIMARY: state-space mapping =====
    state_mapping: Optional[StateMapping] = None

    # ===== Numeric semantics (orthogonal to state mapping) =====
    quantity_semantic: QuantitySemantic = QuantitySemantic.NONE

    # ===== Backward compat (used only when state_mapping is None) =====
    _direction_fallback: DirectionSemantic = DirectionSemantic.CONFIG
    transform_hint: Optional[TransformHint] = None

    # ===== Special handling flags =====
    config_context: Optional[str] = None  # "fuel" | "log" | "tool" | "block" | None
    is_fuel_param: bool = False
    flexible_compatible: bool = True

    # ===== Object parameter support =====
    field_directions: Optional[Dict[str, DirectionSemantic]] = None

    # ===== Metadata =====
    inferred_by: str = "unknown"  # "llm" | "parameter_extractor" | "parameter_engine" | etc.
    confidence: float = 0.0  # 0.0 to 1.0

    @property
    def direction(self) -> DirectionSemantic:
        """Direction derived from state_mapping; falls back to stored value."""
        if self.state_mapping:
            return self.state_mapping.derived_direction
        return self._direction_fallback

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        result: Dict[str, Any] = {
            "quantity_semantic": self.quantity_semantic.value,
            "inferred_by": self.inferred_by,
            "confidence": self.confidence,
        }

        if self.state_mapping:
            result["state_mapping"] = self.state_mapping.to_dict()
        # Always emit direction for downstream readers / backward compat
        result["direction"] = self.direction.value

        if self.transform_hint:
            result["transform_hint"] = self.transform_hint.to_dict()

        if self.config_context:
            result["config_context"] = self.config_context

        if self.is_fuel_param:
            result["is_fuel_param"] = True

        if not self.flexible_compatible:
            result["flexible_compatible"] = False

        if self.field_directions:
            result["field_directions"] = {
                k: v.value for k, v in self.field_directions.items()
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSemantic":
        """
        Deserialize from dictionary.

        Handles:
        - Old string format ("target_total")
        - Old dict format (direction field, no state_mapping)
        - New dict format (state_mapping canonical)
        """
        if not data:
            return cls()

        # Handle old format: semantic was just a string like "target_total"
        if isinstance(data, str):
            return cls(
                quantity_semantic=QuantitySemantic.from_string(data),
                _direction_fallback=DirectionSemantic.CONFIG,
                inferred_by="legacy_migration",
                confidence=0.5,
            )

        # Parse state_mapping (canonical when present)
        state_mapping = StateMapping.from_dict(data.get("state_mapping"))

        # Parse quantity_semantic
        qs_value = data.get("quantity_semantic", "none")
        if isinstance(qs_value, str):
            quantity_semantic = QuantitySemantic.from_string(qs_value)
        else:
            quantity_semantic = QuantitySemantic.NONE

        # Backward compat: read direction only when no state_mapping
        direction_fallback = DirectionSemantic.CONFIG
        if not state_mapping:
            dir_value = data.get("direction", "config")
            if isinstance(dir_value, str):
                direction_fallback = DirectionSemantic.from_string(dir_value)

        # Parse transform_hint
        transform_hint = None
        if "transform_hint" in data and data["transform_hint"]:
            transform_hint = TransformHint.from_dict(data["transform_hint"])

        # Parse field_directions
        field_directions = None
        if "field_directions" in data and data["field_directions"]:
            field_directions = {
                k: DirectionSemantic.from_string(v)
                for k, v in data["field_directions"].items()
            }

        return cls(
            state_mapping=state_mapping,
            quantity_semantic=quantity_semantic,
            _direction_fallback=direction_fallback,
            transform_hint=transform_hint,
            config_context=data.get("config_context"),
            is_fuel_param=data.get("is_fuel_param", False),
            flexible_compatible=data.get("flexible_compatible", True),
            field_directions=field_directions,
            inferred_by=data.get("inferred_by", "unknown"),
            confidence=data.get("confidence", 0.0),
        )

    def is_quantity_param(self) -> bool:
        """Check if this is a quantity parameter."""
        return self.quantity_semantic != QuantitySemantic.NONE

