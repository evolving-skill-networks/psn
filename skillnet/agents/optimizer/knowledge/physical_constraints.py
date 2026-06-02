"""
Physical Constraint Types

Domain-agnostic data types for constraint violations.
The constraint checking framework was removed — it was dead code
(instantiated but never called from production). ConstraintViolation and
ConstraintType are retained as general-purpose constraint types.
"""

from dataclasses import dataclass
from typing import Any
from enum import Enum


class ConstraintType(Enum):
    """Constraint type"""
    DEPTH = "depth"
    BIOME = "biome"
    TIME = "time"
    DISTANCE = "distance"
    TOOL = "tool"


@dataclass
class ConstraintViolation:
    """Constraint violation result"""
    constraint_type: ConstraintType
    severity: str  # "critical", "high", "medium", "low"
    description: str
    current_value: Any
    required_value: Any
    fix_suggestion: str
    evidence: str
