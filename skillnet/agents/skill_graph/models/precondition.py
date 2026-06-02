"""
Dataclasses related to preconditions and effects.

Includes SkillPrecondition, SkillEffect, ActualEffect.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class SkillPrecondition:
    """Precondition: the prerequisites for a skill to be callable."""
    description: str  # Human-readable description
    code: str  # Executable code (used to check the condition)
    state_representation: Dict[str, Any] = None  # Structured state representation, same format as goal
    # Supported formats:
    # 1. Legacy (AND logic): [{"type": "inventory", "item": "oak_log", "count": 1, "operation": "require"}, ...]
    # 2. New (OR logic): {"logic": "OR", "conditions": [{"type": "inventory", "item": "oak_log", ...}, ...]}
    # 3. New (AND logic): {"logic": "AND", "conditions": [{"type": "inventory", "item": "oak_log", ...}, ...]}
    # 4. Nested: {"logic": "AND", "conditions": [..., {"logic": "OR", "conditions": [...]}, ...]}
    # operation can be: "require" (required), "require_or_better" (required or better, e.g. tool tier)

    # Value-function-related fields (used for confidence tracking)
    confidence_value: float = 0.0  # Confidence value computed by the value function
    confidence_level: str = "unknown"  # "high" / "medium" / "low" / "uncertain" / "unknown"
    inference_stats: Dict[str, Any] = None  # Inference statistics
    # Format: {
    # "n_success_with": 9,      # Successes where this condition was present
    # "n_success_total": 10,    # Total successful samples
    # "n_failure_missing": 3,   # Failures where this condition was missing
    # "n_failure_total": 5,     # Total failure samples
    # "code_verified": True,    # Whether verified in code
    # "last_updated": "2024-..."
    # }

    # Conditional support: parameter-dependent preconditions on general skills
    condition: Dict[str, Any] = None  # Condition expression; precondition only applies when parameter matches
    # Format: {"parameter_name": "value"} or {"parameter_name": ["value1", "value2"]}
    # E.g. {"targetBlockNames": "diamond_ore"} means the precondition only applies when targetBlockNames="diamond_ore"
    # If None, the precondition is unconditional (applies for all parameter values)
    source_skill: str = None  # Origin skill (recorded during refactor for traceability)

    def __init__(self, description: str = "", code: str = "", state_representation: Dict[str, Any] = None,
                 confidence_value: float = 0.0, confidence_level: str = "unknown",
                 inference_stats: Dict[str, Any] = None,
                 condition: Dict[str, Any] = None, source_skill: str = None):
        # Root null-safety: LLM-extracted preconditions can arrive with
        # ``"description": null`` or ``"code": null`` in the JSON, which
        # propagate through dict.get() as None. Downstream callers call
        # ``.lower()`` / ``[:80]`` on these fields and crash. Coerce once
        # at construction so the str type hint is actually upheld.
        self.description = description or ""
        self.code = code or ""
        self.state_representation = state_representation or {}
        self.confidence_value = confidence_value
        self.confidence_level = confidence_level
        self.inference_stats = inference_stats or {}
        self.condition = condition
        self.source_skill = source_skill


@dataclass
class SkillEffect:
    """Expected effect: the expected key effects after a skill succeeds."""
    description: str  # Human-readable description
    code: str  # Executable code (used to verify the effect)
    state_representation: Dict[str, Any] = None  # Structured state representation, same format as goal
    # Supported formats:
    # 1. Legacy (list, default OR logic): [{"type": "inventory", "item": "oak_log", "count": 3, "operation": "add"}, ...]
    # 2. New (OR logic): {"logic": "OR", "conditions": [{"type": "inventory", "item": "oak_log", ...}, ...]}
    # 3. New (AND logic): {"logic": "AND", "conditions": [{"type": "inventory", "item": "oak_log", ...}, ...]}
    # 4. Nested: {"logic": "OR", "conditions": [..., {"logic": "AND", "conditions": [...]}, ...]}
    # operation can be: "add", "remove", "place", "equip"

    # Value-function-related fields (used for confidence tracking)
    confidence_value: float = 0.0  # Confidence value computed by the value function
    confidence_level: str = "unknown"  # "high" / "medium" / "low" / "uncertain" / "unknown"
    importance: str = "secondary"  # "core" / "secondary" / "incidental"
    is_primary: bool = False  # Whether this is the skill's primary effect (directly related to the function name/task goal)
    # is_primary=True effects are the skill's core output (e.g. crafting_table for craftOakCraftingTable)
    # is_primary=False effects are intermediates or by-products (e.g. oak_planks produced along the way)
    # EffectMatcher should prefer/match only is_primary=True effects
    inference_stats: Dict[str, Any] = None  # Inference statistics
    # Format: {
    # "n_occurrences": 9,       # Number of occurrences
    # "n_total": 10,            # Total sample count
    # "avg_change": 3.5,        # Average change magnitude
    # "last_updated": "2024-..."
    # }

    # Conditional support: parameter-dependent effects on general skills
    condition: Dict[str, Any] = None  # Condition expression; effect only applies when parameter matches
    # Same format as SkillPrecondition.condition
    source_skill: str = None  # Origin skill (recorded during refactor for traceability)

    def __init__(self, description: str = "", code: str = "", state_representation: Dict[str, Any] = None,
                 confidence_value: float = 0.0, confidence_level: str = "unknown",
                 importance: str = "secondary", is_primary: bool = False,
                 inference_stats: Dict[str, Any] = None,
                 condition: Dict[str, Any] = None, source_skill: str = None):
        # Root null-safety (same rationale as SkillPrecondition).
        self.description = description or ""
        self.code = code or ""
        self.state_representation = state_representation or {}
        self.confidence_value = confidence_value
        self.confidence_level = confidence_level
        self.importance = importance
        self.is_primary = is_primary
        self.inference_stats = inference_stats or {}
        self.condition = condition
        self.source_skill = source_skill


@dataclass
class ActualEffect:
    """Actual effect: the real state changes produced by a skill execution."""
    execution_id: str  # Associated execution ID
    timestamp: str  # Execution timestamp
    description: str  # Human-readable description (auto-generated)
    state_changes: List[Dict[str, Any]] = None  # Structured list of state changes (same format as goal)
    # Format: [{"type": "inventory", "item": "oak_log", "count": 3, "operation": "add"}, ...]
    # Per-category change projections (read directly by effect-verification
    # and the critic's state-delta logic; state_changes is the high-level list).
    inventory_changes: Dict[str, int] = None  # Item changes {item_name: count_change}
    position_changes: Dict[str, float] = None  # Position changes {x, y, z}
    equipment_changes: Dict[str, Any] = None  # Equipment changes
    block_changes: List[Dict[str, Any]] = None  # Block changes
    pre_state: Dict[str, Any] = None  # State before execution
    post_state: Dict[str, Any] = None  # State after execution
    verified: bool = False  # Whether verified (via expected_effects code)

    def __init__(
        self,
        execution_id: str = None,
        timestamp: str = None,
        description: str = "",
        state_changes: List[Dict[str, Any]] = None,
        inventory_changes: Dict[str, int] = None,
        position_changes: Dict[str, float] = None,
        equipment_changes: Dict[str, Any] = None,
        block_changes: List[Dict[str, Any]] = None,
        pre_state: Dict[str, Any] = None,
        post_state: Dict[str, Any] = None,
        verified: bool = False,
    ):
        self.execution_id = execution_id or f"effect_{datetime.now().timestamp()}"
        self.timestamp = timestamp or datetime.now().isoformat()
        self.description = description
        self.state_changes = state_changes or []
        self.inventory_changes = inventory_changes or {}
        self.position_changes = position_changes or {}
        self.equipment_changes = equipment_changes or {}
        self.block_changes = block_changes or []
        self.pre_state = pre_state or {}
        self.post_state = post_state or {}
        self.verified = verified
