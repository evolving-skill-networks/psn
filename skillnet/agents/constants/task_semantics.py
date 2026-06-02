"""Task semantic type constants - shared across PSN components

Semantic Consistency Design Notes:
==================================

The PSN system uses an explicit-semantic design; all components share a unified TARGET semantic:

1. When PSN Curriculum generates tasks:
   - ResourceTracker computes the target based on a threshold system
   - For example: target_count=26, current=5
   - Generates the task "Ensure you have 26 wood logs" (explicit TARGET semantic)

2. Graph Planner selects a skill:
   - Recognizes the "Ensure" verb -> directly interpreted as TARGET semantic
   - Selects the ensureLogs skill (naming-consistent)

3. PSN Critic evaluates the result:
   - Recognizes the "Ensure" verb -> directly uses TARGET-semantic evaluation
   - Verifies that the final inventory >= target count

Design principles:
- Front-end explicit semantics: PSN Curriculum uses the "Ensure you have" sentence pattern to explicitly express TARGET intent
- Naming consistency: the task "Ensure you have X logs" <-> the ensureLogs skill
- Back-end inference as fallback: compatible with legacy tasks or LLM fallback scenarios

Semantic flow (explicit):
  PSN Curriculum ("Ensure you have 26 wood logs")
    -> Graph Planner (recognize Ensure -> TARGET)
    -> PSN Critic (recognize Ensure -> TARGET evaluation)
"""


class TaskSemanticType:
    """Task semantic type enumeration"""
    DELTA = "delta"           # Delta semantic: need to collect X new items
    TARGET_TOTAL = "target"   # Target semantic: ultimately hold X items
    FIND = "find"             # Find semantic: locate X targets (check nearby blocks)
    EQUIP = "equip"           # Equip semantic: wear armor/equipment
    PLACE = "place"           # Place semantic: place items as world blocks (check nearby blocks)


# Semantic detection keyword mapping
# PSN Curriculum now uniformly uses the "Ensure" verb to generate tasks, explicitly expressing TARGET semantic
# Back-end inference logic is retained as a fallback, compatible with legacy tasks
DELTA_KEYWORDS = ["mine", "collect", "harvest", "kill", "catch", "hunt"]
TARGET_KEYWORDS = ["ensure", "have", "gather", "get", "obtain", "prepare"]
FIND_KEYWORDS = ["find", "locate", "discover", "search", "explore"]
EQUIP_KEYWORDS = ["equip", "wear", "put on"]
PLACE_KEYWORDS = ["place", "put", "set up", "setup"]

# Skill-name semantic inference (highest priority)
# These skill names indicate use of TARGET semantic
ENSURE_SKILL_PATTERNS = ["ensure", "prepare", "setup", "get_ready"]


def detect_task_semantic(task: str, executed_skill: str = None) -> str:
    """
    Detect the task semantic type.

    Priority:
    1. Executed skill name contains "ensure" etc. -> TARGET_TOTAL
    2. Task-leading vocabulary match -> corresponding semantic
    3. Default -> DELTA

    Important notes:
    - PSN Curriculum now uniformly generates tasks in the "Ensure you have X" format
    - This format directly expresses TARGET semantic, no back-end inference required
    - Back-end inference logic is kept for compatibility with legacy tasks or external input

    Args:
        task: Task description string
        executed_skill: Executed skill name (optional)

    Returns:
        TaskSemanticType.DELTA or TaskSemanticType.TARGET_TOTAL
    """
    # Priority 1: Skill name (the most reliable semantic indicator)
    if executed_skill:
        skill_lower = executed_skill.lower()
        for pattern in ENSURE_SKILL_PATTERNS:
            if pattern in skill_lower:
                return TaskSemanticType.TARGET_TOTAL

    # Priority 2: Task-leading vocabulary
    task_lower = task.lower().strip()

    # Priority 2.5: Find keywords (FIND semantic takes precedence over TARGET)
    for keyword in FIND_KEYWORDS:
        if task_lower.startswith(keyword):
            return TaskSemanticType.FIND

    # Priority 2.6: Equip keywords
    for keyword in EQUIP_KEYWORDS:
        if task_lower.startswith(keyword):
            return TaskSemanticType.EQUIP

    # Priority 2.7: Place keywords
    for keyword in PLACE_KEYWORDS:
        if task_lower.startswith(keyword):
            return TaskSemanticType.PLACE

    for keyword in TARGET_KEYWORDS:
        if task_lower.startswith(keyword):
            return TaskSemanticType.TARGET_TOTAL

    for keyword in DELTA_KEYWORDS:
        if task_lower.startswith(keyword):
            return TaskSemanticType.DELTA

    # Default: DELTA
    return TaskSemanticType.DELTA


def format_task_with_semantic(task: str, semantic: str, target_count: int = None) -> str:
    """Format the task description, optionally appending a semantic annotation.

    Used when PSN Curriculum generates tasks and wants to optionally annotate the semantic explicitly.

    Args:
        task: Original task description
        semantic: Semantic type (TaskSemanticType.DELTA or TARGET_TOTAL)
        target_count: Target count (used for TARGET semantic)

    Returns:
        The formatted task description

    Example:
        format_task_with_semantic("Mine 21 logs", TaskSemanticType.TARGET_TOTAL, 26)
        -> "Mine 21 logs [TARGET: ensure 26 total]"
    """
    if semantic == TaskSemanticType.TARGET_TOTAL and target_count:
        return f"{task} [TARGET: ensure {target_count} total]"
    return task


def infer_semantic_from_planning_result(
    task: str,
    plan_type: str,
    skill_sequence: list = None,
    executed_skills: list = None
) -> str:
    """Infer task semantic from a PlanningResult and execution results.

    This is the most complete semantic-inference function; it considers every possible information source.

    Priority:
    1. Executed skill name (most reliable, since it was actually executed)
    2. Planned skill sequence (returned by Graph Planner)
    3. Task-text keywords
    4. Default DELTA

    Args:
        task: Task description
        plan_type: "graph" or "llm"
        skill_sequence: List of SkillCallContext returned by Graph Planner
        executed_skills: List of skill names actually executed

    Returns:
        TaskSemanticType.DELTA or TaskSemanticType.TARGET_TOTAL
    """
    # Priority 1: Actually executed skills
    if executed_skills:
        for skill in executed_skills:
            skill_lower = skill.lower()
            for pattern in ENSURE_SKILL_PATTERNS:
                if pattern in skill_lower:
                    return TaskSemanticType.TARGET_TOTAL

    # Priority 2: Planned skill sequence (Graph Planner)
    if skill_sequence:
        for ctx in skill_sequence:
            skill_name = getattr(ctx, 'skill_name', '') if hasattr(ctx, 'skill_name') else str(ctx)
            skill_lower = skill_name.lower()
            for pattern in ENSURE_SKILL_PATTERNS:
                if pattern in skill_lower:
                    return TaskSemanticType.TARGET_TOTAL

    # Priority 3: Task-text keywords
    return detect_task_semantic(task, None)


# Semantic-inference strategy under PSN mode
class PSNSemanticStrategy:
    """Semantic-inference strategy for the PSN system

    Under PSN mode, tasks uniformly use the TARGET semantic:
    1. ResourceTracker generates tasks based on the threshold system
    2. Tasks are formatted as "Ensure you have X items"
    3. Tasks directly express TARGET intent; evaluation checks the final inventory

    This class provides PSN-specific semantic-inference logic while remaining compatible with legacy tasks.
    """

    @staticmethod
    def should_use_target_semantic(
        task: str,
        plan_type: str = None,
        skill_sequence: list = None,
        executed_skills: list = None
    ) -> bool:
        """Decide whether TARGET semantic should be used.

        Detection rules (in priority order):
        1. Task begins with "Ensure" -> TARGET (PSN Curriculum standard format)
        2. An "ensure"-type skill was executed -> TARGET
        3. The skill sequence contains an "ensure"-type skill -> TARGET
        4. Otherwise infer from task keywords

        Args:
            task: Task description
            plan_type: "graph" or "llm"
            skill_sequence: Planned skill sequence
            executed_skills: List of executed skills

        Returns:
            True if TARGET semantic should be used
        """
        # Use the complete inference function
        semantic = infer_semantic_from_planning_result(
            task, plan_type, skill_sequence, executed_skills
        )
        return semantic == TaskSemanticType.TARGET_TOTAL

    @staticmethod
    def get_semantic_type(
        task: str,
        plan_type: str = None,
        skill_sequence: list = None,
        executed_skills: list = None
    ) -> str:
        """Return the semantic type"""
        return infer_semantic_from_planning_result(
            task, plan_type, skill_sequence, executed_skills
        )


# ========== P1: Unified semantic-contract system ==========
# The data structures below explicitly carry semantic information between components,
# eliminating the need for inference.

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union


@dataclass
class TargetEffect:
    """Structured expected effect

    Used to describe the effect a task is expected to produce, containing item, quantity, and semantic type.

    Attributes:
        item: Item name (e.g. "oak_planks", "iron_ingot")
        quantity: Count
        semantic: Semantic type (DELTA or TARGET_TOTAL)
        item_group: Item group (e.g. "logs" contains oak_log, birch_log, etc.)
    """
    item: str
    quantity: int
    semantic: str  # TaskSemanticType.DELTA or TARGET_TOTAL
    item_group: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
        return {
            "item": self.item,
            "quantity": self.quantity,
            "semantic": self.semantic,
            "item_group": self.item_group
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TargetEffect':
        """Create from dict for deserialization"""
        return cls(
            item=data.get("item", ""),
            quantity=data.get("quantity", 0),
            semantic=data.get("semantic", TaskSemanticType.DELTA),
            item_group=data.get("item_group")
        )


@dataclass
class TaskWithSemantic:
    """Task object carrying semantic information - the single source of truth for semantics

    This is the core data structure of the P1 unified semantic-contract system.
    Curriculum creates this object when generating tasks; Planner and Critic directly use the semantic information in it,
    without needing to infer.

    Design principles:
    1. Semantic defined at the source - Curriculum determines the semantic type
    2. Semantic passed explicitly - propagated between components via this object
    3. Direct use downstream - Planner and Critic no longer infer

    Attributes:
        task: Task description string (e.g. "Ensure you have 8 oak_planks")
        semantic_type: Overall task semantic type
        target_effects: List of expected effects
        context: Context information (inventory snapshot, current count, etc.)
        source: Source-component identifier
        metadata: Additional metadata
    """
    task: str
    semantic_type: str  # TaskSemanticType.DELTA or TARGET_TOTAL

    # Effect specifications
    target_effects: List[TargetEffect] = field(default_factory=list)

    # Context information
    context: Dict[str, Any] = field(default_factory=dict)
    # Common fields in context:
    # - current_inventory: Dict[str, int]  # current inventory snapshot
    # - current_count: int                  # current count of the target item
    # - delta_needed: int                   # delta required (target - current)

    # Source identifier
    source: str = "curriculum"  # "curriculum" | "planner" | "user" | "legacy_conversion"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_primary_target(self) -> Optional[TargetEffect]:
        """Return the primary target item (the first effect)"""
        return self.target_effects[0] if self.target_effects else None

    def get_target_count(self) -> int:
        """Return the target count"""
        primary = self.get_primary_target()
        return primary.quantity if primary else 0

    def get_delta_needed(self) -> int:
        """Return the required delta"""
        return self.context.get("delta_needed", self.get_target_count())

    def get_current_count(self) -> int:
        """Return the current count"""
        return self.context.get("current_count", 0)

    def to_legacy_string(self) -> str:
        """Backward compatibility: convert to legacy-format task string"""
        return self.task

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization/logging"""
        return {
            "task": self.task,
            "semantic_type": self.semantic_type,
            "target_effects": [e.to_dict() for e in self.target_effects],
            "context": self.context,
            "source": self.source,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskWithSemantic':
        """Create from dict for deserialization"""
        effects = [TargetEffect.from_dict(e) for e in data.get("target_effects", [])]
        return cls(
            task=data.get("task", ""),
            semantic_type=data.get("semantic_type", TaskSemanticType.DELTA),
            target_effects=effects,
            context=data.get("context", {}),
            source=data.get("source", "legacy_conversion"),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_legacy_string(
        cls,
        task_str: str,
        inferred_semantic: str = None,
        current_count: int = 0,
        target_count: int = 0
    ) -> 'TaskWithSemantic':
        """Backward compatibility: create from a legacy-format task string.

        Used to support incremental migration; when upstream components have not yet been updated, the object can be created from a string.

        Args:
            task_str: Task description string
            inferred_semantic: Inferred semantic type (if not provided, uses detect_task_semantic)
            current_count: Current count
            target_count: Target count

        Returns:
            A TaskWithSemantic object
        """
        semantic = inferred_semantic or detect_task_semantic(task_str)

        # Try to parse target information from the task string
        effects = []
        import re
        # Match the pattern "Ensure you have X item" or "Craft X item"
        match = re.search(r'(\d+)\s+(\w+)', task_str)
        if match:
            quantity = int(match.group(1))
            item = match.group(2)
            effects.append(TargetEffect(
                item=item,
                quantity=quantity,
                semantic=semantic
            ))
            if target_count == 0:
                target_count = quantity

        delta_needed = max(0, target_count - current_count) if semantic == TaskSemanticType.TARGET_TOTAL else target_count

        return cls(
            task=task_str,
            semantic_type=semantic,
            target_effects=effects,
            context={
                "current_count": current_count,
                "delta_needed": delta_needed,
            },
            source="legacy_conversion"
        )

    def __str__(self) -> str:
        """Friendly string representation"""
        return f"TaskWithSemantic(task='{self.task}', semantic={self.semantic_type}, source={self.source})"


# Type alias to support incremental migration.
# Components may accept str or TaskWithSemantic.
TaskOrSemantic = Union[str, TaskWithSemantic]


def is_task_with_semantic(task: TaskOrSemantic) -> bool:
    """Check whether the task is a TaskWithSemantic object"""
    return isinstance(task, TaskWithSemantic)


def ensure_task_with_semantic(
    task: TaskOrSemantic,
    current_count: int = 0,
    target_count: int = 0
) -> TaskWithSemantic:
    """Ensure the task is a TaskWithSemantic object.

    If a string is passed in, it is automatically converted to a TaskWithSemantic.
    Used at component entry points for unified handling.

    Args:
        task: Task (string or TaskWithSemantic)
        current_count: Current count (used only when task is a string)
        target_count: Target count (used only when task is a string)

    Returns:
        A TaskWithSemantic object
    """
    if isinstance(task, TaskWithSemantic):
        return task
    return TaskWithSemantic.from_legacy_string(
        task,
        current_count=current_count,
        target_count=target_count
    )
