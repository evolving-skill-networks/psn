"""
Forward Propagation Data Types

Data classes for optimization result communication:
- InterfaceChange: Describes parameter/return changes
- EffectChange: Describes skill effect changes
- OptimizationForwardFeedback: Full optimization result for parent visibility
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class InterfaceChange:
    """Description of an interface change."""
    change_type: str  # "parameter_added", "parameter_removed", "parameter_modified", "return_changed"
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    description: str = ""

    @property
    def requires_parent_update(self) -> bool:
        """Whether the parent skill must update its call site."""
        # Adding/removing/modifying parameters all require parent updates
        return self.change_type in (
            "parameter_added", "parameter_removed", "parameter_modified", "return_changed"
        )


@dataclass
class EffectChange:
    """Description of an effect change."""
    change_type: str  # "added", "removed", "modified"
    effect_description: str = ""
    old_effect: Optional[str] = None
    new_effect: Optional[str] = None

    @property
    def may_affect_parent(self) -> bool:
        """Whether this change may affect the parent skill."""
        # Removing or modifying an effect may affect the parent
        return self.change_type in ("removed", "modified")

    @property
    def impact_description(self) -> str:
        """Impact description."""
        return self.effect_description or f"{self.change_type}: {self.old_effect} -> {self.new_effect}"


@dataclass
class OptimizationForwardFeedback:
    """
    Forward-propagating optimization result (full version, unified definition).

    When child skill C finishes optimization, parent skill B needs to receive:
    1. Base state: whether the optimization succeeded.
    2. Modification details: what was actually changed.
    3. Interface changes: whether parameter signature or return value changed.
    4. Effect changes: whether the skill's effects changed.
    5. Suggestions for the parent: adjustments the parent may need.

    NOTE: This is the sole definition of OptimizationForwardFeedback.
    All modules should import from here and not redeclare it.
    """
    skill_name: str
    optimization_successful: bool

    # Modification details
    changes_made: List[str] = field(default_factory=list)
    code_diff: Optional[str] = None  # Code diff (optional)
    new_code: str = ""               # New code after optimization (used for persistence)
    old_code: str = ""               # Old code before optimization (used for rollback)

    # Interface changes
    interface_changed: bool = False
    interface_changes: List[InterfaceChange] = field(default_factory=list)
    old_signature: Optional[Dict[str, Any]] = None
    new_signature: Optional[Dict[str, Any]] = None

    # Effect changes
    effects_changed: bool = False
    effect_changes: List[EffectChange] = field(default_factory=list)

    # Suggestions for the parent skill
    parent_suggestions: List[str] = field(default_factory=list)
    parent_warnings: List[str] = field(default_factory=list)
    required_parent_updates: List[str] = field(default_factory=list)  # Mandatory updates

    # Metadata
    optimization_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Linkage info
    parent_skill: Optional[str] = None  # The parent skill that triggered this optimization
    task: Optional[str] = None

    # Phase 13.B-9: trigger context for detailed-log diagnostics.
    # Populated by engine.py before invoking on_skill_optimized /
    # on_optimization_failed callbacks. The optimization_tracker writes
    # these to opt_*.json so future runs can answer "what did Phase 1
    # actually see when it produced this feedback?".
    error_message: Optional[str] = None      # current execution error
    error_stack: Optional[str] = None        # per-skill execution error stack
    llm_prompt: Optional[str] = None         # Phase 1 prompt (if captured)
    llm_response: Optional[str] = None       # Phase 1 raw response (if captured)
    execution_context: Optional[Dict[str, Any]] = None  # bot/inventory state

    # P(update s) skip info for parent visibility
    analysis_available: bool = False    # True if Phase 1 analysis was completed
    skipped_reason: str = ""            # Non-empty when optimization was skipped by P(update s)

    @property
    def has_breaking_changes(self) -> bool:
        """Whether there are breaking changes."""
        # Check interface changes
        for c in self.interface_changes:
            if c.change_type in ("parameter_added", "parameter_removed", "parameter_modified"):
                return True
        # Check effect changes
        for c in self.effect_changes:
            if c.change_type in ("removed", "modified"):
                return True
        return False

    @property
    def impact_level(self) -> str:
        """Impact level."""
        if not self.optimization_successful:
            return "failed"
        if self.has_breaking_changes:
            return "breaking"
        if self.interface_changed or self.effects_changed:
            return "moderate"
        return "minimal"


