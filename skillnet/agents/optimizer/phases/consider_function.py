"""
Consider Function - formalized Consider implementation.

Implements the core formula in Bottom-Up optimization:
    B := B + Consider(δ_B, [opt_forward_feedback(C) for C in FaultySubSkills(B)])

Core idea:
The Consider function "considers" the optimization results of child skills and
adjusts the optimization strategy of the current skill accordingly.

Mathematical meaning:
- δ_B is the modification suggestion for B from the Top-Down analysis
- opt_forward_feedback(C) is the feedback produced after optimizing child skill C
- The Consider function computes the final modification direction, taking into account:
  1. Whether the child skill's interface has changed
  2. Whether the child skill's effects have changed
  3. How to adjust δ_B to accommodate these changes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime

from .pure_reflection import SkillDelta, Gradient, GradientType

# Import Forward Feedback types from the unified location
# NOTE: OptimizationForwardFeedback's only definition lives in forward_propagation.py
from ..feedback.forward_propagation import (
    InterfaceChange,
    EffectChange,
    OptimizationForwardFeedback,
)

# ============================================================
# Part 2: Consider result
# ============================================================

@dataclass
class DeltaAdjustment:
    """Delta adjustment entry."""
    original_gradient: Gradient
    adjusted_gradient: Gradient
    adjustment_reason: str

    @property
    def magnitude_change(self) -> float:
        """Magnitude delta."""
        return self.adjusted_gradient.magnitude - self.original_gradient.magnitude


@dataclass
class DependencyUpdate:
    """Dependency-update entry."""
    child_skill: str
    update_type: str          # "call_signature", "error_handling", "return_processing"
    description: str
    code_suggestion: str      # Suggested code modification
    priority: int = 1         # Priority (1 is highest)


@dataclass
class ConsiderResult:
    """
    Output of the Consider function.

    Contains:
    1. The adjusted delta (after considering child skill changes)
    2. Dependency calls that need to be updated
    3. Interface changes that need to be adapted
    """
    # Adjusted delta
    adjusted_delta: SkillDelta

    # Delta adjustment details
    adjustments: List[DeltaAdjustment]

    # Dependency updates
    dependency_updates: List[DependencyUpdate]

    # Overall recommendation
    overall_impact: str         # minimal, moderate, breaking
    action_summary: str         # Action summary
    confidence: float = 0.5

    @property
    def needs_significant_changes(self) -> bool:
        """Whether significant changes are needed."""
        return (
            self.overall_impact in ["moderate", "breaking"] or
            len(self.dependency_updates) > 0
        )


# ============================================================
# Part 3: Consider Function implementation
# ============================================================

class ConsiderFunction:
    """
    Formalized Consider function.

    Implements: Consider(δ_B, [opt_forward_feedback(C) for C in FaultySubSkills(B)])

    Core algorithm:
    1. Analyze child-skill changes
    2. Identify adaptations required in the parent skill
    3. Adjust the delta to reflect these needs
    4. Generate concrete update suggestions

    Usage:
        consider = ConsiderFunction()
        result = consider.consider(delta, child_feedbacks)

        # result contains:
        # - adjusted_delta: adjusted modification suggestions
        # - dependency_updates: dependency calls that need updating
        # - overall_impact: overall impact assessment
    """

    def __init__(self, logger=None):
        self.logger = logger

    def consider(
        self,
        delta: SkillDelta,
        child_forward_feedbacks: List[OptimizationForwardFeedback],
        parent_code: Optional[str] = None,
    ) -> ConsiderResult:
        """
        Execute the Consider operation.

        Args:
            delta: Modification suggestions from Top-Down analysis (δ_B)
            child_forward_feedbacks: Optimization-result feedback from child skills
            parent_code: Parent skill's code (optional, used to generate more precise suggestions)

        Returns:
            ConsiderResult: Result of the Consider operation
        """
        # Step 1: analyze child-skill changes
        interface_changes = self._collect_interface_changes(child_forward_feedbacks)
        effect_changes = self._collect_effect_changes(child_forward_feedbacks)

        # Step 2: identify required dependency updates
        dependency_updates = self._identify_dependency_updates(
            child_forward_feedbacks,
            parent_code,
        )

        # Step 3: adjust the delta
        adjusted_delta, adjustments = self._adjust_delta(
            delta,
            interface_changes,
            effect_changes,
            child_forward_feedbacks,
        )

        # Step 4: evaluate the overall impact
        overall_impact = self._evaluate_overall_impact(
            child_forward_feedbacks,
            interface_changes,
            effect_changes,
        )

        # Step 5: generate the action summary
        action_summary = self._generate_action_summary(
            adjusted_delta,
            dependency_updates,
            overall_impact,
        )

        return ConsiderResult(
            adjusted_delta=adjusted_delta,
            adjustments=adjustments,
            dependency_updates=dependency_updates,
            overall_impact=overall_impact,
            action_summary=action_summary,
            confidence=self._calculate_confidence(adjustments, child_forward_feedbacks),
        )

    def _collect_interface_changes(
        self,
        feedbacks: List[OptimizationForwardFeedback]
    ) -> List[InterfaceChange]:
        """Collect all interface changes."""
        changes = []
        for fb in feedbacks:
            if fb.interface_changed:
                changes.extend(fb.interface_changes)
        return changes

    def _collect_effect_changes(
        self,
        feedbacks: List[OptimizationForwardFeedback]
    ) -> List[EffectChange]:
        """Collect all effect changes."""
        changes = []
        for fb in feedbacks:
            if fb.effects_changed:
                changes.extend(fb.effect_changes)
        return changes

    def _identify_dependency_updates(
        self,
        feedbacks: List[OptimizationForwardFeedback],
        parent_code: Optional[str],
    ) -> List[DependencyUpdate]:
        """Identify required dependency updates."""
        updates = []

        for fb in feedbacks:
            # Check whether interface changes require call updates
            for change in fb.interface_changes:
                if change.requires_parent_update:
                    updates.append(DependencyUpdate(
                        child_skill=fb.skill_name,
                        update_type="call_signature",
                        description=f"Update call to {fb.skill_name}: {change.description}",
                        code_suggestion=self._generate_call_update_suggestion(
                            fb.skill_name, change, parent_code
                        ),
                        priority=1,
                    ))

            # Check whether effect changes require handling updates
            for change in fb.effect_changes:
                if change.may_affect_parent:
                    updates.append(DependencyUpdate(
                        child_skill=fb.skill_name,
                        update_type="effect_handling",
                        description=f"Handle changed effect from {fb.skill_name}: {change.impact_description}",
                        code_suggestion=self._generate_effect_handling_suggestion(
                            fb.skill_name, change
                        ),
                        priority=2,
                    ))

            # Check warnings
            for warning in fb.parent_warnings:
                updates.append(DependencyUpdate(
                    child_skill=fb.skill_name,
                    update_type="warning_action",
                    description=warning,
                    code_suggestion="Review and address the warning",
                    priority=3,
                ))

            # Check required updates
            for required in fb.required_parent_updates:
                updates.append(DependencyUpdate(
                    child_skill=fb.skill_name,
                    update_type="required_update",
                    description=required,
                    code_suggestion="Must update to maintain compatibility",
                    priority=1,
                ))

        # Sort by priority
        updates.sort(key=lambda u: u.priority)

        return updates

    def _generate_call_update_suggestion(
        self,
        child_skill: str,
        change: InterfaceChange,
        parent_code: Optional[str],
    ) -> str:
        """Generate a call-update suggestion."""
        if change.change_type == "parameter_added":
            return f"Add new parameter when calling {child_skill}: {change.description}"
        elif change.change_type == "parameter_removed":
            return f"Remove parameter from call to {child_skill}: {change.description}"
        elif change.change_type == "parameter_modified":
            return f"Update parameter: {change.old_value} -> {change.new_value}"
        return "Review and update the call"

    def _generate_effect_handling_suggestion(
        self,
        child_skill: str,
        change: EffectChange,
    ) -> str:
        """Generate an effect-handling suggestion."""
        if change.change_type == "removed":
            return f"Effect no longer produced by {child_skill}: {change.effect_description}"
        elif change.change_type == "modified":
            return f"Effect behavior changed: {change.impact_description}"
        return "Review effect handling"

    def _adjust_delta(
        self,
        delta: SkillDelta,
        interface_changes: List[InterfaceChange],
        effect_changes: List[EffectChange],
        feedbacks: List[OptimizationForwardFeedback],
    ) -> Tuple[SkillDelta, List[DeltaAdjustment]]:
        """Adjust the delta."""
        adjustments = []
        new_gradients = []

        # Keep existing gradients; magnitudes may be adjusted
        for gradient in delta.gradients:
            adjusted = self._adjust_gradient(
                gradient,
                interface_changes,
                effect_changes,
                feedbacks,
            )
            new_gradients.append(adjusted)

            if adjusted.magnitude != gradient.magnitude:
                adjustments.append(DeltaAdjustment(
                    original_gradient=gradient,
                    adjusted_gradient=adjusted,
                    adjustment_reason="Adjusted based on child skill changes",
                ))

        # Add a new gradient (if child-skill changes require one)
        if any(c.requires_parent_update for c in interface_changes):
            new_gradient = Gradient(
                gradient_type=GradientType.PARAMETER_TYPE,
                magnitude=0.8,
                direction="Update calls to child skills with changed interfaces",
                evidence=f"Child skills changed: {[c.description for c in interface_changes]}",
            )
            new_gradients.append(new_gradient)
            adjustments.append(DeltaAdjustment(
                original_gradient=Gradient(
                    gradient_type=GradientType.PARAMETER_TYPE,
                    magnitude=0.0,
                    direction="",
                    evidence="",
                ),
                adjusted_gradient=new_gradient,
                adjustment_reason="Added due to child interface changes",
            ))

        if any(c.may_affect_parent for c in effect_changes):
            new_gradient = Gradient(
                gradient_type=GradientType.EFFECT,
                magnitude=0.7,
                direction="Handle changed effects from child skills",
                evidence=f"Child effects changed: {[c.effect_description for c in effect_changes]}",
            )
            new_gradients.append(new_gradient)
            adjustments.append(DeltaAdjustment(
                original_gradient=Gradient(
                    gradient_type=GradientType.EFFECT,
                    magnitude=0.0,
                    direction="",
                    evidence="",
                ),
                adjusted_gradient=new_gradient,
                adjustment_reason="Added due to child effect changes",
            ))

        # Build the new delta
        adjusted_delta = SkillDelta(
            skill_name=delta.skill_name,
            gradients=new_gradients,
            source_feedback=delta.source_feedback,
            analysis_depth=delta.analysis_depth,
        )

        return adjusted_delta, adjustments

    def _adjust_gradient(
        self,
        gradient: Gradient,
        interface_changes: List[InterfaceChange],
        effect_changes: List[EffectChange],
        feedbacks: List[OptimizationForwardFeedback],
    ) -> Gradient:
        """Adjust a single gradient."""
        # If child-skill optimizations succeeded, magnitudes of some gradients may be lowered
        successful_children = sum(1 for fb in feedbacks if fb.optimization_successful)
        total_children = len(feedbacks)

        if total_children == 0:
            return gradient

        success_rate = successful_children / total_children

        # If all child-skill optimizations succeeded and the gradient type is child-skill related, lower the magnitude.
        # Expanded gradient-type coverage: LOGIC/ERROR_HANDLING may also be alleviated by child-skill fixes
        if success_rate > 0.8 and gradient.gradient_type in [
            GradientType.PARAMETER_SEMANTIC,
            GradientType.PARAMETER_TYPE,
            GradientType.LOGIC,           # Expanded: logic errors
            GradientType.ERROR_HANDLING,  # Expanded: error handling
        ]:
            # Conservative reduction: changed from 0.5 to 0.6 (retain 40% instead of the previous 50%)
            new_magnitude = gradient.magnitude * (1 - success_rate * 0.6)
            return Gradient(
                gradient_type=gradient.gradient_type,
                magnitude=new_magnitude,
                direction=gradient.direction,
                evidence=gradient.evidence + " (adjusted for child success)",
                suggested_fix=gradient.suggested_fix,
                affected_lines=gradient.affected_lines,
            )

        # When there are breaking changes, increase the magnitude of related gradients
        if any(c.requires_parent_update for c in interface_changes):
            if gradient.gradient_type == GradientType.PARAMETER_TYPE:
                new_magnitude = min(gradient.magnitude * 1.3, 1.0)
                return Gradient(
                    gradient_type=gradient.gradient_type,
                    magnitude=new_magnitude,
                    direction=gradient.direction,
                    evidence=gradient.evidence + " (increased for interface changes)",
                    suggested_fix=gradient.suggested_fix,
                    affected_lines=gradient.affected_lines,
                )

        return gradient

    def _evaluate_overall_impact(
        self,
        feedbacks: List[OptimizationForwardFeedback],
        interface_changes: List[InterfaceChange],
        effect_changes: List[EffectChange],
    ) -> str:
        """Evaluate the overall impact."""
        # Check whether any optimization failed
        if any(not fb.optimization_successful for fb in feedbacks):
            return "failed"

        # Check whether there are breaking changes
        if any(c.requires_parent_update for c in interface_changes):
            return "breaking"

        # Check whether there are effect changes
        if any(c.may_affect_parent for c in effect_changes):
            return "moderate"

        # Check whether there are any changes at all
        if interface_changes or effect_changes:
            return "moderate"

        return "minimal"

    def _generate_action_summary(
        self,
        adjusted_delta: SkillDelta,
        dependency_updates: List[DependencyUpdate],
        overall_impact: str,
    ) -> str:
        """Generate the action summary."""
        lines = []

        if overall_impact == "failed":
            lines.append("⚠️ Some child optimizations failed. Review required.")
        elif overall_impact == "breaking":
            lines.append("🔴 Breaking changes detected. Parent skill MUST be updated.")
        elif overall_impact == "moderate":
            lines.append("🟡 Moderate changes. Parent skill should be reviewed.")
        else:
            lines.append("🟢 Minimal impact. Standard optimization applies.")

        if adjusted_delta.is_significant:
            lines.append(f"Primary issue: {adjusted_delta.primary_type.value if adjusted_delta.primary_type else 'unknown'}")

        if dependency_updates:
            lines.append(f"Required updates: {len(dependency_updates)}")
            for update in dependency_updates[:3]:
                lines.append(f"  - [{update.child_skill}] {update.description[:50]}...")

        return "\n".join(lines)

    def _calculate_confidence(
        self,
        adjustments: List[DeltaAdjustment],
        feedbacks: List[OptimizationForwardFeedback],
    ) -> float:
        """Compute confidence."""
        if not feedbacks:
            return 0.5

        # Based on the child-skill optimization success rate
        success_rate = sum(1 for fb in feedbacks if fb.optimization_successful) / len(feedbacks)

        # Based on how reasonable the adjustments are
        adjustment_factor = 1.0 - (len(adjustments) * 0.1)

        return max(0.3, min(0.9, success_rate * 0.7 + adjustment_factor * 0.3))


