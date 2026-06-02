"""
Runtime Effect Validator

Validates whether the effects extracted by the LLM are actually produced,
based on execution history. Resolves the problem that static analysis cannot
handle dynamic parameters.

Design principles:
1. When static analysis cannot confirm, query the runtime history
2. Compute confidence from actual execution data
3. No longer binary discard — assign a confidence score instead
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..models.execution import SkillExecutionTrace, SkillStatistics
from ..models.precondition import ActualEffect, SkillEffect


class ValidationStatus(Enum):
    """Validation status."""
    CONFIRMED = "confirmed"       # Confirmed by static analysis or runtime data
    LIKELY = "likely"             # Runtime data indicates likely (50-70%)
    UNLIKELY = "unlikely"         # Runtime data indicates unlikely (<50%)
    INSUFFICIENT_DATA = "insufficient_data"  # Insufficient data
    UNCONFIRMED = "unconfirmed"   # Cannot confirm


@dataclass
class EffectValidationResult:
    """Effect validation result."""
    status: ValidationStatus
    confidence_value: float      # 0.0~1.0
    confidence_level: str        # "high" / "medium" / "low" / "uncertain"
    stats: Dict[str, Any]        # Statistics
    reason: str                  # Explanation of the validation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "confidence_value": self.confidence_value,
            "confidence_level": self.confidence_level,
            "stats": self.stats,
            "reason": self.reason,
        }


class RuntimeEffectValidator:
    """Runtime effect validator.

    Aggregates actual_effects from the execution history to verify whether
    the expected_effect is actually produced.

    Use cases:
    1. When static analysis (code_has_effect_implementation) returns False
    2. When effect confidence needs to be updated
    3. When detecting LLM hallucinations
    """

    def __init__(self, min_samples: int = 3, logger: logging.Logger = None):
        """
        Args:
            min_samples: Minimum number of samples; fewer is treated as insufficient data
            logger: Logger
        """
        self.min_samples = min_samples
        self.logger = logger or logging.getLogger(__name__)

    def validate_effect(
        self,
        expected_effect: SkillEffect,
        execution_traces: List[SkillExecutionTrace],
    ) -> EffectValidationResult:
        """Validate whether a single effect is produced in the execution history.

        Args:
            expected_effect: The expected effect extracted by the LLM
            execution_traces: Execution history of the skill

        Returns:
            EffectValidationResult containing confidence and statistics
        """
        # Extract the item name
        item_name = self._extract_item_from_effect(expected_effect)
        if not item_name:
            return EffectValidationResult(
                status=ValidationStatus.UNCONFIRMED,
                confidence_value=0.3,
                confidence_level="uncertain",
                stats={},
                reason="Unable to extract item name from effect",
            )

        # Aggregate execution history
        stats = self._aggregate_actual_effects(execution_traces, item_name)

        # Compute confidence
        return self._calculate_confidence(stats, item_name)

    def update_effect_confidence(
        self,
        effect: SkillEffect,
        validation_result: EffectValidationResult,
    ) -> SkillEffect:
        """Update the effect's confidence fields based on the validation result.

        Args:
            effect: Effect to update
            validation_result: Validation result

        Returns:
            Updated effect (modified in place)
        """
        effect.confidence_value = validation_result.confidence_value
        effect.confidence_level = validation_result.confidence_level
        effect.inference_stats = {
            **(effect.inference_stats or {}),
            **validation_result.stats,
            "validation_status": validation_result.status.value,
            "validation_reason": validation_result.reason,
            "last_updated": datetime.now().isoformat(),
        }
        return effect

    def _extract_item_from_effect(self, effect: SkillEffect) -> Optional[str]:
        """Extract the item name from an effect.

        Supports multiple formats:
        1. state_representation = {"item": "oak_log", ...}
        2. state_representation = {"logic": "OR", "conditions": [{"item": "oak_log"}, ...]}
        """
        state_repr = effect.state_representation
        if not state_repr:
            return None

        # Format 1: contains item directly
        if isinstance(state_repr, dict):
            if "item" in state_repr:
                return state_repr["item"]

            # Format 2: OR/AND logic; take the item of the first condition
            if "conditions" in state_repr:
                conditions = state_repr.get("conditions", [])
                if conditions and isinstance(conditions[0], dict):
                    return conditions[0].get("item")

        # Format 3: list format (legacy)
        if isinstance(state_repr, list) and len(state_repr) > 0:
            return state_repr[0].get("item")

        return None

    def _aggregate_actual_effects(
        self,
        traces: List[SkillExecutionTrace],
        expected_item: str,
    ) -> Dict[str, Any]:
        """Aggregate actual-production statistics for an item across the execution history.

        Args:
            traces: Execution history
            expected_item: Expected item name

        Returns:
            Statistics dictionary
        """
        # Only count records that were actually executed and successful
        success_traces = [
            t for t in traces
            if t.success and t.was_executed
        ]

        if not success_traces:
            return {
                "occurrences": 0,
                "total_success": 0,
                "occurrence_rate": 0.0,
                "total_count": 0,
                "avg_count": 0.0,
            }

        occurrences = 0
        total_count = 0

        for trace in success_traces:
            found_in_trace = False

            # Check actual_effects
            for actual in (trace.actual_effects or []):
                # Approach 1: check inventory_changes
                if actual.inventory_changes:
                    item_change = actual.inventory_changes.get(expected_item, 0)
                    if item_change > 0:
                        total_count += item_change
                        found_in_trace = True

                # Approach 2: check state_changes
                for change in (actual.state_changes or []):
                    if (change.get("item") == expected_item and
                        change.get("operation") == "add" and
                        change.get("count", 0) > 0):
                        count = change.get("count", 1)
                        total_count += count
                        found_in_trace = True

            # Approach 3: compare post_state vs pre_state
            if not found_in_trace and trace.pre_state and trace.post_state:
                pre_inv = trace.pre_state.get("inventory", {})
                post_inv = trace.post_state.get("inventory", {})

                pre_count = pre_inv.get(expected_item, 0)
                post_count = post_inv.get(expected_item, 0)

                if post_count > pre_count:
                    total_count += (post_count - pre_count)
                    found_in_trace = True

            if found_in_trace:
                occurrences += 1

        n_success = len(success_traces)
        return {
            "occurrences": occurrences,
            "total_success": n_success,
            "occurrence_rate": occurrences / n_success if n_success > 0 else 0.0,
            "total_count": total_count,
            "avg_count": total_count / occurrences if occurrences > 0 else 0.0,
        }

    def _calculate_confidence(
        self,
        stats: Dict[str, Any],
        item_name: str,
    ) -> EffectValidationResult:
        """Compute confidence based on statistics.

        Confidence calculation rules:
        - Insufficient data (< min_samples): uncertain, 0.3
        - occurrence_rate >= 0.8: high, 0.9
        - occurrence_rate >= 0.5: medium, 0.6
        - occurrence_rate >= 0.2: low, 0.4
        - occurrence_rate < 0.2: unlikely, 0.1
        """
        n_success = stats.get("total_success", 0)
        rate = stats.get("occurrence_rate", 0.0)

        # Insufficient data
        if n_success < self.min_samples:
            return EffectValidationResult(
                status=ValidationStatus.INSUFFICIENT_DATA,
                confidence_value=0.3,
                confidence_level="uncertain",
                stats=stats,
                reason=f"Insufficient data: only {n_success} successful executions; at least {self.min_samples} are required",
            )

        # High confidence
        if rate >= 0.8:
            return EffectValidationResult(
                status=ValidationStatus.CONFIRMED,
                confidence_value=0.9,
                confidence_level="high",
                stats=stats,
                reason=f"Runtime confirmed: {item_name} produced in {stats['occurrences']}/{n_success} successful executions ({rate:.0%})",
            )

        # Medium confidence
        if rate >= 0.5:
            return EffectValidationResult(
                status=ValidationStatus.LIKELY,
                confidence_value=0.6,
                confidence_level="medium",
                stats=stats,
                reason=f"Likely produced: {item_name} produced in {stats['occurrences']}/{n_success} successful executions ({rate:.0%})",
            )

        # Low confidence
        if rate >= 0.2:
            return EffectValidationResult(
                status=ValidationStatus.UNLIKELY,
                confidence_value=0.4,
                confidence_level="low",
                stats=stats,
                reason=f"Unlikely: {item_name} produced in only {stats['occurrences']}/{n_success} successful executions ({rate:.0%})",
            )

        # Very low confidence
        return EffectValidationResult(
            status=ValidationStatus.UNLIKELY,
            confidence_value=0.1,
            confidence_level="unlikely",
            stats=stats,
            reason=f"Almost never produced: {item_name} produced in only {stats['occurrences']}/{n_success} successful executions ({rate:.0%})",
        )


def get_runtime_validator(min_samples: int = 3) -> RuntimeEffectValidator:
    """Get a runtime-validator instance."""
    return RuntimeEffectValidator(min_samples=min_samples)
