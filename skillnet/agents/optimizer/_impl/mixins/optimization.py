"""
OptimizationMixin - Core optimization logic and loop detection.

Extracted from optimizer_impl.py for better modularity.

Methods included (non-shadowed delegates):
- _detect_parameter_semantic_mismatch: Detect parameter semantic mismatch

Note: The following methods remain in optimizer_impl.py (authoritative versions):
- quick_optimize_skill, _retry_with_minimal_fix,
- _detect_optimization_loop, _handle_optimization_loop
apply_optimization → mixins/apply_optimization.py
         _analyze_feedback_for_edits → mixins/edit_analysis.py
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from skillnet.agents.optimizer.feedback.types import SkillFeedback

# Delegate imports
from skillnet.agents.optimizer._impl.helpers import (
    detect_parameter_semantic_mismatch as _detect_parameter_semantic_mismatch_impl,
)
if TYPE_CHECKING:
    from skillnet.agents.skill_graph import SkillGraphManager
    from ..optimizer_impl import SkillGraphOptimizer


class OptimizationMixin:
    """
    Optimization Core Mixin - Delegate methods for semantic analysis and loop classification.

    These methods delegate to helper modules and are NOT shadowed by optimizer_impl.py.

    Requires self attributes (from SkillGraphOptimizer):
    - self.CRITICAL_CONFLICT_KEYWORDS: frozenset
    - self.UNADDRESSED_ISSUE_KEYWORDS: frozenset
    - self.logger: Logger instance
    """

    def _detect_parameter_semantic_mismatch(
        self: "SkillGraphOptimizer",
        error_text: str,
        feedback: List[SkillFeedback] = None,
        call_context: Dict[str, Any] = None,
        skill_graph: "SkillGraphManager" = None
    ) -> Optional[Dict[str, Any]]:
        """Delegate to helpers.semantic_analysis"""
        return _detect_parameter_semantic_mismatch_impl(
            error_text, call_context, skill_graph
        )
