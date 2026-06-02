"""
Skip Checker

Skip-optimization checker: decides whether to skip optimization for a skill.

Skip conditions:
1. Too many versions (above threshold).
2. Probabilistic skip (based on Value Function).
3. CALLER_FIX_MARKER (the problem is in the caller, not the callee).

Note: covered skills are no longer skipped!
- The wrapper itself may have bugs (parameter-passing errors).
- The prompt guides the LLM to handle wrappers correctly.
- BloatChecker prevents wrapper bloat.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

from ..config import SkipOptimizationConfig, DEFAULT_SKIP_CONFIG

if TYPE_CHECKING:
    from skillnet.agents.skill_graph import SkillNode
    from skillnet.agents.optimizer.feedback.types import SkillFeedback


@dataclass
class SkipCheckResult:
    """Skip-check result."""
    should_skip: bool
    reason: str = ""
    details: List[str] = field(default_factory=list)


class SkipChecker:
    """
    Skip-optimization checker.

    Decides whether to skip optimization for a skill.

    Usage:
        checker = SkipChecker(skill_graph_manager=manager)

        result = checker.should_skip(
            skill_name="craftOakPlanks",
            current_task="craft oak boat"
        )

        if result.should_skip:
            print(f"skipping {skill_name}: {result.reason}")
    """

    def __init__(
        self,
        skill_graph_manager=None,
        config: Optional[SkipOptimizationConfig] = None,
        logger=None,
    ):
        self.skill_graph_manager = skill_graph_manager
        self.config = config or DEFAULT_SKIP_CONFIG
        self.logger = logger

    def _log(self, message: str, level: str = "info"):
        """Emit a log line."""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)

    def should_skip(
        self,
        skill_name: str,
        current_task: Optional[str] = None,
        force_optimize: bool = False,
        feedbacks: Optional[List['SkillFeedback']] = None,
    ) -> SkipCheckResult:
        """
        Check whether optimization should be skipped.

        Args:
            skill_name: skill name
            current_task: current task (used for relevance checks)
            force_optimize: whether to force optimization (skip most checks)
            feedbacks: list of feedbacks for the skill (used to check CALLER_FIX_MARKER)

        Returns:
            SkipCheckResult: check result.
        """
        if force_optimize:
            return SkipCheckResult(should_skip=False)

        if not self.skill_graph_manager:
            return SkipCheckResult(should_skip=False)

        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return SkipCheckResult(
                should_skip=True,
                reason="skill not found",
            )

        details = []

        # [Covered skills are no longer skipped]
        # Covered skills now go through normal optimization, with wrapper context
        # added to the prompt so the LLM knows it is a wrapper and avoids bloat.
        if getattr(node, 'is_covered', False):
            covered_by = getattr(node, 'covered_by', None)
            coverage_type = getattr(node, 'coverage_type', None)
            self._log(
                f"[SkipChecker] '{node.name}' is covered by '{covered_by}' "
                f"(type: {coverage_type}), but will NOT skip optimization",
                "debug"
            )
            details.append(f"is_covered=True, covered_by={covered_by}")

        # Check 1: too many versions
        skip_result = self._check_version_skip(node)
        if skip_result.should_skip:
            return skip_result
        if skip_result.details:
            details.extend(skip_result.details)

        # Check 2: CALLER_FIX_MARKER (problem is in the caller)
        if feedbacks:
            skip_result = self._check_caller_fix_marker(skill_name, feedbacks, current_task)
            if skip_result.should_skip:
                return skip_result
            if skip_result.details:
                details.extend(skip_result.details)

        # Check 3: probabilistic skip (based on Value Function)
        if self.config.ENABLE_PROBABILITY_SKIP:
            skip_result = self._check_probability_skip(node)
            if skip_result.should_skip:
                return skip_result
            if skip_result.details:
                details.extend(skip_result.details)

        return SkipCheckResult(
            should_skip=False,
            details=details,
        )

    def _check_version_skip(self, node: 'SkillNode') -> SkipCheckResult:
        """Check whether to skip due to too many versions."""
        # Get version count
        version_count = len(node.versions) if hasattr(node, 'versions') else 0

        if version_count >= self.config.MAX_VERSIONS_BEFORE_SKIP:
            self._log(
                f"[SkipChecker] skipping '{node.name}': "
                f"version count {version_count} >= {self.config.MAX_VERSIONS_BEFORE_SKIP}",
                "warning"
            )
            return SkipCheckResult(
                should_skip=True,
                reason=f"too many versions ({version_count})",
                details=[f"version_count={version_count}"]
            )

        return SkipCheckResult(should_skip=False)

    def _check_probability_skip(self, node: 'SkillNode') -> SkipCheckResult:
        """
        Probabilistic skip based on the Value Function.

        Formula: P(optimize(s)) = (1-epsilon) * sigma(gamma(threshold - V(s))) + epsilon

        Where:
        - epsilon: minimum optimization probability
        - gamma: sigmoid slope parameter
        - threshold: value-function threshold
        - V(s): the skill's value-function value
        - sigma: sigmoid function

        Skills with high V(s) (performing well) have a lower optimization probability.
        """
        # Get value function
        v_s = getattr(node, 'value_function', 0.5)

        # Compute optimization probability
        probability = self._compute_optimization_probability(v_s)

        # Randomly decide whether to optimize
        should_optimize = random.random() < probability

        if not should_optimize:
            self._log(
                f"[SkipChecker] probabilistic skip for '{node.name}': "
                f"V(s)={v_s:.3f}, P(optimize)={probability:.3f}",
                "info"
            )
            return SkipCheckResult(
                should_skip=True,
                reason=f"probability skip (V(s)={v_s:.3f}, P={probability:.3f})",
                details=[f"value_function={v_s:.3f}", f"probability={probability:.3f}"]
            )

        return SkipCheckResult(
            should_skip=False,
            details=[f"value_function={v_s:.3f}", f"probability={probability:.3f}"]
        )

    def _compute_optimization_probability(self, v_s: float) -> float:
        """
        Compute the probability of optimizing the skill.

        P(optimize(s)) = (1-epsilon) * sigma(gamma(threshold - V(s))) + epsilon

        Args:
            v_s: the skill's value-function value

        Returns:
            float: optimization probability in [0, 1].
        """
        epsilon = self.config.OPTIMIZATION_EPSILON
        gamma = self.config.OPTIMIZATION_GAMMA
        threshold = self.config.OPTIMIZATION_THRESHOLD

        # Compute sigmoid(gamma(threshold - V(s)))
        x = gamma * (threshold - v_s)
        # Guard against numeric overflow
        if x > 500:
            sigmoid_value = 1.0
        elif x < -500:
            sigmoid_value = 0.0
        else:
            sigmoid_value = 1.0 / (1.0 + math.exp(-x))

        # Compute final probability
        probability = (1.0 - epsilon) * sigmoid_value + epsilon

        return probability

    def _check_caller_fix_marker(
        self,
        skill_name: str,
        feedbacks: List['SkillFeedback'],
        current_task: Optional[str] = None,
    ) -> SkipCheckResult:
        """
        Check for a CALLER_FIX_MARKER.

        When backpropagation analysis finds the problem is in the caller (not
        the callee), it emits a CALLER_FIX_MARKER feedback. In that case, the
        called child skill should not be optimized.

        Args:
            skill_name: skill name
            feedbacks: list of feedbacks for the skill
            current_task: current task (used for filtering)

        Returns:
            SkipCheckResult: check result.
        """
        for fb in feedbacks:
            # Check whether this is a CALLER_FIX_MARKER
            fb_type = getattr(fb, 'feedback_type', None)
            if fb_type != "caller_fix_marker":
                continue

            # If a task is specified, ensure it matches
            if current_task:
                fb_task = getattr(fb, 'task', None)
                if fb_task and fb_task != current_task:
                    continue

            # Get caller info
            content = getattr(fb, 'content', '')
            caller_skill = None
            if "caused by its caller" in content:
                # Extract caller name from content
                import re
                match = re.search(r"caller '([^']+)'", content)
                if match:
                    caller_skill = match.group(1)

            self._log(
                f"[SkipChecker] skipping '{skill_name}': problem is in caller '{caller_skill}'",
                "info"
            )
            return SkipCheckResult(
                should_skip=True,
                reason=f"problem is in caller '{caller_skill}'",
                details=[
                    "caller_fix_marker found",
                    f"fix_target={caller_skill}",
                ]
            )

        return SkipCheckResult(should_skip=False)

