"""
Bloat Checker

Anti-bloat checker: checks whether code growth exceeds thresholds.

Rules:
1. Hard limit: more than 300% growth is rejected outright (non-WRAPPER types only)
2. Soft limit: more than 200% growth triggers a warning
3. Absolute limit: no skill may exceed 800 lines
4. Wrapper skill (< 25 lines): at most 50 lines (skip the ratio hard limit because the growth ratio of small functions is naturally high)
5. Normal skill (25-100 lines): at most NORMAL_GROWTH_LIMIT_RATIO (default 2.0x) of original lines
6. Complex skill (> 100 lines): at most COMPLEX_GROWTH_LIMIT_RATIO (default 1.75x) of original lines

Special rules for covered wrappers:
- Wrapper skills marked is_covered by refactor have stricter limits
- Max 20 lines; growth not to exceed 50%
- Prevents wrappers from bloating during optimization
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

from ..config import BloatPreventionConfig, DEFAULT_BLOAT_CONFIG

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import SkillNode, CoverageType


class SkillType(Enum):
    """Skill type"""
    WRAPPER = "wrapper"
    NORMAL = "normal"
    COMPLEX = "complex"


class FixPriority(Enum):
    """
    Fix priority — decides whether the bloat limit may be exceeded

    Priority rules:
    - CRITICAL: must-fix blocking errors (TDZ, variable redeclaration, SyntaxError) → allow up to 5x overage
    - HIGH: runtime errors (TypeError, ReferenceError) → allow up to 3x overage
    - MEDIUM: logic errors (semantic mismatches) → warn but do not block
    - LOW: code-quality issues (style, readability) → constrained by limits
    """
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class FixType:
    """
    Fix type

    Used to classify the type of code fix the LLM generated,
    so we can decide whether to allow exceeding the bloat limit.
    """
    priority: FixPriority
    reason: str

    # Overage cap (multiplier)
    @property
    def max_override_ratio(self) -> float:
        """Return the maximum allowed overage multiplier for this priority"""
        limits = {
            FixPriority.CRITICAL: 5.0,  # Up to 5x
            FixPriority.HIGH: 3.0,      # Up to 3x
            FixPriority.MEDIUM: 3.0,    # Up to 3.0x (matches hard limit)
            FixPriority.LOW: 1.0,       # Overage not allowed
        }
        return limits.get(self.priority, 1.0)


@dataclass
class BloatCheckResult:
    """Bloat check result"""
    passed: bool
    skill_type: SkillType
    old_lines: int
    new_lines: int
    max_allowed_lines: int
    growth_ratio: float

    # Status
    is_warning: bool = False
    message: str = ""

    # Detailed list of issues
    issues: List[str] = field(default_factory=list)


class BloatChecker:
    """
    Code bloat checker

    Prevents skill code from growing without bound.

    Usage:
        checker = BloatChecker()

        result = checker.check(
            skill_name="craftOakBoat",
            old_code=original_code,
            new_code=optimized_code
        )

        if not result.passed:
            print(f"Code bloat rejected: {result.message}")
        elif result.is_warning:
            print(f"Warning: {result.message}")
    """

    def __init__(
        self,
        config: Optional[BloatPreventionConfig] = None,
        logger=None,
    ):
        self.config = config or DEFAULT_BLOAT_CONFIG
        self.logger = logger

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            colors = {
                "info": "\033[36m",
                "warning": "\033[33m",
                "error": "\033[31m",
                "success": "\033[32m",
            }
            reset = "\033[0m"
            print(f"{colors.get(level, '')}{message}{reset}")

    def check(
        self,
        skill_name: str,
        old_code: str,
        new_code: str,
    ) -> BloatCheckResult:
        """
        Check whether the code growth exceeds the threshold

        Args:
            skill_name: skill name
            old_code: code before optimization
            new_code: code after optimization

        Returns:
            BloatCheckResult: check result
        """
        old_lines = len(old_code.strip().split('\n'))
        new_lines = len(new_code.strip().split('\n'))
        growth_ratio = new_lines / max(old_lines, 1)

        skill_type = self._determine_skill_type(old_code, old_lines)
        max_allowed = self._get_max_allowed_lines(old_lines, skill_type)

        issues = []

        # Log growth information
        self._log(
            f"[BloatChecker] {skill_name}: {old_lines} -> {new_lines} lines "
            f"(growth {growth_ratio:.0%}, type: {skill_type.value}, limit: {max_allowed} lines)",
            "info"
        )

        # Prompt-aligned absolute line limit (from code_optimization.txt ANTI-BLOAT RULES)
        # These rules define the absolute ceiling that the LLM is instructed to follow.
        # raised <50→150 (was 100) because stub functions (6-line naive code)
        # legitimately need ~40-70 lines of placeItem/precondition logic when optimized.
        # r10 data: 42→137 lines was rejected at 100 but contained correct placeItem code.
        if old_lines < 50:
            prompt_limit = 150
        elif old_lines <= 100:
            prompt_limit = 200
        else:
            prompt_limit = int(old_lines * 1.5)

        # If within prompt limit, exempt from ratio hard limit and type-based limit
        within_prompt_limit = new_lines <= prompt_limit

        # Check 1: hard-limit ratio
        # Note: WRAPPER types skip the ratio check because they already have a 50-line absolute limit (check 3)
        # Small functions naturally have a high growth ratio (e.g., 13→40 is 308% but only adds 27 lines)
        # Prompt-aligned exemption: skip the ratio check if within the prompt absolute line limit
        if (skill_type != SkillType.WRAPPER
                and not within_prompt_limit
                and growth_ratio > self.config.GROWTH_HARD_LIMIT_RATIO):
            msg = (
                f"Code growth {growth_ratio:.0%} exceeds hard limit "
                f"{self.config.GROWTH_HARD_LIMIT_RATIO:.0%}"
            )
            issues.append(msg)
            self._log(f"[BloatChecker] {skill_name}: rejected - {msg}", "error")

            return BloatCheckResult(
                passed=False,
                skill_type=skill_type,
                old_lines=old_lines,
                new_lines=new_lines,
                max_allowed_lines=max_allowed,
                growth_ratio=growth_ratio,
                message=msg,
                issues=issues,
            )

        # Check 2: absolute line-count limit
        if new_lines > self.config.ABSOLUTE_MAX_LINES:
            msg = (
                f"Code at {new_lines} lines exceeds absolute limit "
                f"{self.config.ABSOLUTE_MAX_LINES} lines"
            )
            issues.append(msg)
            self._log(f"[BloatChecker] {skill_name}: rejected - {msg}", "error")

            return BloatCheckResult(
                passed=False,
                skill_type=skill_type,
                old_lines=old_lines,
                new_lines=new_lines,
                max_allowed_lines=max_allowed,
                growth_ratio=growth_ratio,
                message=msg,
                issues=issues,
            )

        # Check 3: type-specific line-count limit
        # Prompt-aligned exemption: skip the type-based check if within the prompt absolute line limit
        if new_lines > max_allowed and not within_prompt_limit:
            msg = (
                f"{skill_type.value} skill grew from {old_lines} to "
                f"{new_lines} lines, exceeding limit {max_allowed} lines"
            )
            issues.append(msg)
            self._log(f"[BloatChecker] {skill_name}: rejected - {msg}", "error")

            return BloatCheckResult(
                passed=False,
                skill_type=skill_type,
                old_lines=old_lines,
                new_lines=new_lines,
                max_allowed_lines=max_allowed,
                growth_ratio=growth_ratio,
                message=msg,
                issues=issues,
            )

        # Check 4: soft limit (warn but pass)
        if growth_ratio > self.config.GROWTH_SOFT_LIMIT_RATIO:
            msg = (
                f"Code growth {growth_ratio:.0%} exceeds soft limit "
                f"{self.config.GROWTH_SOFT_LIMIT_RATIO:.0%}; additional validation required"
            )
            self._log(f"[BloatChecker] {skill_name}: warning - {msg}", "warning")

            return BloatCheckResult(
                passed=True,
                skill_type=skill_type,
                old_lines=old_lines,
                new_lines=new_lines,
                max_allowed_lines=max_allowed,
                growth_ratio=growth_ratio,
                is_warning=True,
                message=msg,
                issues=issues,
            )

        self._log(
            f"[BloatChecker] {skill_name}: passed - growth within reasonable range",
            "success"
        )

        return BloatCheckResult(
            passed=True,
            skill_type=skill_type,
            old_lines=old_lines,
            new_lines=new_lines,
            max_allowed_lines=max_allowed,
            growth_ratio=growth_ratio,
            message="Code growth within reasonable range",
        )

    def _determine_skill_type(self, code: str, line_count: int) -> SkillType:
        """Determine the skill type"""
        if self._is_wrapper_skill(code, line_count):
            return SkillType.WRAPPER
        elif line_count > 100:
            return SkillType.COMPLEX
        else:
            return SkillType.NORMAL

    def _is_wrapper_skill(self, code: str, line_count: int) -> bool:
        """Check whether this is a wrapper skill"""
        if line_count >= self.config.WRAPPER_LINE_THRESHOLD:
            return False

        # Count await calls
        await_count = code.count('await ')

        return await_count <= self.config.WRAPPER_AWAIT_THRESHOLD

    def _get_max_allowed_lines(
        self,
        old_lines: int,
        skill_type: SkillType
    ) -> int:
        """Compute the maximum allowed lines based on skill type"""
        if skill_type == SkillType.WRAPPER:
            return self.config.WRAPPER_MAX_LINES
        elif skill_type == SkillType.NORMAL:
            return int(old_lines * self.config.NORMAL_GROWTH_LIMIT_RATIO)
        else:  # COMPLEX
            return int(old_lines * self.config.COMPLEX_GROWTH_LIMIT_RATIO)

    def _looks_like_wrapper(self, code: str) -> bool:
        """Check whether the code looks like a wrapper"""
        lines = [l.strip() for l in code.split('\n') if l.strip()]

        # Wrapper traits:
        # 1. Mostly calls other functions
        # 2. Little complex logic
        # 3. Not many if/for/while statements

        control_flow_count = 0
        for line in lines:
            if re.match(r'^\s*(if|for|while|switch)\s*\(', line):
                control_flow_count += 1

        # If there are more than 3 control-flow statements, probably not a wrapper
        return control_flow_count <= 3

    def _generate_constraint_message(
        self,
        skill_type: SkillType,
        old_lines: int,
        max_allowed: int
    ) -> str:
        """Generate a constraint message (used in the LLM prompt)"""
        if skill_type == SkillType.WRAPPER:
            return (
                f"This is a WRAPPER SKILL ({old_lines} lines). "
                f"It should REMAIN a simple wrapper with < {max_allowed} lines. "
                f"Do NOT add complex logic or many helper functions."
            )
        elif skill_type == SkillType.COMPLEX:
            return (
                f"This is a COMPLEX SKILL ({old_lines} lines). "
                f"Max allowed: {max_allowed} lines ({self.config.COMPLEX_GROWTH_LIMIT_RATIO}x growth). "
                f"Focus on targeted fixes, avoid adding too much new code."
            )
        else:
            return (
                f"This is a NORMAL SKILL ({old_lines} lines). "
                f"Max allowed: {max_allowed} lines ({self.config.NORMAL_GROWTH_LIMIT_RATIO}x growth). "
                f"Max {self.config.MAX_NEW_HELPERS} new helper functions."
            )

