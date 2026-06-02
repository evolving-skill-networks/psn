"""
CodeQualityValidationMixin - Code growth, duplication, and fix classification.

Extracted from validation.py for modularity.
Contains 9 methods for code bloat detection, fix type classification,
requirement verification, and minimal change validation.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Delegate imports
# Fix prioritization for bloat override
from skillnet.agents.optimizer.validators.bloat_checker import FixPriority, FixType
from skillnet.agents.optimizer.transforms.diff_engine import (
    basic_syntax_check as _basic_syntax_check_impl,
)

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer

logger = logging.getLogger(__name__)


class CodeQualityValidationMixin:
    """
    Code quality validation methods.

    Provides checks for code growth/bloat, fix type classification,
    code duplication detection, requirement verification, and minimal change validation.

    Required self attributes:
        - self.logger: Logger instance
        - self._bloat_checker: BloatChecker instance
    """

    # ========== Code Quality Checks ==========

    def _check_code_growth(
        self: "SkillGraphOptimizer",
        old_code: str,
        new_code: str,
        skill_name: str,
        pre_analysis: Optional[Dict[str, Any]] = None,
        current_error: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Two-phase code-growth check

        Phase 1: standard bloat check
        Phase 2: only when phase 1 fails, detect whether this is a critical fix and allow override

        Args:
            old_code: original code
            new_code: new code
            skill_name: skill name
            pre_analysis: pass-through Layer 1/2/3 detection result (optional)
            current_error: current error message (optional)

        Returns:
            (passed, message) tuple
        """
        # Phase 1: standard bloat check
        result = self._bloat_checker.check(skill_name, old_code, new_code)
        if result.passed:
            return True, result.message

        # Phase 2: only when phase 1 fails, detect whether this is a critical fix
        if pre_analysis or current_error:
            fix_type = self._classify_fix_type(old_code, new_code, pre_analysis, current_error)

            if fix_type.priority in (FixPriority.CRITICAL, FixPriority.HIGH):
                # Check whether it is within the priority's allowed cap
                max_ratio = fix_type.max_override_ratio
                if result.growth_ratio <= max_ratio:
                    override_msg = (
                        f"[BloatOverride] {skill_name}: override allowed - "
                        f"{fix_type.priority.name} fix ({fix_type.reason}), "
                        f"growth {result.growth_ratio:.1f}x <= {max_ratio}x cap"
                    )
                    self.logger.info(override_msg)
                    return True, override_msg
                else:
                    # Exceeds the priority cap, still rejected
                    reject_msg = (
                        f"[BloatOverride] {skill_name}: rejected - "
                        f"even as a {fix_type.priority.name} fix ({fix_type.reason}), "
                        f"growth {result.growth_ratio:.1f}x > {max_ratio}x cap"
                    )
                    self.logger.warning(reject_msg)
                    return False, reject_msg
            elif fix_type.priority == FixPriority.MEDIUM:
                medium_max_ratio = self._bloat_checker.config.MEDIUM_FIX_OVERRIDE_RATIO
                if result.growth_ratio <= medium_max_ratio:
                    override_msg = (
                        f"[BloatOverride] {skill_name}: override allowed - "
                        f"MEDIUM fix ({fix_type.reason}), "
                        f"growth {result.growth_ratio:.1f}x <= {medium_max_ratio}x cap"
                    )
                    self.logger.info(override_msg)
                    return True, override_msg
                else:
                    self.logger.warning(
                        f"[BloatOverride] {skill_name}: MEDIUM fix still rejected - "
                        f"({fix_type.reason}), growth {result.growth_ratio:.1f}x > {medium_max_ratio}x cap"
                    )

        return False, result.message

    def _classify_fix_type(
        self: "SkillGraphOptimizer",
        old_code: str,
        new_code: str,
        pre_analysis: Optional[Dict[str, Any]],
        current_error: Optional[str],
    ) -> FixType:
        """
        Classify the fix type (only called when bloat check fails)

        By analyzing pre_analysis (Layer 1/2/3 detection results) and current_error,
        determine the priority of the LLM-generated code fix.

        Args:
            old_code: original code
            new_code: new code
            pre_analysis: Layer 1/2/3 detection result, containing:
                - tdz_check: TDZ detection result
                - underlying_error: underlying library error
                - semantic_mismatch: semantic mismatch info
            current_error: current error message

        Returns:
            FixType object, containing the priority and reason
        """
        pre_analysis = pre_analysis or {}
        current_error = current_error or ""

        # 1. TDZ fix detection (CRITICAL)
        old_tdz = pre_analysis.get("tdz_check", {})
        if not old_tdz.get("valid", True):  # Old code has a TDZ issue
            # Reuse existing validate_tdz_issues to check the new code
            try:
                from skillnet.agents.optimizer.validators.code_validator import validate_tdz_issues
                new_tdz = validate_tdz_issues(new_code)
                if new_tdz.get("valid", False):  # New code fixed the TDZ
                    return FixType(FixPriority.CRITICAL, "TDZ bug fix")
            except Exception:
                pass  # Skip on detection failure

        # 2. SyntaxError fix detection (CRITICAL)
        if "SyntaxError" in current_error or "Identifier.*has already been declared" in current_error:
            if self._adds_syntax_fix(old_code, new_code, current_error):
                return FixType(FixPriority.CRITICAL, "SyntaxError fix")

        # 3. TypeError detection (HIGH)
        if "TypeError" in current_error or "Cannot read properties" in current_error:
            if self._adds_type_checking(old_code, new_code):
                return FixType(FixPriority.HIGH, "TypeError fix")

        # 4. ReferenceError detection (HIGH)
        if "ReferenceError" in current_error or "is not defined" in current_error:
            return FixType(FixPriority.HIGH, "ReferenceError fix")

        # 5. Semantic mismatch fix detection (MEDIUM)
        if pre_analysis.get("semantic_mismatch"):
            if self._adds_semantic_conversion(old_code, new_code):
                return FixType(FixPriority.MEDIUM, "Semantic mismatch fix")

        # 6. Underlying library error fix (MEDIUM)
        underlying_error = pre_analysis.get("underlying_error")
        if underlying_error:
            return FixType(FixPriority.MEDIUM, f"Underlying library error fix: {underlying_error.get('type', 'unknown')}")

        # Default: low priority
        return FixType(FixPriority.LOW, "Unrecognized fix type")

    def _adds_syntax_fix(
        self: "SkillGraphOptimizer",
        old_code: str,
        new_code: str,
        current_error: str,
    ) -> bool:
        """Detect whether the new code fixes a syntax error."""
        # Simple detection: old code has a syntax error, new code does not
        try:
            # Validate using basic_syntax_check
            old_valid, _ = _basic_syntax_check_impl(old_code)
            new_valid, _ = _basic_syntax_check_impl(new_code)
            return not old_valid and new_valid
        except Exception:
            return False

    def _adds_type_checking(
        self: "SkillGraphOptimizer",
        old_code: str,
        new_code: str,
    ) -> bool:
        """
        Detect whether the new code adds type checks

        Detection patterns:
        1. Added typeof check
        2. Added instanceof check
        3. Added optional chaining ?.
        4. Added nullish coalescing ??
        5. Added explicit null/undefined check
        """
        # List of type-check patterns
        type_check_patterns = [
            r'\btypeof\b',           # typeof x === 'number'
            r'\binstanceof\b',       # x instanceof Array
            r'\?\.',                 # x?.property (optional chaining)
            r'\?\?',                 # x ?? default (nullish coalescing)
            r'!==?\s*null',          # x !== null
            r'!==?\s*undefined',     # x !== undefined
            r'===?\s*null',          # x === null
            r'===?\s*undefined',     # x === undefined
            r'if\s*\(\s*\w+\s*\)',   # if (x) truthy check
        ]

        old_matches = 0
        new_matches = 0

        for pattern in type_check_patterns:
            old_matches += len(re.findall(pattern, old_code))
            new_matches += len(re.findall(pattern, new_code))

        # If the new code adds at least 2 new type checks, treat it as a type fix
        return (new_matches - old_matches) >= 2

    def _adds_semantic_conversion(
        self: "SkillGraphOptimizer",
        old_code: str,
        new_code: str,
    ) -> bool:
        """
        Detect whether the new code adds semantic conversions

        Detection patterns:
        1. Added parseInt/parseFloat
        2. Added Number() conversion
        3. Added String() conversion
        4. Added .id / .name property access (Minecraft item type conversion)
        5. Added mcData.itemsByName / mcData.blocksByName lookup
        """
        # List of semantic conversion patterns (general JS patterns)
        conversion_patterns = [
            r'\bparseInt\b',                # parseInt(x)
            r'\bparseFloat\b',              # parseFloat(x)
            r'\bNumber\(',                  # Number(x)
            r'\bString\(',                  # String(x)
            r'\.id\b',                      # item.id
            r'\.name\b',                    # item.name
        ]
        # Domain-specific registry access patterns
        _dk = getattr(self, '_domain_knowledge', None)
        if _dk:
            conversion_patterns.extend(_dk.get_registry_access_patterns())

        old_matches = 0
        new_matches = 0

        for pattern in conversion_patterns:
            old_matches += len(re.findall(pattern, old_code))
            new_matches += len(re.findall(pattern, new_code))

        # If the new code adds at least 1 new semantic conversion, treat it as a fix
        return (new_matches - old_matches) >= 1
