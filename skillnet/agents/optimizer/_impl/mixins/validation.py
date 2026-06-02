"""
ValidationMixin - Code validation and consistency checking methods

Extracted from optimizer_impl.py for better modularity.
Further decomposed into sub-mixins via composition:
  - CodeQualityValidationMixin: Code growth, duplication, fix classification
  - ResponsibilityValidationMixin: Skill responsibility boundary validation

This module retains: syntax validation, consistency checking, and contract management.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from langchain.schema import SystemMessage, HumanMessage

from skillnet.agents.optimizer.feedback.types import SkillFeedback
from skillnet.agents.optimizer.transforms.diff_engine import (
    basic_syntax_check as _basic_syntax_check_impl,
)

from .code_quality_validation import CodeQualityValidationMixin
from .responsibility_validation import ResponsibilityValidationMixin

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer

logger = logging.getLogger(__name__)


class ValidationMixin(
    CodeQualityValidationMixin,
    ResponsibilityValidationMixin,
):
    """
    Validation Mixin - Code validation and consistency checking methods

    Composed from sub-mixins:
    - CodeQualityValidationMixin: Code quality checks
    - ResponsibilityValidationMixin: Skill responsibility boundary verification

    This class provides:
    - JavaScript syntax validation
    - Optimization consistency with feedback history
    - Contract violation detection and repair

    Required self attributes:
        - self.logger: Logger instance
        - self.llm: LLM instance
        - self.skill_graph_manager: SkillGraphManager instance
        - self._bloat_checker: BloatChecker instance
        - self.feedback_history: List[SkillFeedback]
        - self._invoke_llm_with_stats(): Method for LLM invocation
        - self._robust_json_parse(): Method for JSON parsing
        - self._format_feedbacks_for_llm(): Method for formatting feedbacks
        - self._generate_diff_for_consistency_check(): Method for diff generation
    """

    # ========== JavaScript Syntax Validation ==========

    def _validate_javascript_syntax(
        self: "SkillGraphOptimizer",
        code: str
    ) -> Tuple[bool, str]:
        """
        Validate code syntax using language-specific parser.

        Args:
            code: Code to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if syntax is valid, False otherwise
            - error_message: Empty string if valid, error description if invalid
        """
        if not code or not code.strip():
            return False, "Empty code"

        # Use skill_language if available
        lang = getattr(self, '_skill_language', None)
        if lang is None:
            # Registry-fallback: resolve via the active DomainKnowledge.
            # A.4 will replace this with an agent-side cached `self._skill_language`.
            from skillnet.core.dk_registry import get_domain_knowledge
            dk = get_domain_knowledge()
            if dk is None:
                raise RuntimeError("No DomainKnowledge registered; cannot parse skill code")
            lang = dk.get_skill_language_impl()

        try:
            result = lang.validate_syntax(code)
            if result.valid:
                return True, ""
            error_msg = result.errors[0] if result.errors else ""

            # Preserve previous "fall back to basic check for richer info" behavior
            # when the language backend reports invalid but with no useful detail.
            if not error_msg:
                basic_valid, basic_error = self._basic_syntax_check(code)
                if not basic_valid:
                    return False, basic_error
            return False, error_msg
        except Exception as e:
            error_str = str(e)
            basic_valid, basic_error = self._basic_syntax_check(code)
            if not basic_valid:
                return False, basic_error
            return False, f"Parse error: {error_str[:200]}"

    def _basic_syntax_check(
        self: "SkillGraphOptimizer",
        code: str
    ) -> Tuple[bool, str]:
        """delegates to the pure function transforms.basic_syntax_check"""
        return _basic_syntax_check_impl(code)

    def identify_syntax_error_source(
        self: "SkillGraphOptimizer",
        error_message: str,
        skill_names: List[str]
    ) -> Optional[str]:
        """Public API: Identify which skill in a sequence caused a syntax error."""
        return self._identify_syntax_error_source(error_message, skill_names)

    def _identify_syntax_error_source(
        self: "SkillGraphOptimizer",
        error_message: str,
        skill_names: List[str]
    ) -> Optional[str]:
        """
        When a syntax error is encountered, scan the code of all involved skills to find the source.

        Used to pre-localize the real source of a syntax error before LLM analysis.

        Args:
            error_message: error message
            skill_names: list of involved skill names

        Returns:
            Optional[str]: name of the skill with a syntax error, or None if none have problems
        """
        # Check whether this is a syntax error
        syntax_error_patterns = [
            "Unexpected end of input",
            "Unexpected token",
            "SyntaxError",
            "missing",  # missing } or )
        ]

        is_syntax_error = any(p.lower() in error_message.lower() for p in syntax_error_patterns)
        if not is_syntax_error:
            return None

        self.logger.info(f"\033[36m[Syntax Precheck] Syntax error detected; pre-checking each skill's code...\033[0m")

        for skill_name in skill_names:
            # Use the new get_skill_code method that supports both the main graph and task-specific skills
            code = self.skill_graph_manager.get_skill_code(skill_name)
            if code:
                is_valid, syntax_error = self._validate_javascript_syntax(code)
                if not is_valid:
                    self.logger.info(f"\033[33m[Syntax Precheck] Syntax error found in '{skill_name}': {syntax_error}\033[0m")
                    return skill_name

        self.logger.info(f"\033[36m[Syntax Precheck] No syntax error found in any known skill\033[0m")
        return None

    # ========== Optimization Consistency Check ==========

    def _check_optimization_consistency(
        self: "SkillGraphOptimizer",
        skill_name: str,
        new_code: str,
        last_k_feedbacks: int = 5,
    ) -> Dict[str, Any]:
        """
        Check whether the optimized code remains consistent with the last k feedbacks (momentum mechanism).

        Uses the unified-diff format to show code changes; more efficient and precise than passing full code.

        Args:
            skill_name: skill name
            new_code: optimized new code
            last_k_feedbacks: number of recent feedbacks to check (default 5)

        Returns:
            Dict[str, Any]: {
                "consistent": bool,  # whether consistent
                "conflicts": List[str],  # descriptions of conflicting feedbacks
                "consistency_score": float,  # consistency score (0-1)
            }
        """
        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return {"consistent": True, "conflicts": [], "consistency_score": 1.0}

        # Get the last k feedbacks (taking from unused feedbacks)
        recent_feedbacks = []
        all_feedbacks = node.gradients.feedback

        # Take the last k feedbacks from the end and convert them to SkillFeedback objects
        for i in range(len(all_feedbacks) - 1, max(-1, len(all_feedbacks) - last_k_feedbacks - 1), -1):
            if i >= 0:
                gradient_item = all_feedbacks[i]
                # Convert SkillGradientItem to SkillFeedback
                content = gradient_item.content
                # Parse feedback type and content (FIX: extended type detection)
                feedback_type = "error"
                if content.startswith("Error:"):
                    feedback_type = "error"
                    content = content[6:].strip()
                elif content.startswith("Critique:"):
                    feedback_type = "critique"
                    content = content[9:].strip()
                elif content.startswith("Performance:"):
                    feedback_type = "performance"
                    content = content[12:].strip()
                elif content.startswith("[CALLER FIX NEEDED]"):
                    feedback_type = "caller_error"
                    # Do not strip the prefix; keep the full content
                elif content.startswith("LLM Analysis:") or "**LLM Deep Analysis Result**" in content:
                    feedback_type = "llm_analysis"
                    # Do not strip the prefix; keep the full content

                feedback = SkillFeedback(
                    skill_name=skill_name,
                    feedback_type=feedback_type,
                    content=content,
                    source="gradients",
                    timestamp=gradient_item.timestamp,
                    severity="high" if feedback_type in ("error", "caller_error") else "medium",
                )
                recent_feedbacks.append(feedback)

        # FIX: supplement caller_error and llm_analysis feedbacks from feedback_history
        # These may not have been synced to gradients
        existing_contents = {fb.content for fb in recent_feedbacks}
        feedback_from_history = [
            fb for fb in self.feedback_history
            if fb.skill_name == skill_name
            and fb.feedback_type in ("caller_error", "llm_analysis")
            and fb.content not in existing_contents  # avoid duplicates
        ][-last_k_feedbacks:]

        if feedback_from_history:
            self.logger.info(f"\033[36m[Consistency Check] Supplemented {len(feedback_from_history)} caller_error/llm_analysis feedbacks from feedback_history\033[0m")
            recent_feedbacks.extend(feedback_from_history)

        if not recent_feedbacks:
            return {"consistent": True, "conflicts": [], "consistency_score": 1.0}

        # Strip misleading eval-bundle line numbers from
        # feedback before the consistency check.  Mineflayer executes skills as
        # a combined JS bundle; errors report the BUNDLE line (e.g., "line 763")
        # not the skill's internal line.  When the consistency-check LLM sees
        # "error at line 763" but the diff only covers 136 lines, it concludes
        # "fix doesn't address the error" (score=0.00) — a false rejection.
        # Stripping "Your code:NNN\n(line not available)\n" removes the noise.
        import re as _re
        for fb in recent_feedbacks:
            fb.content = _re.sub(
                r'Your code:\d+\s*\n\s*\(line not available\)\s*\n\s*',
                '', fb.content
            )
            # Also strip "at evaluateCode...index.js:NNN" stack frames
            fb.content = _re.sub(
                r'\s*at (?:async )?(?:evaluateCode|eval) \([^\)]*\)\s*', ' ', fb.content
            )

        # Use the LLM to check consistency
        feedback_summary = self._format_feedbacks_for_llm(recent_feedbacks)

        # Choose an appropriate diff format based on code length
        # Short code (<1000 lines): use annotated diff (full code + markers, similar to Cursor)
        # Long code (>=1000 lines): use unified diff (saves tokens)
        diff_text, diff_type = self._generate_diff_for_consistency_check(
            node.code, new_code, skill_name, short_code_threshold=1000
        )

        # Compute change statistics
        old_lines = len(node.code.splitlines())
        new_lines = len(new_code.splitlines())

        # Count additions and deletions based on diff type
        if diff_type == "annotated":
            additions = diff_text.count('\n[+]')
            deletions = diff_text.count('\n[-]')
        else:
            additions = diff_text.count('\n+') - 1  # subtract the +++ line
            deletions = diff_text.count('\n-') - 1  # subtract the --- line

        # Use the externalized prompt
        from skillnet.agents.optimizer.prompts import OptimizerPromptLoader
        prompt_name = "consistency_check_annotated" if diff_type == "annotated" else "consistency_check_unified"
        template = OptimizerPromptLoader.load(prompt_name)

        # Build available functions note for the consistency checker
        # This prevents false "undefined reference" rejections when the optimizer
        # correctly refactors local helpers to call external skills or primitives
        available_skills = set(self.skill_graph_manager.get_all_skill_names())
        available_skills.discard(skill_name)  # Don't list the skill being optimized
        _dk = getattr(self, '_domain_knowledge', None)
        _dk_fns = _dk.get_known_functions() if _dk else None
        known_primitives = _dk_fns.get("primitives", set()) if _dk_fns else set()
        known_helpers = _dk_fns.get("helpers", set()) if _dk_fns else set()
        # v12 fix B2: pathfinder Goal classes (GoalPlaceBlock, GoalNear, ...)
        # are destructured into the /step handler scope at
        # mineflayer/index.js:497-523 and accessible to skill code via
        # JavaScript scope chain. Without listing them here, the consistency
        # check LLM previously flagged valid `new GoalPlaceBlock(...)` calls
        # as "undefined reference" (v12 log line 11442), false-rejecting
        # entire Phase 2 optimizations including correct architecture-aware
        # fixes. Including them prevents this class of false positive.
        pathfinder_globals = _dk_fns.get("pathfinder_globals", set()) if _dk_fns else set()
        all_available = sorted(available_skills | known_primitives | known_helpers | pathfinder_globals)
        if all_available:
            available_functions_note = (
                "\n=== AVAILABLE EXTERNAL FUNCTIONS ===\n"
                "The following functions are defined in the execution namespace and can be "
                "called via `await funcName(bot, ...)` (or via `new ClassName(...)` for "
                "constructor-style names like Goal*/Movements which are destructured at "
                "the enclosing route handler scope and reachable through the scope chain). "
                "They are NOT undefined references. "
                "However, their existence does not mean calling them is the correct fix "
                "for the issues described in the feedback.\n"
                f"{', '.join(all_available)}\n"
            )
        else:
            available_functions_note = ""

        system_prompt, human_prompt = template.format(
            skill_name=skill_name,
            old_lines=old_lines,
            new_lines=new_lines,
            additions=additions,
            deletions=deletions,
            diff_text=diff_text,
            last_k_feedbacks=last_k_feedbacks,
            feedback_summary=feedback_summary,
            available_functions_note=available_functions_note,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            response = self._invoke_llm_with_stats(
                messages=messages,
                process_type="consistency_check",
                function_name="_check_optimization_consistency",
                skill_name=skill_name,
            )
            response_content = response.content

            # Parse JSON (using the robust parser)
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = self._robust_json_parse(json_match.group(), "Consistency Check")
                if result:
                    return result

            # On parse failure, conservatively assume consistent
            return {"consistent": True, "conflicts": [], "consistency_score": 0.5, "explanation": "Failed to parse consistency check"}
        except Exception as e:
            print(f"\033[33mWarning: Consistency check failed: {e}\033[0m")
            # Conservative strategy: assume consistent
            return {"consistent": True, "conflicts": [], "consistency_score": 0.5, "explanation": f"Consistency check error: {e}"}
