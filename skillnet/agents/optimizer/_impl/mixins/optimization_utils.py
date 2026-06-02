"""
OptimizationUtilsMixin - Thin delegates and utility methods for optimization.

Extracted from optimizer_impl.py for better modularity.

Methods included (13 thin delegates/utilities):
- _invoke_llm_with_stats: Invoke LLM and record statistics
- _robust_json_parse: Robust JSON parsing for LLM responses
- _extract_issue_type_from_content: Extract issue type from content string
- _add_feedback_to_history: Add feedback with deduplication
- _extract_constraint_feedback_from_critique: Extract constraint feedback from critique
- _remove_unused_helper_functions: Remove helper functions not called by main function
- _sanitize_python_to_js: Sanitize Python-style literals in generated JS code
- _fix_tdz_issues: Fix Temporal Dead Zone issues in code
- _generate_unified_diff: Generate unified diff between old and new code
- _generate_annotated_diff: Generate annotated diff between old and new code
- _generate_diff_for_consistency_check: Choose diff format based on code length
- _format_feedbacks_for_llm: Format feedbacks for LLM consumption
- _print_optimization_log: Print detailed optimization process log
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from skillnet.agents.optimizer.feedback import (
    SkillFeedback,
    extract_constraint_feedback_from_critique as _extract_constraint_feedback_impl,
    extract_issue_type_from_content as _extract_issue_type_impl,
    format_feedbacks_for_llm as _format_feedbacks_for_llm_impl,
)

from skillnet.agents.optimizer.transforms import (
    remove_unused_helper_functions as _remove_unused_helper_functions_impl,
    generate_unified_diff as _generate_unified_diff_impl,
    generate_annotated_diff as _generate_annotated_diff_impl,
)

from skillnet.agents.optimizer.validators import (
    fix_tdz_issues as _fix_tdz_issues_impl,
)

from skillnet.agents.skill_graph.utils import (
    sanitize_python_to_js as _sanitize_python_to_js_impl,
)

from skillnet.agents.optimizer.core import (
    robust_json_parse,
)

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer


class OptimizationUtilsMixin:
    """Thin delegates and utility methods for the optimizer."""

    def _invoke_llm_with_stats(
        self,
        messages,
        process_type: str,
        function_name: str,
        task: str = None,
        skill_name: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Invoke the LLM and record statistics.

        Delegates to LLMInvoker.invoke().

        Args:
            messages: list of LLM messages
            process_type: process type
            function_name: function name
            task: associated task
            skill_name: associated skill name
            metadata: extra metadata

        Returns:
            LLM response object.
        """
        return self._llm_invoker.invoke(
            messages=messages,
            process_type=process_type,
            function_name=function_name,
            task=task,
            skill_name=skill_name,
            metadata=metadata,
        )

    def _robust_json_parse(self, json_str: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Robust JSON parsing for handling malformed JSON returned by the LLM.

        Delegates to the robust_json_parse() function.

        Args:
            json_str: JSON string to parse
            context: context info (used for logging)

        Returns:
            Parsed dict, or None if parsing fails.
        """
        return robust_json_parse(json_str, context, log_func=self.logger.warning)

    # ========== Phase 3: distinguish errors from suggestions ==========

    def _extract_issue_type_from_content(self, content: str) -> str:
        """delegates to feedback.parser.extract_issue_type_from_content."""
        return _extract_issue_type_impl(content)

    def _add_feedback_to_history(self, feedback: SkillFeedback) -> bool:
        """
        P2: Add feedback to feedback_history with deduplication.

        For internal-analysis feedback, dedupe by issue type to avoid repeated
        additions of the same unresolved issue.

        Args:
            feedback: feedback to add

        Returns:
            bool: True if added, False if an identical feedback already exists.
        """
        # For internal-analysis feedback, perform enhanced dedup by issue type
        if feedback.feedback_type == "internal_analysis" and feedback.source == "backpropagation_analysis":
            issue_type = self._extract_issue_type_from_content(feedback.content)
            for existing in self.feedback_history:
                if (existing.skill_name == feedback.skill_name and
                    existing.feedback_type == "internal_analysis" and
                    existing.source == "backpropagation_analysis" and
                    self._extract_issue_type_from_content(existing.content) == issue_type and
                    not getattr(existing, 'resolved', False)):
                    self.logger.debug(f"[Feedback Dedup] skipping same-type internal analysis: {feedback.skill_name}/{issue_type}")
                    return False

        # Generic dedup (based on skill_name, content, feedback_type)
        for existing in self.feedback_history:
            if (existing.skill_name == feedback.skill_name and
                existing.content == feedback.content and
                existing.feedback_type == feedback.feedback_type):
                self.logger.debug(f"[Feedback Dedup] skipping duplicate feedback: {feedback.skill_name}/{feedback.feedback_type}")
                return False

        self.feedback_history.append(feedback)
        return True

    def _extract_constraint_feedback_from_critique(
        self,
        skill_name: str,
        current_critique: str,
        children: List[str],
        current_task: str = None,
    ) -> Optional[SkillFeedback]:
        """delegates to feedback.parser.extract_constraint_feedback_from_critique pure function."""
        return _extract_constraint_feedback_impl(
            skill_name, current_critique, children, current_task, self.logger
        )

    def _remove_unused_helper_functions(
        self,
        code: str,
        main_func_name: str
    ) -> Tuple[str, List[str]]:
        """Remove helper functions not called by main function."""
        if self._skill_language:
            return self._skill_language.remove_unused_helpers(code, main_func_name)
        return _remove_unused_helper_functions_impl(code, main_func_name)

    def _sanitize_python_to_js(self, code: str) -> str:
        """Sanitize Python-style literals in generated JS code."""
        if self._skill_language:
            return self._skill_language.sanitize_llm_output(code)
        result = _sanitize_python_to_js_impl(code)
        if result != code:
            self.logger.warning(f"\033[33m[Optimizer] detected Python syntax and converted it to JavaScript\033[0m")
        return result

    def _fix_tdz_issues(self, code: str) -> tuple:
        """delegates to validators.fix_tdz_issues."""
        return _fix_tdz_issues_impl(code)

    def _generate_unified_diff(self, old_code: str, new_code: str, skill_name: str) -> str:
        """delegates to transforms.generate_unified_diff pure function."""
        return _generate_unified_diff_impl(old_code, new_code, skill_name)

    def _generate_annotated_diff(self, old_code: str, new_code: str, skill_name: str,
                                  collapse_unchanged: int = 15) -> str:
        """delegates to transforms.generate_annotated_diff pure function."""
        return _generate_annotated_diff_impl(old_code, new_code, skill_name, collapse_unchanged)

    def _generate_diff_for_consistency_check(self, old_code: str, new_code: str,
                                              skill_name: str, short_code_threshold: int = 1000) -> tuple:
        """
        Choose a diff format based on code length.

        Args:
            old_code: original code
            new_code: new code
            skill_name: skill name
            short_code_threshold: short-code threshold in lines (default 1000)

        Returns:
            tuple: (diff_text, diff_type) — diff content and type ("annotated" or "unified")
        """
        new_lines_count = len(new_code.splitlines())

        if new_lines_count < short_code_threshold:
            # Short code: use annotated diff (full code + markers)
            diff_text = self._generate_annotated_diff(old_code, new_code, skill_name)
            return diff_text, "annotated"
        else:
            # Long code: use unified diff (save tokens)
            diff_text = self._generate_unified_diff(old_code, new_code, skill_name)
            return diff_text, "unified"

    def _format_feedbacks_for_llm(self, feedbacks: List[SkillFeedback]) -> str:
        """delegates to feedback.parser.format_feedbacks_for_llm."""
        return _format_feedbacks_for_llm_impl(feedbacks)


    def _print_optimization_log(
        self,
        skill_name: str,
        old_code: str,
        new_code: str,
        feedbacks: List[SkillFeedback],
        diagnosis: Dict[str, Any] = None,
        optimization_result: Dict[str, Any] = None,
    ):
        """
        Print a detailed log of the optimization process.

        Args:
            skill_name: skill name
            old_code: code before optimization
            new_code: code after optimization
            feedbacks: feedbacks used for optimization
            diagnosis: diagnostic result (optional)
            optimization_result: optimization result (optional)
        """
        print(f"\n{'='*80}")
        print(f"\033[36m[Optimization Log] Skill: {skill_name}\033[0m")
        print(f"{'='*80}\n")

        # 1. Feedbacks
        if feedbacks:
            print(f"\033[33m[Feedbacks] feedbacks used for optimization ({len(feedbacks)} item(s)):\033[0m")
            for i, fb in enumerate(feedbacks, 1):
                print(f"  {i}. [{fb.feedback_type.upper()}] {fb.severity.upper()}")
                print(f"     source: {fb.source}")
                print(f"     content: {fb.content[:200]}{'...' if len(fb.content) > 200 else ''}")
                print()
        else:
            print(f"\033[33m[Feedbacks] no feedbacks\033[0m\n")

        # 2. Diagnosis
        if diagnosis:
            print(f"\033[33m[Diagnosis] issue analysis:\033[0m")
            issues = diagnosis.get("issues", [])
            if issues:
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. [{issue.get('type', 'unknown')}] {issue.get('description', '')}")
            root_cause = diagnosis.get("root_cause", "")
            if root_cause:
                print(f"  root cause: {root_cause}")
            print()

        # 3. Code Diff — use annotated diff format to show the full code change
        print(f"\033[33m[Code Changes] code changes:\033[0m")
        if old_code == new_code:
            print("  code unchanged")
        else:
            old_lines = old_code.split('\n')
            new_lines = new_code.split('\n')

            print(f"  old code line count: {len(old_lines)}")
            print(f"  new code line count: {len(new_lines)}")

            # Show change summary
            if optimization_result:
                change_summary = optimization_result.get("change_summary", "")
                if change_summary:
                    print(f"  change summary: {change_summary}")

            # Use annotated diff format to show the code change
            print(f"\n  \033[36m=== Annotated Diff (full code change) ===\033[0m")
            annotated_diff = self._generate_annotated_diff(old_code, new_code, skill_name, collapse_unchanged=10)
            # Print each line with indentation
            for line in annotated_diff.split('\n'):
                # Add color based on marker
                if line.startswith('[+]'):
                    print(f"  \033[32m{line}\033[0m")  # Green for additions
                elif line.startswith('[-]'):
                    print(f"  \033[31m{line}\033[0m")  # Red for deletions
                elif line.startswith('[ ]'):
                    print(f"  {line}")  # Normal color for unchanged
                elif '... //' in line:
                    print(f"  \033[33m{line}\033[0m")  # Yellow for collapsed
                else:
                    print(f"  {line}")

        print(f"\n{'='*80}\n")
