"""
Code Edit Mixin - Code transformation and fix operations.

Extracted from optimizer_impl.py for better modularity.
Removed dead deterministic transform system (_apply_single_edit and all wrapper methods).

Methods included (non-shadowed):
- _format_generated_code: Format generated code with Prettier
- _fix_syntax_error: Fix syntax error using LLM
- _basic_syntax_check: Basic syntax validation

Note: _apply_patch, _apply_unified_diff, _apply_simple_diff remain in
optimizer_impl.py (authoritative versions).
"""

import logging
from typing import Optional, Tuple, TYPE_CHECKING

from skillnet.agents.optimizer.transforms import basic_syntax_check as _basic_syntax_check_impl

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer

logger = logging.getLogger(__name__)


class CodeEditMixin:
    """
    Code Edit Mixin - Code formatting, syntax fixing, and validation.

    Required self attributes:
    - self.logger: Logger instance
    - self._invoke_llm_with_stats: Method for LLM invocation
    - self._robust_json_parse: Method for JSON parsing
    - self._validate_javascript_syntax: Method for syntax validation
    """

    def _format_generated_code(
        self: "SkillGraphOptimizer",
        code: str
    ) -> str:
        """
        Unified formatting exit - Format LLM generated code.

        All optimization methods should call this before returning code.
        Uses SkillLanguage (or Prettier fallback) to prevent single-line long code issues.
        """
        if not code or not code.strip():
            return code

        # Remove potential markdown code block markers
        from skillnet.utils.code_block import strip_code_block_markers
        _dk = getattr(self, '_domain_knowledge', None)
        _lang = _dk.get_skill_language() if _dk else "javascript"
        code = strip_code_block_markers(code, _lang)

        # Use skill_language if available; otherwise resolve via the active
        # DomainKnowledge (registry-fallback). A.4 will replace this with an
        # agent-side cached `self._skill_language` so the lookup happens once.
        skill_lang = getattr(self, '_skill_language', None)
        if skill_lang is None:
            from skillnet.core.dk_registry import get_domain_knowledge
            dk = get_domain_knowledge()
            if dk is None:
                raise RuntimeError("No DomainKnowledge registered; cannot format skill code")
            skill_lang = dk.get_skill_language_impl()

        formatted = skill_lang.format_code(code)
        if formatted and formatted.strip():
            self.logger.debug(
                f"[Code Format] SkillLanguage formatting succeeded, "
                f"lines: {len(code.splitlines())} -> {len(formatted.splitlines())}"
            )
            return formatted
        return code

    def _fix_syntax_error(
        self: "SkillGraphOptimizer",
        code: str,
        syntax_error: str,
        skill_name: str
    ) -> Optional[str]:
        """
        Request LLM to fix syntax errors in code.

        Args:
            code: Code with syntax error
            syntax_error: Syntax error message
            skill_name: Skill name

        Returns:
            Fixed code, or None if fix failed
        """
        from langchain.schema import SystemMessage, HumanMessage
        from skillnet.agents.optimizer.prompts import OptimizerPromptLoader

        template = OptimizerPromptLoader.load("syntax_fix")
        system_prompt, human_prompt = template.format(
            skill_name=skill_name,
            code=code,
            syntax_error=syntax_error,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            response = self._invoke_llm_with_stats(
                messages=messages,
                process_type="syntax_fix",
                function_name="_fix_syntax_error",
                skill_name=skill_name,
            )
            result = self._robust_json_parse(response.content, "Syntax Fix")
            if result and result.get("fixed_code"):
                fixed_code = result["fixed_code"]
                is_valid, _ = self._validate_javascript_syntax(fixed_code)
                if is_valid:
                    self.logger.info(
                        f"\033[32m[Syntax Fix] Successfully fixed: "
                        f"{result.get('fix_description', 'N/A')}\033[0m"
                    )
                    return fixed_code
                else:
                    self.logger.warning(
                        "\033[33m[Syntax Fix] Fixed code still has syntax errors\033[0m"
                    )
        except Exception as e:
            self.logger.warning(f"[Syntax Fix] Fix failed: {e}")

        return None

    def _basic_syntax_check(
        self: "SkillGraphOptimizer",
        code: str
    ) -> Tuple[bool, str]:
        """Delegates to transforms.basic_syntax_check"""
        return _basic_syntax_check_impl(code)
