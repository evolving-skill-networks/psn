"""
ParameterizedActionAgent - main class definition.

This module extends ActionAgent to support parameterized skills.
Skills can accept parameters beyond just the bot object, allowing
for more flexible and reusable skill calls.

The class composes functionality from mixin modules:
- SkillNamingMixin: Skill name normalization (rule-based and LLM)
- PromptRenderingMixin: System message rendering with parameterized skills
- CodeValidationMixin: Code validation (brackets, TDZ, recursion)
- CodeParsingMixin: Babel-based code parsing and main function identification
- SkillSelectionMixin: Skill selection based on parameterization analysis
- CodeAssemblyMixin: Code assembly, name replacement, reparameterization
- CodeFinalizationMixin: Final validation and result dict construction
"""

import time

from langchain.schema import AIMessage

from ..action import ActionAgent
from .mixins import (
    SkillNamingMixin,
    PromptRenderingMixin,
    CodeValidationMixin,
    CodeParsingMixin,
    SkillSelectionMixin,
    CodeAssemblyMixin,
    CodeFinalizationMixin,
)


class ParameterizedActionAgent(
    SkillNamingMixin,
    PromptRenderingMixin,
    CodeValidationMixin,
    CodeParsingMixin,
    SkillSelectionMixin,
    CodeAssemblyMixin,
    CodeFinalizationMixin,
    ActionAgent,
):
    """
    Parameterized Action Agent that supports parameterized skills.

    Extends ActionAgent to handle skills with parameters. When skills
    have parameter metadata, the agent will format them appropriately
    and guide the LLM to use parameters when calling skills.
    """

    def __init__(
        self,
        model_name="gpt-5-mini",
        temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
        use_llm_for_normalization=False,
        include_skill_code=False,  # controls whether to include the full skill code in the prompt
        openai_api_base=None,
        openai_api_key=None,
        max_tokens=None,
    ):
        """
        Initialize ParameterizedActionAgent

        Args:
            model_name: LLM model name
            temperature: LLM temperature
            request_timout: API request timeout
            ckpt_dir: Checkpoint directory
            resume: Whether to resume from checkpoint
            chat_log: Whether to show chat log
            execution_error: Whether to show execution errors
            use_llm_for_normalization: Whether to use LLM for skill name normalization (default: False, uses rule-based)
            include_skill_code: Whether to include full skill code in prompt (default: False, only signatures)
            openai_api_base: Custom API base URL for vLLM or compatible endpoints
            openai_api_key: Custom API key
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            request_timout=request_timout,
            ckpt_dir=ckpt_dir,
            resume=resume,
            chat_log=chat_log,
            execution_error=execution_error,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            max_tokens=max_tokens,
        )
        self.use_llm_for_normalization = use_llm_for_normalization
        self.include_skill_code = include_skill_code
        self._skill_manager_ref = None  # Set via set_skill_manager() for function reference validation

    def set_skill_manager(self, skill_manager):
        """Set reference to skill manager for function reference validation."""
        self._skill_manager_ref = skill_manager

    def extract_code_from_message(self, message, task: str = None):
        """Public API: Extract code, function name, and full code from an AI message."""
        return self._extract_code_from_message(message, task=task)

    def process_ai_message(self, message, existing_skills=None):
        """
        Process AI message - same as ActionAgent but allows skills to have parameters

        The main function must still take only 'bot' as argument, but
        skills can have additional parameters.

        Args:
            message: AI message to process
            existing_skills: Optional list of existing skill codes to check for redundancy
        """
        assert isinstance(message, AIMessage)
        if existing_skills is None:
            existing_skills = []

        retry = 3
        error = None
        while retry > 0:
            try:
                # Resolve skill language: cache-first, registry-fallback.
                skill_lang = getattr(self, "_skill_language", None)
                if skill_lang is None:
                    from skillnet.core.dk_registry import get_domain_knowledge
                    dk = get_domain_knowledge()
                    if dk is None:
                        raise RuntimeError("No DomainKnowledge registered; cannot parse skill code")
                    skill_lang = dk.get_skill_language_impl()

                # _parse_and_identify_main resolves the SkillLanguage internally
                # (cache-first via self._skill_language, registry-fallback).
                functions, main_function = self._parse_and_identify_main(
                    message.content, skill_lang
                )

                selection = self._select_skill_to_save(main_function, functions, existing_skills)
                if isinstance(selection, dict):
                    return selection  # early return (no-save case)
                skill_to_save, functions_to_include, wrapper_removed = selection

                program_code, exec_code, normalized_name, all_top_level_declarations = self._assemble_and_normalize(
                    skill_to_save, functions_to_include, functions, main_function, wrapper_removed
                )

                return self._validate_and_finalize(
                    program_code, exec_code, normalized_name,
                    skill_to_save, functions, functions_to_include,
                    skill_lang, main_function
                )
            except Exception as e:
                retry -= 1
                error = e
                time.sleep(1)
        return f"Error parsing action response (before program execution): {error}"
