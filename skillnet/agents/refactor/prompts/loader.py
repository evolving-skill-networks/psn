"""
Prompt Loader

Utility class for loading and managing LLM prompts.
"""

import os
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from pathlib import Path


@dataclass
class PromptTemplate:
    """Prompt template."""
    system: str
    human: str

    def format(self, **kwargs) -> tuple:
        """
        Format the prompt template.

        Args:
            **kwargs: Template variables

        Returns:
            tuple: (formatted_system, formatted_human)
        """
        import re

        def safe_format(template: str, **kw) -> str:
            """Format using only placeholders present in the template, avoiding KeyError."""
            # Find all placeholders in the template
            placeholders = set(re.findall(r'\{(\w+)\}', template))
            if not placeholders:
                # No placeholders; return as-is
                return template
            if not kw:
                # No arguments provided; return the raw template (keeping placeholders)
                return template
            # Provide a value for each placeholder in the template: use the value from kw, or keep the original placeholder string
            format_kwargs = {}
            for ph in placeholders:
                if ph in kw:
                    format_kwargs[ph] = kw[ph]
                else:
                    # Keep unprovided placeholders (in their escaped form)
                    format_kwargs[ph] = '{' + ph + '}'
            return template.format(**format_kwargs)

        return (
            safe_format(self.system, **kwargs),
            safe_format(self.human, **kwargs)
        )


class PromptLoader:
    """
    Prompt loader.

    Loads LLM prompts from files; supports caching and template substitution.

    Usage:
        loader = PromptLoader()
        template = loader.load("parametric_wrapper")
        system, human = template.format(
            source_name="craftOakBoat",
            target_name="craftBoat",
            ...
        )
    """

    # Prompts directory
    PROMPTS_DIR = Path(__file__).parent

    # Cache
    _cache: Dict[str, PromptTemplate] = {}

    # Thread-safe lock
    _lock = threading.Lock()

    # Logger
    _logger = logging.getLogger(__name__)

    @classmethod
    def load(cls, prompt_name: str) -> PromptTemplate:
        """
        Load the specified prompt template.

        Args:
            prompt_name: Prompt name (without extension)

        Returns:
            PromptTemplate: The loaded template

        Raises:
            FileNotFoundError: if the prompt file does not exist
        """
        # First check the cache (lock-free fast path)
        if prompt_name in cls._cache:
            return cls._cache[prompt_name]

        # Acquire the lock to protect cache writes
        with cls._lock:
            # Double-check to avoid duplicate loading
            if prompt_name in cls._cache:
                return cls._cache[prompt_name]

            prompt_file = cls.PROMPTS_DIR / f"{prompt_name}.txt"

            if not prompt_file.exists():
                available = cls.list_available_prompts()
                raise FileNotFoundError(
                    f"Prompt file not found: {prompt_name}\n"
                    f"Available prompts: {', '.join(available) if available else 'none'}\n"
                    f"Expected path: {prompt_file}"
                )

            content = prompt_file.read_text(encoding="utf-8")
            template = cls._parse_prompt_file(content)

            cls._cache[prompt_name] = template
            return template

    @classmethod
    def load_or_default(
        cls,
        prompt_name: str,
        default_system: str = "You are a helpful assistant.",
        default_human: str = "{task}",
    ) -> PromptTemplate:
        """
        Load the prompt template; return a default template if the file does not exist.

        When the prompt file is missing, return a basic default template instead of
        raising an exception. Useful for LLM fallback logic.

        Args:
            prompt_name: Prompt name
            default_system: Default system prompt
            default_human: Default human prompt

        Returns:
            PromptTemplate: Loaded template or default template
        """
        try:
            return cls.load(prompt_name)
        except FileNotFoundError as e:
            cls._logger.warning(
                f"[PromptLoader] Using default template; reason: {e}"
            )
            return PromptTemplate(system=default_system, human=default_human)

    @classmethod
    def list_available_prompts(cls) -> List[str]:
        """
        List all available prompt names.

        Returns:
            List[str]: List of available prompt names
        """
        return [f.stem for f in cls.PROMPTS_DIR.glob("*.txt")]

    @classmethod
    def _parse_prompt_file(cls, content: str) -> PromptTemplate:
        """
        Parse the contents of a prompt file.

        File format:
        ```
        === SYSTEM ===
        System prompt content...

        === HUMAN ===
        Human prompt content...
        ```
        """
        system_marker = "=== SYSTEM ==="
        human_marker = "=== HUMAN ==="

        if system_marker not in content or human_marker not in content:
            raise ValueError(
                f"Invalid prompt file format. Must contain '{system_marker}' and '{human_marker}'"
            )

        # Split the content
        parts = content.split(human_marker)
        system_part = parts[0].replace(system_marker, "").strip()
        human_part = parts[1].strip() if len(parts) > 1 else ""

        return PromptTemplate(system=system_part, human=human_part)

    @classmethod
    def clear_cache(cls):
        """Clear the cache."""
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def get_all_prompts(cls) -> Dict[str, PromptTemplate]:
        """
        Get all available prompts.

        Loads every .txt file in the directory as a prompt template.
        If a file has invalid format, log a warning instead of aborting loading of other files.

        Returns:
            Dict[str, PromptTemplate]: Mapping from prompt name to template
        """
        prompts = {}
        for file in cls.PROMPTS_DIR.glob("*.txt"):
            name = file.stem
            try:
                prompts[name] = cls.load(name)
            except Exception as e:
                cls._logger.warning(f"[PromptLoader] Unable to load prompt '{name}': {e}")
        return prompts
