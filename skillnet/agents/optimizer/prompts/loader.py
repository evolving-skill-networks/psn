"""Optimizer prompt loader — flat directory of `<name>.txt` files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class PromptTemplate:
    """A two-part LLM prompt template with `{placeholder}` substitution."""

    system: str
    human: str

    def format(self, **kwargs) -> Tuple[str, str]:
        """Return `(system, human)` with all `{placeholders}` substituted."""
        return self.format_system(**kwargs), self.format_human(**kwargs)

    def format_system(self, **kwargs) -> str:
        return _substitute(self.system, kwargs)

    def format_human(self, **kwargs) -> str:
        return _substitute(self.human, kwargs)


def _substitute(text: str, kwargs: Dict[str, object]) -> str:
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))
    return text


class OptimizerPromptLoader:
    """Cached loader for `<prompt_name>.txt` files in this directory.

    File format::

        === SYSTEM ===
        ...system prompt with {placeholders}...

        === HUMAN ===
        ...human prompt with {placeholders}...

    Usage::

        template = OptimizerPromptLoader.load("code_optimization")
        system, human = template.format(skill_name="craftOakBoat", ...)
    """

    PROMPTS_DIR = Path(__file__).parent
    _cache: Dict[str, PromptTemplate] = {}

    @classmethod
    def load(cls, prompt_name: str) -> PromptTemplate:
        """Load `<prompt_name>.txt` from PROMPTS_DIR, with caching."""
        if prompt_name in cls._cache:
            return cls._cache[prompt_name]

        prompt_file = cls.PROMPTS_DIR / f"{prompt_name}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        template = cls._parse(prompt_file.read_text(encoding="utf-8"))
        cls._cache[prompt_name] = template
        return template

    @staticmethod
    def _parse(content: str) -> PromptTemplate:
        system_marker = "=== SYSTEM ==="
        human_marker = "=== HUMAN ==="
        if system_marker not in content or human_marker not in content:
            raise ValueError(
                f"Invalid prompt file format: must contain {system_marker!r} and {human_marker!r}"
            )
        parts = content.split(human_marker)
        system_part = parts[0].replace(system_marker, "").strip()
        human_part = parts[1].strip() if len(parts) > 1 else ""
        return PromptTemplate(system=system_part, human=human_part)
