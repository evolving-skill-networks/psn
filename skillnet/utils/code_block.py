"""
Code Block Extraction Utilities

Domain-agnostic helpers for extracting fenced code blocks from LLM responses.
Replaces hardcoded ``(?:javascript|js)`` patterns with language-aware extraction.

Usage::

    from skillnet.utils.code_block import build_code_block_pattern, strip_code_block_markers

    pattern = build_code_block_pattern("python")  # matches ```python or ```py
    code = strip_code_block_markers(raw_code, "python")
"""

import re
from typing import Tuple


# Language name → tuple of recognized aliases for code fence detection.
# Order matters: first alias is the "canonical" name used in prompts.
_LANGUAGE_ALIASES = {
    "javascript": ("javascript", "js"),
    "python": ("python", "py"),
    "typescript": ("typescript", "ts"),
    "rust": ("rust", "rs"),
    "java": ("java",),
    "cpp": ("cpp", "c++"),
    "csharp": ("csharp", "cs", "c#"),
    "go": ("go",),
    "ruby": ("ruby", "rb"),
    "lua": ("lua",),
}


def get_language_aliases(language: str) -> Tuple[str, ...]:
    """Return recognized aliases for a language name.

    Falls back to ``(language,)`` for unknown languages.
    """
    return _LANGUAGE_ALIASES.get(language.lower(), (language,))


def build_code_block_pattern(language: str = "javascript") -> re.Pattern:
    """Build a compiled regex for extracting fenced code blocks by language.

    The pattern matches ````` followed by any recognized alias of *language*,
    captures everything up to the closing ``````.

    Args:
        language: Skill language name (e.g., "javascript", "python").

    Returns:
        Compiled regex with one capture group containing the code content.

    Example::

        >>> pat = build_code_block_pattern("python")
        >>> pat.findall("```python\\nprint('hi')\\n```")
        ["\\nprint('hi')\\n"]
    """
    aliases = get_language_aliases(language)
    lang_alt = "|".join(re.escape(a) for a in aliases)
    return re.compile(rf"```(?:{lang_alt})(.*?)```", re.DOTALL)


def strip_code_block_markers(code: str, language: str = "javascript") -> str:
    """Remove leading/trailing markdown code block markers.

    Handles both language-specific markers (e.g., ````` ``javascript````) and
    generic ````` ``` ```` markers.

    Args:
        code: Raw code string potentially wrapped in markdown fences.
        language: Skill language for alias detection.

    Returns:
        Code with markers stripped.
    """
    code = code.strip()
    aliases = get_language_aliases(language)
    stripped = False
    for alias in aliases:
        marker = f"```{alias}"
        if code.startswith(marker):
            code = code[len(marker):].strip()
            stripped = True
            break
    if not stripped and code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code
