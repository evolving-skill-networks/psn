"""Minecraft-specific LLM prompt templates.

Each domain owns its prompt files.  This module provides a loader
for the ``.txt`` templates that live alongside it.
"""

from pathlib import Path

_DIR = Path(__file__).parent


def load_minecraft_prompt(name: str) -> str:
    """Load a Minecraft prompt template by name.

    Returns the file contents, or empty string if the file does not exist.
    """
    path = _DIR / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""
