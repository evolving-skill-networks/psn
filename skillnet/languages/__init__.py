"""Skill-language implementations.

Each module here provides a concrete implementation of the
``skillnet.core.skill_language.SkillLanguage`` Protocol for one language.
"""
from skillnet.languages.javascript import JavaScriptLanguage
from skillnet.languages.python import PythonLanguage

__all__ = ["JavaScriptLanguage", "PythonLanguage"]
