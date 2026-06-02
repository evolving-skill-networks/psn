"""
Parameterized Action Agent package.

Facade module re-exporting ParameterizedActionAgent for backward compatibility.
"""

from ..action import ActionAgent
from ._agent import ParameterizedActionAgent

__all__ = ["ParameterizedActionAgent", "ActionAgent"]
