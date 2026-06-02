"""
Skill-graph metadata extraction — preconditions, effects, parameters.

The three Extractor classes consume a skill's source code + task context
and return structured metadata used by the rest of the optimizer.
"""

from .preconditions import PreconditionExtractor
from .effects import EffectExtractor
from .parameters import ParameterExtractor

__all__ = [
    "PreconditionExtractor",
    "EffectExtractor",
    "ParameterExtractor",
]
