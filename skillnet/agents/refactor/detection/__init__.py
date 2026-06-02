"""
Refactor Detection Package

Refactor detection module: pre-screen candidate skills, analyze refactor relationships.

Responsibilities migrated from graph_manager_impl.py:
- Pre-screen candidate skills (prescreener.py)
- Detect refactor relationships (analyzer.py)

v5.0 architectural reorganization
"""

from .prescreener import (
    RefactorPrescreener,
    calculate_name_similarity,
    is_generalization_of,
)

from .analyzer import (
    RefactorRelationshipAnalyzer,
    estimate_functional_scope,
    estimate_implementation_maturity,
    estimate_parameterization_level,
)

__all__ = [
    # Prescreener
    "RefactorPrescreener",
    "calculate_name_similarity",
    "is_generalization_of",

    # Analyzer
    "RefactorRelationshipAnalyzer",
    "estimate_functional_scope",
    "estimate_implementation_maturity",
    "estimate_parameterization_level",
]
