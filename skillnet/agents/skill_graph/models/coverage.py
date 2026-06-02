"""
Coverage Type

Defines skill coverage types, used to distinguish different coverage semantics.

Coverage types fall into two broad categories:
1. GENERALIZATION types: the covered skill is a wrapper that calls a more general skill
2. COMPOSITION types: the covered skill is an independently valuable foundational skill

Usage:
    from skillnet.agents.skill_graph.models.coverage import CoverageType

    node.coverage_type = CoverageType.PARAMETRIC

    if node.coverage_type.is_wrapper:
        # Special handling for wrappers
"""

from enum import Enum
from typing import Optional


class CoverageType(Enum):
    """
    Coverage type enum

    Distinguishes different coverage semantics so the optimizer can provide the correct context.

    Handling by coverage type:
    - No type is ever skipped for optimization
    - Wrapper types get a special hint added to the optimization prompt
    - BloatChecker applies stricter limits to wrapper types
    """

    # === GENERALIZATION types ===
    # The covered skill in these types is a wrapper that calls a more general skill

    PARAMETRIC = "parametric"
    """
    Parametric refactor

    Example: mineOakLogs -> mineLogs(wood_type="oak")
    The wrapper just passes fixed parameters to a more general skill
    """

    SIBLING = "sibling"
    """
    Sibling unification refactor

    Example: craftOakBoat, craftBirchBoat -> craftBoat(wood_type)
    Multiple similar skills are unified into a single general skill
    """

    DUPLICATION = "duplication"
    """
    Duplicate-code refactor

    Duplicate code logic is extracted into a general skill
    """

    FUNCTIONAL_SUPERSET = "functional_superset"
    """
    Functional-superset refactor

    The skill's functionality is fully covered by a more powerful skill
    """

    # === COMPOSITION types ===
    # The covered skill in these types is an independently valuable foundational skill

    COMMON_SUBSKILL = "extract_common_subskill"
    """
    Common subskill extraction

    Example: setupCraftingTable shared by multiple crafting skills
    Not a wrapper — an independent foundational capability
    """

    # === Special types ===

    BEHAVIORAL = "behavioral"
    """
    Behavioral refactor

    The target skill calls the source skill
    The source skill does not set is_covered and remains independent
    """

    @property
    def is_wrapper(self) -> bool:
        """
        Whether this is a wrapper type

        Wrapper-type skills are simple wrappers around other skills.
        They need special handling during optimization:
        - Stay concise (usually < 20 lines)
        - Only fix parameter-passing issues
        - Should not add new logic

        Returns:
            bool: True if this is a wrapper type
        """
        return self in {
            CoverageType.PARAMETRIC,
            CoverageType.SIBLING,
            CoverageType.DUPLICATION,
            CoverageType.FUNCTIONAL_SUPERSET,
        }

    @property
    def is_independent(self) -> bool:
        """
        Whether this is an independently valuable skill

        These skills may be marked as covered, but they
        are independent foundational capabilities, not simple wrappers.

        Returns:
            bool: True if it is an independent skill
        """
        return self in {
            CoverageType.COMMON_SUBSKILL,
            CoverageType.BEHAVIORAL,
        }

    @classmethod
    def from_refactor_type(cls, refactor_type: Optional[str]) -> Optional['CoverageType']:
        """
        Convert from the legacy refactor_type string

        Used for backward compatibility — converts the old string type into the new enum.

        Args:
            refactor_type: legacy refactor_type string

        Returns:
            CoverageType or None: matching CoverageType, or None if no match
        """
        if not refactor_type:
            return None

        mapping = {
            "parametric": cls.PARAMETRIC,
            "sibling": cls.SIBLING,
            "duplication": cls.DUPLICATION,
            "functional_superset": cls.FUNCTIONAL_SUPERSET,
            "extract_common_subskill": cls.COMMON_SUBSKILL,
            "behavioral": cls.BEHAVIORAL,
        }

        return mapping.get(refactor_type.lower())
