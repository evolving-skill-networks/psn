"""
PreconditionChecker Facade (Layer E)

PreconditionChecker main class: composes CheckingMixin and SkillFindingMixin.

extracted from precondition_checker.py.

Classes:
- PreconditionChecker: PreconditionChecker main class (facade)
"""

import logging
from typing import Optional

from ._checking import CheckingMixin
from ._skill_finding import SkillFindingMixin

logger = logging.getLogger(__name__)


class PreconditionChecker(CheckingMixin, SkillFindingMixin):
    """
    Precondition checker.

    Responsibilities:
    1. Check whether preconditions are satisfied
    2. Find Skills that can produce the required preconditions
    3. Support complex preconditions with AND/OR logic
    4. Support tool tier matching (require_or_better)

    refactored to a mixin composition pattern.
    - CheckingMixin: precondition validation, environmental feedback, inventory updates
    - SkillFindingMixin: skill lookup
    """

    def __init__(
        self,
        skill_graph_manager,
        effect_matcher=None,
        custom_logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the precondition checker.

        Args:
            skill_graph_manager: SkillGraphManager instance
            effect_matcher: EffectMatcher instance (used for effect matching)
            custom_logger: optional custom logger
        """
        self.skill_graph_manager = skill_graph_manager
        self.effect_matcher = effect_matcher
        self.logger = custom_logger or logger
