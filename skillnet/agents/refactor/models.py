"""
Refactor Data Models

Data-structure definitions for the refactor module.

Contains 3 dataclasses:
- RefactorType: refactor-type enum (5 types)
- RefactorOpportunity: a refactor opportunity
- RefactorResult: a refactor result

Extracted from base.py for modularity.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime


class RefactorType(Enum):
    """
    Refactor-type enum.

    5 clear types:
    - PARAMETRIC: a generic skill already exists; the specialized one becomes a wrapper
    - MERGE_SIBLINGS: multiple sibling skills; create a generic skill
    - EXTRACT_COMMON: multiple skills share a code fragment; extract a common skill
    - BEHAVIORAL: skill A contains skill B's logic but does not call B; modify A to call B
    - DUPLICATION: two skills are functionally identical; keep the one with higher maturity
    """
    PARAMETRIC = "parametric"
    MERGE_SIBLINGS = "merge_siblings"
    EXTRACT_COMMON = "extract_common"
    BEHAVIORAL = "behavioral"
    DUPLICATION = "duplication"

    # Alternate names retained because production code references both spellings.
    # Python enum semantics make these identical to their canonical members
    # (e.g. RefactorType.SIBLING is RefactorType.MERGE_SIBLINGS).
    SIBLING = "merge_siblings"
    EXTRACT_COMMON_SUBSKILL = "extract_common"
    FUNCTIONAL_SUPERSET = "behavioral"


@dataclass
class RefactorOpportunity:
    """
    Refactor opportunity.

    Represents a detected refactor opportunity, including refactor type, the
    skills involved, and suggestions.

    Field semantics (meaning of source_skill and target_skill in each type):

    PARAMETRIC (parametric coverage):
        - source_skill: specialized skill (the one being covered, e.g. craftOakBoat)
        - target_skill: generic skill (the coverer, e.g. craftBoat)
        - Refactor action: source becomes a wrapper that calls target

    BEHAVIORAL (behavioral containment):
        - source_skill: broader-scope skill (will be modified, e.g. ensureWoodenPickaxe)
        - target_skill: narrower-scope skill (the one being called, e.g. ensureCraftingTable)
        - Refactor action: source is modified to call target

    MERGE_SIBLINGS (merge siblings):
        - source_skill: one of the sibling skills (typically the one that triggered detection)
        - target_skill: name of the newly created generic skill
        - covered_skills: list of all sibling skills
        - Refactor action: create the generic skill and convert all siblings into wrappers

    EXTRACT_COMMON (extract common code):
        - source_skill: one of the skills containing the common code
        - target_skill: name of the newly extracted common skill
        - covered_skills: all skills containing the common code
        - Refactor action: create the common skill and update all related skills

    DUPLICATION (functional duplication):
        - source_skill: skill to be marked deprecated (lower maturity)
        - target_skill: skill that is retained (higher maturity)
        - Refactor action: source is marked as deprecated
    """
    refactor_type: RefactorType

    # Skills involved (meanings documented above)
    source_skill: str
    target_skill: str

    # Refactor details
    reason: str  # Explanation of the refactor reason
    confidence: float = 0.5  # Confidence (0-1); higher is more reliable

    # Parameter mapping (only used for the PARAMETRIC type)
    # Format: {target_param: source_value_or_param}
    # Example: {"boatType": "\"oak\"", "count": "count"}
    parameter_mapping: Optional[Dict[str, str]] = None

    # List of covered/involved skills
    # - PARAMETRIC: specialized skills covered by the generic skill
    # - MERGE_SIBLINGS: all sibling skills
    # - EXTRACT_COMMON: skills containing the common code
    covered_skills: List[str] = field(default_factory=list)

    # Analysis data (intermediate results for the detection algorithm)
    functional_scope_diff: Optional[float] = None  # Functional-scope difference (0-1)
    parameterization_diff: Optional[float] = None  # Parameterization-degree difference (0-1)

    # Metadata
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    analysis_id: str = ""  # Analysis identifier, used to trace the source


@dataclass
class RefactorResult:
    """
    Refactor result.

    Represents the result of a single refactor operation.
    """
    success: bool
    refactor_type: RefactorType

    # Skills involved
    source_skill: str
    target_skill: str

    # Change details
    old_code: Optional[str] = None
    new_code: Optional[str] = None
    changes_made: List[str] = field(default_factory=list)

    # On failure
    error_message: Optional[str] = None

    # Rollback information
    rollback_available: bool = True
    rollback_data: Optional[Dict[str, Any]] = None

    # Caller-update information
    updated_callers: List[str] = field(default_factory=list)
    caller_update_details: Dict[str, str] = field(default_factory=dict)  # {caller_name: change_description}

    # Sub-results (used for BEHAVIORAL Case B: one detection produces multiple covered_skills,
    # each corresponding to an independent refactor operation).
    # The history recorder saves a separate record for each sub_result, avoiding
    # loss of information for batch refactors (r34 bug: 3 craftFurnace sub-refactors merged into 1 record).
    sub_results: List["RefactorResult"] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
