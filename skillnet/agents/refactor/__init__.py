"""
Skill Graph Refactor Package

Aligned with the paper "Evolving Programmatic Skill Networks", Section 2.5.

5 refactor types (Paper-aligned aliases):

┌─────────────────────┬──────────────────────┬─────────────────────────────┐
│ Paper Name          │ Code Alias           │ Original Class              │
├─────────────────────┼──────────────────────┼─────────────────────────────┤
│ Parametric coverage │ ParametricCoverage   │ ParametricRefactor          │
│ Behavioral coverage │ BehavioralCoverage   │ BehavioralRefactor          │
│ Sibling special.    │ SiblingSpecialization│ SiblingRefactor             │
│ Common subskill     │ CommonSubskill       │ SubskillExtractionRefactor  │
│ Duplication         │ Duplication          │ DuplicationRefactor         │
└─────────────────────┴──────────────────────┴─────────────────────────────┘

Refactor type descriptions:

1. PARAMETRIC (Parametric coverage / ParametricCoverage)
   - Precondition: a general skill already exists
   - Refactor action: turn the specialized version into a wrapper that calls the general version
   - Example: craftPickaxe(type) already exists -> craftWoodenPickaxe becomes a call to it

2. MERGE_SIBLINGS (Sibling specialization / SiblingSpecialization)
   - Precondition: several sibling skills exist; no general version
   - Refactor action: create a general skill and turn all siblings into wrappers
   - Example: craftOakBoat, craftBirchBoat -> create craftBoat(type)

3. EXTRACT_COMMON (Common subskill / CommonSubskill)
   - Precondition: several skills share a code fragment
   - Refactor action: extract the common code into a standalone subskill
   - Example: many skills share furnace-setup logic -> extract setupFurnace

4. BEHAVIORAL (Behavioral coverage / BehavioralCoverage)
   - Precondition: skill A contains skill B's logic but does not call B
   - Refactor action: modify A so it calls B (B is a mandatory step for A)
   - Example: ensureWoodenPickaxe contains ensureCraftingTable logic -> modify it to call ensureCraftingTable

5. DUPLICATION (Duplication detection / Duplication)
   - Precondition: two skills have identical behavior
   - Refactor action: keep the higher-maturity one and mark the other as deprecated
   - Example: true functional duplication

Backward-compatible aliases:
- SIBLING = MERGE_SIBLINGS
- EXTRACT_COMMON_SUBSKILL = EXTRACT_COMMON
- FUNCTIONAL_SUPERSET = BEHAVIORAL (merged)

Usage:
    from skillnet.agents.refactor import (
        # Base types
        RefactorType,
        RefactorOpportunity,
        RefactorResult,
        RefactorDetector,

        # Refactors
        ParametricRefactor,
        BehavioralRefactor,
        SubskillExtractionRefactor,
        DuplicationRefactor,
        SiblingRefactor,
    )

    # Detect refactor opportunities
    detector = RefactorDetector(skill_graph_manager=manager)
    opportunities = detector.detect_opportunities("skillName")

    # Apply a single refactor
    refactor = ParametricRefactor(skill_graph_manager=manager)
    result = refactor.apply(opportunity)
"""

from .base import (
    RefactorType,
    RefactorOpportunity,
    RefactorResult,
    RefactorDetector,
    SkillRefactor,
)

from .parametric import ParametricRefactor

from .behavioral import BehavioralRefactor

from .subskill_extraction import (
    SubskillExtractionRefactor,
    CommonCodeBlock,
    ExtractionPlan,
    detect_extraction_opportunities,
)

from .duplication import (
    DuplicationRefactor,
    DuplicationAnalysis,
    DuplicationPlan,
    detect_duplication_opportunities,
)

from .sibling import (
    SiblingRefactor,
    SiblingGroup,
    SiblingPlan,
    detect_sibling_opportunities,
)

from .history import (
    RefactorRecord,
    RefactorStatistics,
    RefactorHistoryTracker,
)

from .transaction_adapter import (
    RefactorTransactionState,
    RefactorTransactionAdapter,
    RefactorTransactionContext,
    RefactorBatchContext,
    refactor_transaction,
)

from .graph_adapter import (
    RefactorExecutor,
    create_executor_for_graph_manager,
)

# Detection module
from .detection import (
    RefactorPrescreener,
    RefactorRelationshipAnalyzer,
    calculate_name_similarity,
    is_generalization_of,
    estimate_functional_scope,
    estimate_implementation_maturity,
    estimate_parameterization_level,
)

# =============================================================================
# Paper-aligned aliases (Section 2.5 Online Refactoring)
# =============================================================================

# 5 refactor types from the paper:
ParametricCoverage = ParametricRefactor        # Parametric coverage
BehavioralCoverage = BehavioralRefactor        # Behavioral coverage
SiblingSpecialization = SiblingRefactor        # Sibling specializations
CommonSubskill = SubskillExtractionRefactor    # Common subskill extraction
Duplication = DuplicationRefactor              # Duplication

__all__ = [
    # Paper-aligned aliases
    "ParametricCoverage",
    "BehavioralCoverage",
    "SiblingSpecialization",
    "CommonSubskill",
    "Duplication",
    # Base types
    "RefactorType",
    "RefactorOpportunity",
    "RefactorResult",
    "RefactorDetector",
    "SkillRefactor",

    # Parametric refactor
    "ParametricRefactor",

    # Behavioral refactor
    "BehavioralRefactor",

    # Subskill extraction
    "SubskillExtractionRefactor",
    "CommonCodeBlock",
    "ExtractionPlan",
    "detect_extraction_opportunities",

    # Duplication refactor
    "DuplicationRefactor",
    "DuplicationAnalysis",
    "DuplicationPlan",
    "detect_duplication_opportunities",

    # Sibling refactor
    "SiblingRefactor",
    "SiblingGroup",
    "SiblingPlan",
    "detect_sibling_opportunities",

    # History tracking (M1, M2)
    "RefactorRecord",
    "RefactorStatistics",
    "RefactorHistoryTracker",

    # Transaction adapter (M5)
    "RefactorTransactionState",
    "RefactorTransactionAdapter",
    "RefactorTransactionContext",
    "RefactorBatchContext",
    "refactor_transaction",

    # Graph Manager Adapter
    "RefactorExecutor",
    "create_executor_for_graph_manager",

    # Detection module
    "RefactorPrescreener",
    "RefactorRelationshipAnalyzer",
    "calculate_name_similarity",
    "is_generalization_of",
    "estimate_functional_scope",
    "estimate_implementation_maturity",
    "estimate_parameterization_level",
]
