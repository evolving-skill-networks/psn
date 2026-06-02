"""
Semantic Compatibility Validation

v7.4 Phase 4: Semantic compatibility validation

Components:
- SignatureCompatibilityChecker: signature compatibility check
- BehaviorEquivalenceValidator: behavior equivalence validation (includes LLM)
- CallChainImpactAnalyzer: call-chain impact analysis
- SemanticEquivalenceValidator: main validator (orchestrates all sub-validators)

Usage:
    from skillnet.agents.optimizer.validators.semantic_compatibility import (
        SemanticEquivalenceValidator,
    )
    validator = SemanticEquivalenceValidator(skill_graph, llm)
    result = await validator.validate(old_code, new_code, ...)
"""

from .semantic_equivalence import (
    SemanticEquivalenceValidator,
    SemanticEquivalenceResult,
)

from .signature_checker import (
    SignatureCompatibilityChecker,
    SignatureCompatibilityResult,
    SignatureInfo,
)

from .behavior_validator import (
    BehaviorEquivalenceValidator,
    BehaviorEquivalenceResult,
    BehaviorFeature,
)

from .call_chain_analyzer import (
    CallChainImpactAnalyzer,
    CallChainImpactResult,
    CallerImpact,
)

__all__ = [
    # Main validator
    "SemanticEquivalenceValidator",
    "SemanticEquivalenceResult",

    # Signature checker
    "SignatureCompatibilityChecker",
    "SignatureCompatibilityResult",
    "SignatureInfo",

    # Behavior validator
    "BehaviorEquivalenceValidator",
    "BehaviorEquivalenceResult",
    "BehaviorFeature",

    # Call chain analyzer
    "CallChainImpactAnalyzer",
    "CallChainImpactResult",
    "CallerImpact",

]
