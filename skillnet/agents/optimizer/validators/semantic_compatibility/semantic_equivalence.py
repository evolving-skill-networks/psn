"""
Semantic Equivalence Validator

v7.4 Phase 4: semantic equivalence validator (top-level)

Features:
1. Orchestrates all sub-validators (signature, behavior, call chain)
2. Comprehensively evaluates semantic equivalence
3. Returns a detailed validation result

Three-layer validation strategy:
- Layer 1: static analysis (100-200ms) — signature + API calls
- Layer 2: effects check (500-1000ms) — effects consistency
- Layer 3: LLM validation (2-5s, optional) — advanced semantic validation
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .signature_checker import SignatureCompatibilityChecker, SignatureInfo
from .behavior_validator import BehaviorEquivalenceValidator, BehaviorFeature
from .call_chain_analyzer import CallChainImpactAnalyzer

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import SkillGraph
    from langchain_core.language_models import BaseChatModel as LLM

logger = logging.getLogger(__name__)


@dataclass
class SemanticEquivalenceResult:
    """
    Semantic equivalence validation result

    Combines the results of all sub-validators into a complete semantic equivalence assessment.
    """
    is_equivalent: bool
    signature_compatible: bool
    behavior_equivalent: bool
    call_chain_safe: bool
    confidence: float  # 0.0 - 1.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    llm_reasoning: Optional[str] = None

    # Detailed results
    signature_issues: List[str] = field(default_factory=list)
    behavior_issues: List[str] = field(default_factory=list)
    call_chain_issues: List[str] = field(default_factory=list)

    # Metadata
    validation_time_ms: float = 0.0
    layers_passed: int = 0

    def __bool__(self) -> bool:
        return self.is_equivalent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return {
            'is_equivalent': self.is_equivalent,
            'signature_compatible': self.signature_compatible,
            'behavior_equivalent': self.behavior_equivalent,
            'call_chain_safe': self.call_chain_safe,
            'confidence': self.confidence,
            'issues': self.issues,
            'warnings': self.warnings,
            'llm_reasoning': self.llm_reasoning,
            'validation_time_ms': self.validation_time_ms,
            'layers_passed': self.layers_passed,
        }

    @property
    def summary(self) -> str:
        """Get a summary of the result"""
        if self.is_equivalent:
            return f"Semantically equivalent (confidence={self.confidence:.2f}, layers={self.layers_passed})"
        else:
            return f"Not semantically equivalent: {self.issues[0] if self.issues else 'unknown'}"


class SemanticEquivalenceValidator:
    """
    Semantic equivalence validator

    Orchestrates all sub-validators to perform complete semantic equivalence validation.

    Usage:
        validator = SemanticEquivalenceValidator(skill_graph, llm)

        result = await validator.validate(
            old_code=source_node.code,
            new_code=new_code,
            old_effects=[str(e) for e in source_node.expected_effects],
            new_effects=[str(e) for e in target_node.expected_effects],
            skill_name=source_skill
        )

        if not result.is_equivalent:
            print(f"Not semantically equivalent: {result.issues}")
    """

    def __init__(
        self,
        skill_graph=None,
        llm: "LLM" = None,
        logger_instance=None,
        enable_llm_validation: bool = True,
        enable_call_chain_analysis: bool = True,
    ):
        """
        Initialize the semantic equivalence validator

        Args:
            skill_graph: SkillGraph or SkillGraphManager instance (used for call-chain analysis, duck typing)
            llm: LLM instance (used for Layer 3 validation)
            logger_instance: logger instance
            enable_llm_validation: whether to enable LLM validation
            enable_call_chain_analysis: whether to enable call-chain analysis
        """
        self.skill_graph = skill_graph
        self.llm = llm
        self.logger = logger_instance or logger
        self.enable_llm_validation = enable_llm_validation
        self.enable_call_chain_analysis = enable_call_chain_analysis

        # Initialize sub-validators
        self.signature_checker = SignatureCompatibilityChecker(logger_instance)
        self.behavior_validator = BehaviorEquivalenceValidator(
            llm=llm,
            logger_instance=logger_instance,
            enable_llm_validation=enable_llm_validation,
        )
        self.call_chain_analyzer = CallChainImpactAnalyzer(
            skill_graph=skill_graph,
            logger_instance=logger_instance,
        ) if skill_graph else None

    async def validate(
        self,
        old_code: str,
        new_code: str,
        old_effects: List[str] = None,
        new_effects: List[str] = None,
        skill_name: str = None,
        skip_call_chain: bool = False,
        skip_llm: bool = False,
    ) -> SemanticEquivalenceResult:
        """
        Perform full semantic equivalence validation

        Args:
            old_code: original code
            new_code: new code
            old_effects: original effects list
            new_effects: new effects list
            skill_name: skill name (used for call-chain analysis)
            skip_call_chain: whether to skip call-chain analysis
            skip_llm: whether to skip LLM validation

        Returns:
            SemanticEquivalenceResult: validation result
        """
        import time
        start_time = time.time()

        old_effects = old_effects or []
        new_effects = new_effects or []

        all_issues = []
        all_warnings = []
        signature_issues = []
        behavior_issues = []
        call_chain_issues = []

        signature_compatible = True
        behavior_equivalent = True
        call_chain_safe = True
        llm_reasoning = None
        layers_passed = 0

        # ============================================================
        # Layer 1: signature compatibility check + static behavior analysis
        # ============================================================
        self.logger.debug(f"[SemanticValidator] Layer 1: signature + static analysis ({skill_name})")

        # 1a. signature compatibility
        old_sig = self.signature_checker.parse_signature(old_code)
        new_sig = self.signature_checker.parse_signature(new_code)

        if old_sig and new_sig:
            sig_result = self.signature_checker.check_compatibility(old_sig, new_sig)
            signature_compatible = sig_result.is_compatible
            signature_issues = sig_result.issues
            all_issues.extend(sig_result.issues)
            all_warnings.extend(sig_result.warnings)

            if not signature_compatible:
                self.logger.info(
                    f"[SemanticValidator] Signature incompatible ({skill_name}): {signature_issues}"
                )
                return self._build_result(
                    is_equivalent=False,
                    signature_compatible=False,
                    behavior_equivalent=True,
                    call_chain_safe=True,
                    confidence=0.2,
                    issues=all_issues,
                    warnings=all_warnings,
                    signature_issues=signature_issues,
                    behavior_issues=behavior_issues,
                    call_chain_issues=call_chain_issues,
                    llm_reasoning=llm_reasoning,
                    layers_passed=0,
                    start_time=start_time,
                )
        else:
            # Cannot parse signatures; warn but continue
            all_warnings.append(f"Could not parse signatures: old={old_sig is not None}, new={new_sig is not None}")

        # 1b. static behavior analysis (Layer 1 of BehaviorValidator)
        old_features = self.behavior_validator.extract_behavior_features(old_code)
        new_features = self.behavior_validator.extract_behavior_features(new_code)

        layer1_equiv, layer1_issues = self.behavior_validator.compare_behaviors(
            old_features, new_features
        )
        if not layer1_equiv:
            behavior_issues.extend(layer1_issues)
            all_issues.extend(layer1_issues)
            behavior_equivalent = False

            self.logger.info(
                f"[SemanticValidator] Behavior not equivalent at Layer 1 ({skill_name}): {layer1_issues}"
            )
            return self._build_result(
                is_equivalent=False,
                signature_compatible=signature_compatible,
                behavior_equivalent=False,
                call_chain_safe=True,
                confidence=0.3,
                issues=all_issues,
                warnings=all_warnings,
                signature_issues=signature_issues,
                behavior_issues=behavior_issues,
                call_chain_issues=call_chain_issues,
                llm_reasoning=llm_reasoning,
                layers_passed=1,
                start_time=start_time,
            )

        layers_passed = 1

        # ============================================================
        # Layer 2: effects check + call-chain analysis
        # ============================================================
        self.logger.debug(f"[SemanticValidator] Layer 2: effects + call chain ({skill_name})")

        # 2a. effects check
        if old_effects and new_effects:
            layer2_equiv, layer2_issues = self.behavior_validator.compare_effects(
                old_effects, new_effects
            )
            if not layer2_equiv:
                behavior_issues.extend(layer2_issues)
                all_issues.extend(layer2_issues)
                behavior_equivalent = False

                self.logger.info(
                    f"[SemanticValidator] Effects mismatch ({skill_name}): {layer2_issues}"
                )
                return self._build_result(
                    is_equivalent=False,
                    signature_compatible=signature_compatible,
                    behavior_equivalent=False,
                    call_chain_safe=True,
                    confidence=0.4,
                    issues=all_issues,
                    warnings=all_warnings,
                    signature_issues=signature_issues,
                    behavior_issues=behavior_issues,
                    call_chain_issues=call_chain_issues,
                    llm_reasoning=llm_reasoning,
                    layers_passed=1,
                    start_time=start_time,
                )

        # 2b. call-chain analysis
        if (self.enable_call_chain_analysis and
            self.call_chain_analyzer and
            skill_name and
            old_sig and new_sig and
            not skip_call_chain):

            call_chain_result = self.call_chain_analyzer.analyze_impact(
                skill_name=skill_name,
                old_signature=old_sig,
                new_signature=new_sig,
            )
            call_chain_safe = call_chain_result.is_safe
            call_chain_issues = call_chain_result.issues
            all_issues.extend(call_chain_result.issues)
            all_warnings.extend(call_chain_result.warnings)

            if not call_chain_safe:
                self.logger.info(
                    f"[SemanticValidator] Call chain unsafe ({skill_name}): {call_chain_issues}"
                )
                return self._build_result(
                    is_equivalent=False,
                    signature_compatible=signature_compatible,
                    behavior_equivalent=behavior_equivalent,
                    call_chain_safe=False,
                    confidence=0.5,
                    issues=all_issues,
                    warnings=all_warnings,
                    signature_issues=signature_issues,
                    behavior_issues=behavior_issues,
                    call_chain_issues=call_chain_issues,
                    llm_reasoning=llm_reasoning,
                    layers_passed=1,
                    start_time=start_time,
                )

        layers_passed = 2

        # ============================================================
        # Layer 3: LLM validation (optional)
        # ============================================================
        if self.enable_llm_validation and self.llm and not skip_llm:
            self.logger.debug(f"[SemanticValidator] Layer 3: LLM validation ({skill_name})")

            layer3_equiv, llm_reasoning = await self.behavior_validator.validate_with_llm(
                old_code=old_code,
                new_code=new_code,
                old_effects=old_effects,
                new_effects=new_effects,
            )

            if not layer3_equiv:
                behavior_equivalent = False
                behavior_issues.append(f"LLM: {llm_reasoning}")
                all_issues.append(f"LLM validation failed: {llm_reasoning}")

                self.logger.info(
                    f"[SemanticValidator] LLM validation failed ({skill_name}): {llm_reasoning}"
                )
                return self._build_result(
                    is_equivalent=False,
                    signature_compatible=signature_compatible,
                    behavior_equivalent=False,
                    call_chain_safe=call_chain_safe,
                    confidence=0.6,
                    issues=all_issues,
                    warnings=all_warnings,
                    signature_issues=signature_issues,
                    behavior_issues=behavior_issues,
                    call_chain_issues=call_chain_issues,
                    llm_reasoning=llm_reasoning,
                    layers_passed=2,
                    start_time=start_time,
                )

            layers_passed = 3

        # ============================================================
        # All validations passed
        # ============================================================
        confidence = 0.7 + (0.1 * layers_passed)  # 0.8 - 1.0
        confidence = min(confidence, 0.95)  # Cap at 0.95

        self.logger.info(
            f"[SemanticValidator] Validation passed ({skill_name}): "
            f"layers={layers_passed}, confidence={confidence:.2f}"
        )

        return self._build_result(
            is_equivalent=True,
            signature_compatible=signature_compatible,
            behavior_equivalent=behavior_equivalent,
            call_chain_safe=call_chain_safe,
            confidence=confidence,
            issues=[],
            warnings=all_warnings,
            signature_issues=[],
            behavior_issues=[],
            call_chain_issues=[],
            llm_reasoning=llm_reasoning,
            layers_passed=layers_passed,
            start_time=start_time,
        )

    def _build_result(
        self,
        is_equivalent: bool,
        signature_compatible: bool,
        behavior_equivalent: bool,
        call_chain_safe: bool,
        confidence: float,
        issues: List[str],
        warnings: List[str],
        signature_issues: List[str],
        behavior_issues: List[str],
        call_chain_issues: List[str],
        llm_reasoning: Optional[str],
        layers_passed: int,
        start_time: float,
    ) -> SemanticEquivalenceResult:
        """Build the validation result"""
        import time
        validation_time_ms = (time.time() - start_time) * 1000

        return SemanticEquivalenceResult(
            is_equivalent=is_equivalent,
            signature_compatible=signature_compatible,
            behavior_equivalent=behavior_equivalent,
            call_chain_safe=call_chain_safe,
            confidence=confidence,
            issues=issues,
            warnings=warnings,
            llm_reasoning=llm_reasoning,
            signature_issues=signature_issues,
            behavior_issues=behavior_issues,
            call_chain_issues=call_chain_issues,
            validation_time_ms=validation_time_ms,
            layers_passed=layers_passed,
        )

    def validate_sync(
        self,
        old_code: str,
        new_code: str,
        old_effects: List[str] = None,
        new_effects: List[str] = None,
        skill_name: str = None,
    ) -> SemanticEquivalenceResult:
        """
        Synchronous validation (without LLM)

        Args:
            old_code: original code
            new_code: new code
            old_effects: original effects list
            new_effects: new effects list
            skill_name: skill name

        Returns:
            SemanticEquivalenceResult: validation result
        """
        import asyncio

        # Run async validation in a synchronous context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If an event loop is already running, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.validate(
                            old_code=old_code,
                            new_code=new_code,
                            old_effects=old_effects,
                            new_effects=new_effects,
                            skill_name=skill_name,
                            skip_llm=True,  # Sync version skips LLM
                        )
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.validate(
                        old_code=old_code,
                        new_code=new_code,
                        old_effects=old_effects,
                        new_effects=new_effects,
                        skill_name=skill_name,
                        skip_llm=True,
                    )
                )
        except RuntimeError:
            # No event loop
            return asyncio.run(
                self.validate(
                    old_code=old_code,
                    new_code=new_code,
                    old_effects=old_effects,
                    new_effects=new_effects,
                    skill_name=skill_name,
                    skip_llm=True,
                )
            )

    def quick_check(
        self,
        old_code: str,
        new_code: str,
    ) -> bool:
        """
        Quick check (Layer 1 only)

        Suitable for scenarios requiring a fast judgment.

        Args:
            old_code: original code
            new_code: new code

        Returns:
            bool: whether possibly equivalent
        """
        # Signature check
        old_sig = self.signature_checker.parse_signature(old_code)
        new_sig = self.signature_checker.parse_signature(new_code)

        if old_sig and new_sig:
            sig_result = self.signature_checker.check_compatibility(old_sig, new_sig)
            if not sig_result.is_compatible:
                return False

        # Behavior check
        old_features = self.behavior_validator.extract_behavior_features(old_code)
        new_features = self.behavior_validator.extract_behavior_features(new_code)

        is_equiv, _ = self.behavior_validator.compare_behaviors(
            old_features, new_features
        )

        return is_equiv
