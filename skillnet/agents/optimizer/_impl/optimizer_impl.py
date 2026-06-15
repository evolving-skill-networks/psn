"""
Skill Graph Optimizer

Implements skill optimization functionality, including:
- Feedback collection and structuring
- Subgraph extraction and traversal
- Three-step optimization strategy (diagnosis, plan generation, code optimization)
- In-depth root-cause analysis and error categorization
- Optimization-history tracking and loop detection
"""

import os
import re
import json
from typing import Dict, List, Set, Optional, Any, Tuple, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime

# Import refactored modules (single source of truth after modular restructure)

# import active code-editing operations from transforms/
from skillnet.agents.optimizer.transforms import (
    remove_unused_helper_functions as _remove_unused_helper_functions_impl,
    # diff generation functions
    generate_unified_diff as _generate_unified_diff_impl,
    generate_annotated_diff as _generate_annotated_diff_impl,
    basic_syntax_check as _basic_syntax_check_impl,
)

# Import error categories and diagnostic types from analysis/
from skillnet.agents.optimizer.analysis import (
    ErrorCategory,
    FixTargetType,
    # interface-analysis functions
    parse_function_params as _parse_function_params_impl,
    analyze_param_changes as _analyze_param_changes_impl,
    classify_interface_change as _classify_interface_change_impl,
    detect_options_object_change as _detect_options_object_change_impl,
    determine_impact_and_strategy as _determine_impact_and_strategy_impl,
)

# Import feedback types and parsing functions from feedback/
from skillnet.agents.optimizer.feedback import (
    SkillFeedback,
    OptimizationRecord,
    ModificationAnalysis,
    extract_constraint_feedback_from_critique as _extract_constraint_feedback_impl,
    extract_issue_type_from_content as _extract_issue_type_impl,
    format_feedbacks_for_llm as _format_feedbacks_for_llm_impl,
)

# Import code-validation functions from validators/
from skillnet.agents.optimizer.validators import (
    validate_tdz_issues,
    validate_code_completeness,
    find_bracket_mismatch_line,
    fix_tdz_issues as _fix_tdz_issues_impl,
    validate_function_implementation,  # Semantic validation: consistency of function name and implementation
    BloatChecker,
    check_naming_conflicts,  # Fix 12: naming-conflict detection
)

# Import CodeBloatTracker, LoopManager, OptimizationTracker from tracking/
from skillnet.agents.optimizer.tracking import (
    CodeBloatTracker,
    LoopManager,
    OptimizationTracker,
)

# Import LLM invocation utilities from core/
from skillnet.agents.optimizer.core import (
    LLMInvoker,
    robust_json_parse,
    # P2 Phase A: pure helper functions for optimization
    build_current_state_info,
    extract_issues_from_feedbacks,
    check_requirements_addressed,
)

# import environmental-feedback helpers from planning
try:
    from skillnet.agents.planning.precondition_checker import (
        PreconditionChecker,
        EnvironmentalFeedback,
        infer_environmental_feedback_from_error,
    )
    _HAS_PRECONDITION_CHECKER = True
except ImportError:
    _HAS_PRECONDITION_CHECKER = False
    PreconditionChecker = None
    EnvironmentalFeedback = None
    infer_environmental_feedback_from_error = None

# import extracted pure functions from helpers/
from skillnet.agents.optimizer._impl.helpers import (
    save_feedback_history as _save_feedback_history_impl,
    load_feedback_history as _load_feedback_history_impl,
    mark_feedbacks_resolved as _mark_feedbacks_resolved_impl,
    detect_parameter_semantic_mismatch as _detect_parameter_semantic_mismatch_impl,
)

# import extracted Mixin classes from mixins/
from skillnet.agents.optimizer._impl.mixins import (
    CodeEditMixin,
    ValidationMixin,
    FeedbackMixin,
    InterfaceManagementMixin,
    OptimizationMixin,
    ApplyOptimizationMixin,
    EditContextMixin,
    EditAnalysisMixin,
    OptimizationUtilsMixin,
    OptimizationLifecycleMixin,
    QuickOptimizationMixin,
)

# import code-processing utilities from skill_graph/utils
from skillnet.agents.skill_graph.utils import (
    sanitize_python_to_js as _sanitize_python_to_js_impl,
)

# Import LangChain
from langchain.schema import HumanMessage, SystemMessage

# Import PSN utilities
import skillnet.utils as U
from skillnet.utils.llm_factory import create_chat_llm
from skillnet.utils.log_utils import LoggerManager

# Import SkillGraph-related types (used for subgraph extraction etc.)
from skillnet.agents.skill_graph import SkillGraph, SkillNode, _strip_comments_and_strings

# Import bloat-prevention configuration
from skillnet.agents.optimizer.config import DEFAULT_BLOAT_CONFIG as BLOAT_CONFIG


import copy





class SkillGraphOptimizer(
    FeedbackMixin,
    ValidationMixin,
    CodeEditMixin,
    InterfaceManagementMixin,
    OptimizationMixin,
    ApplyOptimizationMixin,
    EditContextMixin,
    EditAnalysisMixin,
    QuickOptimizationMixin,
    OptimizationLifecycleMixin,
    OptimizationUtilsMixin,
):
    """
    Skill Graph Optimizer: responsible for optimizing skills.

    Features:
    1. Collect and structure skill-level feedback
    2. Extract and traverse subgraphs
    3. Generate optimization suggestions
    4. Execute the three-step optimization strategy
    """
    
    # NOTE: the consistency check rejects on the LLM's holistic consistency_score
    # alone (see QuickOptimizationMixin._should_reject_for_consistency). The
    # former CRITICAL_CONFLICT_KEYWORDS / UNADDRESSED_ISSUE_KEYWORDS layer that
    # substring-matched the LLM's free-text conflicts was removed: it false-
    # rejected substantively-correct optimizations (any "undefined"/"syntax" in
    # the prose triggered it), and the defect classes it claimed to catch are
    # already covered by dedicated validators (completeness/syntax/reference).

    def __init__(
        self,
        skill_graph_manager: "SkillGraphManager",
        model_name: str = "gpt-5-mini",
        temperature: float = 0,
        request_timeout: int = 240,
        consistency_threshold: float = 0.3,  # P2: consistency threshold; reject optimization below this value
        enable_refactor: bool = True,  # Whether to enable refactor (code restructuring)
        openai_api_base: str = None,
        openai_api_key: str = None,
        pure_reasoning: bool = False,  # Remove domain-data injection (used for evaluation)
        include_reasoning_examples: bool = False,  # Cheat 3 gate (default off per cross-LLM Phase A)
        use_factual_primitive_doc: bool = False,  # Phase 13.B-9 R4 Tier-2 gate
        # B1: Maturity gating parameters
        optimization_threshold: float = 0.6,
        optimization_epsilon: float = 0.05,
        optimization_gamma: float = 8.0,
    ):
        """
        Initialize the optimizer.

        Args:
            skill_graph_manager: SkillGraphManager instance
            model_name: LLM model name
            temperature: LLM temperature
            request_timeout: Request timeout
            consistency_threshold: Consistency-check threshold; reject optimization below this value; default 0.3
            enable_refactor: Whether to enable refactor; default True
            openai_api_base: Custom API base URL for vLLM or compatible endpoints
            openai_api_key: Custom API key
            pure_reasoning: Remove domain-data injection (conclusions, domain data tables); used for evaluation
        """
        self.enable_refactor = enable_refactor
        self._pure_reasoning = pure_reasoning
        self._include_reasoning_examples = include_reasoning_examples
        self._use_factual_primitive_doc = use_factual_primitive_doc
        self.skill_graph_manager = skill_graph_manager
        self._model_name = model_name

        self.llm = create_chat_llm(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            component="optimizer",
        )
        
        # Use skill_graph_manager's stats tracker
        self.stats_tracker = skill_graph_manager.stats_tracker

        # Initialize the logger
        from skillnet.utils.log_utils import LoggerManager
        self.logger = LoggerManager.get_logger("SkillGraphOptimizer", skill_graph_manager.ckpt_dir)

        # Initialize the LLM invocation utility (unified calls and statistics)
        self._llm_invoker = LLMInvoker(
            llm=self.llm,
            stats_tracker=self.stats_tracker,
            instance_logger=self.logger,
        )

        # Graph Planner reference (set by the PSN agent; used for parameter-passing problem detection and learning)
        self.graph_planner = None

        # Feedback storage
        self.feedback_history: List[SkillFeedback] = []

        # consistency threshold (configurable)
        self.consistency_threshold = consistency_threshold
        self.logger.info(f"\033[36m[Optimizer] Consistency threshold: {consistency_threshold}\033[0m")

        self._current_feedback_requirements: List[str] = []
        self._current_phase1_gradients: List = []  # P3: Phase 1 gradients for data bridge
        self._current_session_id: Optional[str] = None  # P1: current optimization session ID

        # LLM analysis result for the current optimization (used to pass key issue info during minimal fixes)
        self._current_llm_analysis: Optional['ModificationAnalysis'] = None

        # Reference Check rejection history for feedback loop
        self._reference_check_rejections: Dict[str, List[Dict]] = {}

        # Responsibility Check rejection history for feedback loop
        # Mirrors reference-check pattern. When RespCheck rejects an optimization
        # because inlined logic belongs to a sibling skill, this history is
        # threaded into the next _build_optimization_prompt call so the LLM can
        # see the prior rejection reason and switch to the composable alternative.
        self._responsibility_check_rejections: Dict[str, List[Dict]] = {}

        # Create the optimizer directory
        optimizer_dir = f"{skill_graph_manager.ckpt_dir}/skill_graph/optimizer"
        U.f_mkdir(optimizer_dir)

        # Create the directory for backpropagation optimization records
        self.backprop_dir = f"{skill_graph_manager.ckpt_dir}/skill_graph/backprop_skills"
        U.f_mkdir(self.backprop_dir)

        # Initialize the optimization-history tracker
        self.optimization_tracker = OptimizationTracker(skill_graph_manager.ckpt_dir)

        # Phase 6: initialize the code-bloat statistics tracker
        self.bloat_tracker = CodeBloatTracker(skill_graph_manager.ckpt_dir)

        # initialize the code-bloat checker
        self._bloat_checker = BloatChecker(config=BLOAT_CONFIG, logger=self.logger)

        # Load existing feedback history
        self._load_feedback_history()

        # Register the code-change callback so that SkillGraphManager notifies us on code changes,
        # which lets us automatically trigger interface checks and caller updates after refactor operations
        self.skill_graph_manager.set_on_code_changed_callback(
            lambda skill_name, old_code, new_code, change_source:
                self.on_skill_code_changed(
                    skill_name=skill_name,
                    old_code=old_code,
                    new_code=new_code,
                    change_source=change_source,
                    auto_update_callers=True,
                )
        )

        # ========== Initialize the two-phase optimization engine (new architecture) ==========
        # This is the new optimization-system entry point, using modular components
        from skillnet.agents.optimizer.engine import (
            TwoPhaseOptimizationEngine,
            TwoPhaseOptimizationConfig,
        )

        self._two_phase_engine = TwoPhaseOptimizationEngine(
            skill_graph_manager=skill_graph_manager,
            llm=self.llm,
            logger=self.logger,
            ckpt_dir=skill_graph_manager.ckpt_dir,
            config=TwoPhaseOptimizationConfig(
                max_reflection_depth=3,
                enable_transactions=True,
                auto_rollback_on_failure=False,  # Preserve partial success
                enable_post_optimization_refactor=self.enable_refactor,  # Pass through refactor toggle
                pure_reasoning=self._pure_reasoning,
                include_reasoning_examples=self._include_reasoning_examples,
                use_factual_primitive_doc=self._use_factual_primitive_doc,
                # B1: Maturity gating parameters
                optimization_threshold=optimization_threshold,
                optimization_epsilon=optimization_epsilon,
                optimization_gamma=optimization_gamma,
            ),
            # Pass the actual optimizer callback, which uses quick_optimize_skill
            optimizer_callback=self._optimizer_callback_wrapper,
            optimization_tracker=self.optimization_tracker,
            # New: callback to record optimization results (fixes detailed_logs missing code_after issue)
            on_skill_optimized=self._on_skill_optimized_callback,
            on_optimization_failed=self._on_optimization_failed_callback,
        )
        self.logger.info(f"\033[36m[Optimizer] Two-phase optimization engine initialized\033[0m")

        # Wire skill_graph_manager to Phase 1 analyzer for composable skills
        try:
            pipeline = getattr(self._two_phase_engine, 'pure_pipeline', None)
            if pipeline:
                reflection = getattr(pipeline, 'pure_reflection', None)
                if reflection and hasattr(reflection, 'analyzer'):
                    reflection.analyzer._skill_graph_manager = skill_graph_manager
        except Exception:
            pass  # Non-critical: Phase 1 still works without composable skills

        # Domain knowledge (set via set_domain_knowledge for from_domain path)
        self._domain_knowledge = None

        # Skill language (set via set_skill_language for domain-agnostic code ops)
        self._skill_language = None

    def set_skill_language(self, language):
        """Set the skill language implementation for code operations."""
        self._skill_language = language

    def set_domain_knowledge(self, knowledge):
        """Wire domain knowledge to internal LLMAnalyzer and refactors via engine chain."""
        self._domain_knowledge = knowledge
        # Chain: _two_phase_engine → pure_pipeline → pure_reflection → analyzer
        engine = getattr(self, '_two_phase_engine', None)
        if engine:
            pipeline = getattr(engine, 'pure_pipeline', None)
            if pipeline:
                reflection = getattr(pipeline, 'pure_reflection', None)
                if reflection and hasattr(reflection, 'analyzer'):
                    reflection.analyzer._domain_knowledge = knowledge
                    # Wire skill_graph_manager for composable skills in Phase 1
                    sgm = getattr(self, 'skill_graph_manager', None)
                    if sgm:
                        reflection.analyzer._skill_graph_manager = sgm
            # Propagate to refactors for function reference validation
            for refactor in getattr(engine, 'refactors', {}).values():
                refactor.domain_knowledge = knowledge
            # Propagate to refactor detector for variant pattern injection
            detector = getattr(engine, 'refactor_detector', None)
            if detector is not None:
                detector.domain_knowledge = knowledge
        # Propagate loop-breaking API hints to LoopManager
        tracker = getattr(self, 'optimization_tracker', None)
        if tracker and knowledge:
            _hints = knowledge.get_known_functions().get("loop_break_api_hints", {})
            if hasattr(tracker, '_loop_manager'):
                tracker._loop_manager._api_hints = _hints
        # Cache skill-language implementation for hot paths (registry-fallback is the
        # backup for module-level helpers).
        if knowledge is not None:
            try:
                self._skill_language = knowledge.get_skill_language_impl()
            except Exception as e:
                # DKs that don't implement get_skill_language_impl yet still load,
                # but agent code that needs it will fall back to the dk_registry.
                print(f"[{type(self).__name__}] Could not cache skill language: {e}")
                self._skill_language = None
        else:
            self._skill_language = None
