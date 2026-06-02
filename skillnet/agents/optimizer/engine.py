"""
Two-Phase Optimization Engine

Two-phase optimization engine that uses TwoPhaseOptimizationPipeline to perform optimization.

Core components:
- TwoPhaseOptimizationPipeline: full two-phase optimization flow
- TransactionManager: transaction management and rollback
- RefactorDetector: post-optimization refactor detection

Usage:
    engine = TwoPhaseOptimizationEngine(
        skill_graph_manager=manager,
        llm=llm,
        logger=logger,
    )

    result = engine.optimize(
        skills_to_optimize=["craftIronSword"],
        current_task="craft iron sword",
        current_error="Missing iron ingot",
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

from .phases.pure_reflection import SkillDelta  # use the new SkillDelta

from .phases.reflection_chain import (
    TwoPhaseOptimizationPipeline,
    create_two_phase_pipeline,
    ChainExecutionResult,
)
from .phases.skill_info_adapter import create_skill_info_getter

from .tracking.transaction import (
    TransactionManager,
    SessionTransaction,
    SubgraphTransaction,
    TransactionType,
    TransactionStatus,
)

# Import refactor module
try:
    from skillnet.agents.refactor import (
        RefactorDetector,
        RefactorType,
        RefactorOpportunity,
        ParametricRefactor,
        BehavioralRefactor,
        SiblingRefactor,
        DuplicationRefactor,
        SubskillExtractionRefactor,
    )
    REFACTOR_AVAILABLE = True
except ImportError:
    REFACTOR_AVAILABLE = False


@dataclass
class TwoPhaseOptimizationConfig:
    """Two-phase optimization configuration"""
    # Top-down configuration
    max_reflection_depth: int = 3

    # Transaction configuration
    enable_transactions: bool = True
    auto_rollback_on_failure: bool = True

    # Refactor configuration
    enable_post_optimization_refactor: bool = True
    refactor_min_success_count: int = 3  # How many successes are required before refactor detection is triggered
    refactor_types_enabled: List[str] = field(default_factory=lambda: [
        "parametric",       # Parametric refactor
        "behavioral",       # Behavioral refactor
        "merge_siblings",   # Merge-siblings refactor
        "duplication",      # Duplicate-code refactor
        "extract_common",   # Extract-common-subskill refactor
    ])

    # Pure-reasoning mode: removes domain-data injection (conclusion, domain data tables), used for evaluation
    pure_reasoning: bool = False

    # Gates the reasoning_examples_section in the Phase 1 prompt. Defaults
    # to False because a cross-LLM ablation showed 0 diagnostic-correctness
    # delta with the guides enabled on both Qwen3 and gpt-5-mini.
    include_reasoning_examples: bool = False

    # Gates the factual primitive-doc section in Phase 1.
    # When True, doc strings for primitives detected in skill_code are
    # appended to the analysis prompt. See
    # skillnet/agents/optimizer/phases/factual_primitive_doc.py.
    use_factual_primitive_doc: bool = False

    # B1: Maturity gating (passed to SkipChecker)
    optimization_threshold: float = 0.6
    optimization_epsilon: float = 0.05
    optimization_gamma: float = 8.0



@dataclass
class TwoPhaseOptimizationResult:
    """Two-phase optimization result"""
    success: bool
    session_id: str

    # Optimization statistics
    skills_analyzed: List[str] = field(default_factory=list)
    skills_optimized: List[str] = field(default_factory=list)
    skills_skipped: List[str] = field(default_factory=list)
    skills_failed: List[str] = field(default_factory=list)

    # Detailed results
    skill_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Transaction info
    transaction_committed: bool = False
    rolled_back_skills: List[str] = field(default_factory=list)

    # Refactor results
    refactor_applied: List[Dict[str, Any]] = field(default_factory=list)
    refactor_opportunities_detected: int = 0

    # Error info
    error_message: Optional[str] = None

    # Metadata
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0


class TwoPhaseOptimizationEngine:
    """
    Two-phase optimization engine

    Integrates all optimization components and provides a unified optimization interface.

    Flow:
    1. Begin transaction
    2. Top-Down analysis: starting from the root node, recursively analyze problems and produce deltas
    3. Bottom-Up optimization: starting from leaf nodes, optimize progressively while propagating forward_feedback
    4. Commit or roll back the transaction
    """

    def __init__(
        self,
        skill_graph_manager=None,
        llm=None,
        logger=None,
        ckpt_dir: Optional[str] = None,
        config: Optional[TwoPhaseOptimizationConfig] = None,
        # Callback functions (optional)
        on_skill_optimized=None,
        on_optimization_failed=None,
        # External optimizer callback (used to invoke the actual LLM optimization logic)
        optimizer_callback=None,
        # Parameter for compatibility with the legacy interface
        optimization_tracker=None,
    ):
        """
        Initialize the engine

        Args:
            skill_graph_manager: skill graph manager
            llm: LLM instance
            logger: logger
            ckpt_dir: checkpoint directory
            config: optimization configuration
            on_skill_optimized: callback after a skill is successfully optimized
            on_optimization_failed: callback after a skill optimization fails
            optimizer_callback: external optimizer callback function with signature
                (skill_name, task, context, state, error, critique) -> Dict[str, Any]
                The return value should contain the 'new_code' and 'success' fields
            optimization_tracker: optimization tracker (legacy interface compatibility)
        """
        self.skill_graph_manager = skill_graph_manager
        self.llm = llm
        self.logger = logger
        self.ckpt_dir = ckpt_dir
        self.config = config or TwoPhaseOptimizationConfig()

        # Callbacks
        self.on_skill_optimized = on_skill_optimized
        self.on_optimization_failed = on_optimization_failed

        # External optimizer callback
        self.optimizer_callback = optimizer_callback

        # Legacy interface compatibility
        self.optimization_tracker = optimization_tracker

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize internal components"""
        # Transaction manager
        self.transaction_manager = TransactionManager(
            skill_graph_manager=self.skill_graph_manager,
            ckpt_dir=self.ckpt_dir,
            logger=self.logger,
        )

        # Refactor components (if available)
        self.refactor_detector = None
        self.refactors = {}

        # === refactor-initialization debug log ===
        self.logger.info(f"\033[36m[TwoPhaseEngine] Refactor initialization status:\033[0m")
        self.logger.info(f"\033[36m[TwoPhaseEngine]   REFACTOR_AVAILABLE = {REFACTOR_AVAILABLE}\033[0m")
        self.logger.info(f"\033[36m[TwoPhaseEngine]   config.enable_post_optimization_refactor = {self.config.enable_post_optimization_refactor}\033[0m")
        self.logger.info(f"\033[36m[TwoPhaseEngine]   config.refactor_types_enabled = {self.config.refactor_types_enabled}\033[0m")
        # === End of debug log ===

        if REFACTOR_AVAILABLE and self.config.enable_post_optimization_refactor:
            self.refactor_detector = RefactorDetector(
                skill_graph_manager=self.skill_graph_manager,
                llm=self.llm,
                logger=self.logger,
            )
            # Initialize refactors
            if "parametric" in self.config.refactor_types_enabled:
                self.refactors["parametric"] = ParametricRefactor(
                    skill_graph_manager=self.skill_graph_manager,
                    llm=self.llm,
                    logger=self.logger,
                )
                self.logger.info(f"\033[32m[TwoPhaseEngine] Initialized parametric refactor\033[0m")
            if "behavioral" in self.config.refactor_types_enabled:
                self.refactors["behavioral"] = BehavioralRefactor(
                    skill_graph_manager=self.skill_graph_manager,
                    llm=self.llm,
                    logger=self.logger,
                )
                self.logger.info(f"\033[32m[TwoPhaseEngine] Initialized behavioral refactor\033[0m")
            if "merge_siblings" in self.config.refactor_types_enabled:
                self.refactors["merge_siblings"] = SiblingRefactor(
                    skill_graph_manager=self.skill_graph_manager,
                    llm=self.llm,
                    logger=self.logger,
                )
                self.logger.info(f"\033[32m[TwoPhaseEngine] Initialized merge_siblings refactor\033[0m")
            if "duplication" in self.config.refactor_types_enabled:
                self.refactors["duplication"] = DuplicationRefactor(
                    skill_graph_manager=self.skill_graph_manager,
                    llm=self.llm,
                    logger=self.logger,
                )
                self.logger.info(f"\033[32m[TwoPhaseEngine] Initialized duplication refactor\033[0m")
            if "extract_common" in self.config.refactor_types_enabled:
                self.refactors["extract_common"] = SubskillExtractionRefactor(
                    skill_graph_manager=self.skill_graph_manager,
                    llm=self.llm,
                    logger=self.logger,
                )
                self.logger.info(f"\033[32m[TwoPhaseEngine] Initialized extract_common refactor\033[0m")

            self.logger.info(f"\033[36m[TwoPhaseEngine] Final initialized refactors: {list(self.refactors.keys())}\033[0m")
        else:
            if not REFACTOR_AVAILABLE:
                self.logger.warning(f"\033[33m[TwoPhaseEngine] Refactors unavailable: REFACTOR_AVAILABLE=False (probably an import failure)\033[0m")
            if not self.config.enable_post_optimization_refactor:
                self.logger.warning(f"\033[33m[TwoPhaseEngine] Refactors disabled: config.enable_post_optimization_refactor=False\033[0m")

        # Pure pipeline
        self.pure_pipeline = None
        self.skill_info_getter = None
        self._init_pure_pipeline()

    def _init_pure_pipeline(self):
        """Initialize the pure pipeline"""
        try:
            # Create skill_info_getter
            self.skill_info_getter = create_skill_info_getter(
                skill_graph_manager=self.skill_graph_manager,
                max_traces=5,
                include_statistics=True,
                logger=self.logger,
            )

            # Create SkipChecker for P(update s) gating at Phase 2
            # B1: Pass maturity gating params from config
            skip_checker = None
            try:
                from .validators.skip_checker import SkipChecker
                from .config import SkipOptimizationConfig
                skip_config = SkipOptimizationConfig(
                    OPTIMIZATION_THRESHOLD=self.config.optimization_threshold,
                    OPTIMIZATION_EPSILON=self.config.optimization_epsilon,
                    OPTIMIZATION_GAMMA=self.config.optimization_gamma,
                )
                skip_checker = SkipChecker(
                    skill_graph_manager=self.skill_graph_manager,
                    logger=self.logger,
                    config=skip_config,
                )
            except Exception as e:
                self._log(f"SkipChecker init failed (P(update s) disabled): {e}", "warning")

            # Create the pipeline (supports pattern learning)
            self.pure_pipeline = create_two_phase_pipeline(
                llm=self.llm,
                optimize_fn=self._pure_pipeline_optimize_fn,
                max_depth=self.config.max_reflection_depth,
                logger=self.logger,
                pure_reasoning=self.config.pure_reasoning,
                include_reasoning_examples=self.config.include_reasoning_examples,
                use_factual_primitive_doc=self.config.use_factual_primitive_doc,
                skip_checker=skip_checker,
                skill_graph_manager=self.skill_graph_manager,
            )

            # Inject KnowledgeRetrieval diagnostic logger into analyzer
            try:
                from skillnet.utils.log_utils import LoggerManager
                if self.ckpt_dir:
                    kr_logger = LoggerManager.get_logger("KnowledgeRetrieval", self.ckpt_dir)
                    reflection = getattr(self.pure_pipeline, 'pure_reflection', None)
                    if reflection and hasattr(reflection, 'analyzer'):
                        reflection.analyzer._kr_logger = kr_logger
            except Exception:
                pass

            self._log("Pure pipeline initialized successfully", "info")
        except Exception as e:
            self._log(f"Pure pipeline initialization failed: {e}", "warning")
            self.pure_pipeline = None

    def _pure_pipeline_optimize_fn(
        self,
        skill_name: str,
        delta,
        context: str,
    ) -> Tuple[str, bool]:
        """
        Optimization function for the pure pipeline

        Args:
            skill_name: skill name
            delta: SkillDelta
            context: optimization context

        Returns:
            Tuple[str, bool]: (new_code, success)
        """
        # Get the node
        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return "", False

        self._log(f"[PurePipeline] Optimizing {skill_name}: {context[:100]}...", "info")

        # If an external optimizer callback is configured, use it for the actual LLM optimization
        if self.optimizer_callback:
            try:
                # Extract feedback info from the delta
                critique = ""
                error = ""
                task = ""

                if delta and hasattr(delta, 'gradients'):
                    for grad in delta.gradients:
                        # The Gradient class stores its description in the 'direction' field, not 'description'
                        if hasattr(grad, 'direction'):
                            critique += grad.direction + "\n"
                            # If a suggested_fix is available, include it in critique too
                            if hasattr(grad, 'suggested_fix') and grad.suggested_fix:
                                critique += f"Suggested fix: {grad.suggested_fix}\n"
                        if hasattr(grad, 'evidence') and grad.evidence:
                            critique += f"Evidence: {grad.evidence}\n"

                # Extract task info from the context
                if 'task' in context.lower():
                    import re
                    task_match = re.search(r'task[:\s]+([^\n]+)', context, re.IGNORECASE)
                    if task_match:
                        task = task_match.group(1).strip()

                # Invoke the external optimizer
                # Disable internal backpropagation because ReflectionChain has already handled it
                # Pass the original skill_delta to preserve structured info (gradient_type, magnitude, etc.)
                result = self.optimizer_callback(
                    skill_name=skill_name,
                    current_task=task,
                    current_context=context,
                    current_state=None,
                    current_error=error,
                    current_critique=critique,
                    skill_delta=delta,  # pass the original SkillDelta to preserve structured Gradient info
                    quality_metrics=self._current_quality_metrics,  # Critic quality metrics
                    chat_log=self._current_chat_log,  # Chat log for diagnostic info
                )

                if result and isinstance(result, dict):
                    new_code = result.get("new_code", "")
                    success = result.get("success", False)

                    if success and new_code and new_code != node.code:
                        self._log(f"[PurePipeline] ✓ {skill_name} optimization succeeded (code updated)", "info")
                        return new_code, True
                    elif success:
                        self._log(f"[PurePipeline] ✓ {skill_name} optimization succeeded (no changes needed)", "info")
                        return node.code, True
                    else:
                        reason = result.get("reason", result.get("error", "unknown"))
                        self._log(f"[PurePipeline] ✗ {skill_name} optimization failed: {reason}", "warning")
                        return node.code, False

            except Exception as e:
                self._log(f"[PurePipeline] Optimizer callback exception: {e}", "error")
                return node.code, False

        # If no external optimizer is configured, return the original code, meaning "no change needed"
        self._log(f"[PurePipeline] {skill_name} has no external optimizer callback; keeping original code", "warning")
        return node.code, True

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            log_fn = getattr(self.logger, level, self.logger.info)
            log_fn(message)

    def get_last_depth_stats(self) -> Optional[Dict[str, Any]]:
        """
        B3: Extract credit assignment depth stats from last reflection chain.

        Returns dict with max_depth, total_nodes, depth_distribution,
        or None if no chain result available.
        """
        chain_result = getattr(self.pure_pipeline, '_last_chain_result', None) if self.pure_pipeline else None
        if not chain_result:
            return None

        # Build depth distribution from per-node depths
        depth_dist: Dict[int, int] = {}
        for node in chain_result.nodes.values():
            d = node.depth
            depth_dist[d] = depth_dist.get(d, 0) + 1

        return {
            "max_depth": chain_result.max_depth,
            "total_nodes": chain_result.total_nodes,
            "depth_distribution": depth_dist,
        }

    def get_last_execution_stats(self) -> Optional[Dict[str, Any]]:
        """
        5C: Extract REFLECT invocation stats from last pipeline execution.

        Returns dict with total_optimized, successful, failed, skipped,
        skipped_zero_gradient counts, or None if no execution result available.
        """
        er = getattr(self, '_last_execution_result', None)
        if not er:
            return None

        return {
            "total_optimized": er.total_optimized,
            "successful": er.successful,
            "failed": er.failed,
            "skipped": er.skipped,
            "skipped_zero_gradient": len(er.skipped_zero_gradient),
        }

    def optimize(
        self,
        skills_to_optimize: List[str],
        current_task: Optional[str] = None,
        current_context: Optional[str] = None,
        current_state: Optional[Dict[str, Any]] = None,
        current_error: Optional[str] = None,
        current_critique: Optional[str] = None,
        momentum_window: int = 5,
        skill_execution_results: Optional[Dict[str, Dict[str, Any]]] = None,
        skill_events_unreliable: bool = False,
        quality_metrics: Optional[Dict[str, Any]] = None,  # Critic quality metrics
        chat_log: str = "",  # Chat log from onChat events
    ) -> TwoPhaseOptimizationResult:
        """
        Execute the two-phase optimization

        Args:
            skills_to_optimize: list of skills to optimize
            current_task: current task
            current_context: current context
            current_state: current environment state
            current_error: current execution error
            current_critique: current critique
            momentum_window: momentum window size
            skill_execution_results: execution result for each skill
            skill_events_unreliable: whether the event mechanism is unreliable
            chat_log: v7.7 Chat log containing diagnostic messages

        Returns:
            TwoPhaseOptimizationResult: optimization result
        """
        import time
        start_time = time.time()

        result = TwoPhaseOptimizationResult(
            success=False,
            session_id="",
        )

        if not skills_to_optimize:
            result.error_message = "No skills to optimize"
            return result

        primary_skill = skills_to_optimize[0]

        # ========== Use Pure Pipeline ==========
        if self.pure_pipeline is not None:
            try:
                return self._optimize_with_pure_pipeline(
                    skills_to_optimize=skills_to_optimize,
                    current_task=current_task,
                    current_error=current_error,
                    current_critique=current_critique,
                    start_time=start_time,
                    quality_metrics=quality_metrics,
                    chat_log=chat_log,
                    skill_execution_results=skill_execution_results,
                )
            except Exception as e:
                self._log(f"[TwoPhaseEngine] Pure Pipeline execution failed: {e}", "error")
                result.error_message = f"Pure pipeline failed: {e}"
                result.completed_at = datetime.now().isoformat()
                result.duration_seconds = time.time() - start_time
                return result
        else:
            # Pure Pipeline not initialized
            result.error_message = "Pure pipeline not initialized"
            result.completed_at = datetime.now().isoformat()
            result.duration_seconds = time.time() - start_time
            return result

    def _optimize_with_pure_pipeline(
        self,
        skills_to_optimize: List[str],
        current_task: Optional[str],
        current_error: Optional[str],
        current_critique: Optional[str],
        start_time: float,
        quality_metrics: Optional[Dict[str, Any]] = None,
        chat_log: str = "",  # Chat log from onChat events
        skill_execution_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> TwoPhaseOptimizationResult:
        """
        Execute optimization via the pure pipeline

        Args:
            skills_to_optimize: list of skills to optimize
            current_task: current task
            current_error: current error
            current_critique: current critique
            start_time: start time
            quality_metrics: v7.5.4 Critic quality metrics
            chat_log: v7.7 Chat log containing diagnostic messages

        Returns:
            TwoPhaseOptimizationResult: optimization result
        """
        import time

        # store quality_metrics for use by _pure_pipeline_optimize_fn
        self._current_quality_metrics = quality_metrics
        # store chat_log for use by _pure_pipeline_optimize_fn
        self._current_chat_log = chat_log

        result = TwoPhaseOptimizationResult(
            success=False,
            session_id=f"pure_{datetime.now().timestamp()}",
        )

        primary_skill = skills_to_optimize[0]

        # ========== Transaction support ==========
        session = None
        subgraph_txn = None

        if self.config.enable_transactions:
            try:
                session = self.transaction_manager.begin_session(
                    session_type="optimization",
                    primary_skill=primary_skill,
                    task=current_task,
                )
                result.session_id = session.session_id

                subgraph_txn = self.transaction_manager.begin_subgraph_transaction(
                    session, primary_skill, current_task
                )
                self._log(f"[PurePipeline] Started transaction {session.session_id}", "info")
            except Exception as e:
                self._log(f"[PurePipeline] Transaction initialization failed: {e}", "warning")
                # Continue executing, but without transactions

        # Build feedback content
        feedback_content = ""
        if current_error:
            feedback_content += f"Error: {current_error}\n"
        if current_critique:
            feedback_content += f"Critique: {current_critique}\n"

        feedback_type = "error" if current_error else "critique"

        self._log(f"[PurePipeline] Starting optimization of {primary_skill}", "info")
        self._log(f"[PurePipeline] Feedback: {feedback_content[:200]}...", "info")

        try:
            # Run the pipeline
            execution_result = self.pure_pipeline.run(
                root_skill_name=primary_skill,
                root_feedback_content=feedback_content,
                root_feedback_type=feedback_type,
                skill_info_getter=self.skill_info_getter,
                chat_log=chat_log,  # Pass chat log to pipeline
            )
            # 5C: Store execution result for operational statistics extraction
            self._last_execution_result = execution_result

            # Extract fix_targets from chain_result
            self._current_fix_targets = {}
            chain_result = getattr(self.pure_pipeline, '_last_chain_result', None)
            if chain_result:
                for name, node in chain_result.nodes.items():
                    ft = getattr(node, 'fix_target', None)
                    if ft:
                        from skillnet.agents.optimizer.feedback.types import FixTargetType
                        self._current_fix_targets[name] = (
                            FixTargetType.CALLER_FIX if ft == 'caller_fix'
                            else FixTargetType.BOTH_FIX if ft == 'both_fix'
                            else FixTargetType.CALLEE_FIX
                        )

            # Convert results
            result.success = execution_result.successful > 0
            result.skills_analyzed = list(execution_result.node_results.keys())
            result.skills_optimized = [
                name for name, fb in execution_result.node_results.items()
                if fb.optimization_successful
            ]
            result.skills_failed = [
                name for name, fb in execution_result.node_results.items()
                if not fb.optimization_successful
            ]

            # Record detailed results and trigger callbacks
            for skill_name, fb in execution_result.node_results.items():
                result.skill_results[skill_name] = {
                    "optimization_successful": fb.optimization_successful,
                    "success": fb.optimization_successful,  # compatibility with psn.py's check
                    "changes_made": fb.changes_made,
                    "interface_changed": fb.interface_changed,
                    "effects_changed": fb.effects_changed,
                    "new_code": fb.new_code,  # pass the new code for persistence
                    "old_code": fb.old_code,  # pass the old code for rollback
                }

                # Phase 13.B-9: stash trigger context on fb so the callback's
                # detailed-log writer can capture what Phase 1 actually saw.
                # Without this, opt_*.json had `task` and `error_message`
                # empty even on real failures (logger only got fb output, not
                # the input that triggered the optimization). Diagnosing
                # Phase 1 hallucinations like 'Add proximity validation
                # before mineBlock' is impossible without this.
                if fb.task is None:
                    fb.task = current_task
                if getattr(fb, 'error_message', None) is None:
                    # OptimizationForwardFeedback may not have this field;
                    # set as attribute for callback consumption either way.
                    try:
                        fb.error_message = current_error
                    except Exception:
                        pass
                if getattr(fb, 'error_stack', None) is None:
                    try:
                        _ser = (skill_execution_results or {}).get(skill_name) or {}
                        fb.error_stack = _ser.get('error_stack')
                    except Exception:
                        pass

                # ========== Callback support ==========
                if fb.optimization_successful:
                    # Phase 8: mark as experimental after optimization, awaiting execution verification
                    # The newly generated code must be verified by a successful execution before refactor is triggered
                    self._mark_skill_as_experimental(skill_name, current_task)

                    if self.on_skill_optimized:
                        try:
                            self.on_skill_optimized(skill_name, fb)
                        except Exception as e:
                            self._log(f"[PurePipeline] on_skill_optimized callback exception: {e}", "warning")
                else:
                    if self.on_optimization_failed:
                        try:
                            self.on_optimization_failed(skill_name, fb)
                        except Exception as e:
                            self._log(f"[PurePipeline] on_optimization_failed callback exception: {e}", "warning")

            # ========== Record zero-gradient events (eliminate survivor bias) ==========
            if execution_result.skipped_zero_gradient and self.optimization_tracker:
                from skillnet.agents.optimizer.feedback.types import (
                    ErrorCategory, FixTargetType,
                )
                for zg_skill in execution_result.skipped_zero_gradient:
                    try:
                        determined_ft = self._current_fix_targets.get(zg_skill, FixTargetType.CALLEE_FIX)
                        zg_node = self.skill_graph_manager.get_node(zg_skill) if self.skill_graph_manager else None
                        self.optimization_tracker.record_optimization(
                            skill_name=zg_skill,
                            error_category=ErrorCategory.UNKNOWN,
                            error_pattern="phase1_zero_gradient",
                            strategy_used="two_phase_pipeline",
                            fix_target=determined_ft,
                            successful=False,
                            task=current_task,
                            error_message=current_error[:500] if current_error else None,
                            error_stack=((skill_execution_results or {}).get(zg_skill) or {}).get('error_stack'),
                            failure_reason="Phase 1 produced zero gradients; Phase 2 skipped",
                            value_at_optimization=zg_node.value_function if zg_node else None,
                        )
                    except Exception as e:
                        self._log(f"[PurePipeline] Failed to record zero-gradient event: {e}", "warning")

            # Record P(update s) rejections in optimization tracker
            if execution_result and self.optimization_tracker:
                for sk_name, fb in execution_result.node_results.items():
                    if getattr(fb, 'skipped_reason', '') and getattr(fb, 'analysis_available', False):
                        try:
                            sk_node = self.skill_graph_manager.get_node(sk_name) if self.skill_graph_manager else None
                            self.optimization_tracker.record_optimization(
                                skill_name=sk_name,
                                error_category=ErrorCategory.UNKNOWN,
                                error_pattern="p_update_s_rejected",
                                strategy_used="two_phase_pipeline",
                                fix_target=self._current_fix_targets.get(sk_name, FixTargetType.CALLEE_FIX),
                                successful=False,
                                task=current_task,
                                failure_reason=f"P(update s) rejected: {fb.skipped_reason}",
                                value_at_optimization=sk_node.value_function if sk_node else None,
                            )
                        except Exception as e:
                            self._log(f"[PurePipeline] P(update s) tracking failed: {e}", "warning")

            # ========== Commit transaction ==========
            if session and subgraph_txn:
                if result.skills_failed and self.config.auto_rollback_on_failure:
                    # There are failures and auto-rollback is configured
                    self.transaction_manager.rollback_subgraph(subgraph_txn)
                    self.transaction_manager.end_session(session, commit=False)
                    result.transaction_committed = False
                    self._log(f"[PurePipeline] Transaction rolled back ({len(result.skills_failed)} failures)", "warning")
                else:
                    # Commit transaction
                    self.transaction_manager.commit_subgraph(subgraph_txn)
                    self.transaction_manager.end_session(session, commit=True)
                    result.transaction_committed = True
                    self._log(f"[PurePipeline] Transaction committed", "info")

        except Exception as e:
            # ========== Roll back transaction on exception ==========
            self._log(f"[PurePipeline] Execution exception: {e}", "error")

            if session and subgraph_txn:
                try:
                    self.transaction_manager.rollback_subgraph(subgraph_txn)
                    self.transaction_manager.end_session(session, commit=False)
                    result.transaction_committed = False
                    self._log(f"[PurePipeline] Transaction rolled back (exception)", "warning")
                except Exception as txn_error:
                    self._log(f"[PurePipeline] Transaction rollback failed: {txn_error}", "error")

            # clear quality_metrics
            self._current_quality_metrics = None
            raise  # re-raise the exception for upper layers to handle

        result.completed_at = datetime.now().isoformat()
        result.duration_seconds = time.time() - start_time

        self._log(
            f"[PurePipeline] Optimization complete: "
            f"succeeded={len(result.skills_optimized)}, "
            f"failed={len(result.skills_failed)}, "
            f"duration={result.duration_seconds:.2f}s",
            "info"
        )

        # Write optimization session summary
        try:
            import json as _json
            from .knowledge.llm_knowledge_retriever import get_kr_stats
            summary = {
                "session_id": result.session_id,
                "task": current_task,
                "primary_skill": primary_skill,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "duration_seconds": result.duration_seconds,
                "skills_analyzed": result.skills_analyzed,
                "skills_optimized": result.skills_optimized,
                "skills_failed": result.skills_failed,
                "skills_skipped": result.skills_skipped,
                "kr_stats": get_kr_stats(),
            }
            summary_dir = os.path.join(self.ckpt_dir, "skill_graph", "optimizer", "session_summaries")
            os.makedirs(summary_dir, exist_ok=True)
            with open(os.path.join(summary_dir, f"{result.session_id}.json"), 'w') as f:
                _json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # clear quality_metrics
        self._current_quality_metrics = None

        return result

    # ========== Experimental skill management ==========

    def _mark_skill_as_experimental(
        self,
        skill_name: str,
        task: Optional[str] = None,
    ) -> None:
        """
        Mark an optimized skill as experimental

        Experimental skills only trigger refactor detection after
        they have been verified by a successful real execution.

        Args:
            skill_name: skill name
            task: associated task
        """
        if not self.skill_graph_manager:
            return

        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return

        # Mark as experimental
        node.is_experimental = True
        node.experimental_task = task

        # If created_for_task is not yet set, this is a newly created skill; set it
        # If it is already set, this is a previously created skill being re-optimized; keep the original value
        if node.created_for_task is None:
            node.created_for_task = task

        self._log(
            f"[TwoPhaseEngine] Marked '{skill_name}' as experimental (created_for_task={node.created_for_task}), awaiting execution verification",
            "info"
        )

    # ========== Delayed refactor (called after skill has been verified) ==========

    def trigger_delayed_refactor(
        self,
        skill_name: str,
        current_task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trigger delayed refactor detection

        This method should be called after an experimental skill has been verified.
        When to call:
        - From psn.py, when record_execution() returns should_trigger_refactor=True

        Args:
            skill_name: name of the skill that was just verified
            current_task: current task

        Returns:
            Dict: {
                "opportunities": number of detected opportunities,
                "applied": list of applied refactors
            }
        """
        if not REFACTOR_AVAILABLE or not self.refactor_detector:
            return {"opportunities": 0, "applied": []}

        self._log(
            f"[TwoPhaseEngine] ========== Delayed refactor: {skill_name} verified ==========",
            "info"
        )

        opportunities_count = 0
        applied_refactors = []

        try:
            # Detect refactor opportunities
            opportunities = self.refactor_detector.detect_opportunities(skill_name)
            opportunities_count = len(opportunities)

            if not opportunities:
                self._log(
                    f"[TwoPhaseEngine] No refactor opportunities found for {skill_name}",
                    "info"
                )
                return {"opportunities": 0, "applied": []}

            self._log(
                f"[TwoPhaseEngine] Found {len(opportunities)} refactor opportunities for {skill_name}",
                "info"
            )

            # Apply refactors
            for opportunity in opportunities:
                refactor_result = self._apply_refactor(opportunity)
                if refactor_result and refactor_result.get("success"):
                    applied_refactors.append(refactor_result)
                    self._log(
                        f"[TwoPhaseEngine] ✓ Applied refactor: {opportunity.refactor_type.value} "
                        f"({opportunity.source_skill} -> {opportunity.target_skill})",
                        "info"
                    )

        except Exception as e:
            self._log(
                f"[TwoPhaseEngine] Refactor detection failed for {skill_name}: {e}",
                "warning"
            )

        if applied_refactors:
            self._log(
                f"[TwoPhaseEngine] Delayed refactor complete: applied {len(applied_refactors)} refactors",
                "info"
            )

        return {
            "opportunities": opportunities_count,
            "applied": applied_refactors,
        }

    def _apply_refactor(
        self,
        opportunity: 'RefactorOpportunity',
    ) -> Optional[Dict[str, Any]]:
        """
        Apply a refactor

        Args:
            opportunity: refactor opportunity

        Returns:
            Optional[Dict]: refactor result
        """
        if not REFACTOR_AVAILABLE:
            return None

        refactor_type = opportunity.refactor_type.value
        refactor = self.refactors.get(refactor_type)

        if not refactor:
            self._log(
                f"[TwoPhaseEngine] Refactor not found: {refactor_type}",
                "warning"
            )
            return None

        try:
            result = refactor.apply(opportunity)

            return {
                "success": result.success,
                "refactor_type": refactor_type,
                "source_skill": opportunity.source_skill,
                "target_skill": opportunity.target_skill,
                "old_code": result.old_code if hasattr(result, 'old_code') else None,
                "new_code": result.new_code if hasattr(result, 'new_code') else None,
                "changes_made": result.changes_made if hasattr(result, 'changes_made') else [],
                "error_message": result.error_message if hasattr(result, 'error_message') else None,
            }

        except Exception as e:
            self._log(
                f"[TwoPhaseEngine] Failed to apply refactor: {e}",
                "error"
            )
            return {
                "success": False,
                "refactor_type": refactor_type,
                "source_skill": opportunity.source_skill,
                "target_skill": opportunity.target_skill,
                "error_message": str(e),
            }

    # ========== Experimental skill lifecycle management ==========

    def on_task_completed(
        self,
        task: str,
        success: bool,
        completing_skills: List[str],
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified entry point for handling task completion

        This is the core method for experimental-skill lifecycle management.
        It performs different handling depending on whether the task succeeded:
        - Success: verify new skills, trigger refactor, clean up unused experimental skills
        - Failure: clean up all experimental skills produced for this task
        - Preflight failure: skip experimental-skill handling (code-generation/interface issue, not a task-difficulty issue)

        **Important rule**: only newly created skills trigger refactor; old skills do not.

        Args:
            task: task name
            success: whether the task succeeded
            completing_skills: list of skills that completed the task (may include new and old skills)
            error_type: failure type; "preflight" indicates a code-generation/interface issue

        Returns:
            Dict[str, Any]: {
                "verified_skills": [...],    # verified skills (new skills, experimental flag removed)
                "cleaned_skills": [...],     # cleaned-up skills
                "refactor_results": [...],   # refactor results
                "skipped_skills": [...],     # skipped skills (old skills, no refactor triggered)
            }
        """
        result = {
            "verified_skills": [],
            "cleaned_skills": [],
            "refactor_results": [],
            "skipped_skills": [],
            "error_type": error_type,
        }

        if not self.skill_graph_manager:
            return result

        # List of system-level error types (these are project code issues, not skill code issues, and should not trigger skill deletion)
        SYSTEM_ERROR_TYPES = {
            "TypeError", "ImportError", "AttributeError", "NameError",
            "ModuleNotFoundError", "KeyboardInterrupt", "SystemExit",
            "SyntaxError", "IndentationError", "TabError",
        }

        # Check whether skill cleanup should be skipped
        if not success and error_type:
            # System-level error: project code issue, not a skill issue
            if error_type in SYSTEM_ERROR_TYPES:
                self._log(
                    f"[Lifecycle] Task '{task}' failure reason={error_type} (system-level error), "
                    f"skipping experimental skill cleanup",
                    "warning"
                )
                return result

            # Preflight rejection: code-generation/interface issue
            if error_type == "preflight":
                self._log(
                    f"[Lifecycle] Task '{task}' failure reason=preflight, skipping experimental skill handling",
                    "info"
                )
                return result

        self._log(
            f"[Lifecycle] ========== on_task_completed: task='{task}', success={success} ==========",
            "info"
        )
        self._log(
            f"[Lifecycle] completing_skills: {completing_skills}",
            "info"
        )

        if not success:
            # Task failed: clean up all experimental skills produced for this task
            task_created_skills = self._get_skills_created_for_task(task)

            # Detailed logging: explain which skills are deleted vs preserved
            skipped_skills = [s for s in completing_skills if s not in task_created_skills]
            self._log(
                f"[Lifecycle] Task failed; cleaning up {len(task_created_skills)} experimental skills created for this task",
                "info"
            )

            # Explain the skipped skills
            for skill_name in skipped_skills:
                node = self.skill_graph_manager.get_node(skill_name)
                if node:
                    is_verified = getattr(node, 'is_verified', False)
                    created_for = getattr(node, 'created_for_task', 'unknown')
                    if is_verified:
                        self._log(
                            f"[Lifecycle] ⏭️ Preserving '{skill_name}': verified reusable skill (created for task: {created_for})",
                            "info"
                        )
                    else:
                        self._log(
                            f"[Lifecycle] ⏭️ Preserving '{skill_name}': not created for the current task (created for: {created_for})",
                            "info"
                        )

            for skill_name in task_created_skills:
                self._cleanup_experimental_skill(skill_name, "task_failed")
                result["cleaned_skills"].append(skill_name)
            return result

        # Task succeeded: process the skills that completed the task
        for skill_name in completing_skills:
            node = self.skill_graph_manager.get_node(skill_name)
            if not node:
                continue

            # Decide whether this is a skill newly created for this task
            is_new_skill = (
                node.is_experimental and
                node.experimental_task == task
            )

            if is_new_skill:
                # Phase 5: three categories A/B/C - distinguish task_specific from non-task_specific
                is_task_specific = getattr(node, 'is_task_specific', False)

                if is_task_specific:
                    # Category C1: task_specific + experimental succeeded
                    # Mark as deprecated; do not trigger refactor (these skills have no cross-task reuse value)
                    self._cleanup_experimental_skill(skill_name, "task_specific_succeeded")
                    result["cleaned_skills"].append(skill_name)

                    self._log(
                        f"[Lifecycle] 🗑️ Cleaning up task_specific skill '{skill_name}' (succeeded but has no reuse value)",
                        "info"
                    )
                else:
                    # Category C2: non-task_specific + experimental succeeded
                    # Remove experimental flag, mark as verified, trigger refactor
                    node.is_experimental = False
                    node.is_verified = True
                    result["verified_skills"].append(skill_name)

                    self._log(
                        f"[Lifecycle] ✓ Verified new skill '{skill_name}' (is_verified=True)",
                        "info"
                    )

                    # Only new skills trigger refactor
                    refactor_result = self.trigger_delayed_refactor(skill_name, task)
                    result["refactor_results"].append({
                        "skill": skill_name,
                        "result": refactor_result,
                    })

                    if refactor_result.get("applied"):
                        self._log(
                            f"[Lifecycle] ✓ Applied {len(refactor_result['applied'])} refactors",
                            "info"
                        )
            else:
                # Old skill (already verified): skip, do not trigger refactor
                result["skipped_skills"].append(skill_name)
                self._log(
                    f"[Lifecycle] ⏭️ Skipping old skill '{skill_name}' (is_verified={getattr(node, 'is_verified', False)})",
                    "info"
                )

        # Clean up unused experimental skills produced for this task
        task_created_skills = self._get_skills_created_for_task(task)
        for skill_name in task_created_skills:
            if skill_name not in completing_skills:
                self._cleanup_experimental_skill(skill_name, "not_used_in_success")
                result["cleaned_skills"].append(skill_name)

        self._log(
            f"[Lifecycle] Complete: verified={len(result['verified_skills'])}, "
            f"skipped={len(result['skipped_skills'])}, cleaned={len(result['cleaned_skills'])}",
            "info"
        )

        # Phase 11: persist changes (consistent with graph_manager_impl.py)
        if result["verified_skills"] or result["cleaned_skills"]:
            if hasattr(self.skill_graph_manager, 'save'):
                self.skill_graph_manager.save()

        return result

    def _get_skills_created_for_task(self, task: str) -> List[str]:
        """
        Get all skills created for the given task

        Checks both fields to make sure all related experimental skills are found:
        - created_for_task: set by _mark_skill_as_experimental on successful optimization
        - experimental_task: set during record_execution (even when optimization fails)

        Args:
            task: task name

        Returns:
            List[str]: list of skill names
        """
        if not self.skill_graph_manager:
            return []

        skills = []
        for name, node in self.skill_graph_manager.iter_skills(include_task_specific=True):
            # Check created_for_task (set on successful optimization)
            if getattr(node, 'created_for_task', None) == task:
                skills.append(name)
            # Also check experimental_task (set during record_execution, even when optimization fails)
            elif (node.is_experimental and
                  getattr(node, 'experimental_task', None) == task):
                skills.append(name)

        return skills

    def _cleanup_experimental_skill(self, skill_name: str, reason: str) -> None:
        """
        Clean up an experimental skill - Phase 9: actually delete it, not merely mark it

        Uses delete_skill() to perform 5 layers of cleanup:
        1. In-memory graph structure
        2. Vector database
        3. Disk files
        4. Graph JSON snapshot
        5. Coverage relationships

        Before deletion, the skill is automatically backed up to the skill_graph/deleted/ directory.

        Args:
            skill_name: skill name
            reason: cleanup reason ("task_failed", "not_used_in_success", "task_specific_succeeded")
        """
        if not self.skill_graph_manager:
            return

        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return

        # Verified skills must not be removed by experimental cleanup
        # Verification means the skill has already succeeded in a real execution, giving it cross-task reuse value
        # Re-optimization may set is_experimental=True, but verified status represents permanent value
        # Without this guard, later task failures would associate the skill via _get_skills_created_for_task and delete it
        if getattr(node, 'is_verified', False):
            self._log(
                f"[Lifecycle] ⏭️ Protecting verified skill '{skill_name}' from deletion "
                f"(reason: {reason}, is_verified=True)",
                "info"
            )
            # Restore non-experimental state so subsequent checkpoints can persist it
            node.is_experimental = False
            return

        # Phase 9: actually delete via delete_skill()
        success = self.skill_graph_manager.delete_skill(
            skill_name,
            reason=f"Experimental skill cleanup: {reason}"
        )

        if success:
            self._log(
                f"[Lifecycle] 🗑️ Deleted experimental skill '{skill_name}': {reason}",
                "info"
            )
        else:
            self._log(
                f"[Lifecycle] ⚠️ Failed to delete experimental skill '{skill_name}'",
                "warning"
            )
