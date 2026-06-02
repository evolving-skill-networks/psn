"""
Reflection Chain - explicit representation of the recursive reflection chain

Implements the recursive application of the chain rule:
    ∂L/∂root = Σ (∂L/∂child · ∂child/∂root) for child in FaultyChildren

Core concepts:
1. A ReflectionChain is an explicit DAG (directed acyclic graph)
2. Each node holds the skill's delta and the feedbacks propagated to its children
3. Edges represent the direction of feedback propagation
4. The chain provides a topological order that ensures the correct optimization sequence

Mathematical form:
- Start at root node A and call psn_reflection recursively
- psn_reflection(A) → (δ_A, {φ_(A→B), φ_(A→C), ...})
- For each B ∈ FaultyChildren(A), continue psn_reflection(B, φ_(A→B))
- Until a leaf node is reached or the maximum depth is hit
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Iterator
from enum import Enum
from datetime import datetime
from collections import deque

from .pure_reflection import (
    PureReflection,
    ReflectionInput,
    ReflectionOutput,
    SkillDelta,
    PropagatedFeedback,
    Gradient,
    GradientType,
)
from .consider_function import (
    ConsiderFunction,
    ConsiderResult,
    OptimizationForwardFeedback,
)


# ============================================================
# Section 1: chain nodes and edges
# ============================================================

class NodeStatus(Enum):
    """Node status"""
    PENDING = "pending"              # awaiting analysis
    ANALYZED = "analyzed"            # analyzed (Top-Down)
    OPTIMIZING = "optimizing"        # optimization in progress (Bottom-Up)
    OPTIMIZED = "optimized"          # optimized
    SKIPPED = "skipped"              # skipped
    FAILED = "failed"                # failed


@dataclass
class ChainNode:
    """
    A node in the reflection chain.

    Represents a skill that needs to be analyzed and optimized.
    """
    skill_name: str
    status: NodeStatus = NodeStatus.PENDING

    # Top-Down analysis result
    delta: Optional[SkillDelta] = None
    reflection_output: Optional[ReflectionOutput] = None

    # Bottom-Up optimization result
    consider_result: Optional[ConsiderResult] = None
    forward_feedback: Optional[OptimizationForwardFeedback] = None

    # Graph structure
    parents: Set[str] = field(default_factory=set)     # parent nodes
    children: Set[str] = field(default_factory=set)    # child nodes
    depth: int = 0                                      # depth in the chain

    # Metadata
    analysis_timestamp: Optional[str] = None
    optimization_timestamp: Optional[str] = None
    skill_info: Optional[Dict[str, Any]] = None  # full skill info (including code)

    # LLM-driven caller/callee attribution
    fix_target: Optional[str] = None  # 'caller_fix', 'both_fix', or None (=callee_fix)

    @property
    def is_leaf(self) -> bool:
        """Whether this is a leaf node"""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Whether this is a root node"""
        return len(self.parents) == 0

    @property
    def total_gradient_magnitude(self) -> float:
        """Total gradient magnitude"""
        if self.delta:
            return self.delta.total_magnitude
        return 0.0


# ============================================================
# Section 2: Reflection Chain
# ============================================================

@dataclass
class ReflectionChainResult:
    """
    The complete result of a ReflectionChain.

    Contains the full subgraph produced by Top-Down analysis and the optimization order.
    """
    # Graph structure
    nodes: Dict[str, ChainNode]

    # Root node
    root_skill: str

    # Optimization order (Bottom-Up)
    optimization_order: List[str]

    # Statistics
    total_nodes: int
    max_depth: int
    total_gradient_magnitude: float



class ReflectionChain:
    """
    Recursive reflection chain.

    Implements the full Top-Down analysis flow:
    1. Start from the root skill
    2. Recursively invoke psn_reflection
    3. Build a DAG containing all problematic skills
    4. Provide a Bottom-Up optimization order

    Usage:
        chain = ReflectionChain(
            pure_reflection=reflection,
            max_depth=3
        )

        # Build the reflection chain
        result = chain.build(
            root_skill_name="mySkill",
            root_feedback=feedback,
            skill_info_getter=get_skill_info
        )

        # Iterate in Bottom-Up order
        for skill_name in result.optimization_order:
            node = result.nodes[skill_name]
            # Use node.delta to optimize
    """

    def __init__(
        self,
        pure_reflection: PureReflection,
        max_depth: int = 3,
        logger=None,
    ):
        """
        Initialize the ReflectionChain

        Args:
            pure_reflection: PureReflection instance
            max_depth: maximum recursion depth
            logger: logger
        """
        self.pure_reflection = pure_reflection
        self.max_depth = max_depth
        self.logger = logger

        # Internal state
        self._nodes: Dict[str, ChainNode] = {}

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(f"[{level.upper()}] {message}")

    def build(
        self,
        root_skill_name: str,
        root_feedback_content: str,
        root_feedback_type: str = "error",
        skill_info_getter=None,
        chat_log: str = "",  # Chat log from onChat events
    ) -> ReflectionChainResult:
        """
        Build the reflection chain.

        Starting from the root skill, recursively call psn_reflection to construct the full DAG.

        Args:
            root_skill_name: root skill name
            root_feedback_content: feedback content for the root skill
            root_feedback_type: feedback type
            skill_info_getter: callback that retrieves skill info
                signature: (skill_name: str) -> Dict[str, Any]
                returns: {"code": str, "description": str, "children": List[str], ...}
            chat_log: v7.7 Chat log containing diagnostic messages

        Returns:
            ReflectionChainResult: the constructed reflection chain
        """
        # Reset state
        self._nodes = {}

        # Build recursively
        self._build_recursive(
            skill_name=root_skill_name,
            feedback_content=root_feedback_content,
            feedback_type=root_feedback_type,
            skill_info_getter=skill_info_getter,
            propagated_feedback=None,
            depth=0,
            chat_log=chat_log,  # Pass chat log to recursive build
        )

        # Compute optimization order (Bottom-Up)
        optimization_order = self._topological_sort_reverse()

        # Compute statistics
        total_magnitude = sum(
            node.total_gradient_magnitude for node in self._nodes.values()
        )
        max_depth = max(node.depth for node in self._nodes.values()) if self._nodes else 0

        # Blind-spot 3 fix: aggregate diagnostic stats - quickly spot abnormal zero-gradient rates
        if self._nodes:
            zero_grad_nodes = [
                name for name, node in self._nodes.items()
                if node.delta and node.delta.total_magnitude == 0
            ]
            analyzed_count = sum(
                1 for node in self._nodes.values()
                if node.status == NodeStatus.ANALYZED
            )
            if self.logger:
                self.logger.info(
                    f"[ReflectionChain] Phase 1 summary: "
                    f"nodes={len(self._nodes)} analyzed={analyzed_count} "
                    f"zero_gradient={len(zero_grad_nodes)}/{analyzed_count} "
                    f"total_magnitude={total_magnitude:.2f} max_depth={max_depth}"
                )
                if zero_grad_nodes:
                    self.logger.warning(
                        f"[ReflectionChain] Zero gradient skills: {zero_grad_nodes}"
                    )
                # Per-skill Phase 1 gradient breakdown
                for name, node in self._nodes.items():
                    if node.delta and node.delta.gradients:
                        grad_types = [g.gradient_type.value for g in node.delta.gradients]
                        self.logger.info(
                            f"[Phase1] {name}: gradients={len(node.delta.gradients)} "
                            f"magnitude={node.delta.total_magnitude:.2f} "
                            f"types={grad_types} "
                            f"fix_target={node.fix_target or 'callee_fix'} "
                            f"faulty_children={list(node.reflection_output.faulty_children) if node.reflection_output else []}"
                        )
                    else:
                        self.logger.info(f"[Phase1] {name}: no gradients (magnitude=0.00)")

        return ReflectionChainResult(
            nodes=self._nodes,
            root_skill=root_skill_name,
            optimization_order=optimization_order,
            total_nodes=len(self._nodes),
            max_depth=max_depth,
            total_gradient_magnitude=total_magnitude,
        )

    def _build_recursive(
        self,
        skill_name: str,
        feedback_content: str,
        feedback_type: str,
        skill_info_getter,
        propagated_feedback: Optional[PropagatedFeedback],
        depth: int,
        parent_skill: Optional[str] = None,
        chat_log: str = "",  # Chat log from onChat events
    ):
        """Recursively build the reflection chain"""
        # Check depth limit
        if depth >= self.max_depth:
            self._log(f"Reached max depth {self.max_depth}; stopping recursion at {skill_name}", "info")
            return

        # Check whether it has already been analyzed
        if skill_name in self._nodes:
            # Add the edge but do not re-analyze
            if parent_skill:
                self._nodes[skill_name].parents.add(parent_skill)
            return

        # Fetch skill info
        skill_info = skill_info_getter(skill_name) if skill_info_getter else {}
        if not skill_info:
            self._log(f"Unable to fetch skill info: {skill_name}", "warning")
            return

        # Create the node
        node = ChainNode(
            skill_name=skill_name,
            depth=depth,
        )
        node.skill_info = skill_info  # store skill_info for later use
        if parent_skill:
            node.parents.add(parent_skill)

        # Build the reflection input
        reflection_input = ReflectionInput(
            skill_name=skill_name,
            skill_code=skill_info.get("code", ""),
            skill_description=skill_info.get("description", ""),
            feedback_content=feedback_content,
            feedback_type=feedback_type,
            execution_traces=skill_info.get("execution_traces", []),
            pre_state=skill_info.get("pre_state"),
            post_state=skill_info.get("post_state"),
            children=skill_info.get("children", []),
            children_info=skill_info.get("children_info", {}),
            propagated_feedback=propagated_feedback,
            chat_log=chat_log,  # Chat log for diagnostic info
            is_task_specific=skill_info.get("is_task_specific", False),
            # Plan v3-rev Fix 1.A: pass refactor-derived wrapper flags so Phase 1
            # can advise redirecting fixes to the covered_by parent instead of
            # expanding the wrapper into an inline reimplementation.
            is_covered=bool(skill_info.get("is_covered", False)),
            covered_by=skill_info.get("covered_by"),
        )

        # Run psn_reflection
        self._log(f"Analyzing skill: {skill_name} (depth={depth})", "info")
        output = self.pure_reflection.reflect(reflection_input)

        # Update the node
        node.delta = output.delta

        # Fallback: parent marks child as faulty, but the child's own analysis produces an insignificant gradient
        if (
            propagated_feedback
            and propagated_feedback.issue_description
            and (not node.delta or not node.delta.is_significant)
        ):
            # Phase 1 always propagates full gradients (no V(s) dampening).
            # P(update s) gates optimization at Phase 2 (ChainExecutor), not here.
            fallback_magnitude = propagated_feedback.weight * 0.5
            self._log(
                f"Fallback gradient: {skill_name}'s own analysis is insignificant "
                f"({node.delta.total_magnitude if node.delta else 0:.2f}), "
                f"but parent {propagated_feedback.source_skill} explicitly flagged it as problematic. "
                f"Constructing a fallback gradient from propagated_feedback (magnitude={fallback_magnitude:.2f})",
                "warning"
            )

            node.delta = SkillDelta(
                skill_name=skill_name,
                gradients=[Gradient(
                    gradient_type=GradientType.LOGIC,
                    direction=propagated_feedback.issue_description,
                    magnitude=fallback_magnitude,
                    evidence=(
                        f"Propagated from {propagated_feedback.source_skill}: {propagated_feedback.responsibility}"
                    ),
                    suggested_fix="",
                )],
            )

        node.reflection_output = output
        node.status = NodeStatus.ANALYZED
        node.analysis_timestamp = datetime.now().isoformat()

        # Set fix_target based on LLM attribution
        has_caller_fix = bool(getattr(output, 'caller_fix_children', set()))
        has_child_issues = bool(output.faulty_children)
        if has_caller_fix and has_child_issues:
            node.fix_target = 'both_fix'
        elif has_caller_fix:
            node.fix_target = 'caller_fix'
        # else: None (default = callee_fix)
        # Fix: do not use faulty_children directly; add them dynamically via recursion
        # Ensures node.children only contains child skills that actually exist in self._nodes
        # The original faulty_children is still accessible via node.reflection_output.faulty_children
        node.children = set()

        self._nodes[skill_name] = node

        # Update the parent's children reference
        if parent_skill and parent_skill in self._nodes:
            self._nodes[parent_skill].children.add(skill_name)

        # Recursively handle problematic child skills
        for child_name, child_feedback in output.propagated_feedbacks.items():
            if child_name in output.faulty_children:
                # Build feedback content for the child skill
                child_feedback_content = child_feedback.to_gradient_context()

                self._build_recursive(
                    skill_name=child_name,
                    feedback_content=child_feedback_content,
                    feedback_type="backpropagated",
                    skill_info_getter=skill_info_getter,
                    propagated_feedback=child_feedback,
                    depth=depth + 1,
                    parent_skill=skill_name,
                    chat_log=chat_log,  # Pass chat log to children
                )

    def _topological_sort_reverse(self) -> List[str]:
        """
        Reverse topological sort (Bottom-Up order).

        Ensures child nodes are optimized before their parents.

        Fix: under multi-parent scenarios, sort by depth to guarantee a deterministic order.
        """
        if not self._nodes:
            return []

        # Compute in-degree (here it's the out-degree, since we are reversing)
        out_degree = {name: len(node.children) for name, node in self._nodes.items()}

        # Start from leaf nodes, sorted by descending depth (deeper nodes processed first)
        leaves = [name for name, degree in out_degree.items() if degree == 0]
        queue = deque(sorted(leaves, key=lambda n: -self._nodes[n].depth))
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # Update the parents' "out-degree", processing in descending depth order for determinism
            node = self._nodes[current]
            parents_sorted = sorted(
                node.parents,
                key=lambda p: -self._nodes[p].depth if p in self._nodes else 0
            )
            for parent in parents_sorted:
                if parent in out_degree:
                    out_degree[parent] -= 1
                    if out_degree[parent] == 0:
                        queue.append(parent)

        # Fallback check: ensure all analyzed nodes are in the result
        # Catches nodes that were not processed because their children list contained invalid references
        missing = set(self._nodes.keys()) - set(result)
        if missing:
            self._log(
                f"[Topological sort] Warning: the following nodes are missing from the sorted result (possibly due to invalid child references): {missing}",
                "warning"
            )
            # Append in descending depth order (so deeper nodes are optimized first)
            missing_sorted = sorted(missing, key=lambda n: self._nodes[n].depth, reverse=True)
            result.extend(missing_sorted)

        return result



# ============================================================
# Section 3: Chain Executor - executes the Bottom-Up optimization
# ============================================================

@dataclass
class ChainExecutionResult:
    """Chain execution result"""
    # Per-node optimization results
    node_results: Dict[str, OptimizationForwardFeedback]

    # Statistics
    total_optimized: int
    successful: int
    failed: int
    skipped: int

    # Final forward feedback (can be propagated to a higher layer)
    final_feedback: Optional[OptimizationForwardFeedback] = None

    # List of skills skipped because of zero gradients in Phase 1 (eliminates survivor bias)
    skipped_zero_gradient: List[str] = field(default_factory=list)


class ChainExecutor:
    """
    Chain executor.

    Executes optimization in Bottom-Up order and correctly propagates forward feedback.

    Usage:
        executor = ChainExecutor(
            consider_function=consider,
            optimize_fn=my_optimizer
        )

        result = executor.execute(chain_result)
    """

    def __init__(
        self,
        consider_function: ConsiderFunction,
        optimize_fn=None,
        logger=None,
        skip_checker=None,
        skill_graph_manager=None,
    ):
        """
        Initialize the ChainExecutor

        Args:
            consider_function: the Consider function
            optimize_fn: the actual optimization function
                signature: (skill_name, delta, context) -> (new_code, success)
            logger: logger
            skip_checker: P(update s) checker, gates optimization at Phase 2
            skill_graph_manager: SkillGraphManager for metadata fixes (inference feedback)
        """
        self.consider_function = consider_function
        self.optimize_fn = optimize_fn
        self.logger = logger
        self.skip_checker = skip_checker
        self.skill_graph_manager = skill_graph_manager

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(f"[{level.upper()}] {message}")

    def execute(
        self,
        chain_result: ReflectionChainResult,
        optimize_fn=None,
    ) -> ChainExecutionResult:
        """
        Execute Bottom-Up optimization.

        Iterates in chain_result.optimization_order:
        1. Optimize the current node
        2. Produce forward feedback
        3. Pass it to the parent for Consider

        Args:
            chain_result: result from ReflectionChain
            optimize_fn: optimization function (optional, overrides the default)

        Returns:
            ChainExecutionResult: execution result
        """
        optimize = optimize_fn or self.optimize_fn
        if not optimize:
            raise ValueError("optimize_fn is required")

        node_results: Dict[str, OptimizationForwardFeedback] = {}
        successful = 0
        failed = 0
        skipped = 0
        skipped_zero_gradient: List[str] = []
        # skip reason accounting
        skip_reasons = {"zero_gradient": 0, "insignificant": 0, "task_wrapper": 0, "p_update_s": 0}

        # Record the full optimization order to aid debugging
        self._log(
            f"Optimization order ({len(chain_result.optimization_order)} nodes): "
            f"{' -> '.join(chain_result.optimization_order)}",
            "info"
        )
        self._log(
            f"Analyzed nodes: {list(chain_result.nodes.keys())}",
            "info"
        )

        # Step 0: Extract parameter corrections from ALL Phase 1 results.
        # This is independent of code optimization — corrections apply even when
        # code optimization is skipped (P(update s), insignificant delta, etc.)
        n_corrections = self._extract_parameter_corrections(chain_result)
        if n_corrections:
            self._log(f"Extracted {n_corrections} parameter corrections from Phase 1", "info")

        for skill_name in chain_result.optimization_order:
            node = chain_result.nodes.get(skill_name)
            if not node:
                continue

            self._log(f"Optimizing skill: {skill_name}", "info")

            # Check whether optimization should run
            if not node.delta or not node.delta.is_significant:
                self._log(f"Skipping {skill_name}: gradient not significant", "info")
                node.status = NodeStatus.SKIPPED
                skipped += 1
                skip_reasons["insignificant"] += 1
                if not node.delta or node.delta.total_magnitude == 0:
                    skipped_zero_gradient.append(skill_name)
                    skip_reasons["zero_gradient"] += 1
                continue

            # Task-specific wrapper: code optimization is moot (regenerated
            # each iteration). Cross-task learning is carried instead by
            # _extract_parameter_corrections, which persists Phase 1's
            # parameter_corrections on SkillGraphManager and is consumed by
            # the planner's PureLLMResolver on the next attempt.
            if node.skill_info and node.skill_info.get("is_task_specific", False):
                # Audit log retains Phase 1 attribution for offline diagnosis
                if node.fix_target in ("caller_fix", "both_fix"):
                    self._log(
                        f"Skipping {skill_name}: task-specific wrapper with "
                        f"fix_target={node.fix_target} (Phase 1 attributed fault "
                        f"to wrapper; corrections persist via parameter_corrections)",
                        "info",
                    )
                else:
                    self._log(
                        f"Skipping {skill_name}: task-specific wrapper "
                        f"(Phase 1 backpropagation already complete)",
                        "info",
                    )
                node.status = NodeStatus.SKIPPED
                skipped += 1
                skip_reasons["task_wrapper"] += 1

                # Attribution deadlock fallback — when wrapper is
                # skipped, delegate its gradients to faulty_children that were
                # previously skipped with zero gradients. This breaks the
                # deadlock where wrapper blames child but can't be fixed itself.
                if (node.status == NodeStatus.SKIPPED
                        and node.reflection_output
                        and node.reflection_output.faulty_children
                        and node.delta
                        and node.delta.total_magnitude > 0):
                    # Bug B fix: do NOT delegate when Phase 1's attribution
                    # says the wrapper itself is the fault site. Delegating
                    # in those cases pushes the wrapper's gradient onto
                    # callees Phase 1 already declared healthy — which is
                    # exactly the loop that produced multiple wasted
                    # craftPickaxe versions in the May 19 diag run
                    # (log:16085→16086→16154, log:27260→27262→27601).
                    if node.fix_target in ("caller_fix", "both_fix"):
                        self._log(
                            f"[Deadlock Fallback] Skipping delegation for "
                            f"{skill_name}: fix_target={node.fix_target} "
                            f"(Phase 1 attributed fault to wrapper, not "
                            f"its children)",
                            "info",
                        )
                        continue
                    for child_name in node.reflection_output.faulty_children:
                        child_node = chain_result.nodes.get(child_name)
                        if not child_node or child_node.status != NodeStatus.SKIPPED:
                            continue
                        self._log(
                            f"[Deadlock Fallback] Delegating {skill_name}'s gradients "
                            f"(mag={node.delta.total_magnitude:.2f}) to child {child_name}",
                            "info",
                        )
                        # Override child's zero-gradient delta with wrapper's
                        child_node.delta = node.delta
                        child_node.status = NodeStatus.PENDING

                        # Run the same optimization path as non-wrapper skills
                        try:
                            child_feedbacks_for = [
                                node_results[c]
                                for c in child_node.children
                                if c in node_results
                            ]
                            consider_result = self.consider_function.consider(
                                delta=child_node.delta,
                                child_forward_feedbacks=child_feedbacks_for,
                            )
                            child_node.consider_result = consider_result
                            child_node.status = NodeStatus.OPTIMIZING

                            context = self._build_optimization_context(
                                child_node, consider_result
                            )
                            new_code, success = optimize(
                                child_name,
                                consider_result.adjusted_delta,
                                context,
                            )
                            if success:
                                child_node.status = NodeStatus.OPTIMIZED
                                successful += 1
                                skipped -= 1  # undo the wrapper's skip count
                                ff = self._generate_forward_feedback(
                                    child_name, child_node, consider_result,
                                    success, new_code,
                                )
                                child_node.forward_feedback = ff
                                node_results[child_name] = ff
                                self._log(
                                    f"[Deadlock Fallback] ✓ {child_name} optimized "
                                    f"with delegated gradients",
                                    "info",
                                )
                            else:
                                child_node.status = NodeStatus.FAILED
                                failed += 1
                                skipped -= 1
                                self._log(
                                    f"[Deadlock Fallback] ✗ {child_name} optimization "
                                    f"failed with delegated gradients",
                                    "warning",
                                )
                        except Exception as e:
                            child_node.status = NodeStatus.FAILED
                            failed += 1
                            skipped -= 1
                            self._log(
                                f"[Deadlock Fallback] ✗ {child_name} exception: {e}",
                                "error",
                            )

                continue

            # P(update s) — probabilistic optimization gate
            # Phase 1 analysis is complete; this gates whether to APPLY the optimization.
            # Rejected skills still produce forward feedback so parents see their analysis.
            if self.skip_checker:
                skip_result = self.skip_checker.should_skip(skill_name)
                if skip_result.should_skip:
                    self._log(
                        f"P(update s) rejected {skill_name}: {skip_result.reason} "
                        f"details={skip_result.details}",
                        "info"
                    )
                    node.status = NodeStatus.SKIPPED
                    skipped += 1
                    skip_reasons["p_update_s"] += 1
                    node_results[skill_name] = OptimizationForwardFeedback(
                        skill_name=skill_name,
                        optimization_successful=False,
                        analysis_available=True,
                        skipped_reason=skip_result.reason,
                    )
                    continue

            # Collect forward feedbacks from child nodes
            child_feedbacks = [
                node_results[child]
                for child in node.children
                if child in node_results
            ]

            # Run Consider
            consider_result = self.consider_function.consider(
                delta=node.delta,
                child_forward_feedbacks=child_feedbacks,
            )

            node.consider_result = consider_result
            node.status = NodeStatus.OPTIMIZING

            # Run the optimization
            try:
                # Build the optimization context
                context = self._build_optimization_context(node, consider_result)

                new_code, success = optimize(
                    skill_name,
                    consider_result.adjusted_delta,
                    context,
                )

                if success:
                    node.status = NodeStatus.OPTIMIZED
                    successful += 1

                    # Produce forward feedback (passing new_code for persistence)
                    ff = self._generate_forward_feedback(
                        skill_name, node, consider_result, success, new_code
                    )
                    node.forward_feedback = ff
                    node_results[skill_name] = ff

                    self._log(f"✓ {skill_name} optimization succeeded", "info")
                else:
                    node.status = NodeStatus.FAILED
                    failed += 1
                    self._log(f"✗ {skill_name} optimization failed", "warning")

            except Exception as e:
                node.status = NodeStatus.FAILED
                failed += 1
                self._log(f"✗ {skill_name} optimization exception: {e}", "error")

            node.optimization_timestamp = datetime.now().isoformat()

        # Phase 2 per-node status breakdown
        for skill_name in chain_result.optimization_order:
            node = chain_result.nodes.get(skill_name)
            if not node:
                continue
            mag = node.delta.total_magnitude if node.delta else 0.0
            self._log(f"[Phase2] {skill_name}: status={node.status.value} magnitude={mag:.2f}", "info")
        self._log(f"[Phase2] skip_breakdown: {skip_reasons}", "info")

        # Use the root node's feedback as the final result
        final_feedback = node_results.get(chain_result.root_skill)

        return ChainExecutionResult(
            node_results=node_results,
            total_optimized=len(chain_result.optimization_order),
            successful=successful,
            failed=failed,
            skipped=skipped,
            final_feedback=final_feedback,
            skipped_zero_gradient=skipped_zero_gradient,
        )

    def _extract_parameter_corrections(
        self, chain_result: ReflectionChainResult
    ) -> int:
        """Extract parameter corrections from ALL Phase 1 results.

        Runs unconditionally for ALL skills — corrections are independent of
        code optimization gating (P(update s), insignificant delta, etc.).
        Stores corrections on SkillGraphManager for PureLLM to consume on retry.

        Returns:
            Number of corrections stored.
        """
        if not self.skill_graph_manager:
            return 0

        count = 0
        for skill_name, node in chain_result.nodes.items():
            if not node.reflection_output:
                continue
            corrections = getattr(node.reflection_output, 'parameter_corrections', [])
            for correction in corrections:
                # Use the LLM's target_skill attribution when present (the typical
                # wrapper case attributes the correction to the callee whose param
                # was wrong, e.g. exploreUntilSkill). Fall back to the reflected
                # node name when target_skill is missing (older LLM responses, or
                # cases where the LLM was unable to attribute).
                attributed_skill = correction.get("target_skill") or skill_name
                self.skill_graph_manager.set_task_parameter_correction(
                    skill_name=attributed_skill,
                    correction=correction,
                )
                self._log(
                    f"Parameter correction: {attributed_skill}.{correction['param_name']} "
                    f"{correction['passed_value']!r} → {correction['suggested_value']!r} "
                    f"(conf={correction.get('confidence', 0):.1f})",
                    "info",
                )
                count += 1
        return count

    def _build_optimization_context(
        self,
        node: ChainNode,
        consider_result: ConsiderResult,
    ) -> str:
        """Build the optimization context"""
        lines = []

        lines.append("## Optimization Context")
        lines.append("")
        lines.append(f"Skill: {node.skill_name}")
        lines.append(f"Depth in chain: {node.depth}")

        if consider_result.adjusted_delta.gradients:
            lines.append("")
            lines.append("### Gradients to address:")
            for g in consider_result.adjusted_delta.gradients:
                lines.append(f"- [{g.gradient_type.value}] (mag={g.magnitude:.2f}) {g.direction}")

        if consider_result.dependency_updates:
            lines.append("")
            lines.append("### Required dependency updates:")
            for update in consider_result.dependency_updates:
                lines.append(f"- [{update.child_skill}] {update.description}")

        lines.append("")
        lines.append(f"### Impact: {consider_result.overall_impact}")
        lines.append(consider_result.action_summary)

        return "\n".join(lines)

    def _generate_forward_feedback(
        self,
        skill_name: str,
        node: ChainNode,
        consider_result: ConsiderResult,
        success: bool,
        new_code: str = "",
    ) -> OptimizationForwardFeedback:
        """Produce forward feedback"""
        # Capture old code for rollback
        old_code = node.skill_info.get("code", "") if node.skill_info else ""

        return OptimizationForwardFeedback(
            skill_name=skill_name,
            optimization_successful=success,
            changes_made=[
                g.direction for g in consider_result.adjusted_delta.gradients
            ],
            new_code=new_code,  # pass the new code for persistence
            old_code=old_code,  # pass the old code for rollback
            interface_changed=any(
                u.update_type == "call_signature"
                for u in consider_result.dependency_updates
            ),
            effects_changed=any(
                u.update_type == "effect_handling"
                for u in consider_result.dependency_updates
            ),
            parent_suggestions=[
                consider_result.action_summary
            ] if consider_result.needs_significant_changes else [],
            parent_warnings=[
                f"Dependency update required: {u.description}"
                for u in consider_result.dependency_updates
                if u.priority == 1
            ],
        )


# ============================================================
# Section 4: the complete two-phase optimization pipeline
# ============================================================

class TwoPhaseOptimizationPipeline:
    """
    The complete two-phase optimization pipeline.

    Integrates the full Top-Down analysis and Bottom-Up optimization flow.

    Usage:
        pipeline = TwoPhaseOptimizationPipeline(
            pure_reflection=reflection,
            consider_function=consider,
            optimize_fn=my_optimizer,
        )

        result = pipeline.run(
            root_skill_name="mySkill",
            root_feedback=feedback,
            skill_info_getter=get_skill_info
        )
    """

    def __init__(
        self,
        pure_reflection: PureReflection,
        consider_function: ConsiderFunction,
        optimize_fn=None,
        max_depth: int = 3,
        logger=None,
        skip_checker=None,
        skill_graph_manager=None,
    ):
        self.pure_reflection = pure_reflection
        self.consider_function = consider_function
        self.optimize_fn = optimize_fn
        self.max_depth = max_depth
        self.logger = logger
        self.skip_checker = skip_checker
        self.skill_graph_manager = skill_graph_manager

        # Internal components
        self.chain = ReflectionChain(
            pure_reflection=pure_reflection,
            max_depth=max_depth,
            logger=logger,
        )
        self.executor = ChainExecutor(
            consider_function=consider_function,
            optimize_fn=optimize_fn,
            logger=logger,
            skip_checker=skip_checker,
            skill_graph_manager=skill_graph_manager,
        )

    def run(
        self,
        root_skill_name: str,
        root_feedback_content: str,
        root_feedback_type: str = "error",
        skill_info_getter=None,
        chat_log: str = "",  # Chat log from onChat events
    ) -> ChainExecutionResult:
        """
        Run the complete two-phase optimization

        Args:
            root_skill_name: root skill name
            root_feedback_content: root skill's feedback
            root_feedback_type: feedback type
            skill_info_getter: callback that retrieves skill info
            chat_log: v7.7 Chat log containing diagnostic messages

        Returns:
            ChainExecutionResult: execution result
        """
        # Phase 1: Top-Down analysis
        self._log("=== Phase 1: Top-Down Analysis ===", "info")
        chain_result = self.chain.build(
            root_skill_name=root_skill_name,
            root_feedback_content=root_feedback_content,
            root_feedback_type=root_feedback_type,
            skill_info_getter=skill_info_getter,
            chat_log=chat_log,  # Pass chat log to reflection chain
        )
        self._last_chain_result = chain_result  # expose for fix_target extraction

        self._log(
            f"Analysis complete: {chain_result.total_nodes} nodes, "
            f"depth {chain_result.max_depth}, "
            f"total gradient {chain_result.total_gradient_magnitude:.2f}",
            "info"
        )

        # Phase 2: Bottom-Up optimization
        self._log("=== Phase 2: Bottom-Up Optimization ===", "info")
        execution_result = self.executor.execute(chain_result)

        self._log(
            f"Optimization complete: succeeded {execution_result.successful}, "
            f"failed {execution_result.failed}, "
            f"skipped {execution_result.skipped}",
            "info"
        )

        # Log Knowledge Retrieval stats
        try:
            from ..knowledge.llm_knowledge_retriever import get_kr_stats
            kr_stats = get_kr_stats()
            if any(v for v in kr_stats.values() if v):
                self._log(f"[KR Stats] {kr_stats}", "info")
        except Exception:
            pass

        return execution_result

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(f"[{level.upper()}] {message}")


# ============================================================
# Section 5: convenience functions
# ============================================================

def create_two_phase_pipeline(
    llm=None,
    optimize_fn=None,
    max_depth: int = 3,
    logger=None,
    pure_reasoning: bool = False,
    include_reasoning_examples: bool = False,
    use_factual_primitive_doc: bool = False,
    skip_checker=None,
    skill_graph_manager=None,
    **kwargs,
) -> TwoPhaseOptimizationPipeline:
    """
    Create a two-phase optimization pipeline

    Args:
        llm: LLM instance
        optimize_fn: optimization function
        max_depth: maximum recursion depth
        logger: logger
        skip_checker: P(update s) skip checker
        skill_graph_manager: SkillGraphManager instance

    Returns:
        TwoPhaseOptimizationPipeline: configured pipeline
    """
    from .pure_reflection import create_pure_reflection

    pure_reflection = create_pure_reflection(
        llm=llm,
        logger=logger,
        mode="llm",
        pure_reasoning=pure_reasoning,
        include_reasoning_examples=include_reasoning_examples,
        use_factual_primitive_doc=use_factual_primitive_doc,
    )
    consider_function = ConsiderFunction(logger=logger)

    return TwoPhaseOptimizationPipeline(
        pure_reflection=pure_reflection,
        consider_function=consider_function,
        optimize_fn=optimize_fn,
        max_depth=max_depth,
        logger=logger,
        skip_checker=skip_checker,
        skill_graph_manager=skill_graph_manager,
    )
