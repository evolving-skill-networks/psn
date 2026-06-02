"""
Learning Dynamics Recorder

Records cumulative learning dynamics snapshots per iteration in JSONL format.
All metrics are cumulative snapshots; per-iteration deltas can be computed
in post-processing by diffing consecutive records.

Output: progress/learning_dynamics.jsonl
"""

import collections
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from skillnet.agents.optimizer.tracking.optimization_tracker import (
        OptimizationTracker,
    )


class LearningDynamicsRecorder:
    """
    Records cumulative learning dynamics snapshots at each iteration boundary.

    Design decisions:
    - All metrics are CUMULATIVE snapshots (not deltas). Post-processing
      computes deltas by diffing consecutive records.
    - Append-only JSONL format. On resume, duplicate iteration numbers may
      appear; consumers should take the last occurrence per iteration.
    - Read-only access to SkillGraphManager, OptimizationTracker, and
      RefactorHistoryTracker (never mutates their state).

    Sections recorded per snapshot:
    - topology: node/edge counts, depth, fanout, root/leaf counts, reuse_ratio
    - value_distribution: V(s) statistics (mean, std, quartiles, frozen count)
    - optimization: cumulative optimization totals and success rate
    - refactoring: cumulative refactoring totals by type
    - skill_invocations: cumulative execution counts per skill
    """

    DEFAULT_MATURITY_THRESHOLD = 0.8

    def __init__(
        self,
        ckpt_dir: str,
        graph_manager,
        optimization_tracker: Optional["OptimizationTracker"] = None,
        maturity_threshold: float = DEFAULT_MATURITY_THRESHOLD,
        graph_snapshot_interval: int = 0,
        optimization_threshold: Optional[float] = None,
        composite_weights: tuple = (0.4, 0.3, 0.2, 0.1),
    ):
        self.graph_manager = graph_manager
        self.optimization_tracker = optimization_tracker
        self.maturity_threshold = maturity_threshold
        self.graph_snapshot_interval = graph_snapshot_interval
        self.optimization_threshold = optimization_threshold
        self.composite_weights = composite_weights
        self._ckpt_dir = ckpt_dir

        self.progress_dir = Path(ckpt_dir) / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.dynamics_path = self.progress_dir / "learning_dynamics.jsonl"

        # J(N): Rolling window for R_task computation
        self._recent_successes: collections.deque = collections.deque(maxlen=20)
        self._warmup_success_deque()

    @staticmethod
    def _try_load_refactor_tracker(ckpt_dir: str):
        """Load RefactorHistoryTracker read-only from disk (if available)."""
        try:
            from skillnet.agents.refactor.history import RefactorHistoryTracker

            return RefactorHistoryTracker(ckpt_dir=ckpt_dir)
        except Exception:
            return None

    # ================================================================
    # Public API
    # ================================================================

    def record_snapshot(
        self,
        iteration: int,
        task: str = "",
        success: bool = False,
        skills_executed_this_iter: Optional[List[str]] = None,
        optimization_depth_stats: Optional[Dict[str, Any]] = None,
        reflect_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a cumulative learning dynamics snapshot.

        Call at the end of each iteration, after all other processing.

        Args:
            optimization_depth_stats: B3 credit assignment depth stats from
                the optimizer's last reflection chain (max_depth, total_nodes,
                depth_distribution).
            reflect_stats: 5C REFLECT invocation stats (total_optimized,
                successful, failed, skipped, skipped_zero_gradient).
        """
        record = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "success": success,
            "topology": self._compute_topology(),
            "value_distribution": self._compute_value_distribution(),
            "optimization": self._compute_optimization_snapshot(),
            "refactoring": self._compute_refactoring_snapshot(),
            "skill_invocations": self._compute_invocation_snapshot(
                skills_executed_this_iter or []
            ),
        }
        if optimization_depth_stats:
            record["credit_assignment"] = optimization_depth_stats
        if reflect_stats:
            record["reflect"] = reflect_stats
        # 7C: Periodic full graph structure snapshot
        interval = getattr(self, 'graph_snapshot_interval', 0)
        if (interval > 0 and iteration % interval == 0):
            record["graph_snapshot"] = self._compute_graph_snapshot()
        # J(N): Composite objective
        self._recent_successes.append(success)
        record["composite_objective"] = self._compute_composite_objective(
            topology=record["topology"],
            value_distribution=record["value_distribution"],
            refactoring=record["refactoring"],
        )
        self._append_jsonl(record)

    # ================================================================
    # Topology
    # ================================================================

    def _compute_topology(self) -> Dict[str, Any]:
        """Extract network topology metrics from the skill graph."""
        gm = self.graph_manager
        all_names = gm.get_all_skill_names()
        node_count = len(all_names)

        if node_count == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "max_depth": 0,
                "avg_fanout": 0.0,
                "leaf_count": 0,
                "root_count": 0,
                "reuse_ratio": 0.0,
            }

        edge_count = 0
        fan_outs = []
        fan_ins: Dict[str, int] = collections.Counter()
        name_set = set(all_names)

        for name in all_names:
            children = [c for c in gm.get_children(name) if c in name_set]
            edge_count += len(children)
            fan_outs.append(len(children))
            for child in children:
                fan_ins[child] += 1

        leaf_count = sum(1 for f in fan_outs if f == 0)
        root_count = sum(1 for name in all_names if fan_ins[name] == 0)
        reuse_ratio = sum(1 for v in fan_ins.values() if v > 1) / node_count

        result = {
            "node_count": node_count,
            "edge_count": edge_count,
            "max_depth": self._compute_max_depth(all_names, name_set),
            "avg_fanout": round(sum(fan_outs) / len(fan_outs), 3),
            "leaf_count": leaf_count,
            "root_count": root_count,
            "reuse_ratio": round(reuse_ratio, 3),
        }

        # B2: Skill cap info
        max_skills = getattr(self.graph_manager, 'max_skills', None)
        if max_skills is not None:
            result["max_skills"] = max_skills
            result["at_capacity"] = node_count >= max_skills

        return result

    def _compute_max_depth(self, all_names: List[str], name_set: set) -> int:
        """Compute max depth of DAG using topological sort + DP (longest path)."""
        gm = self.graph_manager
        if not all_names:
            return 0
        # In-degree only counts parents within name_set
        in_degree = {}
        for n in all_names:
            in_degree[n] = sum(1 for p in gm.get_parents(n) if p in name_set)

        queue = collections.deque(n for n in all_names if in_degree[n] == 0)
        if not queue:
            return 0

        depth = {n: 1 for n in all_names}
        while queue:
            node = queue.popleft()
            for child in gm.get_children(node):
                if child not in name_set:
                    continue
                depth[child] = max(depth[child], depth[node] + 1)
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return max(depth.values()) if depth else 0

    # ================================================================
    # 7C: Graph Structure Snapshot
    # ================================================================

    def _compute_graph_snapshot(self) -> Dict[str, Any]:
        """Full graph structure snapshot for cross-run topology analysis.

        Returns nodes list, edges dict, and per-node V(s) values.
        Only included at configurable intervals (graph_snapshot_interval).
        """
        gm = self.graph_manager
        all_names = gm.get_all_skill_names()
        name_set = set(all_names)

        edges: Dict[str, List[str]] = {}
        for name in all_names:
            children = [c for c in gm.get_children(name) if c in name_set]
            if children:
                edges[name] = children

        node_values: Dict[str, float] = {}
        for name, node in gm.iter_skills():
            node_values[name] = round(node.value_function, 4)

        return {
            "nodes": all_names,
            "edges": edges,
            "node_values": node_values,
        }

    # ================================================================
    # Value Distribution
    # ================================================================

    def _compute_value_distribution(self) -> Dict[str, Any]:
        """Extract V(s) distribution statistics across all skills.

        Added (Phase 4b): ``mean_clamped`` = mean(max(V(s), 0)) per paper
        §O.1 Eq. for R_reliab. Stored alongside the legacy ``mean`` so
        downstream analysis can use either.
        """
        values = []
        frozen_count = 0

        for _name, node in self.graph_manager.iter_skills():
            v = node.value_function
            values.append(v)
            if v >= self.maturity_threshold:
                frozen_count += 1

        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "mean_clamped": 0.0,
                "std": 0.0,
                "min": 0.0,
                "q25": 0.0,
                "median": 0.0,
                "q75": 0.0,
                "max": 0.0,
                "frozen_count": 0,
            }

        arr = np.array(values)
        # Paper R_reliab = mean(max(V(s), 0)) — per-skill clamp, then mean.
        # Differs from plain mean(V(s)) when some V's are negative (small-n
        # skills). Saved explicitly so the composite objective aligns with
        # Appendix O.1 without requiring per-skill values downstream.
        mean_clamped = float(np.mean(np.clip(arr, 0.0, None)))
        return {
            "count": len(values),
            "mean": round(float(np.mean(arr)), 4),
            "mean_clamped": round(mean_clamped, 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "q25": round(float(np.percentile(arr, 25)), 4),
            "median": round(float(np.median(arr)), 4),
            "q75": round(float(np.percentile(arr, 75)), 4),
            "max": round(float(np.max(arr)), 4),
            "frozen_count": frozen_count,
        }

    # ================================================================
    # Optimization Snapshot
    # ================================================================

    def _compute_optimization_snapshot(self) -> Dict[str, Any]:
        """Extract cumulative optimization statistics from OptimizationTracker."""
        if not self.optimization_tracker:
            return {
                "total_optimizations": 0,
                "total_successful": 0,
                "success_rate": 0.0,
                "skills_with_optimizations": 0,
            }

        history = self.optimization_tracker.history
        total = 0
        successful = 0
        for records in history.values():
            for r in records:
                total += 1
                if r.successful:
                    successful += 1

        # B5: V(s) correlation — mean V(s) for successful vs failed optimizations
        success_vs = []
        failure_vs = []
        for records in history.values():
            for r in records:
                v = getattr(r, 'value_at_optimization', None)
                if v is not None:
                    if r.successful:
                        success_vs.append(v)
                    else:
                        failure_vs.append(v)

        vs_correlation = {
            "count": len(success_vs) + len(failure_vs),
            "success_mean_vs": round(float(np.mean(success_vs)), 4) if success_vs else None,
            "failure_mean_vs": round(float(np.mean(failure_vs)), 4) if failure_vs else None,
        }

        result = {
            "total_optimizations": total,
            "total_successful": successful,
            "success_rate": round(successful / total, 4) if total > 0 else 0.0,
            "skills_with_optimizations": len(history),
            "vs_correlation": vs_correlation,
        }
        # B1: Record optimization threshold for cross-run comparison
        if getattr(self, 'optimization_threshold', None) is not None:
            result["optimization_threshold"] = self.optimization_threshold
        return result

    # ================================================================
    # Refactoring Snapshot
    # ================================================================

    def _compute_refactoring_snapshot(self) -> Dict[str, Any]:
        """Extract cumulative refactoring statistics from RefactorHistoryTracker."""
        tracker = self._try_load_refactor_tracker(self._ckpt_dir)
        if not tracker:
            return {
                "total_refactors": 0,
                "successful_refactors": 0,
                "rolled_back_refactors": 0,
                "by_type": {},
                "skills_affected": 0,
            }

        stats = tracker.get_statistics()
        return {
            "total_refactors": stats.total_refactors,
            "successful_refactors": stats.successful_refactors,
            "rolled_back_refactors": stats.rolled_back_refactors,
            "by_type": dict(stats.by_type),
            "skills_affected": stats.skills_affected,
        }

    # ================================================================
    # Invocation Snapshot
    # ================================================================

    def _compute_invocation_snapshot(
        self,
        skills_executed_this_iter: List[str],
    ) -> Dict[str, Any]:
        """
        Record cumulative execution counts per skill, plus this iteration's
        executed set.
        """
        cumulative = {}
        for name, node in self.graph_manager.iter_skills():
            total = node.statistics.total_executions
            if total > 0:
                cumulative[name] = {
                    "total": total,
                    "success": node.statistics.successful_executions,
                    "failed": node.statistics.failed_executions,
                }

        return {
            "this_iteration_skills": skills_executed_this_iter,
            "cumulative_executions": cumulative,
        }

    # ================================================================
    # J(N) Composite Objective
    # ================================================================

    def _warmup_success_deque(self):
        """Populate recent successes deque from existing JSONL on resume."""
        if not self.dynamics_path.exists():
            return
        try:
            with open(self.dynamics_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._recent_successes.append(rec.get("success", False))
        except Exception:
            pass

    def _compute_composite_objective(
        self,
        topology: Dict[str, Any],
        value_distribution: Dict[str, Any],
        refactoring: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute J(N) = w1*R_task + w2*R_reliab + w3*R_struct + w4*R_cons.

        Components:
        - R_task: rolling task success rate (mean of recent window)
        - R_reliab: mean V(s) across skill library
        - R_struct: reuse_ratio (compositional structure quality)
        - R_cons: refactoring success rate (consistency; validation rejection counts as failure)
        """
        w1, w2, w3, w4 = self.composite_weights

        # R_task: rolling success rate
        r_task = (
            sum(self._recent_successes) / len(self._recent_successes)
            if self._recent_successes else 0.0
        )

        # R_reliab: paper §O.1 formula = (1/|S|) * Σ_s max(V(s), 0).
        # Uses the per-skill-clamped mean we now record in value_distribution.
        # Falls back to legacy max(mean(V), 0) when the field is missing, so
        # old checkpoints read under an old recorder still function.
        r_reliab = value_distribution.get("mean_clamped")
        if r_reliab is None:
            r_reliab = max(value_distribution.get("mean", 0.0), 0.0)

        # R_struct: reuse ratio
        r_struct = topology.get("reuse_ratio", 0.0)

        # R_cons: refactoring success rate (validation rejection + rollback both count)
        total_ref = refactoring.get("total_refactors", 0)
        successful_ref = refactoring.get("successful_refactors", 0)
        r_cons = (successful_ref / total_ref) if total_ref > 0 else 1.0

        j = w1 * r_task + w2 * r_reliab + w3 * r_struct + w4 * r_cons

        return {
            "R_task": round(r_task, 4),
            "R_reliab": round(r_reliab, 4),
            "R_struct": round(r_struct, 4),
            "R_cons": round(r_cons, 4),
            "J": round(j, 4),
            "weights": list(self.composite_weights),
        }

    # ================================================================
    # I/O
    # ================================================================

    def _append_jsonl(self, data: Dict[str, Any]):
        """Append a JSON record to the JSONL file."""
        with open(self.dynamics_path, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
