"""
FeedbackMixin - Feedback collection, parsing, and history management.

Extracted from optimizer_impl.py for better modularity.
Contains methods for collecting, organizing, and analyzing skill feedback.

Methods included:
- collect_feedback: Collect skill execution feedback
- get_subgraph_with_dfs: Extract subgraph using DFS
- _sync_feedback_from_gradients: Sync feedback from gradients
- get_feedback_for_skill: Get feedback for a single skill
- get_feedback_for_subgraph: Get feedback for subgraph
- _log_optimization_changes: Log optimization changes with diff
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from skillnet.agents.optimizer.feedback.types import SkillFeedback
from skillnet.agents.skill_graph.models import SkillGraph

# Delegate imports
from skillnet.agents.optimizer._impl.helpers import (
    save_feedback_history as _save_feedback_history_impl,
    load_feedback_history as _load_feedback_history_impl,
    mark_feedbacks_resolved as _mark_feedbacks_resolved_impl,
)

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import SkillExecutionTrace
    from ..optimizer_impl import SkillGraphOptimizer

logger = logging.getLogger(__name__)


class FeedbackMixin:
    """
    Feedback Management Mixin - Feedback collection, parsing, and history management.

    Requires self attributes:
    - self.skill_graph_manager: SkillGraphManager instance
    - self.feedback_history: List[SkillFeedback]
    - self.logger: Logger instance
    """

    def collect_feedback(
        self: "SkillGraphOptimizer",
        skill_name: str,
        execution_trace: Optional["SkillExecutionTrace"] = None,
        critique: Optional[str] = None,
        error_message: Optional[str] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        source: str = "execution",
        task: Optional[str] = None,
    ) -> List[SkillFeedback]:
        """
        Collect and structure skill feedback.

        Args:
            skill_name: Skill name
            execution_trace: Execution trace info
            critique: Critic Agent's critique
            error_message: Error message
            performance_metrics: Performance metrics (execution time, resource usage, etc.)
            source: Feedback source
            task: Current task (for context isolation)

        Returns:
            List[SkillFeedback]: Collected feedback list
        """
        feedbacks = []

        # Collect feedback from execution trace
        if execution_trace:
            trace_task = getattr(execution_trace, 'task', None)
            trace_trajectory_file = getattr(execution_trace, 'trajectory_file', None)
            trace_trajectory_segment_id = getattr(execution_trace, 'trajectory_segment_id', None)
            trace_critique = getattr(execution_trace, 'critique', None)

            if not execution_trace.success and execution_trace.error_message:
                feedbacks.append(SkillFeedback(
                    skill_name=skill_name,
                    feedback_type="error",
                    content=execution_trace.error_message,
                    source=source,
                    execution_id=execution_trace.execution_id,
                    trajectory_segment_id=trace_trajectory_segment_id,
                    trajectory_file=trace_trajectory_file,
                    task=trace_task,
                    timestamp=execution_trace.timestamp,
                    severity="high",
                ))

            # Extract errors from environment events
            if execution_trace.environment_events:
                for event in execution_trace.environment_events:
                    if isinstance(event, dict) and event.get("type") == "error":
                        feedbacks.append(SkillFeedback(
                            skill_name=skill_name,
                            feedback_type="error",
                            content=str(event.get("error", "")),
                            source="mineflayer",
                            execution_id=execution_trace.execution_id,
                            trajectory_segment_id=trace_trajectory_segment_id,
                            trajectory_file=trace_trajectory_file,
                            task=trace_task,
                            timestamp=execution_trace.timestamp,
                            severity="high",
                        ))

            # Collect critique from execution trace
            if trace_critique:
                feedbacks.append(SkillFeedback(
                    skill_name=skill_name,
                    feedback_type="critique",
                    content=trace_critique,
                    source="critic_agent",
                    execution_id=execution_trace.execution_id,
                    trajectory_segment_id=trace_trajectory_segment_id,
                    trajectory_file=trace_trajectory_file,
                    task=trace_task,
                    timestamp=execution_trace.timestamp,
                    severity="medium",
                ))

        # Collect feedback from critique parameter
        if critique and not (execution_trace and getattr(execution_trace, 'critique', None)):
            trace_task = getattr(execution_trace, 'task', None) if execution_trace else None
            trace_trajectory_file = getattr(execution_trace, 'trajectory_file', None) if execution_trace else None
            trace_trajectory_segment_id = getattr(execution_trace, 'trajectory_segment_id', None) if execution_trace else None

            feedbacks.append(SkillFeedback(
                skill_name=skill_name,
                feedback_type="critique",
                content=critique,
                source="critic_agent",
                execution_id=execution_trace.execution_id if execution_trace else None,
                trajectory_segment_id=trace_trajectory_segment_id,
                trajectory_file=trace_trajectory_file,
                task=trace_task,
                timestamp=datetime.now().isoformat(),
                severity="medium",
            ))

        # Collect feedback from error message
        if error_message:
            effective_task = task or (getattr(execution_trace, 'task', None) if execution_trace else None)
            feedbacks.append(SkillFeedback(
                skill_name=skill_name,
                feedback_type="error",
                content=error_message,
                source=source,
                timestamp=datetime.now().isoformat(),
                severity="high",
                task=effective_task,
            ))

        # Collect feedback from performance metrics
        if performance_metrics:
            if performance_metrics.get("execution_time", 0) > 10.0:
                feedbacks.append(SkillFeedback(
                    skill_name=skill_name,
                    feedback_type="performance",
                    content=f"Execution time is too long: {performance_metrics['execution_time']}s",
                    source="performance_monitor",
                    timestamp=datetime.now().isoformat(),
                    severity="medium",
                ))

        # Store feedback to skill's gradients
        if self.skill_graph_manager.has_node(skill_name):
            node = self.skill_graph_manager.get_node(skill_name)
            for feedback in feedbacks:
                if feedback.feedback_type == "error":
                    node.gradients.add_feedback(f"Error: {feedback.content}")
                elif feedback.feedback_type == "critique":
                    node.gradients.add_feedback(f"Critique: {feedback.content}")
                elif feedback.feedback_type == "performance":
                    node.gradients.add_feedback(f"Performance: {feedback.content}")

        # Save to history (deduplicated by skill_name + content + feedback_type)
        for feedback in feedbacks:
            exists = any(
                f.skill_name == feedback.skill_name and
                f.content == feedback.content and
                f.feedback_type == feedback.feedback_type
                for f in self.feedback_history
            )
            if not exists:
                self.feedback_history.append(feedback)
        self._save_feedback_history()

        return feedbacks

    def get_subgraph_with_dfs(
        self: "SkillGraphOptimizer",
        root_name: str,
        max_depth: int = None,
        include_parents: bool = False,
    ) -> SkillGraph:
        """
        Extract subgraph using DFS.

        Args:
            root_name: Root node name
            max_depth: Maximum depth (None for unlimited)
            include_parents: Whether to include parent nodes

        Returns:
            SkillGraph: Extracted subgraph
        """
        if not self.skill_graph_manager.has_node(root_name):
            return SkillGraph()

        manager = self.skill_graph_manager
        subgraph = SkillGraph()
        visited = set()

        def dfs(node_name: str, depth: int = 0):
            if node_name in visited:
                return
            if max_depth is not None and depth > max_depth:
                return

            visited.add(node_name)

            node = manager.get_node(node_name)
            if node:
                import copy
                subgraph.add_node(copy.deepcopy(node))

            for child_name in manager.get_children(node_name):
                if child_name not in visited:
                    dfs(child_name, depth + 1)

            if include_parents:
                for parent_name in manager.get_parents(node_name):
                    if parent_name not in visited:
                        dfs(parent_name, depth + 1)

        dfs(root_name)

        # Add edges within subgraph
        for node_name in subgraph.nodes:
            for child_name in manager.get_children(node_name):
                if child_name in subgraph.nodes:
                    subgraph.add_edge(node_name, child_name)

        return subgraph

    def _sync_feedback_from_gradients(self: "SkillGraphOptimizer", skill_name: str) -> None:
        """
        Sync feedback from skill's gradients to feedback_history.

        Args:
            skill_name: Skill name
        """
        if not self.skill_graph_manager.has_node(skill_name):
            return

        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return

        for gradient_item in node.gradients.feedback:
            content = gradient_item.content
            feedback_type = "error"

            if content.startswith("Error:"):
                feedback_type = "error"
                content = content[6:].strip()
            elif content.startswith("Critique:"):
                feedback_type = "critique"
                content = content[9:].strip()
            elif content.startswith("Performance:"):
                feedback_type = "performance"
                content = content[12:].strip()
            elif content.startswith("[CALLER FIX NEEDED]"):
                feedback_type = "caller_error"
            elif content.startswith("LLM Analysis:") or "**LLM in-depth analysis result**" in content:
                feedback_type = "llm_analysis"
            elif content.startswith("[SKIP OPTIMIZATION]"):
                feedback_type = "caller_fix_marker"

            exists = any(
                f.skill_name == skill_name and
                f.content == content and
                f.feedback_type == feedback_type
                for f in self.feedback_history
            )

            if not exists:
                feedback = SkillFeedback(
                    skill_name=skill_name,
                    feedback_type=feedback_type,
                    content=content,
                    source="gradients",
                    timestamp=gradient_item.timestamp,
                    severity="high" if feedback_type == "error" else "medium",
                )
                self.feedback_history.append(feedback)

    def get_feedback_for_skill(
        self: "SkillGraphOptimizer",
        skill_name: str,
        feedback_types: Optional[List[str]] = None,
        min_severity: str = "low",
        sync_from_gradients: bool = True,
        current_task: Optional[str] = None,
        include_resolved: bool = False,
        max_feedbacks: int = 5,
    ) -> List[SkillFeedback]:
        """
        Get feedback for a specific skill.

        Args:
            skill_name: Skill name
            feedback_types: Feedback type filter (None for no filter)
            min_severity: Minimum severity ("low", "medium", "high")
            sync_from_gradients: Whether to sync from gradients (default True)
            current_task: Current task identifier (for context isolation)
            include_resolved: Whether to include resolved feedback (default False)
            max_feedbacks: Maximum number of feedbacks to return (most recent).
                Prevents stale feedback accumulation from overwhelming LLM.

        Returns:
            List[SkillFeedback]: Feedback list
        """
        if sync_from_gradients:
            self._sync_feedback_from_gradients(skill_name)

        severity_levels = {"low": 0, "medium": 1, "high": 2}
        min_level = severity_levels.get(min_severity, 0)

        feedbacks = [
            f for f in self.feedback_history
            if f.skill_name == skill_name
            and severity_levels.get(f.severity, 0) >= min_level
            and (include_resolved or not getattr(f, 'resolved', False))
        ]

        if feedback_types:
            feedbacks = [f for f in feedbacks if f.feedback_type in feedback_types]

        if current_task:
            task_feedbacks = [
                f for f in feedbacks
                if f.task and current_task.lower() in f.task.lower()
            ]
            feedbacks = task_feedbacks if task_feedbacks else feedbacks

        # Window cap: only return the most recent feedbacks to prevent
        # stale feedback accumulation from confusing the LLM
        if max_feedbacks and len(feedbacks) > max_feedbacks:
            feedbacks = feedbacks[-max_feedbacks:]

        return feedbacks

    def get_feedback_for_subgraph(
        self: "SkillGraphOptimizer",
        root_name: str,
        max_depth: int = None,
        feedback_types: Optional[List[str]] = None,
        current_task: Optional[str] = None,
    ) -> Dict[str, List[SkillFeedback]]:
        """
        Get feedback for all skills in a subgraph.

        Args:
            root_name: Root node name
            max_depth: Maximum depth
            feedback_types: Feedback type filter
            current_task: Current task for context isolation

        Returns:
            Dict[str, List[SkillFeedback]]: Skill name to feedback list mapping
        """
        subgraph = self.get_subgraph_with_dfs(root_name, max_depth=max_depth)
        result = {}

        for skill_name in subgraph.nodes:
            result[skill_name] = self.get_feedback_for_skill(
                skill_name,
                feedback_types=feedback_types,
                current_task=current_task,
            )

        return result


    def _log_optimization_changes(
        self: "SkillGraphOptimizer",
        skill_name: str,
        old_code: str,
        new_code: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Log optimization changes with annotated diff.

        Args:
            skill_name: Skill name
            old_code: Code before optimization
            new_code: Code after optimization
            result: LLM optimization result
        """
        self.logger.info(f"\033[36m[Quick Optimize] ========== Code Changes ==========\033[0m")
        old_lines_count = len(old_code.split('\n'))
        new_lines_count = len(new_code.split('\n'))
        self.logger.info(f"\033[36m[Quick Optimize] Old lines: {old_lines_count}, New lines: {new_lines_count}\033[0m")

        if old_lines_count > 0:
            growth_ratio = new_lines_count / old_lines_count
            if growth_ratio > 1.5:
                self.logger.warning(
                    f"\033[33m[Code Growth Warning] '{skill_name}' code grew significantly: "
                    f"{old_lines_count} -> {new_lines_count} lines ({growth_ratio:.0%} growth). "
                    f"Consider reviewing for redundant code.\033[0m"
                )

        self.logger.info(f"\033[36m[Quick Optimize] Change summary: {result.get('change_summary', 'N/A')}\033[0m")
        if result.get("issues"):
            self.logger.info(f"\033[36m[Quick Optimize] Identified issues:\033[0m")
            for issue in result.get("issues", []):
                self.logger.info(f"\033[36m[Quick Optimize]   - {issue.get('type', 'unknown')}: {issue.get('description', 'N/A')}\033[0m")

        self.logger.info(f"\033[36m[Quick Optimize] === Annotated Diff ===\033[0m")
        annotated_diff = self._generate_annotated_diff(old_code, new_code, skill_name, collapse_unchanged=10)
        for diff_line in annotated_diff.split('\n'):
            if diff_line.startswith('[+]'):
                self.logger.info(f"\033[32m{diff_line}\033[0m")
            elif diff_line.startswith('[-]'):
                self.logger.info(f"\033[31m{diff_line}\033[0m")
            elif diff_line.startswith('[ ]'):
                self.logger.info(diff_line)
            elif '... //' in diff_line:
                self.logger.info(f"\033[33m{diff_line}\033[0m")
            else:
                self.logger.info(diff_line)

    # Thin wrappers delegating to helpers
    def _save_feedback_history(self: "SkillGraphOptimizer") -> None:
        """Delegate to helpers.feedback_persistence"""
        _save_feedback_history_impl(
            self.feedback_history,
            self.skill_graph_manager.ckpt_dir
        )

    def _load_feedback_history(self: "SkillGraphOptimizer") -> None:
        """Delegate to helpers.feedback_persistence"""
        self.feedback_history = _load_feedback_history_impl(
            self.skill_graph_manager.ckpt_dir
        )

    def _mark_feedbacks_resolved(
        self: "SkillGraphOptimizer",
        skill_name: str,
        source: str = None
    ) -> None:
        """Delegate to helpers.feedback_persistence"""
        _mark_feedbacks_resolved_impl(
            self.feedback_history,
            skill_name,
            source,
            log_callback=self.logger.info,
            save_callback=self._save_feedback_history
        )
