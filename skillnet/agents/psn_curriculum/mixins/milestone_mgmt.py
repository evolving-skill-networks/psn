"""
MilestoneManagementMixin - Milestone tracking, failure budgets, and alert handling.

Methods:
    _handle_critical_alerts(alerts, events) -> Optional[Tuple[str, str]]
    _reset_milestone_failures(milestone_tasks) -> None
    _log_milestone_validation(newly_completed, inventory) -> None

Self attributes used:
    goal_planner, resource_tracker, failed_tasks, _milestone_retry_budget,
    milestone_task_failure_threshold
Cross-mixin calls (MRO):
    _handle_critical_alerts -> Facade._count_task_failures
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from skillnet.agents.psn_curriculum.resource_tracker import ResourceAlert
from skillnet.agents.constants.task_semantics import TaskWithSemantic


class MilestoneManagementMixin:
    """Mixin for milestone tracking, failure budgets, and critical alert handling."""

    def _handle_critical_alerts(
        self,
        alerts: List[ResourceAlert],
        events: List
    ) -> Optional[Tuple[str, str]]:
        """
        Handle critical resource alerts by suggesting stockpile tasks.

        Returns:
            (task, reason) if a stockpile task should be done, None otherwise
        """
        if not alerts:
            return None

        # Get failure counts to avoid infinite loops on failed stockpile tasks
        failure_counts = self._count_task_failures()
        consecutive_failure_threshold = self.milestone_task_failure_threshold

        # Sort by priority
        alerts.sort(key=lambda a: -a.threshold.priority)

        # Try each alert in priority order, skip if task has failed too many times
        for alert in alerts:
            task = alert.suggested_action

            # Check if this task has failed too many times
            normalized_task = task.lower().strip()
            normalized_task = re.sub(r'\s+', ' ', normalized_task)
            failure_count = failure_counts.get(normalized_task, 0)

            if failure_count >= consecutive_failure_threshold:
                print(
                    f"\033[33m[PSN Stockpile] Skipping '{task}' - failed {failure_count} times "
                    f"(threshold: {consecutive_failure_threshold})\033[0m"
                )
                continue

            # Verify feasibility
            feasibility = self.goal_planner.check_feasibility(TaskWithSemantic.from_legacy_string(task))
            if feasibility.is_feasible:
                reason = f"Critical: {alert.item} is below minimum ({alert.current_count}/{alert.threshold.min_count})"
                return (task, reason)
            elif feasibility.suggested_prerequisites:
                # Check if prerequisite has also failed too many times
                prereq = feasibility.suggested_prerequisites[0]
                normalized_prereq = prereq.lower().strip()
                normalized_prereq = re.sub(r'\s+', ' ', normalized_prereq)
                prereq_failure_count = failure_counts.get(normalized_prereq, 0)

                if prereq_failure_count >= consecutive_failure_threshold:
                    print(
                        f"\033[33m[PSN Stockpile] Skipping prerequisite '{prereq}' - failed {prereq_failure_count} times\033[0m"
                    )
                    continue

                reason = f"Prerequisite for restocking {alert.item}"
                return (prereq, reason)

        # All alerts have been exhausted (all tasks failed too many times)
        print(f"\033[33m[PSN Stockpile] All stockpile tasks have failed too many times. Skipping stockpile phase.\033[0m")
        return None

    def _reset_milestone_failures(self, milestone_tasks: List[str]) -> None:
        """Clear failure records for milestone tasks, giving them a fresh retry window.

        Called after exploration cooldown to let the bot retry milestones that
        previously failed — the environment may have changed (new resources,
        new position) during exploration.
        """
        milestone_normalized = {
            re.sub(r'\s+', ' ', t.lower().strip()) for t in milestone_tasks
        }
        before = len(self.failed_tasks)
        self.failed_tasks = [
            t for t in self.failed_tasks
            if re.sub(r'\s+', ' ', t.lower().strip()) not in milestone_normalized
        ]
        cleared = before - len(self.failed_tasks)
        if cleared:
            print(
                f"\033[32m[PSN Recovery] Reset {cleared} milestone failure records. "
                f"Retry budget remaining: {self._milestone_retry_budget}\033[0m"
            )

    def _log_milestone_validation(
        self,
        newly_completed: List[str],
        inventory: Dict[str, int]
    ) -> None:
        """
        Log item breakdown for newly completed milestones (defensive validation).

        This helps diagnose Critic-vs-system inconsistencies by showing exactly
        which items satisfied each milestone requirement.
        """
        from skillnet.config.robustness_config import RobustnessConfig

        if not RobustnessConfig.ENABLE_MILESTONE_VALIDATION:
            return

        from skillnet.agents.psn_curriculum.goal_planner import _get_group_count

        for milestone_name in newly_completed:
            # Find the milestone object
            milestone = None
            for m in self.goal_planner.milestones:
                if m.name == milestone_name:
                    milestone = m
                    break

            if not milestone:
                continue

            # Log item breakdown for each required item/group
            breakdown_parts = []
            for item_or_group, required in milestone.required_items.items():
                actual = _get_group_count(inventory, item_or_group)
                status = "OK" if actual >= required else "INSUFFICIENT"
                breakdown_parts.append(
                    f"{item_or_group}: {actual}/{required} ({status})"
                )

            print(
                f"\033[32m[Milestone Validation] '{milestone_name}' verified: "
                f"{', '.join(breakdown_parts)}\033[0m"
            )
