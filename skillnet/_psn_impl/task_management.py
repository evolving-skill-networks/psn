"""
TaskManagementMixin for PSNAgent

Methods for task normalization, failure limits, capability change detection,
and iteration recording.
"""

import re
from typing import TYPE_CHECKING

from skillnet._psn_impl.event_helpers import find_last_observe

if TYPE_CHECKING:
    from ..psn import PSNAgent


class TaskManagementMixin:
    """Task failure tracking, limits, and progress recording."""

    @staticmethod
    def _normalize_task_key(task: str) -> str:
        """Normalize task string for counting (case-insensitive, remove extra spaces, plural→singular).

        This ensures "Mine 3 cobblestone" and "mine 3 Cobblestone" are treated as the same task,
        and that "16 torches" and "16 torch" are also treated as the same task,
        avoiding bypass of failure-count guards due to differing case or pluralization.

        This is a static method so it can be called from other modules
        (e.g., PSNCurriculumAgent._count_task_failures) for consistent normalization.
        """
        normalized = task.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        # Normalize known Minecraft plurals to singular for consistent matching
        try:
            from skillnet.agents.psn_curriculum.item_name_adapter import ItemNameAdapter
            for plural, singular in ItemNameAdapter.PLURAL_TO_SINGULAR.items():
                # Replace plural with singular at word boundaries:
                # 1. Preceded by space
                normalized = normalized.replace(f' {plural}', f' {singular}')
                # 2. At start of string
                if normalized.startswith(plural + ' ') or normalized == plural:
                    normalized = singular + normalized[len(plural):]
                # 3. At end of string
                if normalized.endswith(plural):
                    normalized = normalized[:-len(plural)] + singular
        except ImportError:
            pass
        return normalized

    @staticmethod
    def _is_api_error(exc: Exception, _depth: int = 0) -> bool:
        """Check if an exception is an API/infrastructure error (not a skill logic error).

        API errors should NOT count toward task failure limits since they're
        transient infrastructure issues, not skill capability problems.
        """
        if _depth > 10:
            return False

        error_str = str(exc).lower()
        error_type = type(exc).__name__

        # Check for known API error types
        api_error_types = {
            'RateLimitError', 'APIError', 'APIConnectionError',
            'APITimeoutError', 'AuthenticationError', 'InternalServerError',
        }
        if error_type in api_error_types:
            return True

        # Check for HTTP status codes in error message
        if '429' in error_str or 'rate_limit' in error_str or 'rate limit' in error_str:
            return True
        if 'insufficient_quota' in error_str or 'api_quota' in error_str or 'billing' in error_str:
            return True
        if 'connection' in error_str and ('refused' in error_str or 'timeout' in error_str):
            return True

        # Check exception chain (with depth limit to prevent infinite recursion)
        cause = exc.__cause__ or exc.__context__
        if cause and cause is not exc:
            return TaskManagementMixin._is_api_error(cause, _depth + 1)

        return False

    def _get_task_failure_limit(self, task: str) -> int:
        """Return a dynamic failure threshold based on task complexity.

        Uses domain-provided complexity tiers when available,
        otherwise returns the base limit.

        Returns:
            int: maximum attempts for this task
        """
        if not task:
            return self.global_task_failure_limit
        task_lower = task.lower()
        base = self.global_task_failure_limit

        domain = getattr(self, '_domain', None)
        if domain and hasattr(domain, 'knowledge'):
            tiers = domain.knowledge.get_task_complexity_tiers()
            if tiers:
                for keyword in tiers.get('high', []):
                    if keyword in task_lower:
                        return base * 2
                for keyword in tiers.get('medium', []):
                    if keyword in task_lower:
                        return int(base * 1.5)

        return base

    def _check_and_reset_on_capability_change(self, old_inventory: dict, new_inventory: dict):
        """Detect capability changes (newly acquired tools) and reset failure counts for related tasks.

        Uses domain-provided tool unlock mapping when available.
        Without domain knowledge, no capability-based resets occur.
        """
        domain = getattr(self, '_domain', None)
        tool_unlocks = None
        if domain and hasattr(domain, 'knowledge'):
            tool_unlocks = domain.knowledge.get_tool_unlock_mapping()

        if not tool_unlocks:
            return  # No domain knowledge → no capability-based resets

        # Check for newly acquired tools
        new_tools = []
        for tool in tool_unlocks.keys():
            old_count = old_inventory.get(tool, 0)
            new_count = new_inventory.get(tool, 0)
            if new_count > old_count:
                new_tools.append(tool)

        if not new_tools:
            return

        # Collect task keywords that should be reset
        keywords_to_reset = set()
        for tool in new_tools:
            keywords_to_reset.update(tool_unlocks.get(tool, []))

        # Reset counts for matching tasks
        tasks_to_reset = []
        for normalized_task in list(self.global_task_attempt_counts.keys()):
            for keyword in keywords_to_reset:
                if keyword in normalized_task:
                    tasks_to_reset.append(normalized_task)
                    break

        for task_key in tasks_to_reset:
            del self.global_task_attempt_counts[task_key]

        if tasks_to_reset:
            print(
                f"\033[32m[Capability Change] New tools: {new_tools}. "
                f"Reset {len(tasks_to_reset)} task counters: {tasks_to_reset[:3]}{'...' if len(tasks_to_reset) > 3 else ''}\033[0m"
            )

    def _record_iteration_end(self, info: dict):
        """Record iteration end for progress viewer."""
        try:
            # Extract final inventory and position from last_events
            final_inventory = {}
            position = {}

            observe_data = find_last_observe(self.last_events)
            if observe_data is not None:
                final_inventory = observe_data.get("inventory", {})
                status = observe_data.get("status", {})
                pos = status.get("position", {})
                if pos:
                    position = {
                        "x": pos.get("x", 0),
                        "y": pos.get("y", 0),
                        "z": pos.get("z", 0),
                    }

            # Get skill counts (always use SkillGraphManager)
            skill_count = self.skill_manager.skill_count
            active_skill_count = 0
            # Count non-deprecated skills
            for node_name, skill_node in self.skill_manager.iter_skills(include_task_specific=True):
                if skill_node and not getattr(skill_node, 'is_deprecated', False):
                    if not getattr(skill_node, 'is_covered', False):
                        active_skill_count += 1
                    else:
                        active_skill_count += 1  # Covered skills are still active

            # Record the iteration end
            self.progress_recorder.record_iteration_end(
                success=info.get("success", False),
                final_inventory=final_inventory,
                position=position,
                skill_count=skill_count,
                active_skill_count=active_skill_count,
                error_message=info.get("error_message"),
            )

            # Record skill events for skills executed in this iteration
            if hasattr(self, 'last_skill_execution_results'):
                for skill_name, result in self.last_skill_execution_results.items():
                    self.progress_recorder.record_skill_executed(
                        skill_name=skill_name,
                        success=result.get("success", False),
                        call_depth=1,  # Top-level call
                    )

        except Exception as e:
            # Don't fail the main loop if progress recording fails
            print(f"\033[33m[Progress Recorder] Warning: Failed to record iteration: {e}\033[0m")

        # Record learning dynamics snapshot (independent try/except)
        try:
            if hasattr(self, 'dynamics_recorder') and self.dynamics_recorder:
                # B3: Extract credit assignment depth stats from optimizer
                # 5C: Extract REFLECT invocation stats from optimizer
                depth_stats = None
                reflect_stats = None
                try:
                    engine = getattr(getattr(self, 'optimizer', None), '_two_phase_engine', None)
                    if engine:
                        depth_stats = engine.get_last_depth_stats()
                        reflect_stats = engine.get_last_execution_stats()
                except Exception:
                    pass

                self.dynamics_recorder.record_snapshot(
                    iteration=self.progress_recorder.current_iteration,
                    task=self.progress_recorder.current_task,
                    success=info.get("success", False),
                    skills_executed_this_iter=list(
                        self.progress_recorder.skills_executed_this_iteration
                    ),
                    optimization_depth_stats=depth_stats,
                    reflect_stats=reflect_stats,
                )
        except Exception as e:
            print(f"\033[33m[Learning Dynamics] Warning: {e}\033[0m")
