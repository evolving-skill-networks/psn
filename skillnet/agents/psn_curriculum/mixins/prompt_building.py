"""
PromptBuildingMixin - LLM prompt construction, context rendering, and task decision.

Methods:
    _llm_decide_with_resource_context(...) -> Tuple[str, str]
    _build_psn_context(alerts, milestone_tasks, feasible_task) -> str
    _render_human_message_with_psn(observation, psn_context) -> HumanMessage
    _generate_learning_context(task, learning_path) -> str

Self attributes used:
    llm, psn_system_prompt, goal_planner, resource_tracker,
    curriculum_observations, warm_up, progress, adaptive_planner
Cross-mixin calls (MRO):
    _llm_decide -> TaskValidation._validate_task, _check_environment_requirements
    _build_psn_context -> Facade._count_task_failures
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from langchain.schema import HumanMessage, SystemMessage

from skillnet.agents.psn_curriculum.resource_tracker import ResourceAlert, AlertUrgency
from skillnet.agents.psn_curriculum.adaptive_planner import LearningPath
from skillnet.agents.constants.task_semantics import TaskWithSemantic
from skillnet.utils.stats_tracker import record_llm_usage


class PromptBuildingMixin:
    """Mixin for LLM prompt construction and context rendering."""

    def _llm_decide_with_resource_context(
        self,
        *,
        events: List,
        chest_observation: str,
        alerts: List[ResourceAlert],
        milestone_tasks: List[str],
        feasible_task: Optional[str],
        max_retries: int = 5,
        blacklisted_tasks: Optional[Set[str]] = None,
    ) -> Tuple[str, str]:
        """
        Use LLM to decide on next task with full resource context.

        Provides the LLM with:
        - Current observation (from parent class)
        - Resource alerts and thresholds
        - Milestone progress
        - Suggested feasible tasks
        """
        # Build enhanced human message
        observation = self.render_observation(events=events, chest_observation=chest_observation)

        # Extract current Y coordinate for environment awareness via the
        # active DomainKnowledge (handles Minecraft tuple-events +
        # dict-events transparently).
        current_y = None
        try:
            dk = getattr(self, "_domain_knowledge", None)
            if dk is not None:
                obs = dk.extract_observation(events)
                if obs.position and len(obs.position) >= 2:
                    current_y = obs.position[1]
        except (IndexError, TypeError, AttributeError):
            pass

        # Add PSN-specific context
        psn_context = self._build_psn_context(alerts, milestone_tasks, feasible_task, current_y=current_y, blacklisted_tasks=blacklisted_tasks)

        # Combine messages
        messages = [
            SystemMessage(content=self.psn_system_prompt),
            self._render_human_message_with_psn(observation, psn_context)
        ]

        # Call LLM
        response = None
        for attempt in range(max_retries):
            try:
                print(f"[PSN] LLM attempt {attempt + 1}/{max_retries} starting...")
                _llm_resp = self.llm.invoke(messages)
                record_llm_usage(_llm_resp, process_type="curriculum", function_name="curriculum.prompt_building._llm_decide_with_resource_context")
                response = _llm_resp.content
                parsed = self.parse_ai_message(response)

                if parsed and "next_task" in parsed:
                    task = parsed["next_task"]

                    # Normalize known plurals to singular
                    task = self._normalize_item_plurals(task)

                    # Hard blacklist check (tasks exceeded global limit).
                    # Phase 4f: only enforced during tech-tree milestone
                    # phase. Post-milestone relies on the LLM's own
                    # reasoning from the neutrally-rendered failed-tasks
                    # list (Voyager-style soft constraint) — no hard reject.
                    post_milestone = self.goal_planner.get_next_milestone() is None
                    if blacklisted_tasks and not post_milestone:
                        from skillnet._psn_impl.task_management import TaskManagementMixin
                        normalized_check = TaskManagementMixin._normalize_task_key(task)
                        if normalized_check in blacklisted_tasks:
                            print(f"[PSN] Task '{task}' rejected: blacklisted (exceeded global limit)")
                            continue

                    # Validate task format and content
                    is_valid, validation_error = self._validate_task(task)
                    if not is_valid:
                        print(f"[PSN] Task '{task}' rejected: {validation_error}")
                        continue

                    # Check if task requires unavailable environment conditions
                    env_issue = self._check_environment_requirements(task, chest_observation)
                    if env_issue:
                        print(f"[PSN] Task '{task}' rejected: {env_issue}")
                        # Skip this task and continue to next attempt or fallback
                        continue

                    # Verify feasibility - redirect strategy depends on task type
                    # Goal: Learn reusable sub-skills progressively while avoiding too many steps
                    #
                    # Strategy:
                    # - For TOOL crafting (pickaxe, axe, sword, shovel): Let Action Agent handle
                    # dependencies, as these tasks should call learned sub-skills
                    # - For INTERMEDIATE items (planks, sticks, crafting_table): Redirect to
                    # prerequisites so we learn these reusable skills
                    # - For mining tasks: Redirect if tool is required
                    feasibility = self.goal_planner.check_feasibility(TaskWithSemantic.from_legacy_string(task))
                    if not feasibility.is_feasible and feasibility.suggested_prerequisites:
                        task_lower = task.lower()

                        # Check if this is a tool crafting task (high-level, should use sub-skills)
                        _dk = getattr(self, '_domain_knowledge', None)
                        _tool_names = _dk.get_tool_type_names() if _dk else []
                        is_tool_craft = "craft" in task_lower and any(
                            tool in task_lower for tool in _tool_names
                        )

                        if is_tool_craft:
                            # Don't redirect - Action Agent will call learned sub-skills
                            print(f"[PSN] Tool craft task '{task}' - letting Action Agent handle dependencies")
                        else:
                            # Redirect to learn intermediate skills (planks, sticks, etc.)
                            task = feasibility.suggested_prerequisites[0]
                            print(f"[PSN] Redirecting to prerequisite: {task}")

                    context = self._get_task_context_with_planning(task, events)
                    return task, context

            except Exception as e:
                response_preview = response[:200] if response else "N/A"
                print(f"[PSN] LLM attempt {attempt + 1} failed: {e}")
                print(f"[PSN] Response preview: {response_preview}")

        # Fallback: return the feasible task we identified
        if feasible_task:
            print(f"[PSN] All LLM attempts failed, using fallback task: {feasible_task}")
            context = self._get_task_context_with_planning(feasible_task, events)
            return feasible_task, context

        # Ultimate fallback
        print(f"[PSN] WARNING: All LLM attempts failed and no fallback task available")
        return "Explore the area", ""

    def _build_psn_context(
        self,
        alerts: List[ResourceAlert],
        milestone_tasks: List[str],
        feasible_task: Optional[str],
        current_y: Optional[float] = None,
        blacklisted_tasks: Optional[Set[str]] = None,
    ) -> str:
        """Build PSN-specific context for LLM prompt"""
        lines = []

        # Check if all milestones are completed
        all_milestones_completed = self.goal_planner.get_next_milestone() is None

        # Resource display - differs based on milestone status
        if all_milestones_completed:
            # Post-milestone: show neutral inventory summary, not directive alerts
            # This prevents LLM from being biased toward stockpile tasks by
            # CRITICAL/Suggested language when all milestones are already done
            _dk = getattr(self, '_domain_knowledge', None)
            resource_items = _dk.get_display_resource_groups() if _dk else []
            resource_lines = []
            for item in resource_items:
                count = self.resource_tracker.get_group_count(item)
                if count > 0:
                    resource_lines.append(f"  {item}: {count}")
            if resource_lines:
                lines.append("\n=== Current Resources ===")
                lines.extend(resource_lines)
        elif alerts:
            # Pre-milestone: keep existing CRITICAL/LOW alert logic
            filtered_alerts = []
            for alert in alerts:
                # Always include CRITICAL alerts
                if alert.urgency == AlertUrgency.CRITICAL:
                    filtered_alerts.append(alert)
                else:
                    # Milestones not complete - filter out non-essential alerts
                    # Skip food alerts when we have active milestones (food is optional)
                    if alert.item == "cooked_food":
                        continue  # Don't distract from milestone progression
                    filtered_alerts.append(alert)

            if filtered_alerts:
                lines.append("\n=== Resource Alerts ===")
                for alert in filtered_alerts[:5]:  # Limit to top 5
                    urgency = "[CRITICAL]" if alert.urgency == AlertUrgency.CRITICAL else "[LOW]"
                    lines.append(f"{urgency} {alert.item}: {alert.current_count}/{alert.threshold.target_count}")
                    lines.append(f"  Suggested: {alert.suggested_action}")

        # Milestone progress
        next_milestone = self.goal_planner.get_next_milestone()
        if next_milestone:
            lines.append(f"\n=== Current Milestone ===")
            lines.append(f"Goal: {next_milestone.name} - {next_milestone.description}")
            missing = next_milestone.get_missing_items(self.resource_tracker.current_inventory)
            if missing:
                lines.append("Still needed:")
                for item, count in list(missing.items())[:5]:
                    lines.append(f"  - {item}: {count}")
            else:
                lines.append("All required items are available! Ready to complete this milestone.")
        else:
            # Post-milestone (Phase 4f): Voyager-like open exploration.
            # No hardcoded activity menu, no "DO NOT" prohibitions — just a
            # neutral cue reminding the LLM to use its state (inventory,
            # completed_tasks, failed_tasks, biome, etc.) to pick a novel
            # task. Mirrors Voyager curriculum.txt criterion 5
            # ("novel and interesting ... not doing same thing over"),
            # without imposing a vertical tech-tree bias.
            #
            # Body is loaded from the active DomainKnowledge so a dict-event domain can
            # supply an achievement-aware variant; Minecraft's file is the
            # original prose ported verbatim.
            exploration_template = ""
            _dk = getattr(self, '_domain_knowledge', None)
            if _dk is not None:
                exploration_template = _dk.get_prompt("exploration") or ""
            if not exploration_template:
                # Fallback for domains without an exploration prompt
                exploration_template = (
                    "Propose the next task for the bot to attempt. Respond "
                    "with JSON containing a single key 'task'."
                )
            lines.append("\n" + exploration_template)

        # Suggested tasks with clear emphasis on learning order
        if milestone_tasks:
            lines.append(f"\n=== Suggested Tasks (Learning Order) ===")
            lines.append("(Learn sub-skills progressively: planks → sticks → crafting_table → tools)")

            for task in milestone_tasks[:5]:
                task_lower = task.lower()
                is_mining = "mine" in task_lower or "collect" in task_lower

                if task == feasible_task:
                    if is_mining:
                        lines.append(f">>> [GATHER - IF NEEDED] {task}")
                    else:
                        lines.append(f">>> [FEASIBLE - RECOMMENDED] {task}")
                        lines.append("    (This task can be completed NOW)")
                else:
                    lines.append(f"    {task}")

        # Failure-aware prompting. Pre-milestone keeps the directive
        # "Tasks to AVOID" / "BLACKLISTED — do NOT" rhetoric because the
        # milestone path relies on hard-filtering to converge. Post-milestone
        # (Phase 4f) drops the directive language and renders failed tasks
        # neutrally — mirrors Voyager's "Failed tasks that are too hard: ..."
        # line — letting the LLM decide based on its own reasoning.
        task_failure_counts = self._count_task_failures()
        frequently_failing = [(task, count) for task, count in task_failure_counts.items() if count >= 2]
        if all_milestones_completed:
            if frequently_failing or blacklisted_tasks:
                failed_names = [t for t, _ in frequently_failing]
                if blacklisted_tasks:
                    failed_names.extend(sorted(blacklisted_tasks))
                # Dedup while preserving order
                seen = set()
                uniq = [t for t in failed_names if not (t in seen or seen.add(t))]
                lines.append(f"\nFailed tasks that are too hard: {', '.join(uniq[:15])}")
        else:
            if frequently_failing or blacklisted_tasks:
                lines.append(f"\n=== Tasks to AVOID (Repeatedly Failed) ===")
                lines.append("These tasks have failed multiple times. You may need to try other tasks first.")
                if frequently_failing:
                    frequently_failing.sort(key=lambda x: -x[1])
                    for task, count in frequently_failing[:5]:
                        lines.append(f"  ✗ {task} (failed {count}x)")
                if blacklisted_tasks:
                    lines.append("BLACKLISTED (exceeded global failure limit — do NOT propose these):")
                    for bt in sorted(blacklisted_tasks)[:10]:
                        lines.append(f"  ✗✗ {bt}")

        # Show learned skills so LLM can propose matching tasks
        learned_skills = self._get_learned_skill_names()
        if learned_skills:
            lines.append(f"\n=== Learned Skills (use these when possible) ===")
            lines.append("You have these reusable skills. Prefer tasks that match them:")
            for name in learned_skills[:15]:
                lines.append(f"  ✓ {name}")

        # Dynamic environment note
        if current_y is not None and current_y < 50:
            lines.append(f"\n=== Environment Note ===")
            lines.append(f"You are at Y={current_y:.0f} (underground). Surface features like trees and animals are not present here.")

        return "\n".join(lines)

    def _render_human_message_with_psn(
        self,
        observation: Dict[str, str],
        psn_context: str
    ) -> HumanMessage:
        """Render human message with PSN context added"""
        # Start with standard observation
        content = ""
        for key in self.curriculum_observations:
            if self.progress >= self.warm_up[key]:
                if key in observation:
                    content += observation[key]

        # Add PSN context
        content += f"\n{psn_context}\n"

        # Add instruction
        content += "\nBased on the above information, what should be the next task?"
        content += "\nConsider resource availability and milestone progress."
        content += "\nRespond with: Reasoning: <your reasoning>\nTask: <the task>"

        return HumanMessage(content=content)

    def _generate_learning_context(self, task: str, learning_path: LearningPath) -> str:
        """
        Generate context for a learning task.

        Args:
            task: Current learning task
            learning_path: The learning path being followed

        Returns:
            Context string for the task
        """
        progress_summary = self.adaptive_planner.get_progress_summary()
        completed = progress_summary.get("completed", 0)
        total = progress_summary.get("learning_tasks", 0)

        progress_info = ""
        if total > 0:
            progress_info = f"\nLearning Progress: {completed}/{total} sub-skills completed"

        context = f"""This is a learning task to build skills for: {learning_path.original_task}

Current learning task: {task}
{progress_info}

Focus on learning this specific skill correctly. After mastering it, you will use it
to accomplish more complex tasks.

IMPORTANT: This task is part of an adaptive learning sequence. Complete it successfully
to progress toward the final goal."""

        return context

    @staticmethod
    def _normalize_item_plurals(task: str) -> str:
        """Normalize known Minecraft plurals to singular in task strings."""
        try:
            from skillnet.agents.psn_curriculum.item_name_adapter import ItemNameAdapter
            result = task
            for plural, singular in ItemNameAdapter.PLURAL_TO_SINGULAR.items():
                # Case-insensitive word replacement
                result = re.sub(
                    r'\b' + re.escape(plural) + r'\b',
                    singular,
                    result,
                    flags=re.IGNORECASE
                )
            return result
        except ImportError:
            return task
