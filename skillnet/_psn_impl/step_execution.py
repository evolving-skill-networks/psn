"""
StepExecutionMixin for PSNAgent.

Orchestrates the step() pipeline by composing sub-mixins:
- StepPlanMixin: Phases 1-3 (planning, code extraction, recursive error handling)
- StepExecutePhaseMixin: Phases 4-5 (execution, diagnostics)
- StepOptimizeMixin: Phase 6 (two-phase optimization)

This module provides:
- step(): Main orchestrator
- _step_rebuild_messages(): Phase 7 (message reconstruction)
"""

import copy

from skillnet._psn_impl.event_helpers import (
    find_last_observe,
    get_event_data_dict,
    iter_events,
)
from skillnet._psn_impl.step_context import StepContext
from skillnet._psn_impl.step_plan import StepPlanMixin
from skillnet._psn_impl.step_execute_phase import StepExecutePhaseMixin
from skillnet._psn_impl.step_optimize import StepOptimizeMixin

import skillnet.utils as u


class StepExecutionMixin(StepPlanMixin, StepExecutePhaseMixin, StepOptimizeMixin):
    """Core step execution pipeline for PSNAgent.

    Composes sub-mixins for each pipeline phase and provides
    the orchestrator (step) plus remaining utility methods.
    """

    def _step_rebuild_messages(self, ctx: StepContext):
        """Phase 7: Reset placed blocks if failed + rebuild messages for next iteration."""
        if self.reset_placed_if_failed and not ctx.success:
            # revert all the placing event in the last step
            blocks = []
            positions = []
            for event_type, event_data in iter_events(ctx.events):
                if event_type == "onSave" and isinstance(event_data, dict) and event_data.get("onSave", "").endswith("_placed"):
                    block = event_data["onSave"].split("_placed")[0]
                    position = event_data.get("status", {}).get("position")
                    if position:
                        blocks.append(block)
                        positions.append(position)
            # Domain-injectable revert code, with Minecraft fallback
            domain = getattr(self, '_domain', None)
            revert_code = None
            if domain and hasattr(domain, 'knowledge'):
                revert_code = domain.knowledge.get_revert_failed_action_code(blocks, positions)
            if revert_code is None:
                revert_code = (
                    f"await givePlacedItemBack(bot, {u.json_dumps(blocks)}, {u.json_dumps(positions)})"
                )
            new_events = self.env.step(
                revert_code,
                programs=self.skill_manager.programs,
                is_iteration=False,
            )
            new_observe = find_last_observe(new_events)
            if new_observe is not None:
                last_ctx_data = get_event_data_dict(ctx.events[-1]) if ctx.events else None
                if last_ctx_data is not None:
                    last_ctx_data["inventory"] = new_observe.get("inventory", {})
                    last_ctx_data["voxels"] = new_observe.get("voxels", [])
        # On successful optimization the message was already built in the if optimized: block above; skip rebuilding
        if not ctx.optimized:
            # Retrieve skills with parameter metadata (always use ParameterizedActionAgent + SkillGraphManager)
            # Include task in query to improve skill retrieval for specific tasks
            # This helps match "Mine 3 birch log" with "mineLogs" skill
            retrieval_query = self.task if self.task else ""
            if self.context:
                retrieval_query += "\n\n" + self.context
            if ctx.events:
                chatlog_summary = self.action_agent.summarize_chatlog(ctx.events)
                if chatlog_summary:
                    retrieval_query += "\n\n" + chatlog_summary

            new_skills, skill_metadata = self.skill_manager.retrieve_skills(
                query=retrieval_query,
                return_metadata=True
            )

            # activate the reuse_skill_hint mechanism
            # Find a directly reusable skill and generate a call example
            reuse_skill_hint = self._find_reuse_skill_hint(self.task, skill_metadata)

            system_message = self.action_agent.render_system_message(
                skills=new_skills, skill_metadata=skill_metadata, reuse_skill_hint=reuse_skill_hint,
                task=self.task,
            )

            # Escalation — check if Level 1 regeneration should trigger
            escalation_triggered = self._check_and_apply_escalation(ctx)

            if escalation_triggered:
                # Before Level 1: consult Skill Evolution Manager for cross-skill patterns
                sem = getattr(self, '_evolution_manager', None)
                tracker = self._escalation_tracker
                level2_analysis = None

                if sem and tracker:
                    level2_analysis = sem.analyze_escalation(
                        current_task=self.task,
                        escalation_tracker=tracker,
                    )

                if level2_analysis and level2_analysis.pattern_detected:
                    # Level 2: Systemic pattern detected — store diagnosis for learn_step()
                    self._level2_diagnosis = level2_analysis
                    tracker.get_state(self.task).level = 2
                    print(
                        f"\033[35m[Escalation] Level 2 triggered for task '{self.task}': "
                        f"{level2_analysis.common_cause}\033[0m"
                    )
                    # Still inject failure summary so the next attempt benefits from history
                    failure_summary = tracker.get_failure_summary(self.task)
                    if level2_analysis.common_cause:
                        failure_summary += (
                            f"\n\nSYSTEMIC ISSUE DETECTED: {level2_analysis.common_cause}"
                        )
                else:
                    # Level 1: no systemic pattern — regenerate with failure history
                    self._level2_diagnosis = None
                    failure_summary = tracker.get_failure_summary(self.task)

                critique_with_history = (
                    (ctx.critique + "\n\n" if ctx.critique else "") + failure_summary
                )
                human_message = self.action_agent.render_human_message(
                    events=ctx.events,
                    code="",  # No old code — fresh start
                    task=self.task,
                    context=self.context,
                    critique=critique_with_history,
                )
                tracker.mark_escalated(self.task)
                level_str = "Level 2 (systemic)" if (level2_analysis and level2_analysis.pattern_detected) else "Level 1"
                print(
                    f"\033[35m[Escalation] {level_str} for task '{self.task}': "
                    f"regenerating with failure history\033[0m"
                )
            else:
                human_message = self.action_agent.render_human_message(
                    events=ctx.events,
                    code=ctx.parsed_result["program_code"],
                    task=self.task,
                    context=self.context,
                    critique=ctx.critique,
                )
            self.messages = [system_message, human_message]
        self.last_events = copy.deepcopy(ctx.events)

    def step(self):
        if self.action_agent_rollout_num_iter < 0:
            raise ValueError("Agent must be reset before stepping")

        # Reset step-scoped refactor-modification tracking. Populated by
        # SkillGraphManager.update_skill_code when the source starts with
        # "refactor"; consulted later to skip redundant add_new_skill calls
        # and to block action_agent from overwriting a refactor-modified skill.
        try:
            self.skill_manager.reset_step_refactor_tracking()
        except AttributeError:
            pass

        ctx = StepContext()

        # Phase 1: Planning (graph planner + LLM fallback)
        self._step_plan(ctx)

        # Phase 2: Process AI message (if no graph planner result)
        if ctx.parsed_result is None and ctx.ai_message is not None:
            self._step_process_code(ctx)

        ctx.success = False

        # Phase 3: Handle recursive call errors
        if isinstance(ctx.parsed_result, str) and "Recursive call detected" in ctx.parsed_result:
            self._step_handle_recursive_error(ctx)

        # Phase 4-7: Execute, evaluate, optimize (dict result path)
        if isinstance(ctx.parsed_result, dict):
            self._step_execute(ctx)
            self._step_diagnose_and_record(ctx)

            ctx.optimized = False
            if not ctx.success and self.optimizer:
                self._step_optimize(ctx)

            # Record step result for escalation tracking
            if not ctx.success:
                self._record_escalation_step(ctx)

            self._step_rebuild_messages(ctx)
        else:
            # String error path
            assert isinstance(ctx.parsed_result, str)
            self.recorder.record([], self.task)
            self.trajectory_recorder.record_step(
                action=ctx.parsed_result,
                events=[],
                conversation=self.conversations[-1] if self.conversations else None,
                ai_message=ctx.ai_message.content if ctx.ai_message else None,
                success=False,
            )
            print(f"\033[34m{ctx.parsed_result} Trying again!\033[0m")

        assert len(self.messages) == 2

        # Rollout counter
        self.action_agent_rollout_num_iter += 1

        done = (
            self.action_agent_rollout_num_iter >= self.action_agent_task_max_retries
            or ctx.success
        )

        info = {
            "task": self.task,
            "context": self.context,
            "success": ctx.success,
            "conversations": self.conversations,
            "from_graph_planner": ctx.planning_result is not None and ctx.planning_result.success if ctx.planning_result else False,
            "planner_mode": self.planner_mode,
        }

        if ctx.success:
            # Reset escalation on success
            tracker = getattr(self, '_escalation_tracker', None)
            if tracker:
                tracker.on_task_success(self.task)
            assert (
                "program_code" in ctx.parsed_result and "program_name" in ctx.parsed_result
            ), "program and program_name must be returned when success"
            info["program_code"] = ctx.parsed_result["program_code"]
            info["program_name"] = ctx.parsed_result["program_name"]
            if "should_save_skill" in ctx.parsed_result:
                info["should_save_skill"] = ctx.parsed_result["should_save_skill"]
        else:
            print(
                f"\033[32m****Action Agent human message****\n{self.messages[-1].content}\033[0m"
            )

        if done:
            self.trajectory_recorder.save_trajectory(
                success=ctx.success,
                final_events=self.last_events if hasattr(self, 'last_events') else None,
            )

        return self.messages, 0, done, info

    def _record_escalation_step(self, ctx: StepContext):
        """Record a failed step result for escalation tracking."""
        tracker = getattr(self, '_escalation_tracker', None)
        if tracker is None:
            return

        # Get primary skill's V(s) if available
        v_s = 0.0
        if isinstance(ctx.parsed_result, dict):
            primary_skill = ctx.parsed_result.get("program_name", "")
            if primary_skill and hasattr(self, 'skill_manager') and self.skill_manager.has_node(primary_skill):
                v_s = self.skill_manager.get_node(primary_skill).value_function

        # Collect error from events
        error = ""
        if ctx.events:
            for event_type, event_data in iter_events(ctx.events):
                if event_type == "onError":
                    err_text = event_data.get("onError", "") if isinstance(event_data, dict) else str(event_data)
                    if err_text:
                        error = err_text

        code = ""
        if isinstance(ctx.parsed_result, dict):
            code = ctx.parsed_result.get("program_code", "")

        tracker.record_step_result(
            task=self.task,
            success=False,
            v_s=v_s,
            error=error,
            code=code,
        )

    def _check_and_apply_escalation(self, ctx: StepContext) -> bool:
        """Check if escalation should trigger and return True if Level 1."""
        tracker = getattr(self, '_escalation_tracker', None)
        if tracker is None:
            return False

        if ctx.success:
            return False

        return tracker.should_escalate(self.task)

