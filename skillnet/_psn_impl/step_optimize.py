"""
StepOptimizeMixin: Phase 6 of step() pipeline.

Two-phase skill optimization on task failure.
"""

import traceback

from skillnet._psn_impl.event_helpers import (
    find_last_observe,
    get_event_data_dict,
    iter_events,
)
from skillnet._psn_impl.step_context import StepContext


class StepOptimizeMixin:
    """Phase 6: Two-phase skill optimization on failure."""

    def _step_optimize(self, ctx: StepContext):
        """Phase 6: Two-phase skill optimization on failure."""
        # Extract called skills from code
        full_code = ctx.parsed_result.get("program_code", "") + "\n" + ctx.parsed_result.get("exec_code", "")
        called_skills = self.skill_manager.extract_called_skills(full_code)

        # Extract main function name (newly generated code s_0)
        main_function_name = ctx.parsed_result.get("program_name")
        main_function_code = ctx.parsed_result.get("program_code", "")
        should_save_skill = ctx.parsed_result.get("should_save_skill", True)  # check whether to save

        # ========== Unified failed-skill handling flow ==========
        # Design principles:
        # 1. Always start optimization from the new skill (unless it is a simple wrapper)
        # 2. Rely on backpropagation to automatically locate the problem (new skill / sub-skill / both)
        # 3. Decide the fix target based on root-cause analysis
        skills_to_optimize = []

        if main_function_name and main_function_code:
            # Extract the error message
            error_msg = None
            if ctx.events and len(ctx.events) > 0:
                last_event_data = get_event_data_dict(ctx.events[-1])
                if last_event_data is not None:
                    error_msg = last_event_data.get("error")

            # [Fix 5] remove simple-wrapper skip logic
            # Always save the new skill as an experimental skill and let the lifecycle handle it
            # Reason: simple wrappers need accurate call-relationship tracking; skipping would leave the optimization scope incomplete

            # Save the new skill and start optimization from it
            # Let the backpropagation mechanism analyze the problem automatically
            if not self.skill_manager.has_node(main_function_name):
                print(f"\033[36m[Failed Skill] Saving new skill '{main_function_name}' and starting optimization from it\033[0m")
                print(f"\033[36m[Failed Skill] Backpropagation will automatically locate the issue (new skill / sub-skill / both)\033[0m")

                # Check whether a functionally-equivalent skill already exists (only for log hints)
                existing_skill = self.skill_manager.find_skill_by_task_intent(
                    task=self.task,
                    context=self.context
                )
                if existing_skill and existing_skill != main_function_name:
                    print(f"\033[33m[Failed Skill] Note: a functionally-equivalent skill '{existing_skill}' already exists; still saving the new skill for independent analysis\033[0m")

                try:
                    # Build the info dict; call add_new_skill for full metadata extraction
                    failed_skill_info = {
                        "program_name": main_function_name,
                        "program_code": main_function_code,
                        "task": self.task,
                        "context": self.context,
                        "success": False,  # mark as failure
                        "is_experimental": True,  # mark as experimental
                        "first_failure_reason": error_msg,
                        "should_save_skill": True,  # ensure it will be saved
                    }

                    # Use add_new_skill for full metadata extraction (effects, preconditions, parameters)
                    self.skill_manager.add_new_skill(failed_skill_info)

                    # add_new_skill may rename the skill; fetch the final name
                    final_skill_name = failed_skill_info.get("program_name", main_function_name)

                    # Sync _task_executed_skills so on_task_completed uses the correct name
                    if final_skill_name != main_function_name:
                        if hasattr(self, '_task_executed_skills') and main_function_name in self._task_executed_skills:
                            self._task_executed_skills.discard(main_function_name)
                            self._task_executed_skills.add(final_skill_name)
                            print(f"\033[36m[Skill Rename Sync] Syncing _task_executed_skills after add_new_skill rename: '{main_function_name}' -> '{final_skill_name}'\033[0m")

                    # Get the newly added node and mark the failure (using the final name)
                    if self.skill_manager.has_node(final_skill_name):
                        node = self.skill_manager.get_node(final_skill_name)
                        deprecation_result = node.mark_execution_failed(
                            task=self.task,
                            error=error_msg
                        )
                        if deprecation_result["deprecation_triggered"]:
                            print(f"\033[33m[Progressive Deprecation] Skill '{main_function_name}' reached the deprecation threshold\033[0m")
                        else:
                            print(f"\033[36m[Progressive Deprecation] Skill '{main_function_name}' failure recorded (consecutive failures: {deprecation_result['consecutive_failures']}/{node.deprecation_threshold})\033[0m")

                        self.skill_manager.save()
                        print(f"\033[36m[Failed Skill] Added {main_function_name} to graph (experimental skill, full metadata)\033[0m")

                        # Record the established call relationships
                        if node.children:
                            print(f"\033[36m[Failed Skill] {final_skill_name} calls the following skills: {', '.join(node.children)}\033[0m")

                        # Add the main function to the optimization candidate list (using the final name)
                        skills_to_optimize.insert(0, final_skill_name)

                except Exception as e:
                    print(f"\033[33m[Failed Skill] Warning: failed to add main function to graph: {e}\033[0m")
                    print(f"\033[33m[Failed Skill] Traceback: {traceback.format_exc()}\033[0m")
            else:
                # Main function already in the graph; just update failure records and start optimization from it
                print(f"\033[36m[Failed Skill] {main_function_name} already in graph; updating failure records and optimizing\033[0m")

                # [Fix] ensure the skill is in vectordb
                # pre_register_skill only adds to graph, not to vectordb
                # When the skill fails, since it is already in the graph, record_execution skips the vectordb add
                # We add it here so the skill can be retrieved via retrieve_skills
                self.skill_manager.ensure_skill_in_vectordb(main_function_name)

                node = self.skill_manager.get_node(main_function_name)
                if node:
                    # Clear pending code (if any); on failure, do not adopt new code
                    if node.has_pending_code():
                        result = node.confirm_pending(success=False)
                        print(f"\033[33m[PSN] Execution failed; discarding pending code: {result.message}\033[0m")

                    deprecation_result = node.mark_execution_failed(
                        task=self.task,
                        error=error_msg
                    )
                    if deprecation_result["deprecation_triggered"]:
                        print(f"\033[33m[Progressive Deprecation] Skill '{main_function_name}' reached the deprecation threshold\033[0m")
                    else:
                        print(f"\033[36m[Progressive Deprecation] Skill '{main_function_name}' failure recorded (consecutive failures: {deprecation_result['consecutive_failures']}/{node.deprecation_threshold})\033[0m")
                    self.skill_manager.save()
                skills_to_optimize.insert(0, main_function_name)

        # [Fix 3] re-extract called_skills after skill is added
        # So newly added skills are included
        called_skills = self.skill_manager.extract_called_skills(ctx.code)

        # Add called skills (sub-skills)
        skills_to_optimize.extend(called_skills)

        # [Fix 4] expand the call chain to ensure complete optimization scope
        skills_to_optimize = self.skill_manager.expand_with_dependencies(skills_to_optimize)

        if skills_to_optimize:
            # Extract current state from events
            current_state = None
            opt_observe = find_last_observe(ctx.events)
            if opt_observe is not None:
                current_state = {
                    "inventory": opt_observe.get("inventory", {}),
                    "position": opt_observe.get("status", {}).get("position", {}),
                    "biome": opt_observe.get("status", {}).get("biome", ""),
                    "equipment": opt_observe.get("status", {}).get("equipment", []),
                    # Extra information needed for environment diagnostics
                    "nearby_blocks": opt_observe.get("voxels", []),
                    "nearby_entities": opt_observe.get("status", {}).get("entities", {}),
                }

            # Extract error message (check both "error" and "onError" event types)
            current_error = None
            for event_type, event_data in iter_events(ctx.events):
                if event_type == "error":
                    current_error = str(event_data.get("error", "")) if isinstance(event_data, dict) else str(event_data)
                    break
                elif event_type == "onError":
                    # onError is the actual event type from Mineflayer observation system
                    current_error = event_data.get("onError", "") if isinstance(event_data, dict) else str(event_data)
                    break

            # Extract chat_log from onChat events
            # Chat messages often contain diagnostic information (e.g., "I need at least a stone_pickaxe...")
            # that is crucial for correct problem diagnosis in the optimizer
            chat_log_messages = []
            for event_type, event_data in iter_events(ctx.events):
                if event_type == "onChat":
                    if isinstance(event_data, str):
                        chat_log_messages.append(event_data)
                    elif isinstance(event_data, dict):
                        # Extract message from dict format
                        msg = event_data.get("message") or event_data.get("onChat") or ""
                        if msg:
                            chat_log_messages.append(msg)
            # Limit to last 30 messages to avoid context overflow
            chat_log = "\n".join(chat_log_messages[-30:]) if chat_log_messages else ""

            # Check which skills actually executed (via skill execution events)
            # Extract skill-execution info from events (skillStart event means a skill actually started executing)
            executed_skills = set()
            for event_type, event_data in iter_events(ctx.events):
                if event_type == "skillStart":
                    skill_name = event_data.get("skillName") if isinstance(event_data, dict) else None
                    if skill_name:
                        executed_skills.add(skill_name)

            # Check whether skillStart events are reliable
            # If the code called skills but no skillStart events arrived, the event mechanism may be unreliable
            # In that case we should not rely on executed_skills for filtering
            skill_events_unreliable = bool(called_skills and not executed_skills)

            if skill_events_unreliable:
                print(f"\033[33m[Optimizer] Warning: code called {len(called_skills)} skills but no skill-execution event was detected.\033[0m")
                print(f"\033[33m[Optimizer] Skipping executed_skills check because the event mechanism may be unreliable.\033[0m")

            # Filter: only optimize skills that actually executed
            # Note: for simple wrappers, the first element of skills_to_optimize is the called skill,
            # so treat it as the "main function" and skip the skillStart check
            filtered_skills_to_optimize = []

            # Improved primary_skill selection logic:
            # 1. For syntax errors, first locate the problem skill via code inspection
            # 2. Allow task-specific skills as primary_skill
            # 3. Fall back to the first skill present in the graph
            primary_skill = None
            primary_skill_source = None

            # Step 1: for syntax errors, check all skills' code syntax first
            if current_error and any(p in current_error for p in
                ["Unexpected end of input", "Unexpected token", "SyntaxError"]):
                syntax_error_skill = None
                if hasattr(self.optimizer, 'identify_syntax_error_source'):
                    syntax_error_skill = self.optimizer.identify_syntax_error_source(
                        current_error, skills_to_optimize
                    )
                if syntax_error_skill:
                    primary_skill = syntax_error_skill
                    primary_skill_source = "syntax_error_detected"
                    print(f"\033[33m[Optimizer] Located problem skill via syntax pre-check: {primary_skill}\033[0m")

            # Step 2: if syntax detection did not find one, check in order
            if not primary_skill:
                for s in skills_to_optimize:
                    # Check the main graph
                    if self.skill_manager.has_node(s):
                        primary_skill = s
                        primary_skill_source = "in_main_graph"
                        break
                    # Check task-specific skills
                    elif self.skill_manager.is_task_specific_skill_in_storage(s):
                        primary_skill = s
                        primary_skill_source = "task_specific"
                        print(f"\033[36m[Optimizer] Choosing task-specific skill as primary: {primary_skill}\033[0m")
                        break

            # Use a set to track already-added skills, to avoid duplicates
            added_skills = set()

            for skill_name in skills_to_optimize:
                # Check whether it is in the main graph or in the task-specific directory
                in_main_graph = self.skill_manager.has_node(skill_name)
                is_task_specific = self.skill_manager.is_task_specific_skill_in_storage(skill_name)

                if not in_main_graph and not is_task_specific:
                    continue
                # Skip already-added skills (to avoid duplicate optimization)
                if skill_name in added_skills:
                    continue
                # For the main function or the first skill (the target skill in a simple wrapper), do not require skillStart events
                # For other called skills, check whether they actually executed
                # But if the event mechanism is unreliable (no skillStart events at all), skip this check
                if skill_name != primary_skill and not skill_events_unreliable:
                    if skill_name not in executed_skills:
                        print(f"\033[33m[Optimizer] Skipping {skill_name}: called in code but did not actually execute (may have returned before the call)\033[0m")
                        continue

                # cross-check with last_skill_execution_results
                # Even after passing the executed_skills check, validate the was_executed field
                exec_result = self.last_skill_execution_results.get(skill_name, {})
                # explicitly check the was_executed field so defaults do not mask missing data
                was_executed = exec_result.get("was_executed")
                if was_executed is None:
                    # Missing data: warn but default to "executed" (preserves backward compatibility)
                    print(f"\033[33m[Optimizer] Warning: {skill_name} lacks was_executed info; assuming executed\033[0m")
                    was_executed = True
                if not was_executed:
                    print(f"\033[36m[Optimizer] Skipping {skill_name}: execution status is not_executed\033[0m")
                    continue
                filtered_skills_to_optimize.append(skill_name)
                added_skills.add(skill_name)

                # Mark whether this is a task-specific skill
                if is_task_specific and not in_main_graph:
                    print(f"\033[36m[Optimizer] Including task-specific skill: {skill_name}\033[0m")

            if not filtered_skills_to_optimize:
                print(f"\033[33m[Optimizer] No skills to optimize\033[0m")
            else:
                # Check whether there is an explicit error, critique, or feedback
                has_explicit_feedback = bool(current_error or ctx.critique)
                has_unused_feedbacks = False
                has_persistent_failures = False  # New: check for skills that have failed persistently

                if not has_explicit_feedback:
                    # Check whether there are unused feedbacks
                    for skill_name in filtered_skills_to_optimize:
                        node = self.skill_manager.get_node(skill_name)
                        if node and node.gradients.get_unused_items("feedback"):
                            has_unused_feedbacks = True
                            break

                    # New: check for persistently failing skills (success rate below threshold with sufficient execution records)
                    # This solves the case of "no explicit error but the task keeps failing"
                    FAILURE_THRESHOLD = 0.3  # success-rate threshold
                    MIN_EXECUTIONS = 3  # minimum execution count
                    for skill_name in filtered_skills_to_optimize:
                        node = self.skill_manager.get_node(skill_name)
                        if node and node.statistics.total_executions >= MIN_EXECUTIONS:
                            if node.statistics.success_rate < FAILURE_THRESHOLD:
                                has_persistent_failures = True
                                print(f"\033[33m[Optimizer] Detected persistently failing skill: {skill_name} "
                                      f"(success rate: {node.statistics.success_rate:.2f}, "
                                      f"total executions: {node.statistics.total_executions})\033[0m")
                                break

                    if not has_unused_feedbacks and not has_persistent_failures:
                        print(f"\033[33m[Optimizer] Warning: no explicit error, critique, or unused feedbacks; skipping optimization to avoid unnecessary code changes\033[0m")

                if has_explicit_feedback or has_unused_feedbacks or has_persistent_failures:
                    # Use the two-phase optimization flow
                    print(f"\033[36m[Optimizer] ========== Starting two-phase optimization ==========\033[0m")
                    print(f"\033[36m[Optimizer] Skills to optimize: {', '.join(filtered_skills_to_optimize)}\033[0m")
                    print(f"\033[36m[Optimizer] Trigger reason: task failure (task: {self.task})\033[0m")

                    if current_error:
                        print(f"\033[36m[Optimizer] Execution error: {current_error[:200]}...\033[0m")
                    if ctx.critique:
                        print(f"\033[36m[Optimizer] Critique: {ctx.critique[:200]}...\033[0m")

                    # Collect feedback (ensure skill-level feedback is recorded correctly)
                    # Important: only collect task-level feedback for the primary skill (candidate / directly executed skill)
                    # Sub-skill feedback should come from backpropagation (based on analysis, not directly from task critique)
                    # This avoids dispatching unrelated feedback (e.g. "craft pickaxe") to irrelevant skills (e.g. "collect logs")
                    primary_skill = filtered_skills_to_optimize[0] if filtered_skills_to_optimize else None

                    if primary_skill:
                        collected_feedbacks = self.optimizer.collect_feedback(
                            skill_name=primary_skill,
                            critique=ctx.critique,
                            error_message=current_error,
                            source="task_failure",
                            task=self.task,  # FIX: pass the task parameter
                        )
                        if collected_feedbacks:
                            print(f"\033[36m[Optimizer] Collected {len(collected_feedbacks)} feedbacks for primary skill '{primary_skill}'\033[0m")

                        # For child skills, just record that they were called but do not assign task-level feedback
                        # Sub-skill issues should be analyzed and assigned via the backpropagation mechanism
                        child_skills = [s for s in filtered_skills_to_optimize[1:] if s != primary_skill]
                        if child_skills:
                            print(f"\033[36m[Optimizer] Child skills {child_skills} will receive analyzed feedback via backpropagation\033[0m")

                    optimization_result = self.optimizer.optimize_skills_two_phase(
                        skills_to_optimize=filtered_skills_to_optimize,
                        current_task=self.task,
                        current_context=self.context,
                        current_state=current_state,
                        current_error=current_error,
                        current_critique=ctx.critique,
                        momentum_window=5,
                        skill_execution_results=self.last_skill_execution_results,
                        skill_events_unreliable=skill_events_unreliable,  # pass event-reliability info
                        quality_metrics=ctx.quality_metrics,  # Critic quality metrics
                        chat_log=chat_log,  # Chat log for diagnostic info
                    )

                    # Apply the optimization result
                    if optimization_result.get("success"):
                        results = optimization_result.get("results", {})
                        for skill_name, skill_result in results.items():
                            if skill_result.get("success") and skill_result.get("new_code"):
                                # Original code-optimization logic
                                apply_success = self.optimizer.apply_optimization(
                                    skill_name=skill_name,
                                    optimization_result=skill_result,
                                    create_new_version=True,
                                    backpropagation_info=skill_result.get("backpropagation_info"),
                                )

                                if apply_success:
                                    print(f"\033[32m[Optimizer] Successfully optimized skill: {skill_name}\033[0m")
                                    ctx.optimized = True

                                    # Skill Synthesis — extract reusable helpers
                                    self._try_extract_helpers(
                                        skill_name, skill_result.get("new_code", "")
                                    )
                                else:
                                    print(f"\033[33m[Optimizer] Optimized code for skill '{skill_name}' but apply failed\033[0m")

                    if ctx.optimized:
                        # Re-retrieve skills with updated code
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
                        reuse_skill_hint = self._find_reuse_skill_hint(self.task, skill_metadata)
                        system_message = self.action_agent.render_system_message(
                            skills=new_skills, skill_metadata=skill_metadata, reuse_skill_hint=reuse_skill_hint,
                            task=self.task,
                        )
                        human_message = self.action_agent.render_human_message(
                            events=ctx.events,
                            code=ctx.parsed_result["program_code"],
                            task=self.task,
                            context=self.context,
                            critique=ctx.critique,
                        )
                        self.messages = [system_message, human_message]
                        # Continue to retry with optimized skill
                        print(f"\033[36m[Optimizer] Retrying with optimized skill\033[0m")

    def _try_extract_helpers(self, skill_name: str, new_code: str):
        """Attempt to extract reusable helper functions from optimized code.

        Called after successful optimization. If the optimized code contains
        large inline helper functions, extracts them as independent skills.
        """
        if not new_code or not hasattr(self, 'skill_manager'):
            return

        extractor = getattr(self, '_helper_extractor', None)
        if extractor is None:
            return

        try:
            candidates = extractor.detect_extractable_helpers(
                code=new_code,
                main_function_name=skill_name,
            )

            if not candidates:
                return

            print(
                f"\033[36m[Synthesis] Found {len(candidates)} extractable helper(s) "
                f"in '{skill_name}': {[c.name for c in candidates]}\033[0m"
            )

            extractor.extract_and_register(
                helpers=candidates,
                parent_skill_name=skill_name,
                parent_code=new_code,
                skill_manager=self.skill_manager,
                task=getattr(self, 'task', ''),
                context=getattr(self, 'context', ''),
            )
        except Exception as e:
            print(f"\033[33m[Synthesis] Helper extraction failed: {e}\033[0m")
