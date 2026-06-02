"""
SkillRecordingMixin for PSNAgent

Methods for recording skill executions, finding reusable skills,
extracting skill targets, and generating call examples.

Internal composition:
    _auto_record_skill_executions() — coordinator calling pure functions from recording_helpers.py
    _register_failed_new_skill()   — progressive deprecation for failed new skills
    _find_reuse_skill_hint()       — reuse hint matching
    _extract_skill_target()        — skill name → target item
    _generate_call_example()       — metadata → JS call example
"""

import re
from typing import Optional, Tuple, TYPE_CHECKING

from skillnet._psn_impl.event_helpers import get_event_data_dict, iter_events
from skillnet._psn_impl.recording_helpers import (
    parse_skill_events,
    classify_execution_status,
    extract_trajectory_info,
    build_action_info,
)
from skillnet.agents.skill_graph import SkillNode

if TYPE_CHECKING:
    from ..psn import PSNAgent


class SkillRecordingMixin:
    """Skill execution recording and reuse hint helpers."""

    def _auto_record_skill_executions(
        self,
        code: str,
        parsed_result: dict,
        events: list,
        success: bool,
        critique: str = None,
    ):
        """
        Automatically record skill executions in graph mode (using precise execution info).

        Coordinator that delegates to pure functions in recording_helpers.py:
        - parse_skill_events(): event parsing
        - classify_execution_status(): per-skill success/failure classification
        - extract_trajectory_info(): trajectory file/segment extraction
        - build_action_info(): action info dict construction
        """
        # === Section 1: Initialize ===
        self.last_skill_execution_results = {}

        full_code = parsed_result.get("program_code", "") + "\n" + parsed_result.get("exec_code", "")
        called_skills = self.skill_manager.extract_called_skills(full_code)

        # fetch the main function name to ensure it is always tracked
        main_function_name = parsed_result.get("program_name")

        # before any early return, ensure the main function is added to task-execution tracking
        if main_function_name and hasattr(self, '_task_executed_skills'):
            self._task_executed_skills.add(main_function_name)
            if not called_skills:
                print(f"\033[36m[Skill Tracking] Main function '{main_function_name}' added to task-execution tracking (no other known skill calls)\033[0m")

        # check whether the events contain skill-execution events
        has_skill_events = any(
            event_type in ("skillStart", "skillEnd", "skillError")
            for event_type, _ in iter_events(events)
        )

        if not called_skills and not has_skill_events:
            return

        # === Section 2: Trajectory info ===
        trajectory_file, trajectory_segment_id = extract_trajectory_info(
            self.trajectory_recorder, self.task
        )

        # === Section 3: Event parsing ===
        pr = parse_skill_events(events, self._find_nearby_blocks_from_events)

        # Augment error_message with critique fallback
        error_message = pr.error_message
        if not success and not error_message and critique:
            error_message = f"Task failed: {critique}"

        # === Section 3b: Diagnostic prints ===
        executed_skill_names = set(pr.skill_executions.keys())

        if pr.skill_event_count > 0:
            print(f"\033[36m[Skill Tracking] Detected {pr.skill_event_count} skill-execution events covering {len(pr.skill_executions)} skills\033[0m")
            unexecuted_skills = set(called_skills) - executed_skill_names
            if unexecuted_skills:
                pass  # silent: this is usually normal behavior from a conditional branch
        elif called_skills:
            if success:
                print(f"\033[36m[Skill Tracking] Static analysis found {len(called_skills)} potential skill calls, but none actually executed due to conditionals (task succeeded)\033[0m")
            else:
                print(f"\033[33m[Skill Tracking] Note: code contains {len(called_skills)} skill calls ({', '.join(called_skills[:3])}{'...' if len(called_skills) > 3 else ''}) but no execution events were detected.\033[0m")
                print(f"\033[33m[Skill Tracking] Possible reasons: 1) conditional branch unmet 2) error before skill execution 3) wrapper malfunction\033[0m")

        # === Compute skill sets ===
        not_executed_skills = set(called_skills) - executed_skill_names
        all_executed_skills = set(called_skills) | executed_skill_names

        if hasattr(self, '_task_executed_skills'):
            self._task_executed_skills.update(all_executed_skills)
            if all_executed_skills:
                print(f"\033[36m[Skill Tracking] Tracked {len(all_executed_skills)} skill executions\033[0m")

        # === Main loop: classify, build, record each skill ===
        for skill_name in all_executed_skills:
            skill_exec_info = pr.skill_executions.get(skill_name, {})

            # Section 4: Classify execution status
            classification = classify_execution_status(
                skill_name=skill_name,
                skill_exec_info=skill_exec_info,
                main_function_name=main_function_name,
                overall_success=success,
                error_message=error_message,
                critique=critique,
                not_executed_skills=not_executed_skills,
                called_skills=called_skills,
                check_effect_fn=self._check_skill_effect_achieved,
            )

            # Writeback effect_verification (classifier itself stays pure)
            if classification.effect_verification:
                skill_exec_info["effect_verification"] = classification.effect_verification

            # Section 5: Build action_info
            is_nested_skill = skill_name not in called_skills
            action_info = build_action_info(
                skill_name=skill_name,
                is_nested_skill=is_nested_skill,
                skill_exec_info=skill_exec_info,
                full_code=full_code,
                parsed_result=parsed_result,
                skill_manager=self.skill_manager,
            )

            # Pre/post state for recording
            pre_state = skill_exec_info.get("pre_state")
            post_state = skill_exec_info.get("post_state")

            # Section 6: Register failed new skill (if not in graph and failed)
            if not self.skill_manager.has_node(skill_name) and not classification.skill_success:
                should_skip, skill_name = self._register_failed_new_skill(
                    skill_name, classification.skill_execution_status, action_info, events
                )
                if should_skip:
                    continue

            # === Section 7: Record + rename sync ===
            # Inject call_stack/js_args into action_info for record_execution
            if skill_exec_info:
                action_info = action_info.copy()
                action_info["call_stack"] = skill_exec_info.get("call_stack", [])
                action_info["js_args"] = skill_exec_info.get("js_args")

            self.last_skill_execution_results[skill_name] = {
                "success": classification.skill_success,
                "execution_status": classification.skill_execution_status,
                "was_executed": classification.skill_execution_status != "not_executed",
                "error_message": classification.skill_error_message,
                "error_stack": classification.skill_error_stack if not classification.skill_success else None,
                "effect_verification": skill_exec_info.get("effect_verification", {}),
                "call_stack": skill_exec_info.get("call_stack", []),
                "js_args": skill_exec_info.get("js_args"),
            }

            final_skill_name = self.skill_manager.record_execution(
                skill_name=skill_name,
                success=classification.skill_success,
                execution_status=classification.skill_execution_status,
                trajectory_segment_id=trajectory_segment_id,
                trajectory_file=trajectory_file,
                task=self.task,
                context=self.context,
                environment_events=pr.environment_events if pr.environment_events else None,
                pre_state=pre_state,
                post_state=post_state,
                action_info=action_info,
                error_message=classification.skill_error_message if not classification.skill_success else None,
                error_stack=classification.skill_error_stack if not classification.skill_success else None,
                critique=critique,
            )

            # Rename sync: update parsed_result, _task_executed_skills, last_skill_execution_results
            if final_skill_name and final_skill_name != skill_name:
                if parsed_result.get("program_name") == skill_name:
                    parsed_result["program_name"] = final_skill_name
                    print(f"\033[36m[Skill Rename Sync] Main function name updated: '{skill_name}' -> '{final_skill_name}'\033[0m")

                if hasattr(self, '_task_executed_skills') and skill_name in self._task_executed_skills:
                    self._task_executed_skills.discard(skill_name)
                    self._task_executed_skills.add(final_skill_name)
                    print(f"\033[36m[Skill Rename Sync] _task_executed_skills updated: '{skill_name}' -> '{final_skill_name}'\033[0m")

                if skill_name in self.last_skill_execution_results:
                    self.last_skill_execution_results[final_skill_name] = self.last_skill_execution_results.pop(skill_name)
                    print(f"\033[36m[Skill Rename Sync] last_skill_execution_results updated: '{skill_name}' -> '{final_skill_name}'\033[0m")

            # Debug print: skill execution result
            if pr.skill_event_count > 0:
                status_icon = "+" if classification.skill_success else "x"
                status_color = "\033[32m" if classification.skill_success else "\033[31m"
                print(f"{status_color}[Skill Tracking] {status_icon} {skill_name}: {'success' if classification.skill_success else 'failed'}\033[0m")
                if not classification.skill_success and classification.skill_error_message:
                    err_msg = classification.skill_error_message
                    print(f"\033[33m  |-- Error: {err_msg[:100]}...\033[0m" if len(err_msg) > 100 else f"\033[33m  |-- Error: {err_msg}\033[0m")

    def _register_failed_new_skill(self, skill_name, skill_execution_status, action_info, events):
        """Register a failed new skill to the graph with progressive deprecation.

        Called when a skill is not in the graph and has failed execution.
        Handles not_executed skip, task-specific skip, primitive name conflict
        resolution, and progressive deprecation tracking.

        Args:
            skill_name: Name of the skill.
            skill_execution_status: "failed", "not_executed", "interrupted", etc.
            action_info: Action info dict from build_action_info.
            events: Raw event list (for extracting last error).

        Returns:
            (should_skip, updated_skill_name): should_skip=True means caller
            should ``continue`` to the next skill in the loop.
        """
        # skip skills that did not execute
        if skill_execution_status == "not_executed":
            print(f"\033[36m[Skip] {skill_name}: did not execute; not adding to graph\033[0m")
            return True, skill_name

        program_code = action_info.get("program_code", "")
        if not program_code:
            program_code = action_info.get("code", "")

        # Check whether this is a task-specific skill
        if hasattr(self.skill_manager, '_is_task_specific_skill'):
            if self.skill_manager.is_task_specific(skill_name, program_code):
                print(f"\033[33m[Skip] Task-specific skill '{skill_name}' execution failed; not adding to graph\033[0m")
                return True, skill_name

        # Check and resolve naming conflicts with control primitives
        from skillnet.agents.skill_graph import resolve_primitive_name_conflict
        skill_name, program_code, was_renamed = resolve_primitive_name_conflict(skill_name, program_code)

        # Generate basic description
        skill_description = f"Failed attempt to solve task: {self.task}"
        if program_code:
            try:
                skill_description = self.skill_manager.generate_skill_description(
                    skill_name, program_code, task=self.task
                )
            except:
                pass

        # Add to graph and use a gradual-deprecation strategy
        print(f"\033[33m[Progressive Deprecation] Detected failed temporary skill: {skill_name}; adding to graph\033[0m")
        node = SkillNode(
            name=skill_name,
            code=program_code,
            description=skill_description,
        )

        error_msg = None
        if events and len(events) > 0:
            last_event_data = get_event_data_dict(events[-1])
            if last_event_data is not None:
                error_msg = last_event_data.get("error")
        deprecation_result = node.mark_execution_failed(
            task=self.task,
            error=error_msg
        )
        if deprecation_result["deprecation_triggered"]:
            print(f"\033[33m[Progressive Deprecation] Skill '{skill_name}' reached deprecation threshold: {deprecation_result['reason']}\033[0m")
        else:
            print(f"\033[36m[Progressive Deprecation] Skill '{skill_name}' failure recorded (consecutive failures: {deprecation_result['consecutive_failures']}/{node.deprecation_threshold})\033[0m")

        self.skill_manager.add_skill_node(node)
        self.skill_manager.update_skill_dependencies(skill_name)
        self.skill_manager.ensure_skill_in_vectordb(skill_name)
        self.skill_manager.save()
        print(f"\033[33m[Deprecated Skill] Added and marked as deprecated: {skill_name}\033[0m")

        return False, skill_name

    # reuse_skill_hint helper methods
    def _find_reuse_skill_hint(self, task: str, skill_metadata: dict) -> Optional[Tuple[str, str]]:
        """
        Find a directly-reusable skill and produce a call example.

        Args:
            task: current task description
            skill_metadata: retrieved skill metadata

        Returns:
            (skill_name, call_example) or None
        """
        if not skill_metadata or not task:
            return None

        task_lower = task.lower()

        # Simple matching: check whether the task explicitly mentions a skill's target
        for skill_name, metadata in skill_metadata.items():
            # Extract a skill's target item (from the skill name)
            target = self._extract_skill_target(skill_name)
            if target and target in task_lower:
                # Generate call example
                call_example = self._generate_call_example(skill_name, metadata)
                print(f"\033[36m[Fix10] Found reusable skill: {skill_name} for task: {task}\033[0m")
                return (skill_name, call_example)

        return None

    def _extract_skill_target(self, skill_name: str) -> Optional[str]:
        """Extract the target item from a skill name.

        For example:
        - craftDiamondPickaxe -> diamond_pickaxe
        - ensureCobblestone -> cobblestone
        """
        domain = getattr(self, '_domain', None)
        if domain and hasattr(domain, 'knowledge'):
            prefixes = domain.knowledge.get_skill_verb_prefixes()
        else:
            prefixes = []
        # Strip prefix
        for prefix in prefixes:
            if skill_name.lower().startswith(prefix):
                target = skill_name[len(prefix):]
                # CamelCase -> snake_case
                target = re.sub(r'([A-Z])', r'_\1', target).lower().strip('_')
                return target
        return None

    def _generate_call_example(self, skill_name: str, metadata: dict) -> str:
        """Generate a skill call example."""
        params = metadata.get("parameters", {})
        domain = getattr(self, '_domain', None)
        if domain and hasattr(domain, 'knowledge'):
            entry_param = domain.knowledge.get_entry_parameter_name()
        else:
            entry_param = "bot"
        param_strs = [entry_param]
        for param_name, param_info in params.items():
            default = param_info.get("default")
            if default is not None:
                if isinstance(default, str):
                    param_strs.append(f'{param_name}="{default}"')
                else:
                    param_strs.append(f"{param_name}={default}")
            else:
                # For parameters without defaults, show a type hint
                param_type = param_info.get("type", "value")
                param_strs.append(f"{param_name}=<{param_type}>")
        return f"await {skill_name}({', '.join(param_strs)})"
