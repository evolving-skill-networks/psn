"""
Skill Execution Mixin for SkillGraphManager

Core execution recording: creates temporary nodes, records execution traces,
triggers performance degradation rollback, and delegates to semantics update.

Extracted from graph_manager_impl.py for better modularity.
Split into 3 mixins for further modularity:
  - SkillExecutionMixin (this file): Core execution recording
  - ExecutionLifecycleMixin (execution_lifecycle.py): Refactor lifecycle management
  - SemanticsUpdateMixin (semantics_update.py): Runtime semantics learning engine
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List

from skillnet.agents.skill_graph.models import (
    SkillNode,
    SkillExecutionTrace,
)
from skillnet.agents.skill_graph.utils import resolve_primitive_name_conflict

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class SkillExecutionMixin:
    """Skill Execution Mixin - Core execution recording.

    Methods:
        record_execution: Record skill execution result (~270 lines)

    Note:
        record_execution creates temporary node when skill doesn't exist.
        Coordination with add_new_skill is protected by _graph_update_lock.

    Cross-mixin dependencies (resolved via MRO):
        ExecutionLifecycleMixin: _trigger_delayed_refactor, _handle_refactored_skill_failure
        SemanticsUpdateMixin: _update_semantics_from_executions, _parse_js_args_to_dict,
                              _parse_call_args_from_exec_code

    Attributes (from SkillGraphManager):
        graph: SkillGraph instance
        vectordb: ChromaDB instance
        logger: Logger instance
        auto_semantic_rename: Whether auto semantic rename is enabled
        value_function_lambda: Value function lambda
        value_function_alpha: Value function alpha
        value_function_beta: Value function beta
    """

    def record_execution(
        self: "SkillGraphManager",
        skill_name: str,
        success: bool,
        execution_status: str = None,  # precise execution status (success/failed/not_executed/interrupted)
        trajectory_segment_id: str = None,
        trajectory_file: str = None,
        task: str = None,
        context: str = None,
        environment_events: List[Any] = None,
        environment_state: Dict[str, Any] = None,
        pre_state: Dict[str, Any] = None,
        post_state: Dict[str, Any] = None,
        action_info: Dict[str, Any] = None,
        error_message: str = None,
        error_stack: str = None,
        critique: str = None,
    ) -> str:
        """
        Record skill-execution info, including actual effects.

        Args:
            skill_name: skill name
            success: whether successful
            execution_status: v7.2 precise execution status, possible values:
                - "success": executed successfully
                - "failed": execution failed
                - "not_executed": called in code but never triggered at runtime
                - "interrupted": execution was interrupted
                If None, inferred from success
            trajectory_segment_id: trajectory segment ID
            trajectory_file: trajectory file path
            task: associated task
            context: execution context
            environment_events: Environment events (runtime observations)
            environment_state: Environment state (post-execution, legacy compat)
            pre_state: pre-execution state
            post_state: post-execution state
            action_info: Action info (includes code, program_name, etc.)
            error_message: error message (if failed)
            critique: feedback from the Critic agent

        Returns:
            str: the final skill name used (may have been renamed)
        """
        # If the skill does not exist, try creating a temporary node to record the execution
        if skill_name not in self.graph.nodes:
            # If code info was provided, create a temporary node to record the first execution
            if action_info:
                program_code = action_info.get("program_code", "") or action_info.get("code", "")
                if program_code:
                    # ===== Check and resolve naming conflicts with control primitives =====
                    original_skill_name = skill_name
                    skill_name, program_code, was_renamed = resolve_primitive_name_conflict(skill_name, program_code)
                    if was_renamed:
                        print(f"\033[33m[Execution Recording] Skill '{original_skill_name}' renamed to '{skill_name}'\033[0m")

                    # ===== Check and resolve semantic naming inconsistencies (aligned with add_new_skill) =====
                    if self.auto_semantic_rename:
                        old_name_before_semantic = skill_name
                        skill_name, program_code, was_semantic_renamed = self._check_and_apply_semantic_rename(
                            skill_name, program_code
                        )
                        if was_semantic_renamed:
                            print(f"\033[32m[Execution Recording] Semantic rename: '{old_name_before_semantic}' → '{skill_name}'\033[0m")

                    # [Fix] aligned with add_new_skill: add type normalization
                    # Ensure record_execution and add_new_skill use the same name-normalization logic
                    # Avoid orphan nodes (record_execution creating "mineOakLog" while add_new_skill creates "mineLog")
                    normalized_name = self._try_normalize_skill_name(skill_name, program_code)
                    if normalized_name and normalized_name != skill_name:
                        print(f"\033[36m[Execution Recording] Type normalization: '{skill_name}' → '{normalized_name}'\033[0m")
                        # Update the function name in code
                        program_code = re.sub(
                            rf'async\s+function\s+{re.escape(skill_name)}\s*\(',
                            f"async function {normalized_name}(",
                            program_code
                        )
                        program_code = re.sub(
                            rf'\b{re.escape(skill_name)}\s*\(',
                            f"{normalized_name}(",
                            program_code
                        )
                        skill_name = normalized_name

                    print(f"\033[36m[Execution Recording] Skill '{skill_name}' does not exist; creating a temporary node to record the first execution\033[0m")
                    # Generate a basic description
                    skill_description = f"Temporary skill for task: {task}" if task else f"Temporary skill: {skill_name}"
                    try:
                        skill_description = self.generate_skill_description(skill_name, program_code, task=task)
                    except Exception as e:
                        print(f"\033[33m[Execution Recording] Warning: description generation failed; using default: {e}\033[0m")

                    # Create the temporary node
                    node = SkillNode(
                        name=skill_name,
                        code=program_code,
                        description=skill_description,
                    )
                    # Set value-function parameters
                    node.value_function_lambda = self.value_function_lambda
                    node.value_function_alpha = self.value_function_alpha
                    node.value_function_beta = self.value_function_beta

                    # Mark as temporary node (a later add_new_skill will update it)
                    node.is_deprecated = False  # temporary node is not deprecated; just awaiting persistent save

                    # Phase 3: immediately mark as experimental on creation
                    # Newly created skills are experimental until first successful validation
                    node.is_experimental = True
                    node.created_for_task = task
                    node.experimental_task = task

                    # [Fix] check and set the task-specific flag
                    # Ensure task-specific skills are filtered properly in _find_skills_by_effects
                    # The node must still be created (for execution tracking and top-down optimizer analysis)
                    if self._is_task_specific_skill(skill_name, program_code):
                        node.is_task_specific = True
                        print(f"\033[33m[Execution Recording] Detected task-specific skill '{skill_name}'\033[0m")

                    self.graph.add_node(node)

                    # [P0 Fix] Extract call relations from code and update edges
                    # Establish edges immediately after the temporary node is created so the optimizer can identify sub-skills correctly
                    self._update_graph_from_code(skill_name)

                    # Add to the vector database
                    try:
                        self.vectordb.add_texts(
                            texts=[skill_description],
                            ids=[skill_name],
                            metadatas=[{"name": skill_name}],
                        )
                    except Exception as e:
                        print(f"\033[33m[Execution Recording] Warning: failed to add to vector database: {e}\033[0m")

                    print(f"\033[32m[Execution Recording] Created temporary node '{skill_name}'; will record the first execution\033[0m")
                else:
                    # No code info; cannot create a node; return directly
                    print(f"\033[33m[Execution Recording] Warning: skill '{skill_name}' does not exist and no code info; skipping record\033[0m")
                    return skill_name
            else:
                # No action_info; cannot create a node; return directly
                print(f"\033[33m[Execution Recording] Warning: skill '{skill_name}' does not exist and no action_info; skipping record\033[0m")
                return skill_name

        node = self.graph.get_node(skill_name)

        # If no post_state was provided, use environment_state
        if post_state is None:
            post_state = environment_state or {}

        # Compute actual effects (if pre_state and post_state were provided)
        actual_effects = []
        if pre_state and post_state:
            try:
                actual_effect = self._calculate_state_changes(
                    pre_state=pre_state,
                    post_state=post_state,
                    environment_events=environment_events,
                )
                actual_effects.append(actual_effect)

                # Save into node.actual_effects (cap at most recent 100)
                node.actual_effects.append(actual_effect)
                if len(node.actual_effects) > 100:
                    node.actual_effects = node.actual_effects[-100:]

                print(f"\033[36m[Effect Recording] Skill '{skill_name}' actual effect: {actual_effect.description}\033[0m")
            except Exception as e:
                print(f"\033[33m[Effect Recording] Warning: Failed to calculate actual effect for '{skill_name}': {e}\033[0m")

        # Scheme 6: extract call arguments and call stack
        # Prefer the actual arguments from the JS side (more accurate); fall back to exec_code parsing
        call_args = {}
        call_stack = []
        call_depth = 1

        if action_info:
            # Prefer JS-side data (recorded by the skill wrapper)
            js_args = action_info.get("js_args")
            if js_args:
                call_args = self._parse_js_args_to_dict(js_args, skill_name, node)
            else:
                # Fallback: parse from exec_code
                exec_code = action_info.get("exec_code", "")
                if exec_code:
                    call_args = self._parse_call_args_from_exec_code(exec_code, skill_name, node)

            # Extract call_stack
            call_stack = action_info.get("call_stack", [])
            call_depth = len(call_stack) if call_stack else 1

        trace = SkillExecutionTrace(
            success=success,
            execution_status=execution_status,  # pass the precise execution status
            trajectory_segment_id=trajectory_segment_id,
            trajectory_file=trajectory_file,
            task=task,
            context=context,
            environment_events=environment_events,
            environment_state=post_state,
            pre_state=pre_state,
            post_state=post_state,
            action_info=action_info,
            error_message=error_message,
            error_stack=error_stack,
            critique=critique,
            actual_effects=actual_effects,
            call_args=call_args,
            call_stack=call_stack,
            call_depth=call_depth,
        )
        node.statistics.add_execution(trace)

        # Append execution record to the current version's sliding window
        latest_version = node.get_latest_version()
        if latest_version:
            node.add_execution_to_window(
                execution_id=trace.execution_id,
                success=success,
                version=latest_version.version,
            )

            # Check performance regression and trigger rollback
            is_degraded, previous_version, current_sr, previous_sr = node.check_performance_degradation(
                current_version=latest_version.version,
                degradation_threshold=0.15,  # 15% regression threshold
                min_window_size=5,  # detect only when at least 5 executions exist
            )

            if is_degraded and previous_version:
                print(f"\033[33m[Performance Degradation] Detected performance regression: {skill_name}\033[0m")
                print(f"  Current version {latest_version.version} success rate: {current_sr:.2%}")
                print(f"  Previous version {previous_version} success rate: {previous_sr:.2%}")
                print(f"  Regression amount: {(previous_sr - current_sr):.2%}")
                print(f"\033[33m[Rollback] Triggering auto-rollback to version {previous_version}\033[0m")

                # Execute rollback
                rollback_success = self.rollback_skill_and_subgraph(
                    skill_name=skill_name,
                    target_version=previous_version,
                    rollback_subgraph=True,
                )

                if rollback_success:
                    print(f"\033[32m[Rollback] Successfully rolled back {skill_name} and its subgraph to version {previous_version}\033[0m")
                else:
                    print(f"\033[31m[Rollback] Rollback failed: {skill_name}\033[0m")

        # Online update of preconditions and effects (based on execution data)
        self._update_semantics_from_executions(skill_name)

        # Part 18: detect refactored-skill failures
        if not success and node.is_covered and node.covered_by:
            self._handle_refactored_skill_failure(
                skill_name,
                error_message=error_message,
                context=context,
                task=task
            )

        # [Fix 7 supplement] handle experimental skill success
        # If an experimental skill executes successfully, trigger the delayed refactor
        if success and getattr(node, 'is_experimental', False):
            success_result = node.mark_execution_success(task=task)
            if success_result.get("should_trigger_refactor"):
                print(f"\033[36m[Delayed Refactor] Experimental skill '{skill_name}' validated successfully; triggering delayed refactor\033[0m")
                # [P0-2 fix] check the return value and record the result
                refactor_success = self._trigger_delayed_refactor(skill_name)
                if not refactor_success:
                    print(f"\033[33m[Delayed Refactor] Refactor failed to execute, but skill '{skill_name}' has been validated\033[0m")

        # Save to checkpoint
        self._save_to_checkpoint()

        # Return the final skill name used (may have been renamed)
        return skill_name
