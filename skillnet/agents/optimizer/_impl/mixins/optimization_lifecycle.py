"""
OptimizationLifecycleMixin - Optimization lifecycle callbacks and orchestration.

Extracted from optimizer_impl.py for better modularity.

Methods included (6):
- _optimizer_callback_wrapper: Wrapper for TwoPhaseOptimizationEngine optimizer callback
- _on_skill_optimized_callback: Callback after successful optimization
- _on_optimization_failed_callback: Callback after failed optimization
- on_task_completed: Unified handler for task completion
- optimize_skills_two_phase: Two-phase optimization flow orchestration
- _get_recent_optimization_history: Get recent optimization history for momentum
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from skillnet.agents.optimizer.analysis import (
    ErrorCategory,
    FixTargetType,
)

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer


class OptimizationLifecycleMixin:
    """Optimization lifecycle callbacks and two-phase orchestration."""

    def _optimizer_callback_wrapper(
        self,
        skill_name: str,
        current_task: str = None,
        current_context: str = None,
        current_state: Dict[str, Any] = None,
        current_error: str = None,
        current_critique: str = None,
        skill_delta=None,  # SkillDelta contains structured Gradient info
        quality_metrics: Optional[Dict[str, Any]] = None,  # Critic quality metrics
        chat_log: str = "",  # Chat log from onChat events
    ) -> Dict[str, Any]:
        """
        Optimizer callback wrapper, used by TwoPhaseOptimizationEngine

        Calls quick_optimize_skill to perform the actual LLM optimization

        Args:
            skill_name: skill name
            current_task: current task
            current_context: current context
            current_state: current environment state
            current_error: current execution error
            current_critique: current critique
            skill_delta: SkillDelta object containing structured Gradient info (gradient_type, suggested_fix, etc.)
            chat_log: v7.7 Chat log containing diagnostic messages from skill execution

        Returns:
            Dict: optimization result containing fields like 'new_code', 'success', etc.
        """
        try:
            # If there is structured skill_delta, directly pass Phase 1 LLM direction/suggested_fix
            structured_edit_context = ""
            if skill_delta and hasattr(skill_delta, 'gradients') and skill_delta.gradients:
                for grad in skill_delta.gradients:
                    grad_type = getattr(grad, 'gradient_type', 'unknown')
                    direction = getattr(grad, 'direction', '')
                    suggested_fix = getattr(grad, 'suggested_fix', '')
                    magnitude = getattr(grad, 'magnitude', 0.5)

                    if direction or suggested_fix:
                        structured_edit_context += f"""
## [LAYER 0 - {grad_type}] (magnitude: {magnitude:.2f})
Problem: {direction}
Fix: {suggested_fix}

"""

                if structured_edit_context:
                    self.logger.info(f"\033[36m[OptimizerCallback] Detected structured Gradient:\033[0m")
                    for grad in skill_delta.gradients:
                        grad_type = getattr(grad, 'gradient_type', 'unknown')
                        self.logger.info(f"\033[36m  - Type: {grad_type}, Direction: {getattr(grad, 'direction', '')[:50]}...\033[0m")

            # Store Phase 1 gradients for use in quick_optimize_skill
            if skill_delta and hasattr(skill_delta, 'gradients') and skill_delta.gradients:
                self._current_phase1_gradients = list(skill_delta.gradients)
            else:
                self._current_phase1_gradients = []

            # Prepend the structured edit_context to the critique (if any)
            enhanced_critique = current_critique or ""
            if structured_edit_context:
                enhanced_critique = structured_edit_context + "\n" + enhanced_critique

            # Inject Reference Check rejection history (Fix 7)
            ref_rejections = self._reference_check_rejections.get(skill_name, [])
            if ref_rejections:
                ref_warning = "\n## PREVIOUS OPTIMIZATION REJECTED BY REFERENCE CHECK\n"
                ref_warning += "Your previous optimization(s) were REJECTED because they called functions that DO NOT EXIST.\n"
                ref_warning += "You MUST use only functions from the EXISTING SKILLS list or system control primitives.\n\n"
                for i, rej in enumerate(ref_rejections[-3:], 1):  # Last 3 rejections
                    ref_warning += f"Rejection {i}:\n"
                    for func_name in rej["undefined_functions"]:
                        sugg = rej["suggestions"].get(func_name, [])
                        if sugg:
                            ref_warning += f"  - '{func_name}' does not exist. Did you mean: {', '.join(sugg[:3])}?\n"
                        else:
                            ref_warning += f"  - '{func_name}' does not exist. No similar function found.\n"
                ref_warning += "\nDo NOT invent new function names. Use ONLY existing skills listed below.\n"
                enhanced_critique = ref_warning + "\n" + enhanced_critique

            # Inject Responsibility Check rejection history
            # When RespCheck rejects because inlined logic belongs to a sibling skill,
            # surfacing the rejection reason + suggested composable alternative lets the
            # next optimization round switch strategy instead of repeating the same
            # inline pattern (which led to the r2 infinite loop).
            resp_rejections = self._responsibility_check_rejections.get(skill_name, [])
            if resp_rejections:
                resp_warning = "\n## PREVIOUS OPTIMIZATION REJECTED BY RESPONSIBILITY CHECK\n"
                resp_warning += (
                    "Your previous optimization(s) were REJECTED because the code inlined "
                    "logic that belongs to a DIFFERENT skill's responsibility (e.g. placement "
                    "logic in a crafting-focused skill). DO NOT repeat the same inlining "
                    "pattern — instead CALL the suggested composable skill.\n\n"
                )
                for i, rej in enumerate(resp_rejections[-3:], 1):  # last 3
                    resp_warning += f"Rejection {i}:\n"
                    resp_warning += f"  - reason: {rej['reason'][:300]}\n"
                    if rej.get("violated_helpers"):
                        resp_warning += (
                            f"  - violated inline helpers: {rej['violated_helpers']}\n"
                        )
                    if rej.get("suggested_target_skill"):
                        resp_warning += (
                            f"  - use this COMPOSABLE SKILL instead: "
                            f"`await {rej['suggested_target_skill']}(bot)`\n"
                        )
                resp_warning += (
                    "\nTo converge: find a skill in COMPOSABLE SKILLS that matches the "
                    "rejected logic's purpose, and CALL it with `await skillName(bot, ...)`.\n"
                )
                enhanced_critique = resp_warning + "\n" + enhanced_critique

            # Function reference validation with retry loop.
            # An LLM that hallucinates one callable name on the first attempt
            # tends to switch to a different hallucinated name on the second
            # if the rejection feedback isn't specific enough. Two retries
            # (three attempts total) gives the LLM one more shot to converge
            # on a real callable once the rejection feedback below lists the
            # full set of valid options.
            from skillnet.agents.optimizer.validators.code_validator._references import (
                validate_function_references,
            )
            max_ref_retries = 2
            critique_for_attempt = enhanced_critique

            # Get domain function sets for validation (if domain configured)
            domain_fns = None
            _dk = getattr(self, '_domain_knowledge', None)
            if _dk:
                domain_fns = _dk.get_known_functions() or None

            for ref_attempt in range(max_ref_retries + 1):
                result = self.quick_optimize_skill(
                    skill_name=skill_name,
                    current_task=current_task,
                    current_context=current_context,
                    current_state=current_state,
                    current_error=current_error,
                    current_critique=critique_for_attempt,
                    quality_metrics=quality_metrics,
                    chat_log=chat_log,
                )

                # Only validate references if optimization succeeded with new code
                if not (result and isinstance(result, dict) and result.get("success") and result.get("new_code")):
                    break  # Optimization failed for other reasons, don't retry

                available_skills = set(self.skill_graph_manager.get_all_skill_names())
                available_skills.add(skill_name)
                ref_result = validate_function_references(
                    code=result["new_code"],
                    available_skills=available_skills,
                    domain_functions=domain_fns,
                    strict_mode=True,
                )

                if ref_result["valid"]:
                    break  # Validation passed

                # Validation failed — log and retry if attempts remain
                undefined = [f["name"] for f in ref_result["undefined_functions"]]
                suggestions = ref_result.get("suggestions", {})
                self.logger.warning(
                    f"\033[33m[OptimizerCallback] Function reference validation failed (attempt {ref_attempt + 1}): "
                    f"undefined={undefined}\033[0m"
                )

                if ref_attempt < max_ref_retries:
                    # Build rejection feedback for retry. List the FULL set of
                    # callable names the LLM may choose from so it can pick a
                    # real alternative rather than inventing another name with
                    # a similar prefix to the one we just rejected. Closest
                    # name-similar matches are shown but explicitly labeled to
                    # avoid pulling the LLM toward another prefix-look-alike
                    # when the functionally-correct choice is a different name
                    # entirely.
                    rejection_feedback = (
                        "\n## FUNCTION REFERENCE VALIDATION FAILED\n"
                        "Your code calls functions that DO NOT EXIST:\n"
                    )
                    for func_name in undefined:
                        sugg = suggestions.get(func_name, [])
                        if sugg:
                            rejection_feedback += (
                                f"  - '{func_name}' does not exist. "
                                f"Name-similar candidates: {', '.join(sugg[:3])} "
                                f"(but pick by FUNCTION, not by name similarity).\n"
                            )
                        else:
                            rejection_feedback += f"  - '{func_name}' does not exist.\n"

                    if _dk:
                        all_prims = sorted(_dk.get_control_primitives() or [])
                        prim_line = (
                            f"  - Control primitives ({len(all_prims)}): "
                            f"{', '.join(all_prims)}"
                        )
                    else:
                        prim_line = "  - Control primitives: (DomainKnowledge unavailable)"

                    sorted_skills = sorted(s for s in available_skills if s)
                    skill_line = (
                        f"  - Existing skills ({len(sorted_skills)}, "
                        f"showing up to 30): "
                        f"{', '.join(sorted_skills[:30])}"
                    )

                    rejection_feedback += (
                        "\nYou may ONLY call names from these explicit lists:\n"
                        f"{prim_line}\n"
                        f"{skill_line}\n"
                        "  - Functions you define locally in the same file.\n"
                        "Do NOT invent new helper function names. If no "
                        "existing callable does what you need, inline the "
                        "logic directly rather than calling a non-existent "
                        "helper.\n"
                    )
                    critique_for_attempt = rejection_feedback + "\n" + enhanced_critique
                else:
                    # All retries exhausted — mark as failed
                    self.logger.error(
                        f"\033[31m[OptimizerCallback] Function reference validation failed (after {max_ref_retries + 1} attempts): "
                        f"undefined={undefined}\033[0m"
                    )
                    return {
                        "new_code": "",
                        "success": False,
                        "reason": f"Function reference validation failed: undefined functions {undefined}",
                    }

            # Convert return format
            if result and isinstance(result, dict):
                return {
                    "new_code": result.get("new_code", ""),
                    "success": result.get("success", False),
                    "reason": result.get("reason", result.get("error", "")),
                    "skipped": result.get("skipped", False),
                }
            else:
                return {
                    "new_code": "",
                    "success": False,
                    "reason": "Invalid result from quick_optimize_skill",
                }
        except Exception as e:
            self.logger.error(f"[OptimizerCallback] Optimization failed: {e}")
            return {
                "new_code": "",
                "success": False,
                "reason": str(e),
            }

    def _on_skill_optimized_callback(
        self,
        skill_name: str,
        fb: 'OptimizationForwardFeedback',
    ) -> None:
        """
        Callback after a successful optimization — records to detailed_logs

        Args:
            skill_name: skill name
            fb: OptimizationForwardFeedback object
        """
        try:
            # Generate code diff
            code_diff = None
            if fb.new_code and fb.old_code:
                code_diff = self._generate_unified_diff(fb.old_code, fb.new_code, skill_name)

            # Fetch skill node info
            node = self.skill_graph_manager.get_node(skill_name) if self.skill_graph_manager else None
            version_before = len(node.versions) - 1 if node and node.versions else 0
            version_after = len(node.versions) if node and node.versions else 1

            # Use actual fix_target from Phase 1 analysis
            fix_target = FixTargetType.CALLEE_FIX
            if hasattr(self, '_two_phase_engine') and hasattr(self._two_phase_engine, '_current_fix_targets'):
                fix_target = self._two_phase_engine._current_fix_targets.get(
                    skill_name, FixTargetType.CALLEE_FIX)

            # Capture trigger context (task/error/llm) on the success path
            # so opt_*.json records what Phase 1 actually saw when it
            # produced the feedback. Without this, hallucinated Phase 1
            # suggestions can't be diagnosed post-hoc.
            self.optimization_tracker.record_optimization(
                skill_name=skill_name,
                error_category=ErrorCategory.UNKNOWN,  # Category doesn't matter on success
                error_pattern="optimization_success",
                strategy_used="two_phase_pipeline",
                fix_target=fix_target,
                successful=True,
                version_before=version_before,
                version_after=version_after,
                code_before=fb.old_code,
                code_after=fb.new_code,
                code_diff=code_diff,
                # Record changes as feedback (Phase 1 gradient directions)
                feedback_content="\n".join(fb.changes_made) if fb.changes_made else None,
                # Trigger context (stashed on fb by engine.py before callback)
                task=fb.task,
                error_message=getattr(fb, 'error_message', None),
                error_stack=getattr(fb, 'error_stack', None),
                llm_prompt=getattr(fb, 'llm_prompt', None),
                llm_response=getattr(fb, 'llm_response', None),
                execution_context=getattr(fb, 'execution_context', None),
                value_at_optimization=node.value_function if node else None,
            )

            self.logger.info(f"\033[32m[OptimizationLogger] ✓ Recorded successful optimization: {skill_name}\033[0m")

        except Exception as e:
            self.logger.warning(f"[OptimizationLogger] Failed to record successful optimization: {skill_name}, error={e}")

    def _on_optimization_failed_callback(
        self,
        skill_name: str,
        fb: 'OptimizationForwardFeedback',
    ) -> None:
        """
        Callback after a failed optimization — records to detailed_logs

        Args:
            skill_name: skill name
            fb: OptimizationForwardFeedback object
        """
        try:
            # Fetch skill node info
            node = self.skill_graph_manager.get_node(skill_name) if self.skill_graph_manager else None
            version_before = len(node.versions) if node and node.versions else 0

            # Use actual fix_target from Phase 1 analysis
            fix_target = FixTargetType.CALLEE_FIX
            if hasattr(self, '_two_phase_engine') and hasattr(self._two_phase_engine, '_current_fix_targets'):
                fix_target = self._two_phase_engine._current_fix_targets.get(
                    skill_name, FixTargetType.CALLEE_FIX)

            # Phase 13.B-9: same trigger-context capture as success path.
            self.optimization_tracker.record_optimization(
                skill_name=skill_name,
                error_category=ErrorCategory.UNKNOWN,
                error_pattern="optimization_failed",
                strategy_used="two_phase_pipeline",
                fix_target=fix_target,
                successful=False,
                version_before=version_before,
                code_before=fb.old_code if fb.old_code else (node.code if node else None),
                feedback_content="\n".join(fb.changes_made) if fb.changes_made else None,
                # Trigger context (stashed on fb by engine.py before callback)
                task=fb.task,
                error_message=getattr(fb, 'error_message', None),
                error_stack=getattr(fb, 'error_stack', None),
                llm_prompt=getattr(fb, 'llm_prompt', None),
                llm_response=getattr(fb, 'llm_response', None),
                execution_context=getattr(fb, 'execution_context', None),
                failure_reason="Optimization did not produce valid code",
                value_at_optimization=node.value_function if node else None,
            )

            self.logger.info(f"\033[33m[OptimizationLogger] ✗ Recorded failed optimization: {skill_name}\033[0m")

        except Exception as e:
            self.logger.warning(f"[OptimizationLogger] Failed to record failed optimization: {skill_name}, error={e}")

    def on_task_completed(
        self,
        task: str,
        success: bool,
        completing_skills: List[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified handler for task completion (delegates to TwoPhaseOptimizationEngine)

        This is the core method for Experimental Skill lifecycle management.
        Different handling depending on whether the task succeeded:
        - Success: verify the new skill, trigger refactor, clean up unused experimental skills
        - Failure: clean up all experimental skills produced by the task
        - Preflight failure: skip experimental skill handling (code-gen/interface issue)

        Args:
            task: task name
            success: whether the task succeeded
            completing_skills: list of skills that completed the task
            error_type: failure type; "preflight" indicates a code-gen/interface issue

        Returns:
            Dict[str, Any]: processing result
        """
        if self._two_phase_engine and hasattr(self._two_phase_engine, 'on_task_completed'):
            return self._two_phase_engine.on_task_completed(
                task=task,
                success=success,
                completing_skills=completing_skills or [],
                error_type=error_type,
            )
        else:
            # Fallback: if the engine lacks this method, log a warning and return an empty result
            self.logger.warning(
                f"[Optimizer] on_task_completed called but _two_phase_engine doesn't have the method"
            )
            return {
                "verified_skills": [],
                "cleaned_skills": [],
                "refactor_results": [],
                "skipped_skills": [],
            }

    def optimize_skills_two_phase(
        self,
        skills_to_optimize: List[str],
        current_task: str = None,
        current_context: str = None,
        current_state: Dict[str, Any] = None,
        current_error: str = None,
        current_critique: str = None,
        momentum_window: int = 5,
        skill_execution_results: Dict[str, Dict[str, Any]] = None,
        skill_events_unreliable: bool = False,  # Whether the event mechanism is unreliable
        quality_metrics: Optional[Dict[str, Any]] = None,  # Critic quality metrics
        chat_log: str = "",  # Chat log containing diagnostic messages from skill execution
    ) -> Dict[str, Any]:
        """
        Two-phase optimization flow:
        1. Top-down analysis and feedback propagation: use PureReflection to analyze failure causes
        2. Bottom-up optimization: starting from leaf nodes, optimize upward

        Args:
            skills_to_optimize: list of skills to optimize
            current_task: current task
            current_context: current context
            current_state: current environment state
            current_error: current execution error
            current_critique: current critique
            momentum_window: momentum window size (considers the last n optimization runs)
            skill_execution_results: per-skill execution results (verified by state changes)
                format: {skill_name: {success: bool, effect_verification: {...}}}
            skill_events_unreliable: whether the event mechanism is unreliable (True when there are no skillStart/skillEnd events)
            chat_log: Chat log from onChat events; contains diagnostic messages like
                "I need at least a stone_pickaxe to mine deepslate_iron_ore!"

        Returns:
            Dict[str, Any]: optimization result
        """
        # Log chat_log if present for debugging
        if chat_log:
            self.logger.info(f"\033[36m[Two-Phase Optimize] Chat log available ({len(chat_log)} chars)\033[0m")

        # Use the two-phase optimization engine
        self.logger.info(f"\033[36m[Two-Phase Optimize] Using TwoPhaseOptimizationEngine\033[0m")
        engine_result = self._two_phase_engine.optimize(
            skills_to_optimize=skills_to_optimize,
            current_task=current_task,
            current_context=current_context,
            current_state=current_state,
            current_error=current_error,
            current_critique=current_critique,
            momentum_window=momentum_window,
            skill_execution_results=skill_execution_results,
            skill_events_unreliable=skill_events_unreliable,
            quality_metrics=quality_metrics,  # pass Critic quality metrics
            chat_log=chat_log,  # pass chat log
        )

        # Convert to return value format
        return {
            "success": engine_result.success,
            "session_id": engine_result.session_id,
            "results": engine_result.skill_results,
            "skills_optimized": engine_result.skills_optimized,
            "skills_skipped": engine_result.skills_skipped,
            "skills_failed": engine_result.skills_failed,
            "error": engine_result.error_message,
            "duration_seconds": engine_result.duration_seconds,
        }

    def _get_recent_optimization_history(
        self,
        skill_name: str,
        window_size: int = 5,
    ) -> str:
        """
        Get the recent optimization history (used by the momentum mechanism)

        Args:
            skill_name: skill name
            window_size: window size

        Returns:
            str: summary of recent optimization history
        """
        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return "No optimization history available."

        # Collect recent feedback and optimization suggestions
        recent_items = []

        # Recent items from feedback
        all_feedbacks = node.gradients.feedback
        for i in range(len(all_feedbacks) - 1, max(-1, len(all_feedbacks) - window_size - 1), -1):
            if i >= 0:
                recent_items.append(f"Feedback: {all_feedbacks[i].content[:200]}...")

        # Recent items from optimization_suggestions
        all_suggestions = node.gradients.optimization_suggestions
        for i in range(len(all_suggestions) - 1, max(-1, len(all_suggestions) - window_size - 1), -1):
            if i >= 0:
                recent_items.append(f"Suggestion: {all_suggestions[i].content[:200]}...")

        # Recent changes from versions
        recent_versions = node.versions[-window_size:] if node.versions else []
        for version in recent_versions:
            if version.change_log:
                recent_items.append(f"Version {version.version}: {version.change_log[:200]}...")

        if not recent_items:
            return "No recent optimization history available."

        return "\n".join(recent_items[:window_size])
