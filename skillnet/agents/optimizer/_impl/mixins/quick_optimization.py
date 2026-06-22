"""
QuickOptimizationMixin - Quick optimization pipeline methods.

Extracted from optimizer_impl.py for better modularity.

Methods included (8):
- _collect_optimization_feedback: Collect all feedback for optimization
- _build_optimization_prompt: Build the LLM prompt messages for optimization
- _invoke_optimization_llm: Invoke the LLM and parse the optimization response
- _validate_optimization_result: Validate the LLM optimization result
- quick_optimize_skill: Lightweight optimization entry point
- _validate_code_completeness: Validate code completeness (truncation detection)
- _validate_function_logic_completeness: Validate function name vs code logic consistency
- _log_validation_diagnostics: Log detailed code validation diagnostics
"""

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain.schema import HumanMessage, SystemMessage

from skillnet.agents.optimizer.feedback import (
    SkillFeedback,
)

from skillnet.agents.optimizer.validators import (
    validate_code_completeness,
    validate_function_implementation,
    check_naming_conflicts,
)

from skillnet.agents.optimizer.core import (
    build_current_state_info,
    extract_issues_from_feedbacks,
    check_requirements_addressed,
)

_HAS_PRECONDITION_CHECKER = False
try:
    from skillnet.agents.planning.precondition_checker import (
        PreconditionChecker,
        EnvironmentalFeedback,
        infer_environmental_feedback_from_error,
    )
    _HAS_PRECONDITION_CHECKER = True
except ImportError:
    pass

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer


def _format_validation_rejection_hint(result: Dict[str, Any]) -> str:
    """Render a validator-rejection result into a retry hint for the LLM.

    The validator's result dict carries the rejection reason in three layers:
      - 'error':       generic class name (e.g. "Invalid item names in
                       optimized code")
      - 'item_name_errors' / 'validation_errors': specific instances (the
                       actual names / lines that violated the rule)
      - 'retry_hint':  optional free-form suggestion baked in by some
                       validators (e.g. the require()-rejection path)

    Without the specific instances, the retry LLM only sees the rule name
    and tends to regenerate the same violating code — observed in
    ckpt_diamond_postG0_a iter 2 lines 1463-1479 where Qwen3-Coder emitted
    `mcData.itemsByName['planks']` on retry after the first attempt was
    rejected for ambiguous item name 'planks'. Including the specific
    error list drops broken-rate from 8/10 to 0/10 (Fisher p=0.0007).
    """
    error_msg = result.get("error", "unknown")
    parts = [f"**PREVIOUS ATTEMPT REJECTED**: {error_msg}"]

    for field in ("item_name_errors", "validation_errors"):
        items = result.get(field)
        if items and isinstance(items, list):
            label = field.replace("_", " ")
            parts.append(f"Specific {label}:")
            for it in items[:5]:
                parts.append(f"  - {it}")

    retry_hint = result.get("retry_hint", "")
    if retry_hint:
        parts.append(f"Hint: {retry_hint}")

    return "\n".join(parts)


class QuickOptimizationMixin:
    """Quick optimization pipeline: collect feedback, build prompt, invoke LLM, validate result."""

    def _collect_optimization_feedback(
        self,
        skill_name: str,
        node,
        current_task: str = None,
        current_error: str = None,
        current_critique: str = None,
        current_state: Dict[str, Any] = None,
        enhanced_feedback: List[SkillFeedback] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
        chat_log: str = "",
        current_context: str = None,
    ) -> Dict[str, Any]:
        """
        Collect all feedback for optimization: subgraph feedback, semantic mismatch,
        overclaim detection, constraint feedback, environmental feedback, and state info.

        Returns:
            Dict with keys: root_feedbacks, child_feedbacks, feedbacks, feedback_text,
            semantic_mismatch_info, overclaim_result, optimization_subgraph_skills,
            environmental_feedback, current_state_info, feedback_summary, is_wrapper,
            semantic_analysis.
        """
        # Collect feedback (FIX: pass current_task for context isolation)
        feedbacks = self.get_feedback_for_subgraph(skill_name, max_depth=2, current_task=current_task)
        root_feedbacks = feedbacks.get(skill_name, [])

        # Merge enhanced feedback (if any)
        if enhanced_feedback:
            root_feedbacks = list(root_feedbacks) + enhanced_feedback

        # ===== NEW: Layer 1 underlying library error detection =====
        # Build feedback_text for detection
        # Include chat_log which contains diagnostic messages like
        # "I need at least a stone_pickaxe to mine deepslate_iron_ore! Skip it!"
        feedback_text = ""
        if current_error:
            feedback_text += f"Error: {current_error}\n"
        if chat_log:
            # Chat log often contains crucial diagnostic information
            feedback_text += f"Chat Log:\n{chat_log}\n"
        if current_critique:
            feedback_text += f"Critique: {current_critique}\n"
        for fb in root_feedbacks:
            if fb.content:
                feedback_text += fb.content + "\n"

        # ===== Layer 2 parameter semantic mismatch detection =====
        semantic_mismatch_info = None
        semantic_analysis = None
        try:
            skill_executions = {}
            if node.statistics and node.statistics.execution_traces:
                latest_trace = node.statistics.execution_traces[-1]
                if latest_trace.call_stack:
                    skill_executions[skill_name] = {
                        "call_stack": latest_trace.call_stack,
                        "js_args": latest_trace.call_args,
                        "status": "success" if latest_trace.success else "error",
                    }

            if skill_executions or feedback_text:
                semantic_analysis = self._analyze_feedback_for_edits(
                    feedback=root_feedbacks,
                    current_error=current_error,
                    current_critique=current_critique,
                    skill_name=skill_name,
                    skill_executions=skill_executions,
                )

                if semantic_analysis and semantic_analysis.get("issues"):
                    for issue in semantic_analysis["issues"]:
                        if issue[0] == "semantic_mismatch":
                            semantic_mismatch_info = issue[2] if len(issue) > 2 else None
                            if semantic_mismatch_info:
                                self.logger.warning(
                                    f"\033[33m[Quick Optimize] Detected parameter semantic mismatch: "
                                    f"{semantic_mismatch_info.get('description', 'unknown')}\033[0m"
                                )
        except Exception as e:
            self.logger.debug(f"[Quick Optimize] Semantic analysis failed (non-critical): {e}")

        # ===== Layer 3 Overclaim detection =====
        overclaim_result = None
        try:
            from skillnet.agents.optimizer.validators import OverclaimDetector

            skill_code = node.code if node else ""
            skill_parameters = node.parameters if node and hasattr(node, 'parameters') else {}

            if skill_code and skill_parameters:
                _dk = getattr(self, '_domain_knowledge', None)
                _dk_fns = _dk.get_known_functions() if _dk else None
                overclaim_detector = OverclaimDetector(
                    logger=self.logger,
                    concatenation_patterns=_dk_fns.get("concatenation_patterns") if _dk_fns else None,
                    pattern_supported_values=_dk_fns.get("pattern_supported_values") if _dk_fns else None,
                    semantic_contexts=_dk_fns.get("overclaim_semantic_contexts") if _dk_fns else None,
                )
                overclaim_result = overclaim_detector.detect(
                    func_name=skill_name,
                    func_code=skill_code,
                    parameters=skill_parameters
                )

                if overclaim_result.has_overclaim and overclaim_result.confidence >= 0.7:
                    self.logger.warning(
                        f"\033[33m[Quick Optimize] Detected overclaim design issue: {skill_name}\033[0m"
                    )
                    self.logger.warning(
                        f"\033[33m[Quick Optimize] Problematic parameter: {overclaim_result.problematic_param}\033[0m"
                    )
                    self.logger.warning(
                        f"\033[33m[Quick Optimize] Evidence: {overclaim_result.evidence}\033[0m"
                    )
        except Exception as e:
            self.logger.debug(f"[Quick Optimize] Overclaim detection failed (non-critical): {e}")

        # ===== Constraint-violation detection and feedback injection =====
        if current_critique and node.children:
            constraint_fb = self._extract_constraint_feedback_from_critique(
                skill_name=skill_name,
                current_critique=current_critique,
                children=list(node.children),
                current_task=current_task,
            )
            if constraint_fb:
                root_feedbacks = list(root_feedbacks) + [constraint_fb]
                self._add_feedback_to_history(constraint_fb)
                self.logger.info(
                    f"\033[33m[Constraint→Feedback] Injecting constraint feedback into root_feedbacks\033[0m"
                )

        # Collect feedback for child skills
        child_feedbacks = {}
        for child_name in node.children:
            if child_name in feedbacks:
                child_feedbacks[child_name] = feedbacks[child_name]

        optimization_subgraph_skills = {skill_name}

        # collect environmental feedback (to guide the LLM in adding fallback logic)
        environmental_feedback = None
        if _HAS_PRECONDITION_CHECKER and current_state and current_state.get("biome"):
            try:
                skill_preconditions = node.preconditions if hasattr(node, 'preconditions') else []
                if skill_preconditions:
                    checker = PreconditionChecker()
                    environmental_feedback = checker.collect_environmental_feedback(
                        preconditions=skill_preconditions,
                        current_state=current_state,
                    )
                    if environmental_feedback:
                        self.logger.info(
                            f"\033[36m[Quick Optimize] Detected environment mismatch (from preconditions): "
                            f"biome={environmental_feedback.current_biome}, "
                            f"missing={environmental_feedback.missing_resources}\033[0m"
                        )

                if not environmental_feedback and current_error and infer_environmental_feedback_from_error:
                    environmental_feedback = infer_environmental_feedback_from_error(
                        error_message=current_error,
                        current_state=current_state,
                    )
                    if environmental_feedback:
                        self.logger.info(
                            f"\033[36m[Quick Optimize] Detected environment mismatch (from error): "
                            f"biome={environmental_feedback.current_biome}, "
                            f"missing={environmental_feedback.missing_resources}\033[0m"
                        )
            except Exception as e:
                self.logger.warning(f"[Quick Optimize] Environmental feedback collection failed: {e}")

        # P2 Phase A: build current state info using the extracted pure function
        current_state_info = build_current_state_info(
            current_task=current_task,
            current_context=current_context,
            current_state=current_state,
            current_error=current_error,
            current_critique=current_critique,
            environmental_feedback=environmental_feedback,  # environmental feedback
            quality_metrics=quality_metrics,  # Critic quality metrics
            domain_knowledge=getattr(self, '_domain_knowledge', None),
        )

        # Format feedback
        feedback_summary = self._format_feedbacks_for_llm(root_feedbacks)

        # ========== Phase 4: Wrapper skill protection ==========
        line_count = len(node.code.strip().split('\n'))
        is_wrapper = self._bloat_checker._is_wrapper_skill(node.code, line_count)

        return {
            "root_feedbacks": root_feedbacks,
            "child_feedbacks": child_feedbacks,
            "feedbacks": feedbacks,
            "feedback_text": feedback_text,
            "semantic_mismatch_info": semantic_mismatch_info,
            "semantic_analysis": semantic_analysis,
            "overclaim_result": overclaim_result,
            "optimization_subgraph_skills": optimization_subgraph_skills,
            "environmental_feedback": environmental_feedback,
            "current_state_info": current_state_info,
            "feedback_summary": feedback_summary,
            "is_wrapper": is_wrapper,
        }

    def _build_optimization_prompt(
        self,
        skill_name: str,
        node,
        feedback_data: Dict[str, Any],
        momentum_window: int = 5,
        current_error: str = None,
        current_task: str = None,
        current_context: str = None,
        current_state: Dict[str, Any] = None,
    ) -> list:
        """
        Build the LLM prompt messages for optimization.

        Returns:
            List of LangChain messages ready for LLM invocation.
        """
        root_feedbacks = feedback_data["root_feedbacks"]
        child_feedbacks = feedback_data["child_feedbacks"]
        overclaim_result = feedback_data["overclaim_result"]
        current_state_info = feedback_data["current_state_info"]
        feedback_summary = feedback_data["feedback_summary"]
        is_wrapper = feedback_data["is_wrapper"]

        # Log details of the feedback used
        # Two independent input channels can feed the optimizer:
        # (a) root_feedbacks   — text-feedback from feedback_history (legacy path)
        # (b) structured gradients — Phase 1's Gradient objects (P3 path, passed via
        # skill_delta and stored in _current_phase1_gradients)
        # Only warn when BOTH are empty; otherwise report whichever channel is active.
        structured_gradients = self._current_phase1_gradients or []
        self.logger.info(f"\033[36m[Quick Optimize] ========== Feedback details ==========\033[0m")
        if root_feedbacks:
            self.logger.info(f"\033[36m[Quick Optimize] Current skill '{skill_name}' has {len(root_feedbacks)} feedbacks:\033[0m")
            for i, fb in enumerate(root_feedbacks, 1):
                self.logger.info(f"\033[36m[Quick Optimize]   Feedback {i} ({fb.feedback_type}): {fb.content[:200]}...\033[0m")
        elif structured_gradients:
            grad_types = [
                g.gradient_type.value if hasattr(g.gradient_type, 'value') else str(g.gradient_type)
                for g in structured_gradients
            ]
            self.logger.info(
                f"\033[36m[Quick Optimize] Current skill '{skill_name}' via structured gradients: "
                f"{len(structured_gradients)} items ({', '.join(grad_types)})\033[0m"
            )
        else:
            self.logger.warning(
                f"\033[33m[Quick Optimize] Current skill '{skill_name}' has no input "
                f"(feedback_history AND structured gradients are both empty)\033[0m"
            )

        child_feedback_summary = ""
        for child_name, child_fb in child_feedbacks.items():
            child_feedback_summary += f"\n\nChild skill '{child_name}':\n"
            child_feedback_summary += self._format_feedbacks_for_llm(child_fb)

        # Generate external skill warning
        external_skills_warning = self._generate_external_skills_warning(skill_name)

        # Use Phase 1 gradients directly instead of redundant LLM call
        feedback_requirements = []
        for g in (self._current_phase1_gradients or []):
            fix = getattr(g, 'suggested_fix', '') or ''
            direction = getattr(g, 'direction', '') or ''
            grad_type = g.gradient_type.value if hasattr(g, 'gradient_type') else 'unknown'
            magnitude = getattr(g, 'magnitude', 0.0)

            req_text = fix.strip() or direction.strip()
            if req_text:
                prefix = f"[{grad_type}, priority={magnitude:.1f}]"
                feedback_requirements.append(f"{prefix} {req_text}")
        feedback_requirements = feedback_requirements[:5]
        # Fallback: if Phase 1 provided no gradients, use stored requirements
        if not feedback_requirements:
            feedback_requirements = self._current_feedback_requirements or []

        # P2 Phase A: extract issues and fix suggestions from feedback using the extracted pure function
        issues, suggested_fixes = extract_issues_from_feedbacks(
            feedbacks=root_feedbacks,
            logger=self.logger,
        )

        # Build the layered context
        edit_context = self._build_edit_context(
            skill_name=skill_name,
            code=node.code,
            issues=issues,
            suggested_fixes=suggested_fixes,
            feedback_requirements=feedback_requirements or self._current_feedback_requirements or [],
        )

        # Model-aware prompt template selection
        from skillnet.agents.optimizer.prompts import OptimizerPromptLoader
        from skillnet.core.model_profile import detect_model_profile
        _profile = detect_model_profile(getattr(self, '_model_name', ''))
        if _profile.use_structured_output:
            template = OptimizerPromptLoader.load("code_optimization")
        else:
            template = OptimizerPromptLoader.load("code_optimization_compact")

        # Build the wrapper warning
        wrapper_warning = ""
        if is_wrapper:
            wrapper_warning = '''
## ⚠️ WRAPPER SKILL WARNING ⚠️
This is a WRAPPER SKILL (< 25 lines). It should REMAIN a simple wrapper.
- DO NOT add retry loops, fallback logic, or helper functions
- The optimized code should be < 50 lines
- Keep it simple: just call the underlying skill with correct parameters
'''

        # Build the overclaim warning (principle-based, no code templates)
        overclaim_warning = ""
        if overclaim_result and overclaim_result.has_overclaim and overclaim_result.confidence >= 0.7:
            overclaim_warning = f'''
## ⚠️ OVERCLAIM DESIGN ISSUE DETECTED ⚠️
The function declares support for parameter values it CANNOT actually handle.

**Problem Parameter**: `{overclaim_result.problematic_param}`
**Evidence**: {overclaim_result.evidence}

Analyze why the function fails for certain parameter values and fix accordingly.
'''

        # build existing skill name list (for naming-conflict guidance)
        # Compact template doesn't use these — skip to save tokens
        if _profile.use_structured_output:
            existing_skill_names_list = sorted(self.skill_graph_manager.get_all_skill_names(include_task_specific=True))
            existing_skill_names_str = "\n".join(f"  - {name}" for name in existing_skill_names_list[:50])
            if len(existing_skill_names_list) > 50:
                existing_skill_names_str += f"\n  ... and {len(existing_skill_names_list) - 50} more skills"
        else:
            existing_skill_names_str = ""

        # Load control primitives context (full JS implementations)
        dk = getattr(self, '_domain_knowledge', None)
        ctx = dk.load_control_primitive_context(for_optimizer=True) if dk else []
        control_primitives_ctx = "\n\n".join(ctx)

        # Generate composable skills context (guide LLM to call existing skills instead of inlining)
        composable_skills_context = self._generate_composable_skills_context(skill_name)

        system_prompt, human_prompt = template.format(
            skill_name=skill_name,
            edit_context=edit_context,
            skill_code=node.code,
            control_primitives_context=control_primitives_ctx,
            wrapper_warning=wrapper_warning,
            overclaim_warning=overclaim_warning,  # overclaim design issue warning
            composable_skills_context=composable_skills_context,
            existing_skill_names=existing_skill_names_str,  # Fix 12b: existing skill name list
            skill_description=node.description,
            feedback_summary=feedback_summary,
            child_feedback_summary=child_feedback_summary,
            current_state_info=current_state_info if current_state_info else "",
            momentum_window=momentum_window,
            optimization_history=self._get_recent_optimization_history(skill_name, momentum_window),
            total_executions=node.statistics.total_executions,
            success_rate=f"{node.statistics.success_rate:.2%}",
            failed_executions=node.statistics.failed_executions,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        # Store feedback_requirements for use by _validate_optimization_result
        self._last_feedback_requirements = feedback_requirements

        return messages

    def _parse_delimited_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse delimited-section response format (for weak models).

        Expected format:
            === ISSUES ===
            ...
            === OPTIMIZED_CODE ===
            ```javascript
            ...code...
            ```
            === CHANGE_SUMMARY ===
            ...
            === REQUIREMENTS_ADDRESSED ===
            ...
        """
        # Section header: === WORD === (all-caps word distinguishes from JS === operator)
        _SECTION_BOUNDARY = r'===\s*[A-Z_]+\s*==='

        # Extract OPTIMIZED_CODE section
        code_section_match = re.search(
            r'===\s*OPTIMIZED_CODE\s*===(.*?)(?:' + _SECTION_BOUNDARY + r'|$)',
            response, re.DOTALL,
        )
        if not code_section_match:
            return None

        code_text = code_section_match.group(1).strip()

        # Extract code from fenced block if present
        from skillnet.utils.code_block import strip_code_block_markers
        optimized_code = strip_code_block_markers(code_text, "javascript")
        # Handle unclosed fenced blocks (model truncation)
        if optimized_code.startswith("```"):
            optimized_code = re.sub(r'^```\w*\s*\n?', '', optimized_code)
        if not optimized_code or not optimized_code.strip():
            return None

        # Extract other sections (best-effort)
        issues = []
        issues_match = re.search(
            r'===\s*ISSUES\s*===(.*?)(?:' + _SECTION_BOUNDARY + r'|$)',
            response, re.DOTALL,
        )
        if issues_match:
            for line in issues_match.group(1).strip().split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    # Parse "1. [type] description"
                    type_match = re.match(r'\d+\.\s*\[(\w+)\]\s*(.*)', line)
                    if type_match:
                        issues.append({"type": type_match.group(1), "description": type_match.group(2)})
                    else:
                        issues.append({"type": "unknown", "description": line})

        change_summary = ""
        summary_match = re.search(
            r'===\s*CHANGE_SUMMARY\s*===(.*?)(?:' + _SECTION_BOUNDARY + r'|$)',
            response, re.DOTALL,
        )
        if summary_match:
            change_summary = summary_match.group(1).strip()

        requirements_addressed = []
        req_match = re.search(
            r'===\s*REQUIREMENTS_ADDRESSED\s*===(.*?)(?:' + _SECTION_BOUNDARY + r'|$)',
            response, re.DOTALL,
        )
        if req_match:
            for line in req_match.group(1).strip().split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    parts = line.split('|', 1)
                    how = re.sub(r'^\d+\.\s*', '', parts[0]).strip()
                    loc = parts[1].strip() if len(parts) > 1 else ""
                    requirements_addressed.append({
                        "requirement_index": len(requirements_addressed) + 1,
                        "how_addressed": how,
                        "code_location": loc,
                    })

        return {
            "issues": issues,
            "optimized_code": optimized_code,
            "change_summary": change_summary,
            "requirements_addressed": requirements_addressed,
        }

    def _invoke_optimization_llm(
        self,
        messages: list,
        skill_name: str,
        current_task: str = None,
    ) -> Dict[str, Any]:
        """
        Invoke the LLM and parse the optimization response.

        Returns:
            Dict with 'result' (parsed JSON dict), 'optimized_code' (str), and 'response' (raw str),
            or a dict with 'error' key on failure.
        """
        try:
            response_obj = self._invoke_llm_with_stats(
                messages=messages,
                process_type="optimization",
                function_name="quick_optimize_skill",
                task=current_task,
                skill_name=skill_name,
                metadata={"mode": "quick"},
            )
            response = response_obj.content

            # Model-aware parsing: delimited sections for weak models, JSON for strong
            from skillnet.core.model_profile import detect_model_profile
            profile = detect_model_profile(getattr(self, '_model_name', ''))

            if not profile.use_structured_output:
                # Try delimited format first (compact prompt output)
                result = self._parse_delimited_response(response)
                if result is not None:
                    optimized_code = result["optimized_code"]
                    optimized_code = self._sanitize_python_to_js(optimized_code)
                    optimized_code = self._format_generated_code(optimized_code)
                    return {
                        "result": result,
                        "optimized_code": optimized_code,
                        "response": response,
                    }
                # Fallback to JSON parse if delimited failed
                self.logger.debug("[Quick Optimize] Delimited parse failed, falling back to JSON")

            # Parse JSON (using robust parsing)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = self._robust_json_parse(json_match.group(), "LLM Optimization")
                if result is None:
                    # Log a snippet of the malformed response so future diagnostics
                    # don't need to add ad-hoc print statements. Cap at 600 chars
                    # because gpt-5-mini can emit 13K+ chars on one line when the
                    # optimized_code field contains a large JS snippet (observed
                    # in v11 + v3.E runs at char 1979 and char 13889).
                    self.logger.warning(
                        f"\033[33m[Quick Optimize] JSON parse failed for {skill_name}; "
                        f"response_len={len(response)}, preview={response[:400]!r}\033[0m"
                    )
                    return {
                        "error": "Failed to parse optimization response JSON",
                        "raw_response": response,
                        # Mark as parse-failure so the caller's retry path can
                        # fire with an escape-aware hint instead of bailing out.
                        "parse_failure": True,
                    }

                optimized_code = result.get("optimized_code", "")
                if not optimized_code:
                    return {
                        "error": "No optimized code in response",
                        "raw_response": response,
                        "parse_failure": True,
                    }

                # Extract code (may be inside a code block)
                code_match = re.search(r'```(?:javascript|js)?\s*\n(.*?)\n```', optimized_code, re.DOTALL)
                if code_match:
                    optimized_code = code_match.group(1).strip()

                # Convert Python syntax to JavaScript syntax
                optimized_code = self._sanitize_python_to_js(optimized_code)

                # Unified formatting exit: format the code (prevents long single-line code)
                optimized_code = self._format_generated_code(optimized_code)

                return {
                    "result": result,
                    "optimized_code": optimized_code,
                    "response": response,
                }
            else:
                self.logger.warning(
                    f"\033[33m[Quick Optimize] No JSON object in response for {skill_name}; "
                    f"response_len={len(response)}, preview={response[:400]!r}\033[0m"
                )
                return {
                    "error": "Failed to parse quick optimization response",
                    "raw_response": response,
                    "parse_failure": True,
                }
        except Exception as e:
            return {"error": f"Quick optimization failed: {e}"}

    def _should_reject_for_consistency(self, consistency_check):
        """Decide whether to reject an optimization based on the consistency check.

        The consistency check is LLM-based: it returns `consistent` (bool) and a
        holistic `consistency_score` (0-1). We trust that judgment and reject only
        when the score falls below the configured threshold. We deliberately do
        NOT re-scan the LLM's free-text `conflicts` for keywords (the former
        CRITICAL_CONFLICT/UNADDRESSED_ISSUE keyword layer): words like
        "undefined"/"syntax" appear in almost any risk discussion, so prose
        substring matching false-rejected substantively-correct optimizations.
        The real defect classes are caught by dedicated validators, not here:
          - truncation/incompleteness -> _validate_code_completeness (runs earlier)
          - JS syntax errors          -> validate_code_syntax (at skill save)
          - undefined references       -> validate_function_references (Reference Check)

        Returns (reject: bool, reasons: list[str]).
        """
        if consistency_check.get("consistent", True):
            return False, []
        score = consistency_check.get("consistency_score", 0)
        if score < self.consistency_threshold:
            return True, [f"low score ({score:.2f} < {self.consistency_threshold})"]
        return False, []

    def _validate_optimization_result(
        self,
        llm_output: Dict[str, Any],
        skill_name: str,
        node,
        feedback_data: Dict[str, Any],
        current_error: str = None,
        current_task: str = None,
    ) -> Dict[str, Any]:
        """
        Validate the LLM optimization result: completeness, TDZ fixing, semantic consistency,
        error propagation, item names, requirements addressed, consistency check, naming
        conflicts, and code growth.

        Returns:
            Dict with 'success' key on success, or 'error' key on failure.
        """
        result = llm_output["result"]
        optimized_code = llm_output["optimized_code"]
        response = llm_output["response"]
        root_feedbacks = feedback_data["root_feedbacks"]
        optimization_subgraph_skills = feedback_data["optimization_subgraph_skills"]
        semantic_analysis = feedback_data.get("semantic_analysis")
        feedback_requirements = self._last_feedback_requirements

        # Validate code completeness (guard against truncation); pass LLM response for diagnostics
        code_validation = self._validate_code_completeness(optimized_code, node.code, llm_response=response)
        if not code_validation.get("complete", True):
            validation_errors = code_validation.get("errors", [])
            diagnostics = code_validation.get("diagnostics", {})

            # Log detailed diagnostic information
            self._log_validation_diagnostics(skill_name, diagnostics, validation_errors)

            # Check whether a TDZ issue can be auto-fixed
            if code_validation.get("can_fix_tdz", False):
                self.logger.info(f"\033[36m[Quick Optimize] TDZ issue detected; attempting auto-fix...\033[0m")
                fixed_code, was_fixed, fix_description = self._fix_tdz_issues(optimized_code)

                if was_fixed:
                    self.logger.info(f"\033[32m[Quick Optimize] TDZ auto-fix: {fix_description}\033[0m")
                    optimized_code = fixed_code

                    # Re-validate
                    code_validation = self._validate_code_completeness(optimized_code, node.code, llm_response=response)
                    if code_validation.get("complete", True):
                        self.logger.info(f"\033[32m[Quick Optimize] TDZ issue fixed successfully, code validation passed\033[0m")
                        # Continue with the rest of the logic (do not return an error)
                    else:
                        # Still issues after the fix
                        remaining_errors = code_validation.get("errors", [])
                        self.logger.error(f"\033[31m[Quick Optimize] Code validation failed (after TDZ fix): {', '.join(remaining_errors)}\033[0m")
                        return {
                            "error": "Code validation failed",
                            "validation_errors": remaining_errors,
                            "diagnostics": code_validation.get("diagnostics", {}),
                            "raw_response": response,
                            "should_retry": True
                        }
                else:
                    self.logger.warning(f"\033[33m[Quick Optimize] TDZ auto-fix failed: {fix_description}\033[0m")
                    self.logger.error(f"\033[31m[Quick Optimize] Code validation failed: {', '.join(validation_errors)}\033[0m")
                    return {
                        "error": "Code validation failed",
                        "validation_errors": validation_errors,
                        "diagnostics": diagnostics,
                        "raw_response": response,
                        "should_retry": True
                    }
            else:
                self.logger.error(f"\033[31m[Quick Optimize] Code validation failed: {', '.join(validation_errors)}\033[0m")
                # Note: quick_optimize_skill is single-skill optimization; retries are handled by the caller
                return {
                    "error": "Code validation failed",
                    "validation_errors": validation_errors,
                    "raw_response": response,
                    "should_retry": True  # Mark as retryable
                }

        # ===== semantic consistency validation =====
        semantic_validation = self._validate_function_logic_completeness(
            skill_name=skill_name,
            code=optimized_code,
            strict=False,  # Non-strict mode: log warnings only, do not reject
        )
        if not semantic_validation.get("semantic_match", True):
            warning_msg = semantic_validation.get("warning", "")
            self.logger.warning(f"\033[33m[Quick Optimize] Semantic validation warning: {warning_msg}\033[0m")
            self.logger.warning(
                f"\033[33m[Quick Optimize] Function '{skill_name}' may be missing required "
                f"{semantic_validation.get('detected_prefix', '')} operation logic\033[0m"
            )
            if semantic_validation.get("delegate_only"):
                delegate_target = semantic_validation.get("delegate_target", "unknown")
                self.logger.warning(
                    f"\033[33m[Quick Optimize] Detected pure-delegation pattern: only calls '{delegate_target}'\033[0m"
                )

        # ===== Error swallowing detection (warning only) =====
        from skillnet.agents.optimizer.validators.code_validator import (
            validate_error_propagation, validate_item_names,
        )
        err_prop_valid, err_prop_msg = validate_error_propagation(optimized_code)
        if not err_prop_valid:
            self.logger.warning(f"\033[33m[Quick Optimize] {err_prop_msg}\033[0m")

        # ===== Item name validation (reject on invalid names) =====
        _dk = getattr(self, '_domain_knowledge', None)
        _dk_fns = _dk.get_known_functions() if _dk else None
        _inv_names = _dk_fns.get("invalid_item_names") if _dk_fns else None
        _amb_names = _dk_fns.get("ambiguous_item_names") if _dk_fns else None
        _prims = _dk.get_control_primitives() if _dk else None
        item_names_valid, item_name_errors = validate_item_names(
            optimized_code, invalid_names=_inv_names, ambiguous_names=_amb_names,
            primitives=_prims,
        )
        if not item_names_valid:
            for err in item_name_errors:
                self.logger.error(f"\033[31m[Quick Optimize] {err}\033[0m")
            return {
                "error": "Invalid item names in optimized code",
                "item_name_errors": item_name_errors,
                "raw_response": response,
                "should_retry": True,
            }

        # Reject require("./skillName") — skills are global functions, not modules.
        # Qwen3 sometimes treats composable skills as Node.js modules (r32 bug).
        import re as _re
        _require_matches = _re.findall(r'require\s*\(\s*["\']\./', optimized_code)
        if _require_matches:
            self.logger.error(
                f"\033[31m[Quick Optimize] Rejected: code uses require() to import "
                f"skills ({len(_require_matches)} occurrences). "
                f"Skills are global functions, not modules.\033[0m"
            )
            return {
                "error": "Code uses require() to import skills — call them directly as global functions",
                "should_retry": True,
                "retry_hint": "Do NOT use require() for composable skills. Call directly: await skillName(bot, args)",
            }

        # P2 Phase A: check whether requirements are addressed using the extracted pure function
        # Skip for weak models — they generate correct code but
        # often omit the REQUIREMENTS_ADDRESSED metadata section, causing
        # false rejections. Consistency check (below) validates semantically.
        from skillnet.core.model_profile import detect_model_profile as _detect_mp
        _profile = _detect_mp(getattr(self, '_model_name', ''))
        if _profile.use_structured_output:
            requirements_addressed = result.get("requirements_addressed", [])
            should_reject, missing_requirements, _ = check_requirements_addressed(
                requirements_addressed=requirements_addressed,
                feedback_requirements=feedback_requirements or [],
                logger=self.logger,
            )
            if should_reject:
                return {
                    "error": "Optimization failed to address requirements",
                    "missing_requirements": missing_requirements,
                    "addressed_requirements": requirements_addressed,
                    "raw_response": response,
                    "should_retry": True
                }
        else:
            self.logger.info(
                "\033[36m[Quick Optimize] Skipping requirements self-report check "
                "(weak model — relying on consistency check)\033[0m"
            )

        # Check consistency (momentum mechanism)
        consistency_check = self._check_optimization_consistency(
            skill_name, optimized_code, last_k_feedbacks=5,
        )

        if not consistency_check.get("consistent", True):
            conflicts = consistency_check.get("conflicts", [])
            consistency_score = consistency_check.get("consistency_score", 0)
            self.logger.warning(f"\033[33m[Quick Optimize] Warning: optimization conflicts with recent feedback: {', '.join(conflicts)}\033[0m")
            self.logger.warning(f"\033[33m[Quick Optimize] Consistency score: {consistency_score:.2f}\033[0m")

            # Reject only on the LLM's holistic consistency score (see
            # _should_reject_for_consistency). No keyword re-scan of the LLM's
            # prose: that false-rejected correct optimizations; real defects are
            # caught by the dedicated completeness/syntax/reference validators.
            reject, rejection_reasons = self._should_reject_for_consistency(consistency_check)
            if reject:
                self.logger.error(
                    f"\033[31m[Quick Optimize] Rejecting optimization: {', '.join(rejection_reasons)}\033[0m"
                )
                return {
                    "error": "Optimization rejected due to conflicts",
                    "conflicts": conflicts,
                    "consistency_score": consistency_score,
                    "raw_response": response,
                    "should_retry": True  # Mark as retryable
                }

        # Analyze interface impact
        interface_impact = self._analyze_interface_impact(node.code, optimized_code)

        # Log before/after code comparison — delegate to _log_optimization_changes
        self._log_optimization_changes(skill_name, node.code, optimized_code, result)

        backpropagation_info = None

        # Sync feedback history and save
        self._sync_feedback_from_gradients(skill_name)
        self._save_feedback_history()

        # ===== Dead code detection: remove unused helper functions =====
        cleaned_code, removed_funcs = self._remove_unused_helper_functions(optimized_code, skill_name)
        if removed_funcs:
            self.logger.info(f"\033[36m[Dead Code] Removed unused functions: {removed_funcs}\033[0m")
            optimized_code = cleaned_code

        # ===== Fix 12: naming-conflict detection =====
        existing_skill_names = set(self.skill_graph_manager.get_all_skill_names(include_task_specific=True))
        # A skill's own code legitimately defines `function <skill_name>(...)`; only
        # redefining a DIFFERENT skill is a real shadowing conflict. Without this
        # exclusion, optimizing a skill flags its own definition as a conflict with
        # itself and the optimization is permanently rejected.
        existing_skill_names.discard(skill_name)
        no_conflict, conflict_messages = check_naming_conflicts(optimized_code, existing_skill_names)
        if not no_conflict:
            self.logger.error(f"\033[31m[Quick Optimize] Naming conflict detection failed:\033[0m")
            for msg in conflict_messages:
                self.logger.error(f"\033[31m  - {msg}\033[0m")
            return {
                "error": f"Naming conflict detected: {'; '.join(conflict_messages)}",
                "should_retry": True,
                "retry_hint": "avoid_shadowing_existing_skills",
            }

        # ===== P1: early code-line-count check (before saving) =====
        pre_analysis = {
            "tdz_check": {
                "valid": code_validation.get("complete", True),
                "can_fix_tdz": code_validation.get("can_fix_tdz", False),
            },
            "semantic_mismatch": (
                semantic_analysis.get("issues")
                if semantic_analysis
                else None
            ),
        }
        growth_valid, growth_reason = self._check_code_growth(
            node.code, optimized_code, skill_name,
            pre_analysis=pre_analysis,
            current_error=current_error,
        )
        if not growth_valid:
            self.logger.error(f"\033[31m[Quick Optimize] Code growth exceeded limit: {growth_reason}\033[0m")
            return {
                "error": f"Code growth exceeded: {growth_reason}",
                "should_retry": True,
                "retry_hint": "minimize_changes",
                "old_lines": len(node.code.split('\n')),
                "new_lines": len(optimized_code.split('\n')),
            }

        return {
            "success": True,
            "skill_name": skill_name,
            "old_code": node.code,
            "new_code": optimized_code,
            "change_summary": result.get("change_summary", "Code optimized"),
            "issues": result.get("issues", []),
            "backpropagation_info": backpropagation_info,
            "optimization_subgraph": list(optimization_subgraph_skills),
            "mode": "quick",
            "interface_impact": interface_impact,
            "consistency_check": consistency_check,
        }

    def quick_optimize_skill(
        self,
        skill_name: str,
        current_task: str = None,
        current_context: str = None,
        current_state: Dict[str, Any] = None,
        current_error: str = None,
        current_critique: str = None,
        momentum_window: int = 5,  # Consider the last N optimization runs
        enhanced_feedback: List[SkillFeedback] = None,  # Enhanced feedback generated by LLM analysis
        quality_metrics: Optional[Dict[str, Any]] = None,  # Critic quality metrics
        chat_log: str = "",  # Chat log from onChat events
    ) -> Dict[str, Any]:
        """
        Lightweight optimization: combine diagnosis and optimization into a single step

        Args:
            skill_name: skill name
            current_task: current task
            current_context: current context
            current_state: current environment state
            current_error: current execution error
            current_critique: current critique
            enhanced_feedback: enhanced feedback produced by LLM analysis (with specific fix suggestions)
            chat_log: v7.7 Chat log containing diagnostic messages (e.g., "I need at least a stone_pickaxe...")

        Returns:
            Dict[str, Any]: optimization result
        """
        if not self.skill_graph_manager.has_node(skill_name):
            return {"error": f"Skill '{skill_name}' not found"}

        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return {"error": f"Skill '{skill_name}' not found"}

        # [P0 Fix] defensive edge refresh: ensure edge relations are up-to-date
        self.skill_graph_manager.update_skill_dependencies(skill_name)
        # Re-fetch the node to get the updated children
        node = self.skill_graph_manager.get_node(skill_name)

        # FIX: clear class-level state variables to prevent contamination
        self._current_feedback_requirements = []
        self._current_llm_analysis = None

        self.logger.info(f"\033[36m[Quick Optimize] Starting lightweight optimization for skill: {skill_name}\033[0m")

        # Step 1: Collect all feedback
        feedback_data = self._collect_optimization_feedback(
            skill_name=skill_name,
            node=node,
            current_task=current_task,
            current_error=current_error,
            current_critique=current_critique,
            current_state=current_state,
            enhanced_feedback=enhanced_feedback,
            quality_metrics=quality_metrics,
            chat_log=chat_log,
            current_context=current_context,
        )

        is_wrapper = feedback_data["is_wrapper"]

        # ========== Phase 4: wrapper skill protection ==========
        if is_wrapper:
            print(f"\033[36m[Wrapper Protection] Detected {skill_name} is a wrapper skill ({len(node.code.split(chr(10)))} lines)\033[0m")
            print(f"\033[36m[Wrapper Protection] Will use the dedicated wrapper optimization logic\033[0m")
        # ========== End wrapper detection ==========

        # Check iterative gradient mode
        from skillnet.core.model_profile import detect_model_profile as _detect_profile
        _prof = _detect_profile(getattr(self, '_model_name', ''))
        all_gradients = list(self._current_phase1_gradients or [])
        use_iterative = _prof.iterative_gradients and len(all_gradients) > 1

        if use_iterative:
            return self._iterative_gradient_optimize(
                skill_name=skill_name,
                node=node,
                feedback_data=feedback_data,
                all_gradients=all_gradients,
                momentum_window=momentum_window,
                current_error=current_error,
                current_task=current_task,
                current_context=current_context,
                current_state=current_state,
            )

        # Single-shot path (default for gpt-5-mini)
        # Step 2: Build the LLM prompt
        messages = self._build_optimization_prompt(
            skill_name=skill_name,
            node=node,
            feedback_data=feedback_data,
            momentum_window=momentum_window,
            current_error=current_error,
            current_task=current_task,
            current_context=current_context,
            current_state=current_state,
        )

        # Step 3: Invoke the LLM
        llm_output = self._invoke_optimization_llm(
            messages=messages,
            skill_name=skill_name,
            current_task=current_task,
        )

        # If LLM invocation failed AND it's not a parse failure, bail.
        # Parse failures (parse_failure=True) fall through to the retry
        # path below — gpt-5-mini stochastically emits malformed JSON
        # when the optimized_code field contains a long JS snippet
        # (observed in v11 + v3.E runs: parse error at char 1979 and
        # char 13889 in two separate craftPickaxe/setupCraftingTable
        # calls). A retry with an escape-aware hint usually fixes it.
        if "error" in llm_output and not llm_output.get("parse_failure"):
            return llm_output

        # If parse failure: convert to retry-eligible result for the
        # downstream retry block.
        if llm_output.get("parse_failure"):
            result = {
                "success": False,
                "error": llm_output["error"],
                "raw_response": llm_output.get("raw_response", ""),
                "should_retry": True,
                "retry_hint": (
                    "Your previous response had INVALID JSON. Respond with ONLY a valid "
                    "JSON object. CRITICAL: when embedding JS code in any string field "
                    "(e.g. optimized_code), escape every backslash as \\\\, every double "
                    "quote as \\\", and every newline as \\n. Keep the JSON compact — "
                    "no markdown fences, no prose outside the object."
                ),
            }
        else:
            # Step 4: Validate the result
            result = self._validate_optimization_result(
                llm_output=llm_output,
                skill_name=skill_name,
                node=node,
                feedback_data=feedback_data,
                current_error=current_error,
                current_task=current_task,
            )

        # Retry once on validation failure (single-shot path).
        # Previously only the iterative path had retry. This caused 2 wasted
        # iterations in r31 when Qwen3 used "planks"/"sticks" instead of
        # "oak_planks"/"stick" — validator rejected, no retry, code unchanged.
        # v3-rev follow-up: also retry on JSON parse failure (parse_failure=True
        # path above) with an escape-aware hint.
        if not result.get("success") and result.get("should_retry"):
            error_msg = result.get("error", "unknown")
            self.logger.info(
                f"\033[33m[Single-Shot] Validation failed, retrying: "
                f"{error_msg[:80]}\033[0m"
            )
            augmented = dict(feedback_data)
            extra = "\n\n" + _format_validation_rejection_hint(result)
            # Write the rejection hint into `feedback_summary`, NOT
            # `feedback_text`. The prompt template reads only `feedback_summary`
            # (`code_optimization{,_compact}.txt`: `Feedback: {feedback_summary}`);
            # `feedback_text` is a dead field for prompt purposes — see
            # _build_optimization_prompt at line ~284 where feedback_summary is
            # the value that lands in the rendered prompt.
            augmented["feedback_summary"] = (
                augmented.get("feedback_summary", "") + extra
            )

            retry_msgs = self._build_optimization_prompt(
                skill_name=skill_name, node=node,
                feedback_data=augmented,
                momentum_window=momentum_window,
                current_error=current_error,
                current_task=current_task,
                current_context=current_context,
                current_state=current_state,
            )
            retry_out = self._invoke_optimization_llm(
                messages=retry_msgs, skill_name=skill_name,
                current_task=current_task,
            )
            if "error" not in retry_out:
                retry_result = self._validate_optimization_result(
                    llm_output=retry_out, skill_name=skill_name,
                    node=node, feedback_data=feedback_data,
                    current_error=current_error,
                    current_task=current_task,
                )
                if retry_result.get("success"):
                    self.logger.info(
                        f"\033[32m[Single-Shot] Retry succeeded\033[0m"
                    )
                    return retry_result
                else:
                    self.logger.warning(
                        f"\033[33m[Single-Shot] Retry also failed: "
                        f"{retry_result.get('error', '?')[:80]}\033[0m"
                    )
            else:
                self.logger.warning(
                    f"\033[33m[Single-Shot] Retry LLM failed\033[0m"
                )

        return result

    def _iterative_gradient_optimize(
        self,
        skill_name: str,
        node,
        feedback_data: Dict[str, Any],
        all_gradients: list,
        momentum_window: int = 5,
        current_error: str = None,
        current_task: str = None,
        current_context: str = None,
        current_state: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Apply gradients one-at-a-time, chaining outputs.

        For weak models that struggle to fix multiple bugs in one pass.
        Each iteration focuses on a single gradient, using the previous
        iteration's output as input.

        Fail-soft policy: when a step's LLM call errors, its validator
        rejects without should_retry, or its retry path also fails, that
        step is skipped (no current_code update). Remaining steps still
        execute with the last-successful current_code. Phase 1 emits each
        gradient as an independent fix for a distinct root cause, so a
        failed step's gradient does not block independent gradients from
        applying. Pre-patch this loop broke out on any step failure,
        leaving 0/N gradients applied even when steps 2..N would have
        succeeded — observed in ckpt_diamond_postG0_a iter 2 lines
        1463-1479 where craftCraftingTable had 4 gradients, step 1 retry
        failed, and steps 2/3/4 never ran.
        """
        self.logger.info(
            f"\033[36m[Iterative Optimize] {skill_name}: "
            f"{len(all_gradients)} gradients to apply sequentially\033[0m"
        )

        current_code = node.code
        original_code = node.code
        applied_count = 0
        last_successful_result = None

        for i, grad in enumerate(all_gradients):
            grad_type = grad.gradient_type.value if hasattr(grad, 'gradient_type') else 'unknown'
            magnitude = getattr(grad, 'magnitude', 0.0)
            self.logger.info(
                f"\033[36m[Iterative Optimize] Step {i+1}/{len(all_gradients)}: "
                f"{grad_type} (mag={magnitude:.2f})\033[0m"
            )

            # Temporarily set single gradient + swap node code
            self._current_phase1_gradients = [grad]
            node.code = current_code

            try:
                messages = self._build_optimization_prompt(
                    skill_name=skill_name,
                    node=node,
                    feedback_data=feedback_data,
                    momentum_window=momentum_window,
                    current_error=current_error,
                    current_task=current_task,
                    current_context=current_context,
                    current_state=current_state,
                )

                llm_output = self._invoke_optimization_llm(
                    messages=messages,
                    skill_name=skill_name,
                    current_task=current_task,
                )

                if "error" in llm_output and not llm_output.get("parse_failure"):
                    self.logger.warning(
                        f"\033[33m[Iterative Optimize] Step {i+1} LLM failed: "
                        f"{llm_output['error'][:100]}\033[0m"
                    )
                    continue

                # Parse failure → fall through to the existing should_retry path
                # below with an escape-aware hint.
                if llm_output.get("parse_failure"):
                    result = {
                        "success": False,
                        "error": llm_output["error"],
                        "raw_response": llm_output.get("raw_response", ""),
                        "should_retry": True,
                        "retry_hint": (
                            "Your previous response had INVALID JSON. Respond with ONLY a valid "
                            "JSON object. CRITICAL: when embedding JS code in any string field "
                            "(e.g. optimized_code), escape every backslash as \\\\, every double "
                            "quote as \\\", and every newline as \\n."
                        ),
                    }
                else:
                    result = self._validate_optimization_result(
                        llm_output=llm_output,
                        skill_name=skill_name,
                        node=node,
                        feedback_data=feedback_data,
                        current_error=current_error,
                        current_task=current_task,
                    )

                if result.get("success"):
                    current_code = result["new_code"]
                    applied_count += 1
                    last_successful_result = result
                    self.logger.info(
                        f"\033[32m[Iterative Optimize] Step {i+1} succeeded\033[0m"
                    )
                elif result.get("should_retry"):
                    # Retry once with validation error as hint
                    error_msg = result.get("error", "unknown")
                    self.logger.info(
                        f"\033[33m[Iterative Optimize] Step {i+1} validation failed, retrying: "
                        f"{error_msg[:80]}\033[0m"
                    )
                    augmented = dict(feedback_data)
                    extra = "\n\n" + _format_validation_rejection_hint(result)
                    # See note in single-shot retry above: write to
                    # feedback_summary, the only field the prompt template
                    # reads. Pre-patch this wrote to feedback_text (dead),
                    # which is why iter 2 in ckpt_diamond_postG0_a observed
                    # the retry LLM output as identical to the first attempt
                    # — the LLM never saw "PREVIOUS ATTEMPT REJECTED".
                    augmented["feedback_summary"] = (
                        augmented.get("feedback_summary", "") + extra
                    )

                    node.code = current_code
                    retry_msgs = self._build_optimization_prompt(
                        skill_name=skill_name, node=node,
                        feedback_data=augmented,
                        momentum_window=momentum_window,
                        current_error=current_error,
                        current_task=current_task,
                        current_context=current_context,
                        current_state=current_state,
                    )
                    retry_out = self._invoke_optimization_llm(
                        messages=retry_msgs, skill_name=skill_name,
                        current_task=current_task,
                    )
                    if "error" not in retry_out:
                        retry_result = self._validate_optimization_result(
                            llm_output=retry_out, skill_name=skill_name,
                            node=node, feedback_data=feedback_data,
                            current_error=current_error,
                            current_task=current_task,
                        )
                        if retry_result.get("success"):
                            current_code = retry_result["new_code"]
                            applied_count += 1
                            last_successful_result = retry_result
                            self.logger.info(
                                f"\033[32m[Iterative Optimize] Step {i+1} retry succeeded\033[0m"
                            )
                        else:
                            self.logger.warning(
                                f"\033[33m[Iterative Optimize] Step {i+1} retry also failed: "
                                f"{retry_result.get('error', '?')[:80]}\033[0m"
                            )
                            continue
                    else:
                        self.logger.warning(
                            f"\033[33m[Iterative Optimize] Step {i+1} retry LLM failed\033[0m"
                        )
                        continue
                else:
                    self.logger.warning(
                        f"\033[33m[Iterative Optimize] Step {i+1} validation failed: "
                        f"{result.get('error', '?')[:100]}\033[0m"
                    )
                    continue
            finally:
                node.code = original_code

        # Restore all gradients
        self._current_phase1_gradients = all_gradients

        self.logger.info(
            f"\033[36m[Iterative Optimize] Done: {applied_count}/{len(all_gradients)} "
            f"gradients applied\033[0m"
        )

        if applied_count > 0 and last_successful_result:
            # Return the final accumulated result
            last_successful_result["new_code"] = current_code
            last_successful_result["change_summary"] = (
                f"Iterative optimization: {applied_count}/{len(all_gradients)} gradients applied. "
                + last_successful_result.get("change_summary", "")
            )
            return last_successful_result

        # No gradient succeeded — return the last error
        return {"error": f"Iterative optimization: 0/{len(all_gradients)} gradients applied"}

    def _validate_code_completeness(
        self,
        optimized_code: str,
        original_code: str,
        llm_response: str = None,
    ) -> Dict[str, Any]:
        """delegate to the validators.validate_code_completeness pure function"""
        return validate_code_completeness(optimized_code, original_code, llm_response)

    def _validate_function_logic_completeness(
        self,
        skill_name: str,
        code: str,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate semantic consistency between the function name and the code logic

        Prevents generating code where "the function name implies some operation but the code is an empty delegating shell".
        Example: craftWoodenPickaxe should contain actual crafting logic, not just call setupCraftingTable.

        Args:
            skill_name: skill name (used to detect the function prefix)
            code: code to validate
            strict: in strict mode, validation failure marks as invalid

        Returns:
            Dict[str, Any]: validation result
            - valid: bool - whether validation passed
            - semantic_match: bool - whether semantics match
            - warning: str - warning message
            - error: str - error message
        """
        _dk = getattr(self, '_domain_knowledge', None)
        _dk_fns = _dk.get_known_functions() if _dk else None
        _op_patterns = _dk_fns.get("operation_patterns") if _dk_fns else None
        return validate_function_implementation(skill_name, code, strict, operation_patterns=_op_patterns)

    def _log_validation_diagnostics(
        self,
        skill_name: str,
        diagnostics: Dict[str, Any],
        errors: List[str]
    ):
        """
        Log detailed code-validation diagnostic information

        Args:
            skill_name: skill name
            diagnostics: diagnostics dictionary
            errors: list of errors
        """
        self.logger.error(f"\033[31m[Validation Diagnostics] ========== {skill_name} validation failure details ==========\033[0m")

        # 1. Code statistics comparison
        orig_stats = diagnostics.get("original_code_stats", {})
        opt_stats = diagnostics.get("optimized_code_stats", {})
        self.logger.error(f"\033[31m[Validation Diagnostics] Original code: {orig_stats.get('lines', '?')} lines, {orig_stats.get('chars', '?')} chars\033[0m")
        self.logger.error(f"\033[31m[Validation Diagnostics] Optimized code: {opt_stats.get('lines', '?')} lines, {opt_stats.get('chars', '?')} chars\033[0m")

        # 2. Bracket analysis
        bracket_analysis = diagnostics.get("bracket_analysis", {})
        if bracket_analysis:
            parens = bracket_analysis.get("parentheses", {})
            brackets = bracket_analysis.get("brackets", {})
            braces = bracket_analysis.get("braces", {})

            self.logger.error(f"\033[31m[Validation Diagnostics] Bracket analysis:\033[0m")
            self.logger.error(f"\033[31m[Validation Diagnostics]   parentheses (): open {parens.get('open', '?')} close {parens.get('close', '?')} (diff: {parens.get('diff', '?')})\033[0m")
            self.logger.error(f"\033[31m[Validation Diagnostics]   square brackets []: open {brackets.get('open', '?')} close {brackets.get('close', '?')} (diff: {brackets.get('diff', '?')})\033[0m")
            self.logger.error(f"\033[31m[Validation Diagnostics]   curly braces {{}}: open {braces.get('open', '?')} close {braces.get('close', '?')} (diff: {braces.get('diff', '?')})\033[0m")

            # Problem line
            problem_line = bracket_analysis.get("problem_line")
            if problem_line:
                self.logger.error(f"\033[31m[Validation Diagnostics]   Problem location: line {problem_line.get('line_num', '?')} - {problem_line.get('issue', '?')}\033[0m")
                self.logger.error(f"\033[31m[Validation Diagnostics]   Problem line content: {problem_line.get('line_content', '?')}\033[0m")

        # 3. LLM response analysis
        llm_stats = diagnostics.get("llm_response_stats", {})
        if llm_stats:
            self.logger.error(f"\033[31m[Validation Diagnostics] LLM response analysis:\033[0m")
            self.logger.error(f"\033[31m[Validation Diagnostics]   Response length: {llm_stats.get('total_chars', '?')} chars\033[0m")
            self.logger.error(f"\033[31m[Validation Diagnostics]   Ends with code block: {llm_stats.get('ends_with_code_block', '?')}\033[0m")

        # 4. Truncation indicators
        truncation_indicators = diagnostics.get("truncation_indicators", [])
        if truncation_indicators:
            self.logger.error(f"\033[31m[Validation Diagnostics] Truncation indicators:\033[0m")
            for indicator in truncation_indicators:
                self.logger.error(f"\033[31m[Validation Diagnostics]   ⚠ {indicator}\033[0m")

        # 5. Specific errors
        self.logger.error(f"\033[31m[Validation Diagnostics] Validation errors ({len(errors)}):\033[0m")
        for error in errors:
            self.logger.error(f"\033[31m[Validation Diagnostics]   ✗ {error}\033[0m")

        self.logger.error(f"\033[31m[Validation Diagnostics] ========== End diagnostics ==========\033[0m")
