"""
ApplyOptimizationMixin - Apply optimization pipeline for skills.

Extracted from optimizer_impl.py for better modularity.

Methods included:
- apply_optimization: Save optimized code, validate, handle interface changes
- _has_substantive_code_change: Detect meaningful code changes (vs comment/whitespace only)
- _save_backpropagation_optimization: Persist backpropagation optimization records

Cross-mixin dependencies (resolved via MRO):
    ValidationMixin: _validate_javascript_syntax, _validate_skill_responsibility
    CodeEditMixin: _fix_syntax_error
    FeedbackMixin: get_feedback_for_skill, _mark_feedbacks_resolved
    InterfaceManagementMixin: on_skill_code_changed
    (stays in optimizer_impl.py): _print_optimization_log
"""

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import skillnet.utils as U

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer


class ApplyOptimizationMixin:
    """Apply Optimization Mixin - Code persistence and validation pipeline.

    Attributes (from SkillGraphOptimizer):
        logger: Logger instance
        skill_graph_manager: SkillGraphManager instance
        _current_llm_analysis: Current LLM analysis result (shared state, cleared on completion)
        _reference_check_rejections: Dict tracking reference check rejections per skill
        _responsibility_check_rejections: Dict tracking responsibility check rejections per skill
        bloat_tracker: CodeBloatTracker instance
        backprop_dir: Directory for backpropagation records
    """

    def _extract_violated_helpers(self, reason: str) -> List[str]:
        """Best-effort extraction of helper function names mentioned as violations.

        Looks for patterns like `Violations: ['X', 'Y', 'Z']` or camelCase
        identifiers cited as violating responsibility boundaries. Used as
        part of the rejection entry stored for the feedback loop.
        """
        # Primary: explicit `Violations: [...]` list
        m = re.search(r"Violations?\s*:\s*\[([^\]]+)\]", reason)
        if m:
            items = re.findall(r"['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]", m.group(1))
            # Filter to camelCase helper-ish names (skip bare skill names)
            return [x for x in items if any(c.isupper() for c in x)]
        # Fallback: match `<camelCaseName> (<num> lines?)` pattern — matches each
        # helper individually regardless of whether "helper functions" appears once.
        helpers = re.findall(r"\b([a-z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*)\s*\(\s*\d+\s*lines?\s*\)", reason)
        if helpers:
            # Dedupe preserving order
            seen = set()
            result = []
            for h in helpers:
                if h not in seen:
                    seen.add(h)
                    result.append(h)
            return result
        return []

    def _extract_suggested_skill(
        self, reason: str, available_skills, exclude_self: Optional[str] = None,
    ) -> Optional[str]:
        """Best-effort extraction of a sibling skill name that should handle
        the inlined logic. The responsibility-check LLM often phrases the
        suggestion as `belongs to a separate skill like 'X'` or `should call
        ... skill like 'Y'`. Match such patterns against `available_skills`.

        Args:
            reason: Rejection reason text from ResponsibilityCheck.
            available_skills: List of skill names in the graph.
            exclude_self: Skill being optimized — excluded from matches
                (so we don't suggest calling itself).

        Returns:
            Name of an existing sibling skill that best matches the rejection
            reason, or None if nothing matches.
        """
        if not reason:
            return None
        available = [n for n in (available_skills or []) if n != exclude_self]
        if not available:
            return None

        # Priority 1: exact match on quoted candidates
        candidates = re.findall(r"['\"]([a-zA-Z_][a-zA-Z0-9_]{2,})['\"]", reason)
        for cand in candidates:
            if cand == exclude_self:
                continue
            if cand in available:
                return cand

        # Priority 2: bidirectional substring match (e.g., quoted "placeCraftingTable"
        # should match available "placeCraftingTableNearby" and vice versa)
        for cand in candidates:
            if len(cand) < 5 or cand == exclude_self:
                continue
            for name in available:
                if cand in name or name in cand:
                    return name

        # Priority 3: any available skill name mentioned verbatim in reason
        reason_lower = reason.lower()
        for name in available:
            if len(name) >= 5 and name.lower() in reason_lower:
                return name

        return None

    def _has_substantive_code_change(self: "SkillGraphOptimizer", old_code: str, new_code: str) -> bool:
        """
        Detect whether code has substantive changes (requires updating metadata).

        Substantive changes: function signatures, child skill calls
        Non-substantive changes: pure comment / whitespace formatting changes

        Args:
            old_code: Original code
            new_code: New code

        Returns:
            bool: Whether there is a substantive change
        """
        if not old_code or not new_code:
            return True

        # Normalize: remove comments and whitespace
        def normalize(code: str) -> str:
            code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
            code = re.sub(r'/\*[\s\S]*?\*/', '', code)
            return re.sub(r'\s+', ' ', code.strip())

        if normalize(old_code) == normalize(new_code):
            return False

        # Signature change detection
        def get_params(code):
            m = re.search(r'async\s+function\s+\w+\s*\(([^)]*)\)', code)
            if m:
                return set(p.strip().split('=')[0].strip() for p in m.group(1).split(',') if p.strip())
            return set()

        if get_params(old_code) != get_params(new_code):
            return True

        # Child skill call changes
        def get_calls(code):
            return set(re.findall(r'await\s+(\w+)\s*\(', code))

        if get_calls(old_code) != get_calls(new_code):
            return True

        return False

    def apply_optimization(
        self: "SkillGraphOptimizer",
        skill_name: str,
        optimization_result: Dict[str, Any],
        create_new_version: bool = True,
        backpropagation_info: Dict[str, Any] = None,
        old_code: str = None,
        interface_update_mode: str = "auto",
    ) -> bool:
        """
        Apply the optimization result to a skill

        Args:
            skill_name: skill name
            optimization_result: optimization result (from optimize_skill_code)
            create_new_version: whether to create a new version
            backpropagation_info: backpropagation info (if optimization was triggered by backpropagation)
                {
                    "parent_skill": "parent_skill_name",
                    "backpropagated_feedback": {...},
                    "parent_feedbacks": [...],  # feedbacks used by the parent skill to produce the backpropagated feedback
                }
            old_code: code before optimization (if not provided, retrieved from the node)
            interface_update_mode: update strategy when the interface changes
                - "auto": automatically update all callers (default)
                - "warn": only emit a warning indicating which skills need updating, do not auto-update
                - "manual": return the list of skills that need updating, do not auto-update (via the affected_callers field of the return value)

        Returns:
            bool: whether application succeeded
        """
        if "error" in optimization_result or not optimization_result.get("success"):
            print(f"\033[31mError: Cannot apply optimization: {optimization_result.get('error', 'Unknown error')}\033[0m")
            self._current_llm_analysis = None  # Clear analysis result
            return False

        new_code = optimization_result.get("new_code")
        change_summary = optimization_result.get("change_summary", "Code optimized")

        if not new_code:
            print(f"\033[31mError: No new code in optimization result\033[0m")
            self._current_llm_analysis = None  # Clear analysis result
            return False

        # Validate code syntax (before saving)
        syntax_valid, syntax_error = self._validate_javascript_syntax(new_code)
        if not syntax_valid:
            print(f"\033[33m[Apply Optimization] Syntax validation failed, attempting auto-repair...\033[0m")
            print(f"\033[33m[Apply Optimization] Error details: {syntax_error}\033[0m")

            # Attempt to fix the syntax error (retry up to 2 times)
            max_retries = 2
            for attempt in range(max_retries):
                fixed_code = self._fix_syntax_error(new_code, syntax_error, skill_name)
                if fixed_code:
                    new_code = fixed_code
                    syntax_valid, syntax_error = self._validate_javascript_syntax(new_code)
                    if syntax_valid:
                        print(f"\033[32m[Apply Optimization] Syntax fix succeeded (attempt {attempt + 1})\033[0m")
                        # Update the code in optimization_result
                        optimization_result["new_code"] = new_code
                        break

                if attempt == max_retries - 1:
                    print(f"\033[31m[Syntax Check] Refusing to save: syntax repair failed (after {max_retries} attempts)\033[0m")
                    print(f"\033[31m[Syntax Check] Error details: {syntax_error}\033[0m")
                    self._current_llm_analysis = None  # Clear analysis result
                    return False

        # ========== Function reference validation (prevents LLM-hallucinated calls to undefined functions) ==========
        from skillnet.agents.optimizer.validators.code_validator import validate_function_references

        # Get all currently available skill names
        available_skills = set(self.skill_graph_manager.get_all_skill_names())
        # Include the skill currently being optimized (it can call itself, though not recommended)
        available_skills.add(skill_name)

        # Get domain function sets for validation (if domain configured)
        _dk = getattr(self, '_domain_knowledge', None)
        domain_fns = _dk.get_known_functions() if _dk else None
        domain_fns = domain_fns or None  # Normalize empty dict to None

        ref_validation = validate_function_references(
            code=new_code,
            available_skills=available_skills,
            domain_functions=domain_fns,
            strict_mode=True  # Strict mode: reject any undefined function outright
        )

        if not ref_validation["valid"]:
            undefined = ref_validation.get("undefined_functions", [])
            suggestions = ref_validation.get("suggestions", {})

            # Store rejection details for feedback loop (Fix 7)
            rejection_entry = {
                "undefined_functions": [u["name"] for u in undefined],
                "suggestions": {u["name"]: suggestions.get(u["name"], []) for u in undefined},
            }
            if skill_name not in self._reference_check_rejections:
                self._reference_check_rejections[skill_name] = []
            self._reference_check_rejections[skill_name].append(rejection_entry)

            print(f"\033[31m[Reference Check] Refusing to save: code references undefined functions\033[0m")
            for undef in undefined:
                func_name = undef["name"]
                line = undef["line"]
                sugg = suggestions.get(func_name, [])
                if sugg:
                    print(f"\033[31m  - Line {line}: {func_name} (suggestions: {', '.join(sugg[:3])})\033[0m")
                else:
                    print(f"\033[31m  - Line {line}: {func_name}\033[0m")

            self._current_llm_analysis = None  # Clear analysis result
            return False

        # ========== Domain-specific code validators ==========
        _dk = getattr(self, '_domain_knowledge', None)
        _domain_validators = _dk.get_code_validators() if _dk else []
        _validator_context = {
            "primitives": domain_fns.get("primitives") if domain_fns else None,
            "skill_name": skill_name,
            "available_skills": available_skills,
        }
        for validator_fn in _domain_validators:
            validator_errors = validator_fn(new_code, _validator_context)
            if validator_errors:
                rejection_entry = {
                    "undefined_functions": [e.get('message', str(e)) for e in validator_errors],
                    "suggestions": {},
                }
                # Build suggestions from errors that have 'method' key (e.g., bot method check)
                for e in validator_errors:
                    if 'method' in e:
                        key = f"bot.{e['method']}()"
                        rejection_entry["undefined_functions"] = [key]
                        rejection_entry["suggestions"][key] = [f"{e['method']}(bot, ...)"]

                if skill_name not in self._reference_check_rejections:
                    self._reference_check_rejections[skill_name] = []
                self._reference_check_rejections[skill_name].append(rejection_entry)

                print(f"\033[31m[Domain Validator] Refusing to save: domain validator reported issues\033[0m")
                for err in validator_errors:
                    line = err.get('line', '?')
                    msg = err.get('message', str(err))
                    print(f"\033[31m  - Line {line}: {msg}\033[0m")

                self._current_llm_analysis = None
                return False
        # ========== Domain-specific code validators end ==========

        # ========== Circular dependency check ==========
        # Prevent optimizer from creating A→B→A cycles (e.g., locateDeepslateBlocks ↔ locateDeepslate).
        # would_create_cycle() already exists in skill graph (used by refactor) but was not called here.
        from skillnet.agents.optimizer.validators.code_validator import extract_function_call_names

        new_func_names = extract_function_call_names(new_code)
        new_skill_calls = {f for f in new_func_names if f in available_skills and f != skill_name}

        for called_skill in new_skill_calls:
            if self.skill_graph_manager.would_create_cycle(skill_name, called_skill):
                self.logger.warning(
                    f"[Cycle Check] Rejected: {skill_name} → {called_skill} would create a cycle "
                    f"({called_skill} already has a path to {skill_name})"
                )
                print(
                    f"\033[31m[Cycle Check] Refusing to save: {skill_name} → {called_skill} "
                    f"would create a circular dependency\033[0m"
                )
                self._current_llm_analysis = None
                return False
        # ========== Circular dependency check end ==========

        # Preserve old code (for backpropagation records)
        node = self.skill_graph_manager.get_node(skill_name)

        # Validate responsibility boundary (before saving)
        if node:
            old_code_for_validation = old_code if old_code else node.code
            skill_description = node.description or ""

            print(f"\033[36m[Responsibility Check] Starting validation of skill '{skill_name}' responsibility boundary...\033[0m")

            responsibility_valid, responsibility_reason = self._validate_skill_responsibility(
                skill_name=skill_name,
                skill_description=skill_description,
                original_code=old_code_for_validation,
                new_code=new_code
            )

            print(f"\033[36m[Responsibility Check] Validation result: valid={responsibility_valid}\033[0m")

            if not responsibility_valid:
                print(f"\033[31m[Responsibility Check] Refusing to save: optimization exceeds skill responsibility boundary\033[0m")
                print(f"\033[31m[Responsibility Check] Reason: {responsibility_reason}\033[0m")

                # Store rejection for feedback loop
                # Next _build_optimization_prompt call will include this so the
                # LLM can see why its previous attempt was rejected and switch
                # strategy (e.g. call a composable skill instead of inlining).
                rejection_entry = {
                    "reason": responsibility_reason[:500],
                    "rejected_code_preview": new_code[:500] if new_code else "",
                    # Try to extract violated helpers and suggested sibling skill
                    # from the reason string (best-effort; still useful as raw reason).
                    "violated_helpers": self._extract_violated_helpers(responsibility_reason),
                    "suggested_target_skill": self._extract_suggested_skill(
                        responsibility_reason,
                        available_skills=self.skill_graph_manager.get_all_skill_names(),
                        exclude_self=skill_name,
                    ),
                }
                if skill_name not in self._responsibility_check_rejections:
                    self._responsibility_check_rejections[skill_name] = []
                self._responsibility_check_rejections[skill_name].append(rejection_entry)

                self._current_llm_analysis = None  # Clear analysis result
                return False
            else:
                print(f"\033[32m[Responsibility Check] Passed: {responsibility_reason[:100]}...\033[0m")
        if old_code is None:
            old_code = node.code if node else ""

        # Collect feedbacks for logging
        feedbacks = self.get_feedback_for_skill(skill_name)
        diagnosis = optimization_result.get("diagnosis")

        # Print optimization log
        self._print_optimization_log(
            skill_name=skill_name,
            old_code=old_code,
            new_code=new_code,
            feedbacks=feedbacks,
            diagnosis=diagnosis,
            optimization_result=optimization_result,
        )

        # Capture old parameter metadata (for semantic change detection)
        # Must be captured before update_skill_code, since the update overwrites metadata
        old_metadata_for_semantic = None
        if node:
            old_metadata_for_semantic = dict(node.parameters) if node.parameters else None

        if create_new_version:
            # Use unified interface to create a new version (supports both main graph and task-specific skills)
            success = self.skill_graph_manager.update_skill_code(
                skill_name=skill_name,
                new_code=new_code,
                change_log=change_summary,
                source="optimizer",
                skip_metadata=False,        # Auto-update metadata
                skip_interface_check=True,  # Caller updates handled separately here (because they need optimizer resources)
            )
        else:
            # Update code directly (without creating a version) - using the unified interface
            # Fix for issue 9: detect substantive change to decide whether to update metadata
            has_substantive_change = self._has_substantive_code_change(old_code, new_code)

            success = self.skill_graph_manager.update_skill_code(
                skill_name=skill_name,
                new_code=new_code,
                change_log=change_summary,
                source="optimizer:quick_optimize:no_version",
                create_version=False,  # Do not create a new version
                skip_metadata=not has_substantive_change,  # Update metadata when there is a substantive change
                skip_interface_check=True,  # Skip interface check
            )

        if success:
            print(f"\033[32mOptimization applied to '{skill_name}'\033[0m")

            # Clear Reference Check rejection history on success (Fix 7)
            if skill_name in self._reference_check_rejections:
                del self._reference_check_rejections[skill_name]

            # Clear Responsibility Check rejection history on success (Issue Q)
            if skill_name in self._responsibility_check_rejections:
                del self._responsibility_check_rejections[skill_name]

            # [Fix] After optimization succeeds, check and supplement effects (if empty or only provisional effects)
            # provisional effects (code="") are produced by pre_register_skill rule fallback and need to be upgraded to full effects
            node = self.skill_graph_manager.get_node(skill_name)
            if node:
                has_only_provisional_effects = (
                    len(node.expected_effects) > 0 and
                    all(not getattr(e, 'code', '') for e in node.expected_effects)
                )
                if len(node.expected_effects) == 0 or has_only_provisional_effects:
                    reason = "provisional effects need upgrade" if has_only_provisional_effects else "no effects"
                    print(f"\033[33m[Apply Optimization] Skill '{skill_name}' {reason}, attempting extraction...\033[0m")
                    try:
                        extracted_effects, warnings = self.skill_graph_manager.extract_effects(
                            new_code,
                            node.description or f"Skill: {skill_name}",
                            task=None,
                            context=None,
                            skill_name=skill_name,
                            execution_traces=None
                        )
                        if extracted_effects:
                            if has_only_provisional_effects:
                                node.expected_effects.clear()
                            node.expected_effects.extend(extracted_effects)
                            print(f"\033[32m[Apply Optimization] Successfully extracted {len(extracted_effects)} effects\033[0m")
                            for effect in extracted_effects:
                                desc = effect.description if hasattr(effect, 'description') else str(effect)
                                print(f"\033[32m  - {desc}\033[0m")
                            # Save updated effects to checkpoint
                            try:
                                self.skill_graph_manager.save()
                            except Exception as e:
                                print(f"\033[33m[Apply Optimization] Failed to save effects: {e}\033[0m")
                        if warnings:
                            for warning in warnings:
                                print(f"\033[33m[Apply Optimization] Effect extraction warning: {warning}\033[0m")
                    except Exception as e:
                        print(f"\033[33m[Apply Optimization] Effects extraction failed: {e}\033[0m")

            # Phase 6: Record successful optimization statistics
            if old_code:
                old_lines = len(old_code.strip().split('\n'))
                new_lines = len(new_code.strip().split('\n'))
                self.bloat_tracker.record_optimization(
                    skill_name=skill_name,
                    old_lines=old_lines,
                    new_lines=new_lines,
                    accepted=True,
                    rejection_reason=None
                )

            # Use the unified interface-change hook
            # Whether callers are auto-updated depends on interface_update_mode
            auto_update = interface_update_mode == "auto"

            # Get the new parameter metadata (used for semantic-change detection)
            updated_node = self.skill_graph_manager.get_node(skill_name)
            new_metadata = updated_node.parameters if updated_node else None

            hook_result = self.on_skill_code_changed(
                skill_name=skill_name,
                old_code=old_code,
                new_code=new_code,
                change_source="optimize",
                auto_update_callers=auto_update,
                old_metadata=old_metadata_for_semantic,  # Captured before update_skill_code
                new_metadata=new_metadata,
            )

            # If there is an interface change and not in auto mode, record it on the node
            if hook_result.get("has_interface_change"):
                interface_impact = hook_result.get("interface_impact", {})
                parent_skills = self.skill_graph_manager.get_parents(skill_name)

                if parent_skills and interface_update_mode in ("warn", "manual"):
                    if interface_update_mode == "warn":
                        # Only emit warning, do not auto-update
                        print(f"\033[33m[Warning] The following {len(parent_skills)} skills need to be manually updated to adapt to the interface change:\033[0m")
                        for i, parent_skill_name in enumerate(parent_skills, 1):
                            print(f"\033[33m  {i}. {parent_skill_name}\033[0m")

                    elif interface_update_mode == "manual":
                        # Manual mode: return the list of skills that need updating
                        print(f"\033[36m[Manual mode] The following {len(parent_skills)} skills need updating:\033[0m")
                        for i, parent_skill_name in enumerate(parent_skills, 1):
                            print(f"\033[36m  {i}. {parent_skill_name}\033[0m")

                    # Store information on the node
                    if node:
                        if not hasattr(node, '_interface_update_info'):
                            node._interface_update_info = {}
                        node._interface_update_info[skill_name] = {
                            "affected_callers": parent_skills,
                            "mode": interface_update_mode,
                            "timestamp": datetime.now().isoformat(),
                            "old_signature": interface_impact.get("old_signature", ""),
                            "new_signature": interface_impact.get("new_signature", ""),
                            "old_code": old_code if interface_update_mode == "manual" else None,
                            "new_code": new_code if interface_update_mode == "manual" else None,
                        }

                # In auto mode, store the update result
                elif interface_update_mode == "auto" and node:
                    if not hasattr(node, '_interface_update_info'):
                        node._interface_update_info = {}
                    node._interface_update_info[skill_name] = {
                        "updated_callers": hook_result.get("callers_updated", []),
                        "failed_callers": [f["skill"] for f in hook_result.get("callers_failed", [])],
                        "skipped_callers": [s["skill"] if isinstance(s, dict) else s for s in hook_result.get("callers_skipped", [])],
                        "timestamp": datetime.now().isoformat(),
                    }

            # If this is a backpropagation optimization, save the optimization process
            if backpropagation_info:
                self._save_backpropagation_optimization(
                    skill_name=skill_name,
                    old_code=old_code,
                    new_code=new_code,
                    backpropagation_info=backpropagation_info,
                )

            # Mark gradient items as used (rather than clearing them, to preserve history)
            if node:
                optimization_id = f"opt_{datetime.now().isoformat()}"
                # Collect all unused items
                unused_items = []
                unused_items.extend([item for item in node.gradients.feedback if not item.used_in_optimization])
                unused_items.extend([item for item in node.gradients.reflections if not item.used_in_optimization])
                unused_items.extend([item for item in node.gradients.optimization_suggestions if not item.used_in_optimization])
                # Mark them all as used in one batch
                if unused_items:
                    node.gradients.mark_as_used(unused_items, optimization_id)
                self.skill_graph_manager.save()

            # Mark related internal analysis feedback as resolved
            self._mark_feedbacks_resolved(skill_name)

        # Clear LLM analysis result to avoid polluting the next optimization
        self._current_llm_analysis = None
        return success

    def _save_backpropagation_optimization(
        self: "SkillGraphOptimizer",
        skill_name: str,
        old_code: str,
        new_code: str,
        backpropagation_info: Dict[str, Any],
    ):
        """
        Persist the backpropagation optimization process

        Args:
            skill_name: name of the skill being optimized
            old_code: code before optimization
            new_code: code after optimization
            backpropagation_info: backpropagation info
        """
        try:
            # Create skill-specific directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            skill_backprop_dir = f"{self.backprop_dir}/{skill_name}_{timestamp}"
            U.f_mkdir(skill_backprop_dir)

            # Save backpropagated feedback
            backprop_feedback = backpropagation_info.get("backpropagated_feedback", {})
            U.dump_json(backprop_feedback, f"{skill_backprop_dir}/backpropagated_feedback.json")

            # Save parent skill's feedbacks
            parent_feedbacks = backpropagation_info.get("parent_feedbacks", [])
            U.dump_json(parent_feedbacks, f"{skill_backprop_dir}/parent_feedbacks.json")

            # Save parent skill's code (before and after the modification)
            parent_skill = backpropagation_info.get("parent_skill", "")
            parent_old_code = backpropagation_info.get("parent_old_code", "")
            parent_new_code = backpropagation_info.get("parent_new_code", "")

            if parent_skill:
                parent_code_dir = f"{skill_backprop_dir}/parent_skill_{parent_skill}"
                U.f_mkdir(parent_code_dir)
                U.dump_text(parent_old_code, f"{parent_code_dir}/before.js")
                if parent_new_code and parent_new_code != parent_old_code:
                    U.dump_text(parent_new_code, f"{parent_code_dir}/after.js")

            # Save current skill's code (before and after the modification)
            skill_code_dir = f"{skill_backprop_dir}/skill_{skill_name}"
            U.f_mkdir(skill_code_dir)
            U.dump_text(old_code, f"{skill_code_dir}/before.js")
            U.dump_text(new_code, f"{skill_code_dir}/after.js")

            # Save metadata
            metadata = {
                "skill_name": skill_name,
                "parent_skill": parent_skill,
                "timestamp": timestamp,
                "optimization_date": datetime.now().isoformat(),
                "backpropagated_feedback": backprop_feedback,
                "files": {
                    "backpropagated_feedback": "backpropagated_feedback.json",
                    "parent_feedbacks": "parent_feedbacks.json",
                    "parent_skill_code_before": f"parent_skill_{parent_skill}/before.js" if parent_skill else None,
                    "parent_skill_code_after": f"parent_skill_{parent_skill}/after.js" if parent_skill and parent_new_code != parent_old_code else None,
                    "skill_code_before": f"skill_{skill_name}/before.js",
                    "skill_code_after": f"skill_{skill_name}/after.js",
                }
            }
            U.dump_json(metadata, f"{skill_backprop_dir}/metadata.json")

            self.logger.info(f"\033[36m[Backpropagation] Saved optimization process to: {skill_backprop_dir}\033[0m")

        except (IOError, OSError) as e:
            self.logger.warning(f"\033[33mWarning: Failed to save backpropagation optimization: {e}\033[0m")
