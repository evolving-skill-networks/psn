"""
Behavioral Refactor

Behavioral refactor strategy: make a broad-scope skill call narrow-scope skills.

Examples:
- craftAndEquipIronSword(bot) calls craftIronSword(bot) then equipItem(bot, "iron_sword")
- collectWoodAndCraft(bot) calls mineOakLogs(bot, 4) then craftPlanks(bot)

This refactor modifies the broad-scope skill's code so it delegates to the narrow-scope skill.
"""

import re
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from .base import (
    RefactorType,
    RefactorOpportunity,
    RefactorResult,
    SkillRefactor,
    analyze_code_dependencies,
    inject_dependencies,
    ensure_bot_parameter,
)
from ..utils import validate_code_syntax
from ..skill_graph.utils.code_analysis import extract_function_calls
from skillnet.utils.stats_tracker import record_llm_usage

if TYPE_CHECKING:
    from skillnet.agents.skill_graph import SkillNode, SkillVersion


class BehavioralRefactor(SkillRefactor):
    """
    Behavioral refactor.

    Refactors a broad-scope skill into a composition that calls narrow-scope skills.

    Refactor flow:
    1. Analyze the functions of the source (broad) and target (narrow) skills.
    2. Identify the part of source that can be replaced by target.
    3. Generate new code that calls target.
    4. Update the source skill's code.
    5. Add a dependency edge.
    """

    def apply(self, opportunity: RefactorOpportunity) -> RefactorResult:
        """
        Apply behavioral refactor.

        Args:
            opportunity: refactor opportunity

        Returns:
            RefactorResult: refactor result
        """
        if opportunity.refactor_type != RefactorType.BEHAVIORAL:
            return RefactorResult(
                success=False,
                refactor_type=opportunity.refactor_type,
                source_skill=opportunity.source_skill,
                target_skill=opportunity.target_skill,
                error_message="Wrong refactor type for BehavioralRefactor"
            )

        source_skill = opportunity.source_skill
        target_skill = opportunity.target_skill

        # Guard check: prevent self-reference (skill calling itself)
        if source_skill == target_skill:
            self._log(
                f"[BehavioralRefactor] WARN self-reference detected: {source_skill} == {target_skill}, skipping",
                "warning"
            )
            return self._create_failed_result(
                opportunity, f"Cannot refactor a skill to call itself: {source_skill}"
            )

        # Triple cycle guard: prevent circular dependencies
        if self.skill_graph_manager:
            # Guard 1: check reverse edge (graph already has target -> source)
            if self.skill_graph_manager.has_edge(target_skill, source_skill):
                self._log(
                    f"[BehavioralRefactor] FAIL circular dependency risk: "
                    f"graph already has edge {target_skill} -> {source_skill}",
                    "error"
                )
                return self._create_failed_result(
                    opportunity,
                    f"Circular dependency: graph edge {target_skill} -> {source_skill} already exists"
                )

            # Guard 2: check reverse call in code (target's code calls source)
            target_node = self.skill_graph_manager.get_node(target_skill)
            if target_node and target_node.code:
                called_in_target = extract_function_calls(target_node.code)
                if source_skill in called_in_target:
                    self._log(
                        f"[BehavioralRefactor] FAIL code-level cycle detected: "
                        f"{target_skill}'s code already calls {source_skill}",
                        "error"
                    )
                    return self._create_failed_result(
                        opportunity,
                        f"Circular dependency: {target_skill} code already calls {source_skill}"
                    )

            # Guard 3: check indirect cycle (A -> B -> C -> A)
            if self.skill_graph_manager.would_create_cycle(source_skill, target_skill):
                self._log(
                    f"[BehavioralRefactor] FAIL indirect cycle detected: "
                    f"adding edge {source_skill} -> {target_skill} would form a cycle",
                    "error"
                )
                return self._create_failed_result(
                    opportunity,
                    f"Indirect cycle: adding {source_skill} -> {target_skill} would create a cycle"
                )

            # Idempotency check: if source already calls target, return success (no-op).
            # Prevents r34 Mode 4 bug: every skill verification triggers detection,
            # the LLM re-suggests the same refactor, causing redundant edits and wasted tokens.
            # r34 #4: placeCT <- craftOakCT was applied twice within 10 seconds.
            # r34 #15: craftIronPickaxe + 3 wrappers already pointed to placeCT yet were suggested again.
            source_node_for_idempotency = self.skill_graph_manager.get_node(source_skill)
            if source_node_for_idempotency and source_node_for_idempotency.code:
                already_called = target_skill in extract_function_calls(
                    source_node_for_idempotency.code
                )
                edge_exists = self.skill_graph_manager.has_edge(source_skill, target_skill)
                if already_called or edge_exists:
                    self._log(
                        f"[BehavioralRefactor] SKIP idempotent: {source_skill} already calls "
                        f"{target_skill} (code_call={already_called}, "
                        f"graph_edge={edge_exists})",
                        "info"
                    )
                    # Even when code already calls it, ensure the graph edge exists (self-heal)
                    if already_called and not edge_exists:
                        self.skill_graph_manager.add_edge(source_skill, target_skill)
                    return RefactorResult(
                        success=True,
                        refactor_type=RefactorType.BEHAVIORAL,
                        source_skill=source_skill,
                        target_skill=target_skill,
                        old_code=source_node_for_idempotency.code,
                        new_code=source_node_for_idempotency.code,
                        changes_made=[
                            f"Idempotent skip: {source_skill} already calls {target_skill}"
                        ],
                        rollback_available=False,
                    )

        self._log(
            f"[BehavioralRefactor] starting refactor: {source_skill} will call {target_skill}",
            "info"
        )

        if not self.skill_graph_manager:
            return self._create_failed_result(
                opportunity, "No skill_graph_manager available"
            )

        source_node = self.skill_graph_manager.get_node(source_skill)
        target_node = self.skill_graph_manager.get_node(target_skill)

        if not source_node or not target_node:
            missing = []
            if not source_node:
                missing.append(f"source '{source_skill}'")
            if not target_node:
                missing.append(f"target '{target_skill}'")
            available = self.skill_graph_manager.get_all_skill_names(include_task_specific=True)[:5]
            return self._create_failed_result(
                opportunity,
                f"Skill not found: {', '.join(missing)}. Available skills: {available}..."
            )

        # Check whether code is empty
        if not source_node.code or not target_node.code:
            return self._create_failed_result(
                opportunity, "Source or target skill code is empty"
            )

        # functional equivalence check - prevent replacing a valid function with an empty stub
        feasible, reason = self._validate_behavioral_feasibility(source_node, target_node)
        if not feasible:
            self._log(
                f"[BehavioralRefactor] FAIL functional equivalence check: {reason}",
                "error"
            )
            return self._create_failed_result(opportunity, reason)

        # Save rollback data (including target skill for edge removal)
        rollback_data = self._save_rollback_data(
            source_skill,
            source_node.code,
            list(source_node.expected_effects) if source_node.expected_effects else []
        )
        # Save extra info for full rollback
        rollback_data.update({
            "target_skill": target_skill,  # Record target skill for edge removal
        })

        try:
            # Analyze how to integrate the target skill
            integration_plan = self._analyze_integration(
                source_node, target_node
            )

            if not integration_plan:
                return self._create_failed_result(
                    opportunity, "Could not determine how to integrate target skill"
                )

            # Generate new code (with response capture for forensics + retry)
            from ._retry_helpers import (
                capture_llm_responses,
                save_refactor_failure_forensics,
                build_retry_message,
            )

            with capture_llm_responses(self) as cap_first:
                new_code = self._generate_integrated_code(
                    source_node, target_node, integration_plan
                )
            _first_raw = cap_first.raw_responses[-1] if cap_first.raw_responses else ""
            _first_messages = cap_first.last_messages

            if not new_code:
                return self._create_failed_result(
                    opportunity, "Failed to generate integrated code"
                )

            # Validate new code
            if not self._validate_integrated_code(new_code, source_skill, target_skill):
                return self._create_failed_result(
                    opportunity, "Generated code validation failed"
                )

            # Save original code
            old_code = source_node.code

            # Phase 4: semantic equivalence validation
            if self.skill_graph_manager:
                from skillnet.agents.optimizer.validators.semantic_compatibility import (
                    SemanticEquivalenceValidator,
                )
                semantic_validator = SemanticEquivalenceValidator(
                    skill_graph=self.skill_graph_manager,
                    llm=None,  # Synchronous mode; do not use LLM
                    enable_llm_validation=False,
                    enable_call_chain_analysis=True,
                )
                # Prepare effects
                old_effects = [str(e) for e in source_node.expected_effects] if source_node.expected_effects else []
                new_effects = [str(e) for e in target_node.expected_effects] if target_node.expected_effects else []

                # Run quick check (Layer 1 only)
                is_semantically_equivalent = semantic_validator.quick_check(old_code, new_code)

                if not is_semantically_equivalent:
                    self._log(
                        f"[BehavioralRefactor] WARN quick semantic-equivalence check failed (skill: {source_skill}), proceeding with other validation",
                        "warning"
                    )
                    # Note: warning only, does not block the refactor
                    # because behavioral refactor may change implementation while preserving function
                else:
                    self._log(
                        f"[BehavioralRefactor] OK quick semantic-equivalence check passed (skill: {source_skill})",
                        "info"
                    )

            # Phase 2: naming conflict check (before syntax validation)
            if self.skill_graph_manager:
                from skillnet.agents.optimizer.validators import check_naming_conflicts
                existing_skill_names = set(self.skill_graph_manager.get_all_skill_names(include_task_specific=True))
                no_conflict, conflict_messages = check_naming_conflicts(new_code, existing_skill_names)
                if not no_conflict:
                    self._log(
                        f"[BehavioralRefactor] FAIL naming-conflict check (skill: {source_skill}): {conflict_messages}",
                        "error"
                    )
                    return self._create_failed_result(
                        opportunity,
                        f"Naming conflict in refactored code for {source_skill}: {'; '.join(conflict_messages)}"
                    )

            # Validate new code syntax
            is_valid, syntax_error = validate_code_syntax(new_code)

            # v3.H+ Layer 3 — single-shot retry + forensics on validation failure
            if not is_valid:
                save_refactor_failure_forensics(
                    skill_graph_manager=self.skill_graph_manager,
                    refactor_type="behavioral",
                    skill_names=[source_skill, target_skill],
                    attempt=1,
                    raw_response=_first_raw,
                    babel_error=syntax_error or "",
                    prompt_messages=_first_messages,
                    logger=getattr(self, "logger", None),
                )
                self._log(
                    f"[BehavioralRefactor] ⟳ Layer 3 retry — first attempt failed validation: "
                    f"{(syntax_error or '')[:120]}",
                    "warning",
                )
                with capture_llm_responses(self) as cap_retry:
                    new_code2 = self._generate_with_llm(
                        source_node, target_node, integration_plan,
                        retry_hint=build_retry_message(syntax_error or ""),
                    )
                _retry_raw = cap_retry.raw_responses[-1] if cap_retry.raw_responses else ""
                if new_code2:
                    new_code = new_code2
                    is_valid, syntax_error = validate_code_syntax(new_code)
                    if is_valid:
                        self._log("[BehavioralRefactor] ✓ Layer 3 retry succeeded", "info")
                    else:
                        save_refactor_failure_forensics(
                            skill_graph_manager=self.skill_graph_manager,
                            refactor_type="behavioral",
                            skill_names=[source_skill, target_skill],
                            attempt=2,
                            raw_response=_retry_raw,
                            babel_error=syntax_error or "",
                            prompt_messages=cap_retry.last_messages,
                            logger=getattr(self, "logger", None),
                        )

            if not is_valid:
                self._log(
                    f"[BehavioralRefactor] FAIL JS syntax validation (skill: {source_skill}): {syntax_error}",
                    "error"
                )
                return self._create_failed_result(
                    opportunity,
                    f"Syntax error in refactored code for {source_skill}: {syntax_error}"
                )

            # Use unified interface to update code (syntax already validated, skip redundant validation)
            update_success = self.skill_graph_manager.update_skill_code(
                skill_name=source_skill,
                new_code=new_code,
                change_log=f"Refactored to call {target_skill}",
                source="refactor:behavioral",
                skip_validation=True,  # Syntax already validated
                skip_metadata=False,   # Issue 9 fix: behavioral changes may alter effects; metadata must be updated
                skip_interface_check=True,  # Do not trigger cascades
                create_version=True,
            )
            if not update_success:
                return self._create_failed_result(
                    opportunity,
                    f"Failed to update code for {source_skill}"
                )

            # Add dependency edge
            self.skill_graph_manager.add_edge(source_skill, target_skill)

            self._log(
                f"[BehavioralRefactor] OK successfully refactored {source_skill} to call {target_skill}",
                "info"
            )

            # Check callers (BEHAVIORAL keeps the source skill's interface unchanged, callers usually need no update)
            propagation_result = self.propagate_to_callers(
                refactored_skill=source_skill,
                changes={
                    "change_type": "wrapper",  # Interface unchanged
                },
                auto_update=False,
            )

            changes_made = [
                f"Integrated call to {target_skill}",
                f"Added dependency edge to {target_skill}",
                f"Replaced inline implementation with skill call",
            ]

            if propagation_result["callers"]:
                changes_made.append(
                    f"Found {len(propagation_result['callers'])} callers (no update needed)"
                )

            # Save caller rollback data (if any)
            if propagation_result.get("rollback_data"):
                rollback_data["caller_rollback_data"] = propagation_result["rollback_data"]

            return RefactorResult(
                success=True,
                refactor_type=RefactorType.BEHAVIORAL,
                source_skill=source_skill,
                target_skill=target_skill,
                old_code=old_code,
                new_code=new_code,
                changes_made=changes_made,
                rollback_available=True,
                rollback_data=rollback_data,
                updated_callers=[],  # BEHAVIORAL keeps interface unchanged
                caller_update_details={},
            )

        except Exception as e:
            self._log(f"[BehavioralRefactor] refactor failed: {e}", "error")
            return self._create_failed_result(opportunity, str(e))

    def _analyze_integration(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze how to integrate the target skill into the source skill.

        Args:
            source_node: broad-scope skill node
            target_node: narrow-scope skill node

        Returns:
            Optional[Dict]: integration plan including replacement location and method.
        """
        if self.llm:
            return self._analyze_with_llm(source_node, target_node)
        else:
            return self._analyze_simple(source_node, target_node)

    def _analyze_simple(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
    ) -> Optional[Dict[str, Any]]:
        """Simple analysis (no LLM)."""
        # Look for parts in source code functionally similar to target
        target_description = target_node.description or ""
        source_code = source_node.code

        # Simple heuristic: look for potentially related code blocks
        integration_point = None

        # Check whether target is already called
        if target_node.name in source_code:
            return None  # Already integrated, no refactor needed

        # Try to identify an insertion point
        # 1. At the start of the function body
        match = re.search(r'async\s+function\s+\w+\s*\([^)]*\)\s*\{', source_code)
        if match:
            integration_point = match.end()

        if integration_point:
            return {
                "integration_type": "prepend",
                "insertion_point": integration_point,
                "target_call": f"await {target_node.name}(bot);",
            }

        return None

    def _analyze_with_llm(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
    ) -> Optional[Dict[str, Any]]:
        """Analyze integration approach using the LLM."""
        try:
            from langchain.schema import HumanMessage, SystemMessage

            from .prompts import PromptLoader

            template = PromptLoader.load("behavioral_analysis")
            system_prompt, human_prompt = template.format(
                source_name=source_node.name,
                source_description=source_node.description,
                source_code=source_node.code,
                target_name=target_node.name,
                target_description=target_node.description,
                target_code=target_node.code,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.behavioral._analyze_with_llm", skill_name=source_node.name)
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON
            import json
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError as e:
                    self._log(f"[BehavioralRefactor] JSON parse failed: {e}", "warning")
                    # Fall back to simple analysis

            # No JSON found or parsing failed — fall back to simple analysis
            self._log("[BehavioralRefactor] no valid JSON found in LLM response, using simple analysis", "warning")
            return self._analyze_simple(source_node, target_node)

        except Exception as e:
            self._log(f"[BehavioralRefactor] LLM analysis failed: {e}", "warning")
            return self._analyze_simple(source_node, target_node)

    def _generate_integrated_code(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
        integration_plan: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate the integrated code.

        Args:
            source_node: broad-scope skill node
            target_node: narrow-scope skill node
            integration_plan: integration plan

        Returns:
            Optional[str]: the generated new code.
        """
        if self.llm:
            return self._generate_with_llm(source_node, target_node, integration_plan)
        else:
            return self._generate_simple(source_node, target_node, integration_plan)

    def _generate_simple(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
        integration_plan: Dict[str, Any]
    ) -> str:
        """Generate simple integration code (no LLM)."""
        source_code = source_node.code
        integration_type = integration_plan.get("integration_type", "prepend")
        target_call = integration_plan.get("target_call", f"await {target_node.name}(bot);")

        if integration_type == "prepend":
            # Insert at the start of the function body
            insertion_point = integration_plan.get("insertion_point", 0)
            if isinstance(insertion_point, int) and insertion_point > 0:
                new_code = (
                    source_code[:insertion_point] +
                    f"\n    // Call {target_node.name}\n    {target_call}\n" +
                    source_code[insertion_point:]
                )
                return self._finalize_generated_code(new_code)

        elif integration_type == "replace":
            code_to_replace = integration_plan.get("code_to_replace", "")
            if code_to_replace and code_to_replace in source_code:
                new_code = source_code.replace(
                    code_to_replace,
                    f"// Replaced with call to {target_node.name}\n    {target_call}"
                )
                return self._finalize_generated_code(new_code)

        # Default: add an explanatory comment at the start of the function body
        match = re.search(r'(async\s+function\s+\w+\s*\([^)]*\)\s*\{)', source_code)
        if match:
            new_code = source_code.replace(
                match.group(1),
                f"{match.group(1)}\n    // Calls {target_node.name}\n    {target_call}\n"
            )
            return self._finalize_generated_code(new_code)

        return self._finalize_generated_code(source_code)

    def _finalize_generated_code(self, code: str) -> str:
        """
        Finalize the generated code.

        1. Ensure the bot parameter exists (prevent autoInjectBot misalignment).
        2. Analyze and inject missing dependencies (e.g. GoalPlaceBlock).

        Args:
            code: generated code

        Returns:
            str: processed code.
        """
        # 1. Ensure the bot parameter is present
        code = ensure_bot_parameter(code)

        # 2. Analyze and inject missing dependencies
        # (only INJECTABLE_DEPENDENCIES; do not inject those already provided by globalDepsCode)
        missing_deps = analyze_code_dependencies(code)
        if missing_deps:
            self._log(
                f"[BehavioralRefactor] injecting missing dependencies: {missing_deps}",
                "info"
            )
            code = inject_dependencies(code, missing_deps)

        return code

    def _generate_with_llm(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
        integration_plan: Dict[str, Any],
        retry_hint: Optional[str] = None,
    ) -> Optional[str]:
        """Generate integration code using the LLM.

        v3.H+ Layer 3: when retry_hint is set, append it as an extra
        HumanMessage so the LLM can correct an earlier validation failure.
        """
        try:
            from langchain.schema import HumanMessage, SystemMessage
            from .prompts import PromptLoader

            template = PromptLoader.load("behavioral_generation")
            system_prompt, human_prompt = template.format(
                source_code=source_node.code,
                target_code=target_node.code,
                integration_plan=integration_plan,
                source_skill_name=source_node.name,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            if retry_hint:
                messages.append(HumanMessage(content=retry_hint))

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.behavioral._generate_with_llm", skill_name=source_node.name)
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract code
            code_match = re.search(r'```(?:javascript|js)?\s*\n(.*?)\n```', content, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                # Ensure function name is correct (the LLM sometimes generates the wrong name)
                code = self._ensure_correct_function_name(code, source_node.name)
                if code is None:
                    self._log("[BehavioralRefactor] could not find function definition in generated code", "error")
                    return self._generate_simple(source_node, target_node, integration_plan)
                # Also run dependency analysis and bot-parameter handling on LLM-generated code
                return self._finalize_generated_code(code)

            return None

        except Exception as e:
            self._log(f"[BehavioralRefactor] LLM generation failed: {e}", "warning")
            return self._generate_simple(source_node, target_node, integration_plan)

    def _ensure_correct_function_name(self, code: str, expected_name: str) -> Optional[str]:
        """
        Ensure the function name is correct, handling cases where the LLM
        outputs multiple functions.

        The LLM sometimes generates the wrong function name (e.g. exploreUntil
        instead of ensureLogs), or outputs multiple functions (target skill +
        modified source skill). This method detects and corrects such errors.

        Args:
            code: generated code
            expected_name: expected function name (the source skill name)

        Returns:
            str: corrected code, or None if no function definition can be found.
        """
        pattern = r'(async\s+function\s+)(\w+)(\s*\([^)]*\))'
        matches = list(re.finditer(pattern, code))

        if not matches:
            return None  # Could not find a function definition

        # Single function — use original logic
        if len(matches) == 1:
            current_name = matches[0].group(2)
            if current_name == expected_name:
                return code  # Already correct

            self._log(
                f"[BehavioralRefactor] correcting function name: {current_name} -> {expected_name}",
                "info"
            )

            # Only replace the function definition (count=1); leave same-named references in comments/strings alone
            return re.sub(
                rf'(async\s+function\s+){re.escape(current_name)}(\s*\([^)]*\))',
                rf'\g<1>{expected_name}\g<2>',
                code,
                count=1
            )

        # Multiple functions: prefer one named expected_name
        for match in matches:
            if match.group(2) == expected_name:
                # Found correctly-named function — extract code from that position
                func_start = match.start()
                code_from_func = code[func_start:]
                self._log(
                    f"[BehavioralRefactor] detected multiple functions ({len(matches)}), extracting {expected_name}",
                    "warning"
                )
                return code_from_func

        # No correctly-named function — extract the last one and rename
        # (Typically the last is the modified source skill; earlier ones are target skills.)
        last_match = matches[-1]
        func_start = last_match.start()
        current_name = last_match.group(2)
        code_from_func = code[func_start:]

        self._log(
            f"[BehavioralRefactor] detected multiple functions ({len(matches)}), "
            f"extracting and renaming the last one: {current_name} -> {expected_name}",
            "warning"
        )

        return re.sub(
            rf'(async\s+function\s+){re.escape(current_name)}(\s*\([^)]*\))',
            rf'\g<1>{expected_name}\g<2>',
            code_from_func,
            count=1
        )

    def _validate_integrated_code(
        self,
        code: str,
        source_skill: str,
        target_skill: str
    ) -> bool:
        """Validate integrated code."""
        # Check that the code contains the source function definition
        if f"function {source_skill}" not in code:
            self._log(f"[BehavioralRefactor] code missing function definition", "warning")
            return False

        # Check that the code calls the target function
        if target_skill not in code:
            self._log(f"[BehavioralRefactor] code does not call {target_skill}", "warning")
            return False

        return True

    # ========== P0: refactor safety guards - functional equivalence check ==========

    def _is_empty_stub(self, code: str) -> bool:
        """
        Detect whether the function is an empty stub.

        An empty stub function is one with only an empty body or a bare return, e.g.:
        - async function foo() { return; }
        - async function foo() { }
        - function foo(err) { return; }

        Args:
            code: function code

        Returns:
            bool: True if the function is an empty stub.
        """
        if not code:
            return True

        # Strip comments
        clean_code = re.sub(r'//.*', '', code)  # Single-line comments
        clean_code = re.sub(r'/\*[\s\S]*?\*/', '', clean_code)  # Multi-line comments
        clean_code = re.sub(r'\s+', ' ', clean_code).strip()  # Collapse whitespace

        # Detect typical empty-stub patterns
        empty_patterns = [
            r'{\s*return\s*;?\s*}',                    # { return; } or { return }
            r'{\s*}',                                   # { }
            r'{\s*return\s+null\s*;?\s*}',             # { return null; }
            r'{\s*return\s+undefined\s*;?\s*}',        # { return undefined; }
            r'function\s+\w+\s*\([^)]*\)\s*{\s*}',     # function foo() {}
            r'async\s+function\s+\w+\s*\([^)]*\)\s*{\s*return\s*;?\s*}',  # async function foo() { return; }
        ]

        for pattern in empty_patterns:
            if re.search(pattern, clean_code, re.IGNORECASE):
                return True

        # Line-count check (after the function signature, fewer than 3 effective statements)
        # Extract the function body
        body_match = re.search(r'{\s*([\s\S]*)\s*}$', clean_code)
        if body_match:
            body = body_match.group(1).strip()
            # Split on semicolons; filter empty statements
            statements = [s.strip() for s in body.split(';') if s.strip()]
            # If only 0-2 statements, and all are trivial returns or empty
            if len(statements) <= 2:
                non_trivial = [s for s in statements if not re.match(r'^(return\s*|await\s+\w+\s*\([^)]*\)\s*)$', s)]
                if len(non_trivial) == 0:
                    return True

        return False

    def _has_meaningful_effects(self, skill_node: 'SkillNode') -> bool:
        """
        Detect whether the skill has real effects.

        Meaningful effects include: produce, consume, transform, move, etc.

        Args:
            skill_node: skill node

        Returns:
            bool: True if the skill has real effects.
        """
        if not skill_node or not skill_node.expected_effects:
            return False

        # expected_effects is List[SkillEffect]
        effects = skill_node.expected_effects

        # No effects recorded
        if not effects:
            return False

        # Check whether effects have real content
        meaningful_effect_types = ['produce', 'consume', 'transform', 'move',
                                   'equip', 'craft', 'mine', 'smelt',
                                   'inventory', 'add', 'remove']
        for effect in effects:
            # SkillEffect.state_representation is Dict[str, Any]
            sr = effect.state_representation
            if not sr:
                continue

            effect_type = sr.get('type', '')
            operation = sr.get('operation', '')

            if effect_type in meaningful_effect_types:
                return True
            if operation in meaningful_effect_types:
                return True
            # Also check whether the item field exists
            if sr.get('item'):
                return True

        return False

    def _validate_behavioral_feasibility(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode'
    ) -> tuple:
        """
        Verify whether the target can serve as a reusable component for the
        source (functional equivalence check).

        This is the core check of the P0 refactor safety guard, preventing
        erroneous refactors like craftPickaxe -> ignoreError.

        Checks:
        1. Target must not be an empty stub.
        2. Target must have real effects (optional; skipped in lenient mode).

        Args:
            source_node: source skill node
            target_node: target skill node

        Returns:
            tuple: (feasible?, reason)
        """
        target_name = target_node.name if target_node else "unknown"

        # Check 1: target must not be an empty stub (strict)
        if target_node and target_node.code:
            if self._is_empty_stub(target_node.code):
                return False, f"Target '{target_name}' is an empty stub function, cannot be used for behavioral refactor"

        # Check 2: target must have real effects (lenient, warning only)
        # Note: this check is optional, because some utility functions may have no expected_effects;
        # empty stubs are already filtered by check 1.
        if target_node and not self._has_meaningful_effects(target_node):
            self._log(
                f"[BehavioralRefactor] WARN: Target '{target_name}' has no documented effects, "
                f"refactor may not be meaningful",
                "warning"
            )
            # Do not block — warning only

        return True, "OK"

    # ========== End P0 ==========

    def rollback(self, result: RefactorResult) -> bool:
        """
        Roll back a BEHAVIORAL refactor.

        In addition to restoring code, the added dependency edge must be removed.

        Args:
            result: the refactor result to roll back

        Returns:
            bool: True if rolled back successfully.
        """
        if not result.rollback_available or not result.rollback_data:
            self._log(f"[BehavioralRefactor] cannot roll back: no rollback data", "warning")
            return False

        try:
            skill_name = result.source_skill
            rollback_data = result.rollback_data

            if not self.skill_graph_manager:
                self._log(f"[BehavioralRefactor] cannot roll back: no skill_graph_manager", "warning")
                return False

            node = self.skill_graph_manager.get_node(skill_name)
            if not node:
                self._log(f"[BehavioralRefactor] cannot roll back: node {skill_name} does not exist", "warning")
                return False

            # 1. Restore code
            old_code = rollback_data.get("original_code")
            if old_code:
                node.code = old_code
                self._log(f"[BehavioralRefactor] restored code for '{skill_name}'", "info")

            # 2. Remove dependency edge
            target_skill = rollback_data.get("target_skill")
            if target_skill:
                if self.skill_graph_manager.has_edge(skill_name, target_skill):
                    self.skill_graph_manager.remove_edge(skill_name, target_skill)
                    self._log(f"[BehavioralRefactor] removed edge {skill_name} -> {target_skill}", "info")

            # 3. Roll back caller updates (if any)
            caller_rollback_data = rollback_data.get("caller_rollback_data")
            if caller_rollback_data:
                self.rollback_caller_updates(caller_rollback_data)

            self._log(f"[BehavioralRefactor] OK completed rollback for '{skill_name}'", "info")
            return True

        except Exception as e:
            self._log(f"[BehavioralRefactor] rollback failed: {e}", "error")
            return False
