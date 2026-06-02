"""
Parametric Refactor

Parametric refactor strategy: replace a specialized skill with a call to a
general one.

Examples:
- mineOakLogs(bot, count) -> mineLogs(bot, count, "oak_log")
- craftWoodenPickaxe(bot) -> craftItem(bot, "wooden_pickaxe", 1)

This refactor marks the specialized skill as "covered" so it delegates to the
general version.

integrates Minecraft rules from the planning.inference module.
"""

import re
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from .base import (
    RefactorType,
    RefactorOpportunity,
    RefactorResult,
    SkillRefactor,
)
from ..skill_graph.models.coverage import CoverageType
from ..utils import validate_code_syntax
from skillnet.utils.stats_tracker import record_llm_usage

# empty fallbacks for non-Minecraft domains; Minecraft provides
# these via DomainKnowledge.get_inference_rules().
_FALLBACK_WOOD_TYPES = []
_FALLBACK_RAW_ORE_TYPES = []
_FALLBACK_DIRECT_DROP_ORES = []

if TYPE_CHECKING:
    from skillnet.agents.skill_graph import SkillNode, SkillVersion


class ParametricRefactor(SkillRefactor):
    """
    Parametric refactor.

    Refactors a specialized skill into a wrapper that calls a general skill.

    Refactor flow:
    1. Analyze parameter differences between specialized and general skills.
    2. Generate the parameter mapping.
    3. Create the wrapper code.
    4. Update the specialized skill's code.
    5. Mark the specialized skill as "covered".
    """

    def apply(self, opportunity: RefactorOpportunity) -> RefactorResult:
        """
        Apply parametric refactor.

        Args:
            opportunity: refactor opportunity

        Returns:
            RefactorResult: refactor result.
        """
        if opportunity.refactor_type != RefactorType.PARAMETRIC:
            return RefactorResult(
                success=False,
                refactor_type=opportunity.refactor_type,
                source_skill=opportunity.source_skill,
                target_skill=opportunity.target_skill,
                error_message="Wrong refactor type for ParametricRefactor"
            )

        source_skill = opportunity.source_skill
        target_skill = opportunity.target_skill

        self._log(
            f"[ParametricRefactor] starting refactor: {source_skill} -> {target_skill}",
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

        # Save rollback data (including coverage state)
        rollback_data = self._save_rollback_data(
            source_skill,
            source_node.code,
            list(source_node.expected_effects) if source_node.expected_effects else []
        )
        # Save extra coverage state for full rollback
        rollback_data.update({
            "original_is_covered": getattr(source_node, 'is_covered', False),
            "original_covered_by": getattr(source_node, 'covered_by', None),
            "original_coverage_type": getattr(source_node, 'coverage_type', None),
            "target_skill": target_skill,  # Record target skill for edge removal
        })

        try:
            # Analyze parameter mapping
            param_mapping = self._analyze_parameter_mapping(
                source_node, target_node, opportunity.parameter_mapping
            )

            # Generate wrapper code (with response capture for forensics + retry)
            from ._retry_helpers import (
                capture_llm_responses,
                save_refactor_failure_forensics,
                build_retry_message,
            )

            with capture_llm_responses(self) as cap_first:
                wrapper_code = self._generate_wrapper_code(
                    source_node, target_node, param_mapping
                )
            _first_raw = cap_first.raw_responses[-1] if cap_first.raw_responses else ""
            _first_messages = cap_first.last_messages

            if not wrapper_code:
                return self._create_failed_result(
                    opportunity, "Failed to generate wrapper code"
                )

            # Validate wrapper code
            if not self._validate_wrapper_code(wrapper_code, source_skill, target_skill):
                return self._create_failed_result(
                    opportunity, "Generated wrapper code validation failed"
                )

            # Save original code
            old_code = source_node.code

            # Validate wrapper code syntax
            is_valid, syntax_error = validate_code_syntax(wrapper_code)

            # v3.H+ Layer 3 — single-shot retry + forensics on validation failure
            if not is_valid:
                save_refactor_failure_forensics(
                    skill_graph_manager=self.skill_graph_manager,
                    refactor_type="parametric",
                    skill_names=[source_skill, target_skill],
                    attempt=1,
                    raw_response=_first_raw,
                    babel_error=syntax_error or "",
                    prompt_messages=_first_messages,
                    logger=getattr(self, "logger", None),
                )
                self._log(
                    f"[ParametricRefactor] ⟳ Layer 3 retry — first attempt failed validation: "
                    f"{(syntax_error or '')[:120]}",
                    "warning",
                )
                with capture_llm_responses(self) as cap_retry:
                    wrapper_code2 = self._generate_wrapper_with_llm(
                        source_node, target_node, param_mapping,
                        retry_hint=build_retry_message(syntax_error or ""),
                    )
                _retry_raw = cap_retry.raw_responses[-1] if cap_retry.raw_responses else ""
                if wrapper_code2:
                    wrapper_code = wrapper_code2
                    is_valid, syntax_error = validate_code_syntax(wrapper_code)
                    if is_valid:
                        self._log("[ParametricRefactor] ✓ Layer 3 retry succeeded", "info")
                    else:
                        save_refactor_failure_forensics(
                            skill_graph_manager=self.skill_graph_manager,
                            refactor_type="parametric",
                            skill_names=[source_skill, target_skill],
                            attempt=2,
                            raw_response=_retry_raw,
                            babel_error=syntax_error or "",
                            prompt_messages=cap_retry.last_messages,
                            logger=getattr(self, "logger", None),
                        )

            if not is_valid:
                self._log(
                    f"[ParametricRefactor] FAIL JS syntax validation (skill: {source_skill}): {syntax_error}",
                    "error"
                )
                return self._create_failed_result(
                    opportunity,
                    f"Syntax error in wrapper code for {source_skill}: {syntax_error}"
                )

            # Use unified interface to update code (syntax already validated, skip redundant validation)
            update_success = self.skill_graph_manager.update_skill_code(
                skill_name=source_skill,
                new_code=wrapper_code,
                change_log=f"Converted to wrapper calling {target_skill}",
                source="refactor:parametric",
                skip_validation=True,  # Syntax already validated
                skip_metadata=False,   # Issue 9 fix: parametric refactor may alter preconditions; metadata must be updated
                skip_interface_check=True,  # Do not trigger cascades
                create_version=True,
            )
            if not update_success:
                return self._create_failed_result(
                    opportunity,
                    f"Failed to update code for {source_skill}"
                )

            # Mark as covered
            source_node.is_covered = True
            source_node.covered_by = target_skill
            source_node.coverage_type = CoverageType.PARAMETRIC

            # Add dependency edge
            self.skill_graph_manager.add_edge(source_skill, target_skill)

            self._log(
                f"[ParametricRefactor] OK successfully refactored {source_skill}",
                "info"
            )

            # Check callers (PARAMETRIC keeps the interface unchanged; usually no update needed)
            propagation_result = self.propagate_to_callers(
                refactored_skill=source_skill,
                changes={
                    "change_type": "wrapper",  # Became a wrapper; interface unchanged
                    "new_skill_name": target_skill,
                },
                auto_update=False,
            )

            # Even when no update is needed, record caller info
            changes_made = [
                f"Converted to wrapper calling {target_skill}",
                f"Marked as covered by {target_skill}",
                f"Added dependency edge",
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
                refactor_type=RefactorType.PARAMETRIC,
                source_skill=source_skill,
                target_skill=target_skill,
                old_code=old_code,
                new_code=wrapper_code,
                changes_made=changes_made,
                rollback_available=True,
                rollback_data=rollback_data,
                updated_callers=[],  # PARAMETRIC does not need to update callers
                caller_update_details={},
            )

        except Exception as e:
            self._log(f"[ParametricRefactor] refactor failed: {e}", "error")
            return self._create_failed_result(opportunity, str(e))

    def _analyze_parameter_mapping(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
        provided_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Analyze parameter mapping.

        Args:
            source_node: specialized skill node
            target_node: general skill node
            provided_mapping: provided mapping (optional)

        Returns:
            Dict[str, str]: parameter mapping {target_param: value_or_source_param}
        """
        if provided_mapping:
            return provided_mapping

        mapping = {}

        # Extract parameters of both skills
        source_params = self._extract_parameters(source_node.code, source_node.name)
        target_params = self._extract_parameters(target_node.code, target_node.name)

        # Match same-named parameters
        for target_param in target_params:
            if target_param in source_params:
                mapping[target_param] = target_param
            elif target_param == 'bot':
                mapping[target_param] = 'bot'

        # Infer default values for remaining parameters
        unmapped_params = []
        for target_param in target_params:
            if target_param not in mapping:
                # Try inferring from the source skill name
                default_value = self._infer_default_value(
                    source_node.name, target_param
                )
                if default_value:
                    mapping[target_param] = default_value
                else:
                    # Cannot infer — use undefined and record a warning
                    mapping[target_param] = "undefined"
                    unmapped_params.append(target_param)

        if unmapped_params:
            self._log(
                f"[ParametricRefactor] could not infer parameter values: {unmapped_params}, using undefined",
                "warning"
            )

        return mapping

    def _extract_parameters(self, code: str, func_name: str) -> List[str]:
        """Extract function parameters."""
        pattern = rf'async\s+function\s+{re.escape(func_name)}\s*\(([^)]*)\)'
        match = re.search(pattern, code)

        if not match:
            return []

        params_str = match.group(1)
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if '=' in param:
                param = param.split('=')[0].strip()
            if param and param != 'bot':
                params.append(param)

        return params

    def _infer_default_value(self, skill_name: str, param_name: str) -> Optional[str]:
        """
        Infer a parameter's default value from the skill name.

        E.g. mineOakLogs -> logType = "oak_log".

        Prefer LLM inference; fall back to rules if the LLM is unavailable or fails.
        """
        # Prefer the LLM
        if self.llm:
            result = self._infer_default_value_with_llm(skill_name, param_name)
            if result:
                return result
            # LLM failed — fall back to rules
            self._log(
                f"[ParametricRefactor] LLM inference failed, falling back to rules",
                "warning"
            )

        return self._infer_default_value_simple(skill_name, param_name)

    def _infer_default_value_with_llm(self, skill_name: str, param_name: str) -> Optional[str]:
        """
        Infer a parameter default value using the LLM.

        Args:
            skill_name: skill name (e.g. 'mineOakLogs')
            param_name: parameter name (e.g. 'logType')

        Returns:
            Optional[str]: inferred value (e.g. '"oak_log"'), or None on failure.
        """
        try:
            import json
            from langchain.schema import HumanMessage, SystemMessage
            from .prompts import PromptLoader

            template = PromptLoader.load("parametric_infer_value")
            system_prompt, human_prompt = template.format(
                skill_name=skill_name,
                param_name=param_name,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.parametric._infer_default_value_with_llm", skill_name=skill_name)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
                value = data.get("value")
                if value is not None:
                    # Ensure value is in string form
                    if not value.startswith('"'):
                        value = f'"{value}"'
                    self._log(
                        f"[ParametricRefactor] LLM inferred {skill_name}.{param_name} = {value}",
                        "info"
                    )
                    return value
                else:
                    reason = data.get("reason", "unknown")
                    self._log(
                        f"[ParametricRefactor] LLM could not infer {param_name}: {reason}",
                        "warning"
                    )
                    return None

            return None

        except Exception as e:
            self._log(
                f"[ParametricRefactor] LLM inference exception: {e}",
                "warning"
            )
            return None

    def _infer_default_value_simple(self, skill_name: str, param_name: str) -> Optional[str]:
        """
        Infer a parameter default value via rules (simple version).

        E.g. mineOakLogs -> logType = "oak_log".

        integrates unified Minecraft rules.
        Stage 3: domain-injectable via get_inference_rules().
        """
        # Resolve inference rule data (domain-injected or fallback)
        dk = getattr(self, 'domain_knowledge', None)
        rules = dk.get_inference_rules() if dk else {}
        wood_types = rules.get("wood_types", _FALLBACK_WOOD_TYPES)
        raw_ore_types = rules.get("raw_ore_types", _FALLBACK_RAW_ORE_TYPES)
        direct_drop_ores = rules.get("direct_drop_ores", _FALLBACK_DIRECT_DROP_ORES)

        # first try extracting a Minecraft item from the parameter name and skill name
        param_lower = param_name.lower()
        skill_lower = skill_name.lower()

        # Log/wood-related parameters
        if "log" in param_lower or "wood" in param_lower:
            for wood_type in wood_types:
                if wood_type in skill_lower:
                    return f'"{wood_type}_log"'

        # Ore-related parameters
        if "ore" in param_lower:
            for ore_type in raw_ore_types + direct_drop_ores:
                if ore_type in skill_lower:
                    return f'"{ore_type}_ore"'

        # Planks-related parameters
        if "plank" in param_lower:
            for wood_type in wood_types:
                if wood_type in skill_lower:
                    return f'"{wood_type}_planks"'

        # Ingot-related parameters
        if "ingot" in param_lower:
            for ore_type in raw_ore_types:
                if ore_type in skill_lower:
                    return f'"{ore_type}_ingot"'

        # Common patterns (backward compatibility)
        patterns = [
            # mineOakLogs -> oak_log
            (r'mine(\w+)Logs?', lambda m: f'"{m.group(1).lower()}_log"'),
            # craftWoodenPickaxe -> wooden_pickaxe
            (r'craft(\w+)', lambda m: f'"{self._to_snake_case(m.group(1))}"'),
            # smeltIronOre -> iron_ore
            (r'smelt(\w+)', lambda m: f'"{self._to_snake_case(m.group(1))}"'),
            # harvestWheat -> wheat
            (r'harvest(\w+)', lambda m: f'"{m.group(1).lower()}"'),
            # cookBeef -> beef
            (r'cook(\w+)', lambda m: f'"{m.group(1).lower()}"'),
            # collectWater -> water
            (r'collect(\w+)', lambda m: f'"{m.group(1).lower()}"'),
            # placeStone -> stone
            (r'place(\w+)', lambda m: f'"{self._to_snake_case(m.group(1))}"'),
            # findDiamond -> diamond
            (r'find(\w+)', lambda m: f'"{m.group(1).lower()}"'),
        ]

        for pattern, value_fn in patterns:
            match = re.match(pattern, skill_name, re.IGNORECASE)
            if match:
                try:
                    return value_fn(match)
                except (IndexError, AttributeError, KeyError) as e:
                    # value_fn may fail because match.group() accesses an invalid group;
                    # log it but keep trying other patterns.
                    self._log(
                        f"[ParametricRefactor] default-value inference failed ({pattern}): {e}",
                        "warning"
                    )
                    continue

        return None

    # _to_snake_case is inherited from the SkillRefactor base class

    def _generate_wrapper_code(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
        param_mapping: Dict[str, str]
    ) -> Optional[str]:
        """
        Generate wrapper code.

        Args:
            source_node: specialized skill node
            target_node: general skill node
            param_mapping: parameter mapping

        Returns:
            Optional[str]: generated wrapper code.
        """
        if self.llm:
            return self._generate_wrapper_with_llm(
                source_node, target_node, param_mapping
            )
        else:
            return self._generate_wrapper_simple(
                source_node, target_node, param_mapping
            )

    def _generate_wrapper_simple(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
        param_mapping: Dict[str, str]
    ) -> str:
        """Generate simple wrapper code (no LLM)."""
        source_params = self._extract_parameters(source_node.code, source_node.name)
        target_params = self._extract_parameters(target_node.code, target_node.name)

        # Build parameter list
        params_str = "bot"
        if source_params:
            params_str += ", " + ", ".join(source_params)

        # Build call arguments
        call_args = ["bot"]
        for target_param in target_params:
            if target_param in param_mapping:
                call_args.append(param_mapping[target_param])
            else:
                call_args.append("undefined")

        call_str = ", ".join(call_args)

        # Diagnostic — warn when wrapper call args don't match target signature
        if len(call_args) - 1 != len(target_params):  # -1 for bot
            self._log(
                f"WARN wrapper {source_node.name} arg count mismatch: "
                f"call_args={len(call_args) - 1} (excl. bot), target_params={len(target_params)} "
                f"(target: {target_node.name})",
                "warning"
            )

        return f"""async function {source_node.name}({params_str}) {{
    // Wrapper: calls {target_node.name}
    return await {target_node.name}({call_str});
}}"""

    def _generate_wrapper_with_llm(
        self,
        source_node: 'SkillNode',
        target_node: 'SkillNode',
        param_mapping: Dict[str, str],
        retry_hint: Optional[str] = None,
    ) -> Optional[str]:
        """Generate wrapper code using the LLM.

        v3.H+ Layer 3: when retry_hint is set, append it as an extra
        HumanMessage so the LLM can correct an earlier validation failure.
        """
        try:
            from langchain.schema import HumanMessage, SystemMessage
            from .prompts import PromptLoader

            template = PromptLoader.load("parametric_wrapper")
            system_prompt, human_prompt = template.format(
                source_name=source_node.name,
                target_name=target_node.name,
                source_signature=self._extract_signature(source_node.code, source_node.name),
                target_signature=self._extract_signature(target_node.code, target_node.name),
                param_mapping=param_mapping,
                source_description=source_node.description,
                target_description=target_node.description,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            if retry_hint:
                messages.append(HumanMessage(content=retry_hint))

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.parametric._generate_wrapper_with_llm", skill_name=source_node.name)
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract code
            code_match = re.search(r'```(?:javascript|js)?\s*\n(.*?)\n```', content, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

            return None

        except Exception as e:
            self._log(f"[ParametricRefactor] LLM generation failed: {e}", "warning")
            return self._generate_wrapper_simple(source_node, target_node, param_mapping)

    def _extract_signature(self, code: str, func_name: str) -> str:
        """Extract a function signature."""
        pattern = rf'(async\s+function\s+{re.escape(func_name)}\s*\([^)]*\))'
        match = re.search(pattern, code)
        return match.group(1) if match else f"async function {func_name}(...)"

    def _validate_wrapper_code(
        self,
        wrapper_code: str,
        source_skill: str,
        target_skill: str
    ) -> bool:
        """Validate wrapper code."""
        # Check that the function definition is present
        if f"function {source_skill}" not in wrapper_code:
            self._log(f"[ParametricRefactor] wrapper missing function definition", "warning")
            return False

        # Check that the target function is called
        if target_skill not in wrapper_code:
            self._log(f"[ParametricRefactor] wrapper does not call {target_skill}", "warning")
            return False

        return True

    def rollback(self, result: RefactorResult) -> bool:
        """
        Roll back a PARAMETRIC refactor.

        In addition to restoring code, must also:
        1. Restore is_covered, covered_by, coverage_type, refactor_type attributes.
        2. Remove the added dependency edge.

        Args:
            result: the refactor result to roll back

        Returns:
            bool: True if rolled back successfully.
        """
        if not result.rollback_available or not result.rollback_data:
            self._log(f"[ParametricRefactor] cannot roll back: no rollback data", "warning")
            return False

        try:
            skill_name = result.source_skill
            rollback_data = result.rollback_data

            if not self.skill_graph_manager:
                self._log(f"[ParametricRefactor] cannot roll back: no skill_graph_manager", "warning")
                return False

            node = self.skill_graph_manager.get_node(skill_name)
            if not node:
                self._log(f"[ParametricRefactor] cannot roll back: node {skill_name} does not exist", "warning")
                return False

            # 1. Restore code
            old_code = rollback_data.get("original_code")
            if old_code:
                node.code = old_code
                self._log(f"[ParametricRefactor] restored code for '{skill_name}'", "info")

            # 2. Restore coverage state
            node.is_covered = rollback_data.get("original_is_covered", False)
            node.covered_by = rollback_data.get("original_covered_by", None)
            node.coverage_type = rollback_data.get("original_coverage_type", None)
            self._log(f"[ParametricRefactor] restored coverage state for '{skill_name}'", "info")

            # 3. Remove dependency edge
            target_skill = rollback_data.get("target_skill")
            if target_skill:
                if self.skill_graph_manager.has_edge(skill_name, target_skill):
                    self.skill_graph_manager.remove_edge(skill_name, target_skill)
                    self._log(f"[ParametricRefactor] removed edge {skill_name} -> {target_skill}", "info")

            # 4. Roll back caller updates (if any)
            caller_rollback_data = rollback_data.get("caller_rollback_data")
            if caller_rollback_data:
                self.rollback_caller_updates(caller_rollback_data)

            self._log(f"[ParametricRefactor] OK completed rollback for '{skill_name}'", "info")
            return True

        except Exception as e:
            self._log(f"[ParametricRefactor] rollback failed: {e}", "error")
            return False
