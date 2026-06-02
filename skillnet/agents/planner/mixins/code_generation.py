"""Code Generation Mixin - Generate skill composition code."""

from __future__ import annotations
import re
import json
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from langchain.schema import HumanMessage, SystemMessage
from skillnet.agents.planning import (
    strip_js_comments,
    validate_param_name,
    sanitize_python_to_js,
    normalize_task_name,
    generate_function_name,
    generate_plan_description,
    extract_require_declarations,
    check_variable_usage,
    get_standard_declarations,
    get_global_deps_vars,
    is_module_level_declaration,
    parse_function_parameters,
    is_destructured_parameter,
)
from .._types import SkillCallContext
from skillnet.agents.planning.inference.planning_state import PlanningState
from skillnet.core.dk_registry import get_domain_knowledge
from skillnet.utils.stats_tracker import record_llm_usage

if TYPE_CHECKING:
    from ..graph_planner import GraphPlanner


def _resolve_skill_language(agent):
    """Resolve a SkillLanguage instance for the given agent.

    Cache-first (``agent._skill_language``), registry-fallback. Raises
    ``RuntimeError`` if no DomainKnowledge is registered.
    """
    skill_lang = getattr(agent, "_skill_language", None)
    if skill_lang is not None:
        return skill_lang
    dk = get_domain_knowledge()
    if dk is None:
        raise RuntimeError(
            "No DomainKnowledge registered; cannot parse skill code"
        )
    return dk.get_skill_language_impl()


class CodeGenerationMixin:
    """Code Generation Mixin - Generate skill composition code.

    Methods:
        _extract_declared_variables: Extract declared variables
        _generate_composition_code: Generate composition code - core method, ~400 lines
        _sanitize_python_to_js: Sanitize Python to JS syntax
        _generate_skill_call_with_llm: Generate skill call with LLM
        _should_use_llm_for_skill_call: Check if should use LLM for skill call
        _verify_recursive_call_with_llm: Verify recursive call with LLM
        _map_task_params_to_skill_params_with_llm: Map task params to skill params
        _generate_plan_description: Generate plan description
        _parse_function_parameters: Parse function parameters
        _is_destructured_parameter: Check if parameter is destructured
        _strip_js_comments: Strip JS comments
        _validate_param_name: Validate param name
        _parse_destructured_parameter: Parse destructured parameter
        _generate_function_name: Generate function name
        _normalize_task_name: Normalize task name
        _normalize_task_key: Normalize task key

    Note:
        Some pure functions have been extracted to skillnet.agents.planning.code_generator module.
    """

    def _extract_declared_variables(self, skill_sequence: List[SkillCallContext]) -> List[str]:
        """
        Extract variables declared within all called skill functions.

        Important fix: Only extract declarations from inside functions; ignore module-level
        declarations. Module-level declarations (outside any function) are already available
        in the module scope and do not need to be re-declared in the main function. Otherwise
        this causes JavaScript TDZ (Temporal Dead Zone) errors.

        These variables may be declared inside skill functions but also need to be used in
        the main function scope. Primarily focused on variables declared by require statements,
        such as:
        - const mcData = require('minecraft-data')(bot.version);
        - const { Vec3 } = require('vec3');

        Args:
            skill_sequence: Skill execution sequence (list of SkillCallContext)

        Returns:
            List[str]: De-duplicated list of variable declarations (JavaScript code format)
        """
        declared_variables = []
        seen_variable_names = set()  # Used to deduplicate variable names (not declaration strings)
        seen_declarations = set()  # Used to deduplicate full declaration strings
        module_level_vars = set()  # Module-level declared variables, no need to re-declare in main function

        # Mapping from variable name to standard declaration (domain-aware)
        domain_k = getattr(self, '_domain_knowledge', None)
        variable_to_declaration = get_standard_declarations(domain_k)

        for ctx in skill_sequence:
            skill_name = ctx.skill_name
            node = self.skill_graph_manager.get_node(skill_name)
            if not node or not node.code:
                continue

            skill_code = node.code

            # Extract all require statements declared with const/let/var
            # Matching patterns:
            # 1. const mcData = require('minecraft-data')(bot.version);
            # 2. const { Vec3 } = require('vec3');
            # 3. const Vec3 = require('vec3').Vec3;
            # 4. Generic pattern: const varName = require('module');

            # First match specific common patterns (from domain knowledge)
            require_patterns = domain_k.get_require_patterns() if domain_k else []
            for pattern, var_name, default_decl in require_patterns:
                if re.search(pattern, skill_code, re.MULTILINE):
                    # Check if it is a module-level declaration
                    if is_module_level_declaration(skill_code, pattern):
                        # Module-level declaration, record but do not add to declared_variables
                        module_level_vars.add(var_name)
                        self.logger.debug(f"\033[33m[Graph Planner] Skipping module-level declaration: {var_name} (avoid TDZ issue)\033[0m")
                        continue

                    # Deduplicate by variable name, not by declaration string
                    if var_name not in seen_variable_names and var_name not in module_level_vars:
                        seen_variable_names.add(var_name)
                        seen_declarations.add(default_decl)
                        declared_variables.append(default_decl)

            # Then match the generic pattern: const varName = require('module');
            generic_pattern = r'const\s+(\w+)\s*=\s*require\([\'"]([^\'"]+)[\'"]\)[^;]*;'
            matches = re.finditer(generic_pattern, skill_code, re.MULTILINE)
            for match in matches:
                var_name = match.group(1)
                module_name = match.group(2)
                var_decl = match.group(0).strip()

                # Skip variables already matched by specific patterns
                if var_name in seen_variable_names or var_name in module_level_vars:
                    continue

                # Check if it is a module-level declaration
                if is_module_level_declaration(skill_code, re.escape(var_decl)):
                    module_level_vars.add(var_name)
                    self.logger.debug(f"\033[33m[Graph Planner] Skipping module-level declaration: {var_name} (avoid TDZ issue)\033[0m")
                    continue

                # Normalize the declaration (remove extra whitespace)
                var_decl_normalized = re.sub(r'\s+', ' ', var_decl)

                # Skip declarations already seen
                if var_decl_normalized not in seen_declarations:
                    seen_variable_names.add(var_name)
                    seen_declarations.add(var_decl_normalized)
                    declared_variables.append(var_decl_normalized)

        # If no declarations were found but usage of these variables is detected, add default declarations
        if not declared_variables:
            # Combine all skill code for usage detection
            all_code = ""
            for ctx in skill_sequence:
                node = self.skill_graph_manager.get_node(ctx.skill_name)
                if node and node.code:
                    all_code += node.code + "\n"

            usage = check_variable_usage(all_code, domain_k)
            decls = variable_to_declaration
            for var_name, used in usage.items():
                if used and var_name not in seen_variable_names and var_name in decls:
                    declared_variables.append(decls[var_name])
                    seen_variable_names.add(var_name)

        return declared_variables

    def _extract_requires_and_declarations(
        self,
        skill_sequence: List[SkillCallContext],
    ) -> List[str]:
        """
        Extract require statements and variable declarations from skill code,
        filtering out those already provided by globalDepsCode.

        Args:
            skill_sequence: Skill execution sequence

        Returns:
            List of filtered variable declaration strings
        """
        # Extract variables declared within all called skill functions
        declared_variables = self._extract_declared_variables(skill_sequence)

        # filter out variables already provided by globalDepsCode to avoid
        # duplicate declarations that would cause runtime errors
        domain_k = getattr(self, '_domain_knowledge', None)
        global_deps_vars = get_global_deps_vars(domain_k)
        filtered_declared_variables = []
        for var_decl in declared_variables:
            is_global_dep = False
            for global_var in global_deps_vars:
                # Matches const mcData = ... or const Vec3 = ... or const { Vec3 } = ...
                if re.search(rf'\b{global_var}\b', var_decl):
                    is_global_dep = True
                    self.logger.info(f"\033[33m[Graph Planner] Skipping variable declaration (already provided by globalDepsCode): {var_decl[:50]}...\033[0m")
                    break
            if not is_global_dep:
                filtered_declared_variables.append(var_decl)

        return filtered_declared_variables

    def _resolve_composition_parameters(
        self,
        skill_sequence: List[SkillCallContext],
        main_func_name: str,
        skill_metadata: Optional[Dict[str, Any]] = None,
        target_effects: Optional[List[Dict[str, Any]]] = None,
        task: Optional[str] = None,
    ) -> List[Optional[str]]:
        """
        Resolve parameters and generate call strings for each skill in the composition.

        For each skill, resolves parameters from metadata/target_effects/LLM and generates
        the JavaScript call string (e.g., "    await craftPlanks(bot, 4);").

        Args:
            skill_sequence: Skill execution sequence
            main_func_name: Name of the wrapper function (to detect self-references)
            skill_metadata: Skill metadata for parameter lookup
            target_effects: Target effects for parameter determination
            task: Task description

        Returns:
            List of (call_str or None) per skill in skill_sequence. None if skill was skipped.
        """
        call_strings = []
        # Track the expected outputs of the skill chain for subsequent parameter inference
        planning_state = PlanningState()

        for skill_ctx in skill_sequence:
            skill_name = skill_ctx.skill_name
            precondition_context = skill_ctx.precondition_context  # Passed to parameter inference
            call_str = None  # Initialize; set in each branch

            node = self.skill_graph_manager.get_node(skill_name)
            if not node:
                call_strings.append(None)
                continue

            # Get skill code (only function definitions, no execution code)
            skill_code = node.code

            # Check whether the skill code calls the main function name (this would cause recursion)
            if main_func_name in skill_code:
                # Check whether it is a real function call (not a function definition or comment)
                # Exclude the function definition line
                skill_code_without_def = re.sub(
                    rf'async\s+function\s+{re.escape(main_func_name)}\s*\([^)]*\)\s*\{{',
                    '',
                    skill_code
                )
                # Exclude comments
                skill_code_without_comments = re.sub(r'//.*?$|/\*.*?\*/', '', skill_code_without_def, flags=re.MULTILINE | re.DOTALL)
                # Check whether there is a function call
                recursive_call_pattern = rf'\b{re.escape(main_func_name)}\s*\('
                if re.search(recursive_call_pattern, skill_code_without_comments):
                    self.logger.error(f"\033[31m[Graph Planner] Error: Skill '{skill_name}' code contains a call to the main function name '{main_func_name}', which would cause recursion\033[0m")
                    raise ValueError(
                        f"Skill '{skill_name}' code contains a call to main function '{main_func_name}'. "
                        f"This would cause recursive calls. Please check the skill code and ensure it doesn't call the main function."
                    )

            # Prefer parameter names from metadata (ensures consistency)
            # Only fall back to extracting from code if metadata is unavailable
            metadata_params = None
            if skill_metadata and skill_name in skill_metadata:
                metadata_params = skill_metadata[skill_name].get("parameters", {})
            elif node.parameters:
                metadata_params = node.parameters

            # Extract function name and parameters
            main_func_pattern = rf'async\s+function\s+{re.escape(skill_name)}\s*\(([^)]*)\)'
            skill_func_match = re.search(main_func_pattern, skill_code)

            if skill_func_match:
                func_name = skill_name
                params_str = skill_func_match.group(1)
            else:
                # Use the language's parser to recover the first top-level async
                # function's name and raw parameter string. The parameter string
                # must preserve default-value text (e.g. "bot, count = 1") for
                # _parse_function_parameters downstream, so we re-extract it
                # from the function's source via regex on full_code.
                skill_lang = _resolve_skill_language(self)
                parse_result = skill_lang.parse(skill_code)
                first_async = next(
                    (f for f in parse_result.functions if f.is_async), None
                )
                if first_async is not None:
                    params_match = re.search(
                        r'async\s+function\s+\w+\s*\(([^)]*)\)',
                        first_async.full_code or first_async.body or "",
                    )
                    actual_func_name = first_async.name
                    params_str = params_match.group(1) if params_match else ""
                    func_name = skill_name
                    if actual_func_name != skill_name:
                        self.logger.warning(
                            f"\033[33m[Graph Planner] Warning: Skill '{skill_name}' code has main function name '{actual_func_name}', "
                            f"but '{skill_name}' will be used to generate calls so the Skill Wrapper can track correctly\033[0m"
                        )
                        print(f"\033[33m[Graph Planner] Note: generating call await {skill_name}(...) instead of await {actual_func_name}(...)\033[0m")
                    func_match = True
                else:
                    func_match = None

            # Unified handling of the func_match case (for compatibility with subsequent code)
            if skill_func_match or func_match:
                # Check whether the skill function name conflicts with the main function name (defensive: skip rather than crash)
                if func_name == main_func_name:
                    self.logger.warning(
                        f"\033[33m[Graph Planner] Skipping self-referencing skill '{skill_name}' "
                        f"(function name '{func_name}' conflicts with wrapper function name '{main_func_name}')\033[0m"
                    )
                    call_strings.append(None)
                    continue  # Skip this skill to avoid recursion

                # Parse parameters (correctly handles default values containing arrays, objects, and destructured params)
                code_params = self._parse_function_parameters(params_str)

                # Detect whether there is a destructured parameter (second parameter starting with {)
                has_destructured_param = False
                destructured_param_str = None
                for param in code_params:
                    if param != "bot" and self._is_destructured_parameter(param):
                        has_destructured_param = True
                        destructured_param_str = param
                        break

                # If it is a destructured parameter, use the special handling logic
                if has_destructured_param:
                    self.logger.info(f"\033[36m[Graph Planner] Detected destructured parameter: {skill_name}\033[0m")

                    # Parse the properties of the destructured parameter
                    destructured_props = self._parse_destructured_parameter(destructured_param_str)

                    # Hardening: if parsing failed but metadata has parameter info, use metadata as fallback
                    if not destructured_props and metadata_params:
                        destructured_props = {
                            k: v.get('default') if isinstance(v, dict) else v
                            for k, v in metadata_params.items()
                        }
                        self.logger.info(f"\033[33m[Graph Planner] Recovered destructured parameters from metadata: {list(destructured_props.keys())}\033[0m")

                    self.logger.info(f"\033[36m[Graph Planner]   Destructured parameter properties: {list(destructured_props.keys())}\033[0m")

                    # Get parameter values for each property
                    param_values = {}
                    for prop_name, default_value in destructured_props.items():
                        param_value = self._get_parameter_value(
                            prop_name, skill_name, target_effects, skill_metadata,
                            use_llm=self.use_llm_for_parameters,
                            precondition_context=precondition_context,
                            expected_inventory=planning_state.get_effective_inventory()
                        )

                        if param_value not in ["undefined", "null"]:
                            param_values[prop_name] = param_value

                    # Generate an object-form call
                    if param_values:
                        obj_props = []
                        for prop_name, prop_value in param_values.items():
                            obj_props.append(f"{prop_name}: {prop_value}")
                        obj_str = "{ " + ", ".join(obj_props) + " }"
                        call_str = f"    await {func_name}(bot, {obj_str});"
                    else:
                        call_str = f"    await {func_name}(bot);"

                    self.logger.info(f"\033[36m[Graph Planner]   Generated call: {call_str.strip()}\033[0m")
                else:
                    # Handling logic for ordinary positional parameters
                    call_args = []

                    if metadata_params:
                        self.logger.info(f"\033[36m[Graph Planner] Using parameter names from metadata to generate call: {skill_name}\033[0m")
                        self.logger.info(f"\033[36m[Graph Planner]   Metadata parameters: {list(metadata_params.keys())}\033[0m")

                        code_param_names = []
                        for param in code_params:
                            if param != "bot":
                                param_name = param.split('=')[0].strip()
                                code_param_names.append(param_name)

                        self.logger.info(f"\033[36m[Graph Planner]   Parameters in code: {code_param_names}\033[0m")

                        for i, code_param in enumerate(code_params):
                            if code_param == "bot":
                                call_args.append("bot")
                            else:
                                code_param_name = code_param.split('=')[0].strip()

                                if code_param_name in metadata_params:
                                    param_name = code_param_name
                                else:
                                    param_name = code_param_name
                                    self.logger.warning(f"\033[33m[Graph Planner]   Warning: parameter '{code_param_name}' in code is not in metadata\033[0m")

                                param_value = self._get_parameter_value(
                                    param_name, skill_name, target_effects, skill_metadata,
                                    use_llm=self.use_llm_for_parameters,
                                    precondition_context=precondition_context,
                                    expected_inventory=planning_state.get_effective_inventory()
                                )
                                call_args.append(param_value)

                        if call_args:
                            call_str = f"    await {func_name}({', '.join(call_args)});"
                        else:
                            call_str = f"    await {func_name}(bot);"

                        self.logger.info(f"\033[36m[Graph Planner]   Generated call: {call_str.strip()}\033[0m")
                    else:
                        self.logger.warning(f"\033[33m[Graph Planner]   Warning: {skill_name} has no metadata; extracting parameters from code\033[0m")
                        for param in code_params:
                            if param == "bot":
                                call_args.append("bot")
                            else:
                                param_name = param.split('=')[0].strip()

                                param_value = self._get_parameter_value(
                                    param_name, skill_name, target_effects, skill_metadata,
                                    use_llm=self.use_llm_for_parameters,
                                    precondition_context=precondition_context,
                                    expected_inventory=planning_state.get_effective_inventory()
                                )
                                call_args.append(param_value)

                        # Layer 1.5 helper: trim trailing 'undefined' args that
                        # correspond to params with defaults in the function
                        # signature. JS will use the defaults automatically.
                        # This preserves resolved values (e.g. toolType='axe' from
                        # SPECIALIZED_WHEN strategy) instead of falling through to
                        # an LLM call that has no awareness of the binding.
                        # Also keep code_params aligned with call_args so the
                        # subsequent should_use_llm check doesn't trigger on
                        # trimmed-away params (e.g. 'distance' keyword).
                        active_code_params = list(code_params)
                        while len(call_args) > 1:  # keep at least 'bot'
                            last_idx = len(call_args) - 1
                            if call_args[last_idx] != "undefined":
                                break
                            corresponding_code_param = active_code_params[last_idx]
                            if "=" in corresponding_code_param:  # has default
                                call_args.pop()
                                active_code_params.pop()
                            else:
                                break  # required param missing — keep undefined for LLM fallback

                        # Hybrid strategy: check whether the LLM should be used to generate the call
                        # Use the trimmed code_params so optional params we already
                        # decided to omit don't trigger the LLM-fallback rules
                        # (e.g. 'distance' keyword, 3+ params count).
                        non_bot_param_names = [p.split('=')[0].strip() for p in active_code_params if p != "bot"]

                        should_use_llm = self._should_use_llm_for_skill_call(call_args, non_bot_param_names)

                        if should_use_llm and self.llm:
                            self.logger.info(f"\033[36m[Graph Planner] Rule-based match is uncertain; using LLM to generate call: {skill_name}\033[0m")
                            self.logger.info(f"\033[36m[Graph Planner]   Rule-matched parameters: {call_args}\033[0m")

                            skill_description = node.description if node else ""

                            llm_call_str = self._generate_skill_call_with_llm(
                                skill_name=skill_name,
                                func_name=func_name,
                                skill_description=skill_description,
                                parameters_info=metadata_params or {},
                                target_effects=target_effects or [],
                                task=task,
                                skill_effects=node.expected_effects if node else None
                            )

                            if llm_call_str:
                                call_str = "    " + llm_call_str.strip()
                            else:
                                self.logger.warning(f"\033[33m[Graph Planner] LLM generation failed; using rule-matched result\033[0m")
                                if call_args:
                                    call_str = f"    await {func_name}({', '.join(call_args)});"
                                else:
                                    call_str = f"    await {func_name}(bot);"
                        else:
                            if call_args:
                                call_str = f"    await {func_name}({', '.join(call_args)});"
                            else:
                                call_str = f"    await {func_name}(bot);"

            # Unified handling at the end of the loop: update expected inventory
            call_strings.append(call_str)
            self._update_expected_inventory(planning_state.inventory, node)

        return call_strings

    def _assemble_composition_body(
        self,
        main_func_name: str,
        declared_variables: List[str],
        call_strings: List[Optional[str]],
        skill_sequence: List[SkillCallContext],
    ) -> str:
        """
        Assemble the final JavaScript composition function from its parts.

        Wraps skill calls in a try/catch block, checks for recursive calls, and
        sanitizes Python syntax to JavaScript.

        Args:
            main_func_name: Name of the wrapper function
            declared_variables: Variable declarations to include
            call_strings: List of call strings (or None for skipped skills)
            skill_sequence: Original skill sequence (for recursive call checking)

        Returns:
            Complete JavaScript function code string
        """
        code_lines = []
        code_lines.append(f"async function {main_func_name}(bot) {{")
        code_lines.append("  try {")

        # Declare these variables in the main function scope to ensure they are available
        if declared_variables:
            self.logger.info(f"\033[36m[Graph Planner] Detected variables declared in skill code: {declared_variables}\033[0m")
            for var_decl in declared_variables:
                code_lines.append(f"  {var_decl}")
            code_lines.append("")

        # Append the call for each skill
        for call_str in call_strings:
            if call_str:
                code_lines.append(call_str)

        code_lines.append("    bot.chat('Task completed successfully');")
        code_lines.append("  } catch (error) {")
        code_lines.append("    bot.chat(`Error: ${error.message}`);")
        code_lines.append("    throw error;")
        code_lines.append("  }")
        code_lines.append("}")

        # Skill function definitions are not appended here — the runtime
        # environment already has them available.
        generated_code = "\n".join(code_lines)

        # Debug: check whether the generated code contains a recursive call
        if main_func_name in generated_code:
            func_body_pattern = rf'async\s+function\s+{re.escape(main_func_name)}\s*\([^)]*\)\s*\{{(.*?)\}}'
            func_body_match = re.search(func_body_pattern, generated_code, re.DOTALL)
            if func_body_match:
                func_body = func_body_match.group(1)
                recursive_call_pattern = rf'\b{re.escape(main_func_name)}\s*\('
                if re.search(recursive_call_pattern, func_body):
                    self.logger.error(f"\033[31m[Graph Planner] Warning: in the generated code, the body of the main function {main_func_name} contains a call to itself\033[0m")
                    self.logger.error(f"\033[31m[Graph Planner] Generated code:\n{generated_code}\033[0m")
                    skill_lang = _resolve_skill_language(self)
                    for skill_ctx in skill_sequence:
                        skill_name = skill_ctx.skill_name
                        node = self.skill_graph_manager.get_node(skill_name)
                        if node:
                            parse_result = skill_lang.parse(node.code)
                            skill_func_names = [
                                f.name for f in parse_result.functions if f.is_async
                            ]
                            if skill_func_names:
                                skill_func_name = skill_func_names[0]
                                if skill_func_name == main_func_name:
                                    self.logger.error(f"\033[31m[Graph Planner] Error: Skill '{skill_name}' function name '{skill_func_name}' conflicts with main function name '{main_func_name}'!\033[0m")
                                    raise ValueError(
                                        f"Skill function name '{skill_func_name}' conflicts with main function name '{main_func_name}'. "
                                        f"This would cause recursive calls. Please rename the skill function or use a different task name."
                                    )

        # Convert Python syntax to JavaScript syntax
        generated_code = self._sanitize_python_to_js(generated_code)

        return generated_code

    def _generate_composition_code(
        self,
        skill_sequence: List[SkillCallContext],
        skill_metadata: Optional[Dict[str, Any]] = None,
        target_effects: Optional[List[Dict[str, Any]]] = None,
        task: Optional[str] = None,
    ) -> str:
        """
        Generate skill composition code.

        Args:
            skill_sequence: Skill execution sequence (list of SkillCallContext)
            skill_metadata: Skill metadata (used for parameter passing)
            target_effects: Target effects (used for determining parameters)
            task: Task description (used to generate the function name)

        Returns:
            str: The composed JavaScript code
        """
        if not skill_sequence:
            return ""

        # Generate the main function code: prefer task name, otherwise use the skill sequence
        main_func_name = self._generate_function_name(task, skill_sequence)

        # Filter self-referencing skills: the composition wrapper cannot call itself
        original_count = len(skill_sequence)
        skill_sequence = [
            ctx for ctx in skill_sequence
            if ctx.skill_name != main_func_name
        ]
        if len(skill_sequence) < original_count:
            self.logger.warning(
                f"\033[33m[Graph Planner] Filtered out {original_count - len(skill_sequence)} self-referencing skill(s) "
                f"(same as wrapper function name '{main_func_name}')\033[0m"
            )
        if not skill_sequence:
            return ""  # All skills are self-references; cannot generate a valid composition

        # Step 1: Extract requires and declarations
        declared_variables = self._extract_requires_and_declarations(skill_sequence)

        # Step 2: Resolve parameters and generate call strings
        call_strings = self._resolve_composition_parameters(
            skill_sequence=skill_sequence,
            main_func_name=main_func_name,
            skill_metadata=skill_metadata,
            target_effects=target_effects,
            task=task,
        )

        # Step 3: Assemble the final function body
        return self._assemble_composition_body(
            main_func_name=main_func_name,
            declared_variables=declared_variables,
            call_strings=call_strings,
            skill_sequence=skill_sequence,
        )

    def _sanitize_python_to_js(self, code: str) -> str:
        """
        Convert Python syntax to JavaScript syntax.

        delegated to the sanitize_python_to_js pure function.
        """
        return sanitize_python_to_js(code, logger=self.logger)

    def _generate_skill_call_with_llm(
        self,
        skill_name: str,
        func_name: str,
        skill_description: str,
        parameters_info: Dict[str, Any],
        target_effects: List[Dict[str, Any]],
        task: Optional[str] = None,
        inventory_context: Optional[str] = None,
        skill_effects: Optional[List] = None  # [Step 2+] the skill's effects declarations
    ) -> Optional[str]:
        """
        Use the LLM to generate skill call code.

        When rule-based matching cannot correctly determine parameters, use the LLM
        to understand the context and generate the correct call.

        Args:
            skill_name: Skill name
            func_name: Function name (extracted from code)
            skill_description: Description of the skill's functionality
            parameters_info: Parameter info (includes type, default value, description, etc.)
            target_effects: Target effects
            task: Task description
            inventory_context: Inventory context (optional)
            skill_effects: List of the skill's effects declarations (optional)

        Returns:
            str: The generated call statement (e.g. "await craftPickaxe(bot, null);"); returns None on failure
        """
        if not self.llm:
            return None

        try:
            # Build parameter descriptions
            params_description = []
            for param_name, param_info in parameters_info.items():
                param_type = param_info.get("type", "unknown")
                default_val = param_info.get("default", "none")
                desc = param_info.get("description", "")
                supported = param_info.get("supported_values", [])
                schema = param_info.get("schema")  # Internal structure for object-typed values

                param_desc = f"  - {param_name} ({param_type}): {desc}"
                if default_val is not None:
                    param_desc += f" [default: {default_val}]"
                if supported:
                    param_desc += f" [supported: {', '.join(str(v) for v in supported[:5])}...]"

                # Show the schema for object types
                if param_type == "object" and schema:
                    param_desc += "\n    Object fields:"
                    for field_name, field_info in schema.items():
                        field_type = field_info.get("type", "unknown")
                        field_default = field_info.get("default")
                        field_hint = field_info.get("source_hint", {})
                        transform = field_hint.get("transform") if field_hint else None

                        field_desc = f"\n      - {field_name} ({field_type})"
                        if field_default is not None:
                            field_desc += f" [default: {field_default}]"
                        if transform:
                            # Hint about how to extract the value from the task
                            field_desc += f" [extract from task's item/count]"
                        param_desc += field_desc

                params_description.append(param_desc)

            params_str = "\n".join(params_description) if params_description else "  (no parameters except bot)"

            # [Step 2+] Build the skill effects description (factual context for LLM)
            skill_effects_str = ""
            if skill_effects and target_effects:
                target_item = target_effects[0].get("item", "").lower() if target_effects else ""
                target_count = target_effects[0].get("count", 1) if target_effects else 1
                if not isinstance(target_count, (int, float)) or target_count is None:
                    target_count = 1
                effects_info = self._extract_items_from_skill_effects(skill_effects)
                guaranteed = effects_info["guaranteed"]
                or_alternatives = effects_info["or_alternatives"]

                # Check whether the target is among the guaranteed outputs
                all_items = guaranteed[:]
                for group in or_alternatives:
                    all_items.extend(group)
                target_in_guaranteed = target_item and target_item in [i.lower() for i in guaranteed]
                target_in_or = target_item and any(
                    target_item in [i.lower() for i in group] for group in or_alternatives
                )

                if target_in_guaranteed:
                    skill_effects_str = f"""
## Skill Effects Context:
- This skill's effects declare it produces: {target_item}
- Target quantity needed: {target_count}
- Other guaranteed items: {', '.join(i for i in guaranteed if i.lower() != target_item) or 'none'}
"""
                    if or_alternatives:
                        or_desc = "; ".join(f"one of [{', '.join(g)}]" for g in or_alternatives)
                        skill_effects_str += f"- OR-logic effects (may produce one of): {or_desc}\n"
                elif target_in_or:
                    # Target is in OR-logic — not guaranteed to be produced
                    matching_group = next(
                        g for g in or_alternatives if target_item in [i.lower() for i in g]
                    )
                    skill_effects_str = f"""
## Skill Effects Context:
- This skill has OR-logic effects — it may produce ONE of: {', '.join(matching_group)}
- Target item '{target_item}' is among the possible outputs but NOT guaranteed
- Target quantity needed: {target_count}
- Guaranteed items: {', '.join(guaranteed) if guaranteed else 'none'}
"""
                elif all_items:
                    skill_effects_str = f"""
## Skill Effects Context:
- Skill declared effects include: {', '.join(all_items[:5])}{'...' if len(all_items) > 5 else ''}
- Target item '{target_item}' is NOT in the effects list
"""

            system_prompt = """You are a JavaScript code generation expert for Minecraft bots.
Your task is to generate the correct function call statement for a skill.

CRITICAL RULES:
1. The first parameter is ALWAYS 'bot' - do not omit it
2. Analyze what each parameter expects based on its name and description
3. DO NOT pass the target product to a parameter that expects an input material
   - Example: plankPreference expects a plank type like "birch_planks", NOT "wooden_pickaxe"
   - Example: logType expects a log type like "oak_log", NOT "oak_planks"
4. If a parameter is optional and you're unsure of the value, use 'null' or 'undefined'
5. Consider the inventory context when available - prefer using materials already in inventory
6. IMPORTANT: If "Skill Effects Match" section indicates the skill can produce the target item,
   set the quantity parameter (total/count/amount) to match the target count.
   NEVER set quantity to 0 when the skill should produce something!

Return ONLY a JSON object:
{
  "call_statement": "await functionName(bot, arg1, arg2);",
  "reasoning": "brief explanation of parameter choices"
}"""

            # Build function signature for LLM
            sig_params = ["bot"]
            for pname, pinfo in parameters_info.items():
                default = pinfo.get("default")
                if default is not None:
                    sig_params.append(f'{pname} = {json.dumps(default)}')
                else:
                    sig_params.append(pname)
            func_signature = f"async function {func_name}({', '.join(sig_params)})"

            # Bug 6 fix: inject Phase 1 parameter_corrections so the LLM sees
            # prior-failure context. Path A (inference engine) already does
            # this via PureLLMResolver; Path B (this whole-call LLM gen) used
            # to skip corrections entirely, letting the LLM repeat the same
            # mistake across retries within a task.
            corrections_section = ""
            if hasattr(self, "skill_graph_manager") and hasattr(
                self.skill_graph_manager, "get_task_parameter_corrections"
            ):
                correction_lines = []
                for param_name in parameters_info:
                    try:
                        corrs = self.skill_graph_manager.get_task_parameter_corrections(
                            skill_name=skill_name, param_name=param_name,
                        ) or []
                    except Exception:
                        corrs = []
                    for corr in corrs:
                        passed = corr.get("passed_value")
                        suggested = corr.get("suggested_value")
                        reason = corr.get("reason", "")
                        if passed is None or suggested is None:
                            continue
                        correction_lines.append(
                            f"  - {param_name}: value {passed!r} caused failure; "
                            f"suggested {suggested!r}"
                            + (f" (reason: {reason})" if reason else "")
                        )
                if correction_lines:
                    corrections_section = (
                        "\n\nPrevious failures (from optimizer feedback — DO NOT repeat):\n"
                        + "\n".join(correction_lines)
                    )

            human_prompt = f"""Task: {task or 'Complete the target effects'}

Target Effects:
{json.dumps(target_effects, indent=2)}

Skill to Call: {skill_name}
Function Signature: {func_signature}
Description: {skill_description}
{skill_effects_str}
Parameters (in order):
{params_str}

{f"Current Inventory: {inventory_context}" if inventory_context else ""}{corrections_section}

Generate the JavaScript function call statement for this skill.
Remember:
- First parameter is always 'bot'
- DO NOT pass the target item (e.g., "wooden_pickaxe") to preference/type parameters that expect materials
- Use 'null' for optional parameters when the value should be auto-detected
- If skill effects match the target, set quantity parameter to the target count (NOT 0)
- If "Previous failures" section is present, AVOID the listed bad values

Return only JSON."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            _llm_resp = self.llm.invoke(messages)
            record_llm_usage(_llm_resp, process_type="planning", function_name="planner.code_generation._generate_skill_call_with_llm", skill_name=skill_name, task=task)
            response = _llm_resp.content

            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                call_statement = result.get("call_statement", "")
                reasoning = result.get("reasoning", "")

                if call_statement:
                    # Validate the basic format of the call statement
                    if f"await {func_name}(" in call_statement and call_statement.endswith(");"):
                        # [Step 3.3] Rule-based fallback: check and fix count=0 issues
                        call_statement = self._apply_count_parameter_fallback(
                            call_statement, target_effects
                        )
                        self.logger.info(f"\033[36m[LLM Skill Call] Generated call: {call_statement}\033[0m")
                        self.logger.info(f"\033[36m[LLM Skill Call] Reasoning: {reasoning}\033[0m")
                        return call_statement
                    else:
                        self.logger.warning(f"\033[33m[LLM Skill Call] Call format is incorrect: {call_statement}\033[0m")

        except Exception as e:
            self.logger.warning(f"\033[33m[LLM Skill Call] LLM generation failed: {e}\033[0m")

        return None

    def _should_use_llm_for_skill_call(
        self,
        call_args: List[str],
        param_names: List[str]
    ) -> bool:
        """
        Decide whether the LLM should be used to generate a skill call.

        Use the LLM when one of the following conditions is met:
        1. Any parameter value is "undefined"
        2. A parameter name contains preference-style keywords like preference/option
        3. The parameter count is >= 3 (complex call)

        Args:
            call_args: List of parameter values produced by rule-based matching
            param_names: List of parameter names

        Returns:
            bool: Whether to use the LLM
        """
        # Check whether there is an undefined parameter (except for bot)
        non_bot_args = [arg for arg in call_args if arg != "bot"]
        if any(arg == "undefined" for arg in non_bot_args):
            return True

        # Check whether there is a preference-style parameter (config-type parameters)
        config_indicators = [
            "priority", "preference", "prefer", "option", "options",
            "config", "timeout", "distance", "fallback", "default",
            "selection", "choices", "allowed"
        ]
        for param_name in param_names:
            if any(ind in param_name.lower() for ind in config_indicators):
                return True

        # Complex call (3 or more non-bot parameters)
        if len(non_bot_args) >= 3:
            return True

        return False

    def verify_recursive_call(
        self,
        function_name: str,
        function_body: str,
        full_code: str
    ) -> bool:
        """Public API: Use LLM to verify if a function contains genuine recursive calls."""
        return self._verify_recursive_call_with_llm(function_name, function_body, full_code)

    def _verify_recursive_call_with_llm(
        self,
        function_name: str,
        function_body: str,
        full_code: str
    ) -> bool:
        """
        Use the LLM to verify whether the function really has a recursive call.

        Args:
            function_name: Function name
            function_body: Function body code
            full_code: Full code (used for context)

        Returns:
            bool: True if confirmed to be a recursive call, False if it is a false positive
        """
        if not self.llm:
            # If no LLM is available, default to True (preserve previous behavior)
            self.logger.warning(f"\033[33m[LLM Verification] No LLM available; defaulting to assume recursive call\033[0m")
            return True

        try:
            system_prompt = """You are a code analysis expert. Determine if a JavaScript function calls itself recursively.

A function has a RECURSIVE CALL if:
1. The function body contains a call to the same function name (e.g., `functionName()`)
2. This call is NOT inside a comment, string literal, or typeof check
3. This call would actually execute during runtime (not just a reference)

A function does NOT have a recursive call if:
1. The function name appears only in comments or strings
2. The function name appears in a typeof check (e.g., `typeof functionName === 'function'`)
3. The function name appears in a different function's call (e.g., calling a different function with a similar name)
4. The function name is part of a larger identifier (e.g., `myFunctionName` contains `functionName` but is different)

Return ONLY a JSON object with this structure:
{
    "is_recursive": true/false,
    "reason": "brief explanation"
}"""

            human_prompt = f"""Analyze this JavaScript code:

Function name: {function_name}

Function body:
```javascript
{function_body}
```

Full code context:
```javascript
{full_code[:1000]}  // First 1000 chars for context
```

Does the function '{function_name}' call itself recursively in the function body? Return JSON only."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            _llm_resp = self.llm.invoke(messages)
            record_llm_usage(_llm_resp, process_type="planning", function_name="planner.code_generation._verify_recursive_call_with_llm")
            response = _llm_resp.content

            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_recursive = result.get("is_recursive", True)  # Default to True for safety
                reason = result.get("reason", "No reason provided")

                self.logger.info(f"\033[36m[LLM Verification] Recursive-call verification result: {is_recursive}\033[0m")
                self.logger.info(f"\033[36m[LLM Verification] Reason: {reason}\033[0m")

                return is_recursive
            else:
                # If JSON cannot be parsed, default to True (safe default)
                self.logger.warning(f"\033[33m[LLM Verification] Unable to parse LLM response; defaulting to assume recursive call\033[0m")
                self.logger.debug(f"\033[33m[LLM Verification] LLM response: {response[:200]}\033[0m")
                return True

        except Exception as e:
            # If LLM verification fails, default to True (safe default)
            self.logger.warning(f"\033[33m[LLM Verification] LLM verification failed: {e}; defaulting to assume recursive call\033[0m")
            import traceback
            self.logger.debug(f"\033[33m[LLM Verification] Error details: {traceback.format_exc()}\033[0m")
            return True

    def _map_task_params_to_skill_params_with_llm(
        self,
        task_params: Dict[str, Any],
        skill_params: Dict[str, Any],
        skill_name: str,
        task: str
    ) -> Dict[str, Any]:
        """
        Use the LLM to map parameters extracted from the task to skill parameters.

        When rule-based mapping fails, use the LLM for semantic mapping.
        For example: targetBlockNames -> blockName, targetItem -> resultItemName

        Args:
            task_params: Parameters extracted from the task, e.g. {'targetBlockNames': 'coal_ore', 'targetItem': 'coal'}
            skill_params: The skill's parameter definitions, e.g. {'blockName': {...}, 'resultItemName': {...}}
            skill_name: Skill name
            task: Original task description

        Returns:
            Mapped parameter dict {skill_param_name: value}
        """
        if not task_params:
            return {}

        # 1. First try rule-based mapping
        rule_mappings = {
            'targetBlockNames': ['blockName', 'block_name', 'targetBlock', 'blockType'],
            'targetItem': ['resultItemName', 'itemName', 'result_item', 'item', 'targetProduct'],
            'targetCount': ['count', 'amount', 'quantity'],
            'count': ['count', 'amount', 'quantity'],
        }

        mapped = {}
        unmapped_task_params = dict(task_params)

        for task_key, skill_candidates in rule_mappings.items():
            if task_key in unmapped_task_params:
                for candidate in skill_candidates:
                    if candidate in skill_params:
                        mapped[candidate] = unmapped_task_params.pop(task_key)
                        self.logger.info(f"[Param Mapping] Rule mapping: {task_key} -> {candidate}")
                        break

        # If all parameters are mapped, or there is no LLM, return directly
        if not unmapped_task_params or not self.llm:
            return mapped

        # 2. Use the LLM for semantic mapping
        self.logger.info(f"[LLM Mapping] Unmapped parameters: {list(unmapped_task_params.keys())}")

        system_prompt = """You are a parameter mapping expert for Minecraft bot skills.
Map task-extracted parameters to skill function parameters based on semantic meaning.

Return JSON: {"mappings": {"task_param_name": "skill_param_name", ...}}

Rules:
1. Map based on semantic meaning, not just name similarity
2. Only map to parameters that exist in the skill
3. If no good mapping exists, don't include it in the output"""

        human_prompt = f"""Task: {task}
Skill: {skill_name}

Task parameters (need mapping): {json.dumps(unmapped_task_params)}
Skill parameters (available): {list(skill_params.keys())}

Map each task parameter to the most appropriate skill parameter."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            _llm_resp = self.llm.invoke(messages)
            record_llm_usage(_llm_resp, process_type="planning", function_name="planner.code_generation._map_task_params_to_skill_params_with_llm", skill_name=skill_name, task=task)
            response = _llm_resp.content

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                llm_mappings = result.get("mappings", {})

                for task_key, skill_key in llm_mappings.items():
                    if task_key in unmapped_task_params and skill_key in skill_params:
                        mapped[skill_key] = unmapped_task_params[task_key]
                        self.logger.info(f"[LLM Mapping] LLM mapping: {task_key} -> {skill_key}")
        except Exception as e:
            self.logger.warning(f"[LLM Mapping] LLM mapping failed: {e}")

        return mapped

    def _generate_plan_description(
        self,
        skill_sequence: List[SkillCallContext],
        target_effects: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a plan description.

        delegated to the generate_plan_description pure function.
        """
        return generate_plan_description(skill_sequence, target_effects)

    def _parse_function_parameters(self, params_str: str) -> List[str]:
        """
        Parse a function parameter string, correctly handling default values that
        contain arrays, objects, and destructured parameters.

        delegated to the parse_function_parameters pure function.
        """
        return parse_function_parameters(params_str)

    def _is_destructured_parameter(self, param_str: str) -> bool:
        """
        Check whether a parameter is a destructured parameter (starts with {).

        delegated to the is_destructured_parameter pure function.
        """
        return is_destructured_parameter(param_str)

    def _strip_js_comments(self, code: str) -> str:
        """
        Remove JavaScript comments while preserving string contents.

        delegated to the strip_js_comments pure function.
        """
        return strip_js_comments(code)

    def _validate_param_name(self, param_name: str) -> tuple:
        """
        Validate whether a parameter name is a valid JavaScript identifier.

        delegated to the validate_param_name pure function.
        """
        return validate_param_name(param_name)

    def _parse_destructured_parameter(self, param_str: str) -> Dict[str, Any]:
        """
        Parse a destructured parameter, extracting property names and default values.

        Args:
            param_str: Destructured parameter string, e.g. "{ toolName = \"wooden_axe\", count = 1 } = {}"

        Returns:
            Dict[str, Any]: Property dict, e.g. {"toolName": "wooden_axe", "count": 1}
        """
        result = {}
        has_invalid_params = False

        # === Layer 1: first strip JavaScript comments ===
        clean_param_str = self._strip_js_comments(param_str)

        # Extract the content within the braces
        # Format: { prop1 = val1, prop2 = val2 } = {}
        # Use re.DOTALL so . matches newlines, to support multi-line destructured parameters
        match = re.match(r'\{\s*(.*?)\s*\}\s*=', clean_param_str, re.DOTALL)
        if not match:
            # Try the format without a default empty object: { prop1 = val1, prop2 = val2 }
            match = re.match(r'\{\s*(.*?)\s*\}', clean_param_str, re.DOTALL)

        if not match:
            return result

        inner_content = match.group(1)

        # Use the same logic as _parse_function_parameters to parse the inner properties
        props = []
        current_prop = ""
        bracket_depth = 0
        brace_depth = 0
        in_string = False
        string_char = None

        for char in inner_content:
            if char in ['"', "'"] and (len(current_prop) == 0 or current_prop[-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                current_prop += char
            elif in_string:
                current_prop += char
            elif char == '{':
                brace_depth += 1
                current_prop += char
            elif char == '}':
                brace_depth -= 1
                current_prop += char
            elif char == '[':
                bracket_depth += 1
                current_prop += char
            elif char == ']':
                bracket_depth -= 1
                current_prop += char
            elif char == ',':
                if bracket_depth == 0 and brace_depth == 0:
                    if current_prop.strip():
                        props.append(current_prop.strip())
                    current_prop = ""
                else:
                    current_prop += char
            else:
                current_prop += char

        if current_prop.strip():
            props.append(current_prop.strip())

        # Parse each property
        for prop in props:
            # Format: propName = defaultValue or propName
            if '=' in prop:
                parts = prop.split('=', 1)
                prop_name = parts[0].strip()
                default_value = parts[1].strip()

                # === Layer 2: validate the parameter name ===
                is_valid, cleaned_name, error = self._validate_param_name(prop_name)
                if not is_valid:
                    has_invalid_params = True
                    self.logger.warning(f"[Param Validation] Invalid param name: '{prop_name[:50]}...' ({error})")
                    # Try to extract a valid identifier from the end
                    recovery_match = re.search(r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*$', prop_name)
                    if recovery_match:
                        cleaned_name = recovery_match.group(1)
                        self.logger.info(f"[Param Validation] Recovered param name: '{cleaned_name}'")
                    else:
                        self.logger.warning(f"[Param Validation] Could not recover param name, skipping")
                        continue

                prop_name = cleaned_name

                # Try to parse the default value
                try:
                    # Handle strings
                    if (default_value.startswith('"') and default_value.endswith('"')) or \
                       (default_value.startswith("'") and default_value.endswith("'")):
                        result[prop_name] = default_value[1:-1]
                    # Handle numbers
                    elif default_value.isdigit() or (default_value.replace('.', '').isdigit() and '.' in default_value):
                        result[prop_name] = float(default_value) if '.' in default_value else int(default_value)
                    # Handle booleans
                    elif default_value.lower() in ['true', 'false']:
                        result[prop_name] = default_value.lower() == 'true'
                    else:
                        result[prop_name] = default_value
                except (ValueError, AttributeError, TypeError):
                    result[prop_name] = default_value
            else:
                # Property without a default value
                prop_name = prop.strip()

                # === Layer 2: validate the parameter name ===
                is_valid, cleaned_name, error = self._validate_param_name(prop_name)
                if not is_valid:
                    has_invalid_params = True
                    self.logger.warning(f"[Param Validation] Invalid param name (no default): '{prop_name[:50]}...' ({error})")
                    continue

                result[cleaned_name] = None

        return result

    def _generate_function_name(self, task: Optional[str], skill_sequence: List[SkillCallContext]) -> str:
        """
        Generate a function name.

        delegated to the generate_function_name pure function.
        """
        return generate_function_name(task, skill_sequence)

    def _normalize_task_name(self, task: str) -> str:
        """
        Normalize a task name into a function name.

        delegated to the normalize_task_name pure function.
        """
        return normalize_task_name(task)

    def _normalize_task_key(self, task: str) -> str:
        """Normalize a task description into a key used for history lookup."""
        if not task:
            return ""
        # Simple normalization: lowercase, strip extra whitespace
        key = task.lower().strip()
        # Extract key action words
        action_words = ["mine", "craft", "place", "smelt", "kill", "collect"]
        for word in action_words:
            if word in key:
                # Found the main action; use it as part of the key
                return f"{word}_{key[:50]}"  # Truncate to avoid being too long
        return key[:50]
