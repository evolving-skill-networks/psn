"""
PromptRenderingMixin for ParameterizedActionAgent

Methods:
    _format_parameterized_skill: Format a parameterized skill for display
    render_system_message: Render system message with parameterized skills support
"""

import re

from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import SystemMessage


class PromptRenderingMixin:
    """Prompt rendering helpers for ParameterizedActionAgent."""

    def _format_parameterized_skill(self, skill_code: str, skill_name: str, parameters: dict = None) -> str:
        """
        Format a parameterized skill for display in the prompt

        Args:
            skill_code: The skill code
            skill_name: The skill name
            parameters: Dictionary of parameter metadata (optional)

        Returns:
            Formatted skill code with parameter information
        """
        if not parameters:
            return skill_code

        # Extract function signature
        func_pattern = r'async\s+function\s+(\w+)\s*\([^)]*\)\s*{'
        match = re.search(func_pattern, skill_code)

        if not match:
            return skill_code

        # Build parameter documentation
        param_docs = []
        for param_name, param_info in parameters.items():
            param_type = param_info.get('type', 'any')
            default = param_info.get('default', None)
            description = param_info.get('description', '')

            param_doc = f"  // @param {{{param_type}}} {param_name}"
            if default is not None:
                param_doc += f" (default: {default})"
            if description:
                param_doc += f" - {description}"
            param_docs.append(param_doc)

        # Insert parameter documentation after function signature
        if param_docs:
            doc_block = "\n".join(param_docs)
            formatted_code = re.sub(
                func_pattern,
                lambda m: m.group(0) + f"\n{doc_block}\n",
                skill_code,
                count=1
            )
            return formatted_code

        return skill_code

    def render_system_message(self, skills=[], skill_metadata=None, reuse_skill_hint=None, task=""):
        """
        Render system message with parameterized skills support

        Improvement 4: clarify the current function's context when generating code to avoid confusing old and new parameter formats

        Args:
            skills: List of skill code strings
            skill_metadata: Optional dict mapping skill names to parameter metadata
                Format: {
                    "skillName": {
                        "parameters": {
                            "paramName": {
                                "type": "number|string|boolean",
                                "default": default_value,
                                "description": "parameter description"
                            }
                        }
                    }
                }
                If None, behaves like standard ActionAgent
            reuse_skill_hint: Optional tuple (skill_name, call_example) suggesting to reuse an existing skill

        Returns:
            SystemMessage with formatted skills
        """
        # Always use parameterized template for ParameterizedActionAgent to encourage parameterized skills
        # This is important for graph mode where parameterized skills are preferred
        if self._domain_knowledge:
            # Use model-specific template if available
            from skillnet.core.model_profile import detect_model_profile
            _mp = detect_model_profile(getattr(self, 'model_name', '') or '')
            try:
                system_template = self._domain_knowledge.get_system_prompt_template(
                    model_profile=_mp.name
                )
            except TypeError:
                system_template = self._domain_knowledge.get_system_prompt_template()
        else:
            system_template = ""  # domain-owned

        # Improvement 4: add context info to make the current function's context explicit
        # Extract every skill's function name and parameter info to generate the context hint
        # Entry parameter is domain-configurable (e.g. "bot" for Minecraft)
        _dk = getattr(self, '_domain_knowledge', None)
        _entry_param = _dk.get_entry_parameter_name() if _dk else "bot"
        skill_context_info = []
        semantic_annotations = []  # Track semantic annotations
        if skill_metadata:
            for skill_name, metadata in skill_metadata.items():
                params = metadata.get("parameters", {})
                skill_desc = metadata.get("description", "")  # get the skill's overall description
                if params:
                    param_list = [_entry_param]
                    for param_name, param_info in params.items():
                        param_type = param_info.get("type", "unknown")
                        param_default = param_info.get("default", None)
                        param_desc = param_info.get("description", "")
                        param_semantic = param_info.get("semantic", None)

                        # Build the parameter string
                        param_str = f"{param_name} ({param_type})"

                        # Add semantic annotation (only for quantity-related params)
                        if param_semantic == "target_total":
                            param_str += " [ENSURE: target total]"
                            semantic_annotations.append(f"{skill_name}.{param_name}=[ENSURE]")
                        elif param_semantic == "delta":
                            param_str += " [DELTA: additional amount]"
                            semantic_annotations.append(f"{skill_name}.{param_name}=[DELTA]")

                        # Add default value
                        if param_default is not None:
                            if isinstance(param_default, list):
                                param_str += f" = {param_default}"
                            elif isinstance(param_default, str):
                                param_str += f' = "{param_default}"'
                            else:
                                param_str += f" = {param_default}"
                        param_list.append(param_str)

                    # build the signature line; only show skill description when code isn't passed
                    signature = f"- {skill_name}({', '.join(param_list)})"
                    if skill_desc and not self.include_skill_code:
                        signature += f"\n    // {skill_desc}"
                    skill_context_info.append(signature)

        # Log the semantic annotations
        if semantic_annotations:
            print(f"\033[36m[Semantic] Adding semantic annotations to prompt: {', '.join(semantic_annotations[:5])}" +
                  (f"... and {len(semantic_annotations)} more" if len(semantic_annotations) > 5 else "") + "\033[0m")

        # If there is skill context info, add it to the template
        context_note = ""
        if skill_context_info:
            context_note = "\n\nIMPORTANT CONTEXT NOTES:\n"
            context_note += "When generating code, please note:\n"
            context_note += f"1. CRITICAL: ALL skill functions require '{_entry_param}' as the FIRST parameter. Always call skills like: await skillName({_entry_param}, ...other_args)\n"
            context_note += "2. Use the EXACT parameter names and types from the skill metadata above.\n"
            context_note += "3. Do NOT confuse parameter formats between different versions of the same skill.\n"
            context_note += "4. If calling an existing skill, use its CURRENT parameter signature (from metadata).\n"
            context_note += "5. Do NOT call a function recursively (calling itself). If you need retry logic, use a while loop with a max attempts counter instead.\n"
            context_note += f"\nAvailable skills with their parameter signatures ({_entry_param} is always first):\n"
            context_note += "\n".join(skill_context_info[:10])  # Show only the first 10
            if len(skill_context_info) > 10:
                context_note += f"\n... and {len(skill_context_info) - 10} more skills"
            context_note += "\n"

            # add naming guidance to prevent the LLM from using different function names every time it generates code
            # This is one of the root causes of "Identifier has already been declared" errors
            if skill_metadata:
                existing_names = list(skill_metadata.keys())
                context_note += "\n## NAMING CONSISTENCY WARNING ##\n"
                context_note += "The following skill names ALREADY EXIST in the system:\n"
                context_note += ", ".join(existing_names[:20])
                if len(existing_names) > 20:
                    context_note += f", ... and {len(existing_names) - 20} more"
                context_note += "\n\n"
                context_note += "**CRITICAL RULES for helper function naming:**\n"
                context_note += "- If you need functionality X and a skill named 'setupY' already exists, CALL 'setupY' instead of creating 'setupYBlock' or 'setupYNew'\n"
                context_note += "- If no existing skill matches, create a NEW skill with a UNIQUE name (not similar to existing)\n"
                context_note += "- AVOID creating multiple similar skills like: setupCraftingTable, setupCraftingTableBlock, setupCraftingTableNearby\n"
                context_note += "- BAD: Creating 'ensureLogsAvailable' when 'ensureLogs' already exists\n"
                context_note += "- GOOD: Call existing 'ensureLogs' directly, or use a clearly different name if different functionality needed\n"

        # add explicit reuse guidance, emphasizing that existing skills should be called
        # adjust the guidance text based on include_skill_code
        if skills and len(skills) > 0:
            if self.include_skill_code:
                context_note += "\n\n## CODE REUSE GUIDANCE ##\n"
                context_note += "The skill code below contains COMPLETE IMPLEMENTATIONS you should reuse.\n"
                context_note += "**CRITICAL**: Before implementing any logic, check if the skills above already do it.\n"
                context_note += "- For placement logic: Look for functions like `findPlacementPosition`, `ensureCraftingTable`\n"
                context_note += "- For crafting logic: Look for functions like `craftTool`, `craftItem`\n"
                context_note += "- For resource gathering: Look for functions like `ensureResource`, `mineBlock`\n"
                context_note += "- For inventory checks: Look for functions like `getItemCount`, `getByVariant`\n"
                context_note += "\n**DO NOT reimplement logic that already exists in the skills above.**\n"
                context_note += "**ALWAYS call existing functions instead of writing similar code.**\n"
            else:
                # guidance text used when code is not passed
                context_note += "\n\n## SKILL CALLING GUIDANCE ##\n"
                context_note += "The skill signatures above show available functions you can call.\n"
                context_note += "**CRITICAL**: Call existing skills by name instead of reimplementing their logic.\n"
                context_note += "- Check [ENSURE] vs [DELTA] annotations to pass correct count values\n"
                context_note += "- [ENSURE]: pass TARGET TOTAL you want in inventory\n"
                context_note += "- [DELTA]: pass ADDITIONAL amount to produce\n"
                context_note += "\n**DO NOT reimplement logic that already exists in the skills above.**\n"
                context_note += "**ALWAYS call existing functions instead of writing similar code.**\n"

        base_skills = (
            self._domain_knowledge.get_control_primitives()
            if self._domain_knowledge else []
        )

        # Format skills with parameter information
        formatted_skills = []
        for skill_code in skills:
            # Try to extract skill name from code
            func_pattern = r'async\s+function\s+(\w+)\s*\('
            match = re.search(func_pattern, skill_code)

            if match and skill_metadata:
                skill_name = match.group(1)
                if skill_name in skill_metadata and "parameters" in skill_metadata[skill_name]:
                    formatted_skill = self._format_parameterized_skill(
                        skill_code,
                        skill_name,
                        skill_metadata[skill_name]["parameters"]
                    )
                    formatted_skills.append(formatted_skill)
                else:
                    formatted_skills.append(skill_code)
            else:
                formatted_skills.append(skill_code)

        # include_skill_code determines whether to include the full skill code.
        # Model-aware primitive loading via ModelProfile picks the right context
        # directory (e.g. qwen3-specific variants) for the active LLM.
        from skillnet.core.model_profile import detect_model_profile
        _model_name = getattr(self, 'model_name', '') or ''
        _profile = detect_model_profile(_model_name)

        _dk = self._domain_knowledge
        if _profile.name != "default" and _profile.name != "gpt-5" and _dk:
            primitive_code = _dk.load_control_primitive_context(
                base_skills, model_profile=_profile.name
            )
        elif _dk:
            primitive_code = _dk.load_control_primitive_code()
        else:
            primitive_code = []

        if self.include_skill_code:
            programs = "\n\n".join(primitive_code + formatted_skills)
        else:
            # Pass only control primitives, omitting full code of learned skills
            programs = "\n\n".join(primitive_code)
        # domain-aware response format
        response_format = ""
        if self._domain_knowledge:
            response_format = self._domain_knowledge.get_action_response_format()
        if not response_format:
            response_format = ""  # domain-owned

        # Add reuse skill hint if provided
        reuse_hint = ""
        if reuse_skill_hint:
            skill_name, call_example = reuse_skill_hint
            reuse_hint = f"\n\nIMPORTANT: The task can be completed by directly calling the existing skill '{skill_name}' with appropriate parameters. You should call it like this: {call_example}\nDO NOT create a new skill function. Instead, directly call the existing skill with the correct parameters."

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(
            programs=programs, response_format=response_format
        )

        # Improvement 4: add context info to the system message
        if context_note:
            system_message.content += context_note

        # Append reuse hint to system message
        if reuse_hint:
            system_message.content += reuse_hint

        # Dynamic knowledge injection based on task
        if task and self._domain_knowledge:
            skill_names = list(skill_metadata.keys()) if skill_metadata else []
            action_knowledge = self._domain_knowledge.get_action_prompt_knowledge(
                task, skill_names
            )
            if action_knowledge:
                system_message.content += action_knowledge

        assert isinstance(system_message, SystemMessage)
        return system_message
