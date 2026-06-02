"""
StepPlanMixin: Phases 1-3 of step() pipeline.

Phase 1: Graph planning + LLM fallback.
Phase 2: Code extraction + redundancy detection.
Phase 3: Recursive call error handling.
"""

import copy
import re

from skillnet._psn_impl.step_context import StepContext
from skillnet.core.dk_registry import get_domain_knowledge
from skillnet.utils.stats_tracker import record_llm_usage


def _resolve_skill_language(agent):
    """Resolve a SkillLanguage instance for the given step-pipeline agent.

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


class StepPlanMixin:
    """Phases 1-3: Planning, code extraction, and recursive error handling."""

    def _step_plan(self, ctx: StepContext):
        """Phase 1: Graph planning attempt + LLM fallback + conversation append."""
        # Check if we should use graph planner (when planner_mode="graph" or "adaptive")
        if (self.planner and
            self.planner_mode in ["graph", "adaptive"]):

            # Extract current state from last_events
            current_state = self._extract_current_state(self.last_events if hasattr(self, 'last_events') and self.last_events else [])

            # Retrieve available skills (always uses ParameterizedActionAgent + SkillGraphManager)
            available_skills, skill_metadata = self.skill_manager.retrieve_skills(
                query=self.context, return_metadata=True
            )

            # Try graph-based planning
            # pass the TaskWithSemantic object to the Planner
            ctx.planning_result = self.planner.plan(
                task=self._task_semantic,
                context=self.context,
                current_state=current_state,
                available_skills=available_skills,
                skill_metadata=skill_metadata,
                previous_code=self.messages[1].content if len(self.messages) > 1 else "",
                critique="",  # Will be updated after critic check
            )

            if ctx.planning_result.success and ctx.planning_result.code:
                # Graph planning succeeded, use the generated code
                print(f"\033[32m[Graph Planner] Successfully generated plan using {len(ctx.planning_result.skill_sequence or [])} skills\033[0m")
                if ctx.planning_result.plan:
                    print(f"\033[36m[Graph Planner] Plan: {ctx.planning_result.plan[:200]}...\033[0m")

                # Parse the code similar to process_ai_message
                try:
                    code_pattern = re.compile(r"```(?:javascript|js)(.*?)```", re.DOTALL)
                    code = "\n".join(code_pattern.findall(ctx.planning_result.code))
                    if not code:
                        code = ctx.planning_result.code

                    # Route all skill-code parsing through the active SkillLanguage
                    # (cache-first via self._skill_language, registry-fallback otherwise).
                    skill_lang = _resolve_skill_language(self)
                    parse_result = skill_lang.parse(code)
                    if not parse_result.success:
                        raise ValueError(f"Code parsing failed: {parse_result.error}")

                    functions = []
                    for func_info in parse_result.functions:
                        functions.append({
                            "name": func_info.name,
                            "type": "AsyncFunctionDeclaration" if func_info.is_async else "FunctionDeclaration",
                            "body": func_info.full_code or func_info.body,
                            "params": func_info.params,
                        })

                    main_function = None
                    for function in reversed(functions):
                        if function["type"] == "AsyncFunctionDeclaration":
                            main_function = function
                            break

                    if main_function:
                        # Check for recursive calls
                        main_function_name = main_function["name"]
                        main_function_body = main_function["body"]

                        # Improvement: only check the function body; exclude the definition line
                        # Babel generator emits body_code with the full function definition; we need to extract the body
                        func_def_pattern = rf'async\s+function\s+{re.escape(main_function_name)}\s*\([^)]*\)\s*\{{'
                        func_body_match = re.search(func_def_pattern, main_function_body, re.DOTALL)

                        if func_body_match:
                            # Extract the function body (excluding the definition line)
                            func_body_start = func_body_match.end()
                            # Find the matching closing brace (function end)
                            brace_count = 1
                            func_body_end = func_body_start
                            for i in range(func_body_start, len(main_function_body)):
                                if main_function_body[i] == '{':
                                    brace_count += 1
                                elif main_function_body[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        func_body_end = i
                                        break

                            func_body_only = main_function_body[func_body_start:func_body_end]
                        else:
                            # If no function-definition pattern is found, use the whole body (backward compatible)
                            func_body_only = main_function_body

                        # Check whether the function body contains recursive calls
                        recursive_call_pattern = rf'\b{re.escape(main_function_name)}\s*\('
                        if re.search(recursive_call_pattern, func_body_only):
                            # Check whether it is a real recursive call (not a typeof check)
                            body_without_typeof = re.sub(r'typeof\s+\w+\s*===\s*["\']function["\']', '', func_body_only, flags=re.IGNORECASE)
                            if re.search(recursive_call_pattern, body_without_typeof):
                                # Add debug info: print actual matched content
                                matches = re.findall(recursive_call_pattern, body_without_typeof)
                                print(f"\033[33m[Graph Planner] Debug: function body content:\n{func_body_only[:500]}\033[0m")
                                print(f"\033[33m[Graph Planner] Debug: matches: {matches}\033[0m")

                                # Use the LLM to verify whether there really is a recursive call
                                is_recursive_confirmed = True  # default
                                if self.planner and hasattr(self.planner, 'verify_recursive_call'):
                                    print(f"\033[36m[Graph Planner] Using the LLM to verify the recursive call...\033[0m")
                                    is_recursive_confirmed = self.planner.verify_recursive_call(
                                        function_name=main_function_name,
                                        function_body=func_body_only,
                                        full_code=main_function_body
                                    )

                                if is_recursive_confirmed:
                                    error_msg = (
                                        f"Graph Planner's generated code shows the main function {main_function_name} calling itself (recursion); this causes infinite loops.\n"
                                        f"Suggestions:\n"
                                        f"  1. If this is a bug, remove the recursive call\n"
                                        f"  2. If you need to call another skill, use the correct function name\n"
                                        f"  3. If you need to check whether a function exists, use a typeof check rather than calling it directly"
                                    )
                                    print(f"\033[31m[Graph Planner] {error_msg}\033[0m")
                                    raise ValueError(f"Recursive call detected in Graph Planner code: {error_msg}")
                                else:
                                    print(f"\033[32m[Graph Planner] LLM verification confirmed this is a false positive; continuing\033[0m")

                        program_code = "\n\n".join(function["body"] for function in functions)
                        exec_code = f"await {main_function['name']}(bot);"
                        ctx.parsed_result = {
                            "program_code": program_code,
                            "program_name": main_function["name"],
                            "exec_code": exec_code,
                        }

                        # Log the function successfully generated by Graph Planner
                        print(f"\033[36m[Graph Planner] Successfully generated function:\033[0m")
                        print(f"\033[36m[Graph Planner]   function name: {main_function['name']}\033[0m")
                        print(f"\033[36m[Graph Planner]   function code:\n{program_code}\033[0m")
                        print(f"\033[36m[Graph Planner]   exec code: {exec_code}\033[0m")
                        if ctx.planning_result.skill_sequence:
                            skill_names = [sc.skill_name for sc in ctx.planning_result.skill_sequence]
                            print(f"\033[36m[Graph Planner]   skills used: {', '.join(skill_names)}\033[0m")
                    else:
                        raise ValueError("No async function found in graph planner code")

                except Exception as e:
                    print(f"\033[33m[Graph Planner] Code parsing failed; falling back to LLM: {e}\033[0m")
                    ctx.planning_result.success = False
                    ctx.planning_result.error = f"Code parsing failed: {str(e)}"

            if not ctx.planning_result.success or not ctx.planning_result.code:
                # Graph planning failed, fallback to LLM
                if ctx.planning_result.metadata and ctx.planning_result.metadata.get("fallback"):
                    print(f"\033[33m[Graph Planner] Falling back to LLM planner\033[0m")
                ctx.planning_result = None  # Reset to trigger LLM planning

        # Use LLM planning (original approach) if:
        # 1. planner_mode="llm" (default), OR
        # 2. graph planning didn't succeed (fallback)
        if not ctx.planning_result or not ctx.planning_result.success or not ctx.planning_result.code:
            try:
                ctx.ai_message = self.action_agent.llm.invoke(self.messages)
                record_llm_usage(ctx.ai_message, process_type="skill_generation", function_name="step_plan.generate._step_plan")
            except Exception as e:
                error_msg = str(e)
                model_name = getattr(self.action_agent, 'model_name', 'unknown')
                if "model" in error_msg.lower() or "not found" in error_msg.lower() or "invalid" in error_msg.lower() or "does not exist" in error_msg.lower():
                    print(f"\033[31mError: OpenAI API call failed - model may not exist or be invalid\033[0m")
                    print(f"\033[31mModel name: {model_name}\033[0m")
                    print(f"\033[31mError details: {error_msg}\033[0m")
                    print(f"\033[33mHint: please check OPENAI_MODEL in your .env. PSN is verified against\033[0m")
                    print(f"\033[33mgpt-5-mini (OpenAI) and Qwen3-Coder-Next-FP8 (vLLM).\033[0m")
                elif "timeout" in error_msg.lower():
                    print(f"\033[31mError: OpenAI API call timed out\033[0m")
                    print(f"\033[31mModel name: {model_name}\033[0m")
                    print(f"\033[31mError details: {error_msg}\033[0m")
                else:
                    print(f"\033[31mError: OpenAI API call failed\033[0m")
                    print(f"\033[31mModel name: {model_name}\033[0m")
                    print(f"\033[31mError details: {error_msg}\033[0m")
                raise

            # Detect thinking model context exhaustion
            if hasattr(ctx.ai_message, 'response_metadata'):
                meta = ctx.ai_message.response_metadata
                finish_reason = meta.get('finish_reason')
                token_usage = meta.get('token_usage', {})
                if finish_reason == 'length':
                    print(
                        f"\033[33m[Action Agent] finish_reason=length "
                        f"(completion_tok={token_usage.get('completion_tokens')}) — "
                        f"thinking likely exhausted context. "
                        f"Retrying with thinking disabled.\033[0m"
                    )
                    try:
                        llm_no_think = self.action_agent.llm.bind(extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        })
                        ctx.ai_message = llm_no_think.invoke(self.messages)
                        record_llm_usage(ctx.ai_message, process_type="skill_generation", function_name="step_plan.generate._step_plan")
                        if hasattr(ctx.ai_message, 'response_metadata'):
                            retry_meta = ctx.ai_message.response_metadata
                            retry_usage = retry_meta.get('token_usage', {})
                            print(
                                f"\033[32m[Action Agent] No-think retry: "
                                f"finish={retry_meta.get('finish_reason')} "
                                f"completion_tok={retry_usage.get('completion_tokens')}"
                                f"\033[0m"
                            )
                    except Exception as e:
                        print(
                            f"\033[33m[Action Agent] No-think retry failed: {e}, "
                            f"using original response\033[0m"
                        )

            print(f"\033[34m****Action Agent ai message****\n{ctx.ai_message.content}\033[0m")
            self.conversations.append(
                (self.messages[0].content, self.messages[1].content, ctx.ai_message.content)
            )
        else:
            # Use graph planner result
            ctx.ai_message = None  # Not used for graph planner
            self.conversations.append(
                (self.messages[0].content, self.messages[1].content, f"[Graph Planner] {ctx.planning_result.plan or 'Graph-based planning'}")
            )

    def _step_process_code(self, ctx: StepContext):
        """Phase 2: Process AI message when graph planner didn't produce result."""
        # Pass existing skills to process_ai_message for redundancy detection (if using ParameterizedActionAgent)
        if hasattr(self.action_agent, 'process_ai_message'):
            # Improvement: extract new skill code, generate a description, and then use that description to retrieve existing skills
            # Pass task to intelligently identify the main function (avoiding helpers being mis-identified)
            main_func_name, main_func_code, ctx.all_code = self.action_agent.extract_code_from_message(ctx.ai_message, task=self.task)

            if main_func_name and main_func_code:
                # Generate description for the new skill
                # Note: pass all_code (not main_func_code) because we need the full function signature (including parameters)
                print(f"\033[36m[Code Generation] Generating description for new skill {main_func_name} for retrieval...\033[0m")
                new_skill_description = self.skill_manager.generate_skill_description(
                    program_name=main_func_name,
                    program_code=ctx.all_code,  # use full code to extract function signature
                    task=self.task
                )
                print(f"\033[36m[Code Generation] New skill description: {new_skill_description[:300]}...\033[0m")

                # Retrieve existing skills using the new skill description
                retrieval_query = new_skill_description
                print(f"\033[36m[Code Generation] Using the new skill description for retrieval (for coverage detection)\033[0m")
            else:
                # If extraction fails, fall back to using context
                print(f"\033[33m[Code Generation] Could not extract code; falling back to using context for retrieval\033[0m")
                previous_events = self.last_events if hasattr(self, 'last_events') and self.last_events else None
                retrieval_query = self.context + "\n\n" + (self.action_agent.summarize_chatlog(previous_events) if previous_events else "")

            print(f"\033[36m[Code Generation] Skill retrieval query for coverage detection: {retrieval_query[:300]}...\033[0m")
            existing_skills_for_check, existing_metadata = self.skill_manager.retrieve_skills(
                query=retrieval_query,
                return_metadata=True
            )
            if existing_skills_for_check:
                # Use the active SkillLanguage's parser to extract top-level
                # async function names (matches the previous
                # extract_toplevel_function_names(include_async=True,
                # include_regular=False) contract).
                skill_lang = _resolve_skill_language(self)
                skill_names = []
                for skill_code in existing_skills_for_check:
                    parse_result = skill_lang.parse(skill_code)
                    toplevel_funcs = [
                        f.name for f in parse_result.functions if f.is_async
                    ]
                    skill_names.extend(toplevel_funcs)
                print(f"\033[36m[Code Generation] Existing skills for coverage detection: {', '.join(skill_names)}\033[0m")
                if existing_metadata:
                    print(f"\033[36m[Code Generation] Metadata for existing skills: {', '.join(existing_metadata.keys())}\033[0m")
            else:
                # always uses SkillGraphManager; if retrieval is empty, try fetching all non-deprecated skills from the graph
                graph_skills = [
                    name for name, node in self.skill_manager.iter_skills(include_task_specific=True)
                    if not (hasattr(node, 'is_deprecated') and node.is_deprecated)
                    and not (hasattr(node, 'is_control_primitive') and node.is_control_primitive)
                ]
                if graph_skills:
                    print(f"\033[36m[Code Generation] Retrieval empty; using all skills in graph for coverage detection: {', '.join(graph_skills[:5])}{'...' if len(graph_skills) > 5 else ''}\033[0m")
                    # Fetch skill code from the graph
                    existing_skills_for_check = []
                    for skill_name in graph_skills:
                        node = self.skill_manager.get_node(skill_name)
                        if node and hasattr(node, 'code') and node.code:
                            existing_skills_for_check.append(node.code)
                else:
                    print(f"\033[33m[Code Generation] Warning: existing_skills_for_check is empty and no usable skills exist in the graph; cannot perform coverage detection\033[0m")
            try:
                ctx.parsed_result = self.action_agent.process_ai_message(
                    message=ctx.ai_message,
                    existing_skills=existing_skills_for_check
                )
            except ValueError as e:
                if "Recursive call detected" in str(e):
                    print(f"\033[41m[CRITICAL ERROR] Recursive call detected; blocking execution to avoid an infinite loop\033[0m")
                    print(f"\033[41m[CRITICAL ERROR] Error details: {e}\033[0m")
                    print(f"\033[41m[CRITICAL ERROR] Task '{self.task}' will be marked as failed; please regenerate code\033[0m")
                    # Return an error marker so rollout knows this is a code-generation error
                    ctx.parsed_result = f"Error: Recursive call detected - {str(e)}"
                else:
                    raise
        else:
            try:
                ctx.parsed_result = self.action_agent.process_ai_message(message=ctx.ai_message)
            except ValueError as e:
                if "Recursive call detected" in str(e):
                    print(f"\033[41m[CRITICAL ERROR] Recursive call detected; blocking execution to avoid an infinite loop\033[0m")
                    print(f"\033[41m[CRITICAL ERROR] Error details: {e}\033[0m")
                    print(f"\033[41m[CRITICAL ERROR] Task '{self.task}' will be marked as failed; please regenerate code\033[0m")
                    ctx.parsed_result = f"Error: Recursive call detected - {str(e)}"
                else:
                    raise

    def _step_handle_recursive_error(self, ctx: StepContext):
        """Phase 3: Build synthetic error events when recursive call detected."""
        print(f"\033[41m[ROLLOUT] Recursive-call error; constructing a synthetic error event so the LLM retries\033[0m")
        recursive_error = (
            f"RecursiveCallError: {ctx.parsed_result}\n"
            "The generated code calls the main function recursively, which causes infinite loops.\n"
            "Fix: Replace the recursive call with a while loop. For example:\n"
            "  let attempts = 0;\n"
            "  while (condition && attempts < 5) {\n"
            "    // ... attempt logic ...\n"
            "    attempts++;\n"
            "  }\n"
            "Do NOT call the main function from within itself."
        )
        events = self._build_synthetic_error_events(recursive_error)
        # LLM sees its own recursive code (all_code is already extracted in _step_process_code)
        recursive_code_for_feedback = ctx.all_code if ctx.all_code else ""
        # Rebuild messages (simplified)
        retrieval_query = self.task if self.task else ""
        if self.context:
            retrieval_query += "\n\n" + self.context
        new_skills, skill_metadata = self.skill_manager.retrieve_skills(
            query=retrieval_query, return_metadata=True
        )
        reuse_skill_hint = self._find_reuse_skill_hint(self.task, skill_metadata)
        system_message = self.action_agent.render_system_message(
            skills=new_skills, skill_metadata=skill_metadata, reuse_skill_hint=reuse_skill_hint,
            task=self.task,
        )
        human_message = self.action_agent.render_human_message(
            events=events,
            code=recursive_code_for_feedback,
            task=self.task,
            context=self.context,
            critique=recursive_error,
        )
        self.last_events = copy.deepcopy(events)
        self.messages = [system_message, human_message]
        # Fall through to the string error path for normal count+return
