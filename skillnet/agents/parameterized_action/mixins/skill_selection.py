"""
SkillSelectionMixin for ParameterizedActionAgent

Methods:
    _select_skill_to_save: Select which skill to save based on parameterization analysis
    _has_independent_value: Check if main function has independent value (static)
    _is_redundant_wrapper: Check if main function is a redundant wrapper
    _find_all_dependencies: Recursively find all function dependencies (static)
"""

import re

from skillnet.core.dk_registry import get_domain_knowledge


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


class SkillSelectionMixin:
    """Skill selection helpers for ParameterizedActionAgent."""

    @staticmethod
    def _has_independent_value(main_func_body: str, called_skill_name: str) -> bool:
        """
        Determine whether the main function has independent value (not merely calling another skill).

        Stricter criteria:
        - It is not just a simple parameter transformation and forwarded call
        - It must include substantive additional logic (e.g., composing multiple skills, complex business logic, etc.)

        Args:
            main_func_body: main-function body code
            called_skill_name: name of the called skill; if an empty string, check for any independent logic

        Returns:
            True if main function has significant additional logic beyond just calling the skill
        """
        # Remove comments and whitespace
        body_clean = re.sub(r'//.*?$', '', main_func_body, flags=re.MULTILINE)
        body_clean = re.sub(r'/\*.*?\*/', '', body_clean, flags=re.DOTALL)
        body_clean = re.sub(r'\s+', ' ', body_clean).strip()

        if not called_skill_name:
            # If no called skill is specified, check whether there is any independent logic
            has_conditionals = bool(re.search(r'\b(if|else|switch|case|try|catch|while|for)\b', body_clean))
            has_variables = bool(re.search(r'\b(const|let|var)\s+\w+\s*=', body_clean))
            has_calls = bool(re.search(r'\bawait\s+\w+\s*\(', body_clean))
            return has_conditionals or has_variables or has_calls

        # Check whether it's only a simple function call (possibly with a little parameter prep)
        escaped_skill_name = re.escape(called_skill_name)

        # Remove the called skill's invocation and see what remains
        body_without_call = re.sub(rf'await\s+{escaped_skill_name}\s*\([^)]*\)', '', body_clean)
        body_without_call = re.sub(rf'{escaped_skill_name}\s*\([^)]*\)', '', body_without_call)
        body_without_call = re.sub(r'\s+', ' ', body_without_call).strip()

        # Remove common simple wrapper logic (variable assignments, simple if checks, bot.chat, etc.)
        # These are typically just parameter conversions and logging, not independent value
        simple_wrapper_patterns = [
            r'if\s*\([^)]*\)\s*\{[^}]*return[^}]*\}',  # simple if return
            r'const\s+\w+\s*=\s*[^;]+;',  # simple variable assignment
            r'await\s+bot\.chat\([^)]*\);',  # bot.chat call
            r'typeof\s+\w+\s*===\s*["\']undefined["\']',  # typeof check
            r'!\s*\w+',  # simple "!" check
        ]

        for pattern in simple_wrapper_patterns:
            body_without_call = re.sub(pattern, '', body_without_call, flags=re.IGNORECASE)

        body_without_call = re.sub(r'\s+', ' ', body_without_call).strip()

        # If almost no code remains after stripping simple wrappers, treat as no independent value
        if len(body_without_call) < 50:  # threshold: remaining code under 50 characters
            return False

        # Check for substantive additional logic
        # 1. Calling multiple distinct skills (composition logic)
        skill_calls = re.findall(r'\bawait\s+(\w+)\s*\(', body_clean)
        unique_skill_calls = set(skill_calls)
        if len(unique_skill_calls) > 1:  # calls multiple distinct skills
            return True

        # 2. Complex conditional logic (not just a simple if return)
        complex_conditionals = bool(re.search(r'\b(if|else|switch|case|try|catch|while|for)\s*\([^)]{20,}', body_clean))
        if complex_conditionals:
            return True

        # 3. Loop logic
        has_loops = bool(re.search(r'\b(for|while)\s*\(', body_clean))
        if has_loops:
            return True

        # 4. Calls to other non-bot APIs (e.g., data APIs, control primitives, etc.)
        dk = get_domain_knowledge()
        if dk:
            _prims = dk.get_control_primitives()
            _deps = dk.get_global_dependencies()
            _api_names = list(_deps.keys()) + _prims if _deps else _prims
        else:
            _api_names = []
        if _api_names:
            _api_alt = '|'.join(re.escape(n) for n in _api_names)
            other_apis = re.findall(rf'\b({_api_alt})\s*[\.(]', body_clean)
        else:
            other_apis = []
        if other_apis:
            return True

        # Otherwise, treat as no independent value
        return False

    def _is_redundant_wrapper(self, main_func_body: str, main_func_params: list, called_skill_name: str) -> bool:
        """
        Determine whether the main function is a redundant wrapper (its behavior is fully covered by another skill).

        Returns:
            True if main function is just a redundant wrapper
        """
        # If the main function is parameterized, check whether it merely forwards parameters
        if len(main_func_params) > 1:
            # A parameterized wrapper may be valuable unless it just forwards parameters
            # Check for any additional logic
            return not self._has_independent_value(main_func_body, called_skill_name)
        else:
            # Non-parameterized wrapper: check whether it's just a simple call
            return not self._has_independent_value(main_func_body, called_skill_name)

    @staticmethod
    def _find_all_dependencies(func_name: str, func_body: str, all_funcs: list, visited: set) -> set:
        """
        Recursively find all dependencies of a function (including indirect dependencies).

        Example: if A calls B and B calls C, then A's dependencies are {B, C}.
        """
        deps = set()
        for func in all_funcs:
            other_name = func["name"]
            if other_name != func_name and other_name not in visited:
                # Use a regex to detect function calls (more precise)
                # Matches: functionName( or await functionName(
                call_pattern = rf'\b{re.escape(other_name)}\s*\('
                if re.search(call_pattern, func_body):
                    deps.add(other_name)
                    visited.add(other_name)
                    # Recursively look for indirect dependencies
                    indirect_deps = SkillSelectionMixin._find_all_dependencies(other_name, func["body"], all_funcs, visited)
                    deps.update(indirect_deps)
        return deps

    def _select_skill_to_save(self, main_function, functions, existing_skills):
        """Select which skill to save based on parameterization and redundancy analysis.

        Args:
            main_function: The identified main function dict
            functions: All parsed function dicts
            existing_skills: List of existing skill code strings

        Returns:
            Either a result dict (with should_save_skill=False for early-return / no-save cases)
            or a tuple (skill_to_save, functions_to_include, wrapper_removed).
        """
        main_function_body = main_function["body"]
        main_is_parameterized = len(main_function["params"]) > 1

        # Check if main function is just calling an existing skill from existing_skills
        # If so, it's a redundant wrapper and should not be saved
        is_just_calling_existing_skill = False
        if existing_skills:
            # Extract function names from existing_skills - use the language's
            # parser to pull only top-level async function declarations (matches
            # the previous extract_toplevel_function_names(include_async=True,
            # include_regular=False) contract).
            skill_lang = _resolve_skill_language(self)
            existing_skill_names = []
            for skill_code in existing_skills:
                parse_result = skill_lang.parse(skill_code)
                toplevel_funcs = [
                    f.name for f in parse_result.functions if f.is_async
                ]
                existing_skill_names.extend(toplevel_funcs)

            # Check if main function body just calls an existing skill
            for existing_skill_name in existing_skill_names:
                if existing_skill_name in main_function_body:
                    # Check if it's just a simple call without additional logic
                    if not self._has_independent_value(main_function_body, existing_skill_name):
                        is_just_calling_existing_skill = True
                        print(f"\033[33m[Parameterized Mode] Detected that main function {main_function['name']} only calls existing skill {existing_skill_name}; will not save a new skill\033[0m")
                        break

        # Find all parameterized helper functions (excluding main)
        parameterized_helpers = []
        for function in functions:
            if function["type"] == "AsyncFunctionDeclaration" and function["name"] != main_function["name"]:
                params = function["params"]
                if len(params) > 1:  # Has parameters beyond bot
                    parameterized_helpers.append({
                        "function": function,
                        "param_count": len(params),
                        "is_called_by_main": False
                    })

        # Check which helper functions are called by main function
        for pf in parameterized_helpers:
            func_name = pf["function"]["name"]
            if func_name in main_function_body:
                pf["is_called_by_main"] = True

        # Select the best skill to save:
        # Priority 1: Main function if it's parameterized AND has independent value
        # Priority 2: Parameterized helper functions called by main (reusable helpers)
        # Priority 3: Other parameterized helper functions
        # Priority 4: Main function (fallback - only if no parameterized functions exist)
        skill_to_save = None
        functions_to_include = []

        _dk = getattr(self, '_domain_knowledge', None)
        _entry_param = _dk.get_entry_parameter_name() if _dk else "bot"

        is_redundant = False  # track across branches

        if main_is_parameterized:
            # Check if main function has independent value (not just a wrapper)
            # Find which skills are called by main
            called_skills_in_main = []
            for func in functions:
                if func["name"] != main_function["name"]:
                    func_name = func["name"]
                    if func_name in main_function_body:
                        called_skills_in_main.append(func_name)

            # Check if main is redundant wrapper (direct call to helper function)
            if not is_just_calling_existing_skill and called_skills_in_main:
                # Check if main is just a wrapper of the called skill
                for called_skill in called_skills_in_main:
                    if self._is_redundant_wrapper(main_function_body, main_function["params"], called_skill):
                        is_redundant = True
                        print(f"\033[33m[Parameterized Mode] Warning: main function {main_function['name']} is a redundant wrapper (covered by {called_skill}); saving the called skill instead\033[0m")
                        break

            if not is_redundant:
                # Best case: main function itself is parameterized and has value, use it directly
                skill_to_save = main_function
                wrapper_removed = False  # the main function itself is the skill to save; no wrapper was removed
                print(f"\033[36m[Parameterized Mode] Main function {main_function['name']} is parameterized and has independent value; saving the main function directly\033[0m")

                # Include main function and its dependencies (including indirect dependencies)
                functions_to_include.append(main_function)
                # Find all dependencies (including indirect ones)
                all_deps = self._find_all_dependencies(main_function["name"], main_function_body, functions, {main_function["name"]})
                for func in functions:
                    if func["name"] in all_deps:
                        functions_to_include.append(func)

        # If main is not parameterized or is redundant, check parameterized helpers
        # First check: if main function is just calling an existing skill, don't save anything
        if is_just_calling_existing_skill:
            print(f"\033[36m[Parameterized Mode] Main function {main_function['name']} only calls an existing skill; not saving a new skill\033[0m")
            # Return result indicating no new skill should be saved
            result = {
                "program_code": main_function["body"],
                "program_name": main_function["name"],
                "exec_code": f"await {main_function['name']}({_entry_param});",
                "should_save_skill": False,  # Mark that this should not be saved as a new skill
                "already_normalized": False,  # No normalization (because we're not saving the skill)
            }
            return result

        if skill_to_save is None:

            if parameterized_helpers:
                # Main function is not parameterized or is redundant, but there are parameterized helpers
                # Filter out simple helper functions that are just utilities (e.g., getNearestBlockByName)
                # These should not be saved as independent skills
                meaningful_helpers = []
                for pf in parameterized_helpers:
                    helper_func = pf["function"]
                    helper_name = helper_func["name"]
                    helper_body = helper_func["body"]

                    # Skip simple utility functions (get*, find*, check*, etc.)
                    # These are typically just helper functions, not reusable skills
                    if helper_name.startswith(("get", "find", "check", "has", "is", "count")):
                        # Only include if it has significant logic beyond simple lookups
                        # Check if it has loops, conditionals, or multiple operations
                        has_complex_logic = (
                            "for" in helper_body or "while" in helper_body or
                            "if" in helper_body or "try" in helper_body or
                            len(re.findall(r'await\s+\w+\(', helper_body)) > 1
                        )
                        if not has_complex_logic:
                            print(f"\033[33m[Parameterized Mode] Skipping simple utility function {helper_name}; not saving as a standalone skill\033[0m")
                            continue

                    meaningful_helpers.append(pf)

                if meaningful_helpers:
                    # Sort by: called_by_main (True first), then param_count
                    meaningful_helpers.sort(
                        key=lambda x: (not x["is_called_by_main"], -x["param_count"]),
                        reverse=False
                    )
                    skill_to_save = meaningful_helpers[0]["function"]
                    print(f"\033[36m[Parameterized Mode] Main function {main_function['name']} is non-parameterized or redundant; detected meaningful parameterized function {skill_to_save['name']}; saving it preferentially\033[0m")
                else:
                    # No meaningful helpers, fall back to main function even if redundant
                    # But only if it's not just calling an existing skill
                    if not is_redundant or self._has_independent_value(main_function_body, ""):
                        skill_to_save = main_function
                        print(f"\033[36m[Parameterized Mode] No meaningful parameterized helper functions; saving main function {main_function['name']}\033[0m")
                    else:
                        # Main function is redundant and has no independent value, don't save
                        print(f"\033[36m[Parameterized Mode] Main function {main_function['name']} is redundant and has no independent value; not saving a new skill\033[0m")
                        result = {
                            "program_code": main_function["body"],
                            "program_name": main_function["name"],
                            "exec_code": f"await {main_function['name']}({_entry_param});",
                            "should_save_skill": False,  # Mark that this should not be saved as a new skill
                            "already_normalized": False,  # No normalization (because we're not saving the skill)
                        }
                        return result

                # Build dependency graph: find all functions that skill_to_save depends on
                skill_name = skill_to_save["name"]
                skill_body = skill_to_save["body"]

                # Find all function names that skill_to_save calls (including indirect dependencies)
                called_functions = self._find_all_dependencies(skill_name, skill_body, functions, {skill_name})

                # Include the parameterized skill and its dependencies
                functions_to_include.append(skill_to_save)
                for func in functions:
                    if func["name"] in called_functions:
                        functions_to_include.append(func)

                # Check if main function should be excluded (redundant wrapper)
                main_body = main_function["body"]
                is_simple_wrapper = (
                    skill_name in main_body and
                    len(main_function["params"]) == 1 and
                    (f"await {skill_name}({_entry_param})" in main_body or f"{skill_name}({_entry_param})" in main_body)
                )

                # Also check if main is redundant even if parameterized
                if not is_simple_wrapper and len(main_function["params"]) > 1:
                    is_simple_wrapper = self._is_redundant_wrapper(main_body, main_function["params"], skill_name)

                # Used to decide which function exec_code calls
                wrapper_removed = False

                if is_simple_wrapper:
                    print(f"\033[36m[Parameterized Mode] Main function {main_function['name']} is a redundant wrapper; removed from the saved code\033[0m")
                    wrapper_removed = True
                elif skill_to_save["name"] != main_function["name"]:
                    # skill_to_save is the helper function; the main function is its caller
                    # The main function should not be included in the helper's code file (fixes issue 15)
                    wrapper_removed = True
                    print(f"\033[36m[Parameterized Mode] skill_to_save={skill_to_save['name']}; main function {main_function['name']} is the caller and won't be saved in the same file\033[0m")
                elif self._has_independent_value(main_body, skill_name):
                    # skill_to_save is the main function itself and it has independent value
                    functions_to_include.append(main_function)
                    print(f"\033[36m[Parameterized Mode] Main function {main_function['name']} has independent value; saving it as well\033[0m")
                # else: wrapper_removed stays False (the main function is not a redundant wrapper but also has no independent value, and is kept as is)
            else:
                # No parameterized functions, use main function
                skill_to_save = main_function
                functions_to_include = functions
                wrapper_removed = False

        return skill_to_save, functions_to_include, wrapper_removed
