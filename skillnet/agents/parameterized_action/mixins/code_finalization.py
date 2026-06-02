"""
CodeFinalizationMixin for ParameterizedActionAgent

Methods:
    _validate_and_finalize: Validate assembled code and build final result dict

Cross-mixin dependencies:
    → CodeValidationMixin._validate_generated_code, _fix_tdz_issues (via self)
    → SkillNamingMixin._normalize_skill_name (via self)
    → CodeAssemblyMixin._safe_replace_function_name (via self)
"""

import re


class CodeFinalizationMixin:
    """Code finalization helpers for ParameterizedActionAgent."""

    def _validate_and_finalize(self, program_code, exec_code, normalized_name,
                               skill_to_save, functions, functions_to_include,
                               skill_lang, main_function):
        """Validate the assembled code (TDZ fix, reparameterization) and build the
        final result dict.

        Args:
            program_code: The assembled program code string
            exec_code: The execution code string
            normalized_name: The (possibly normalized) skill name
            skill_to_save: The function dict selected to be saved
            functions: All parsed function dicts
            functions_to_include: List of function dicts included in program_code
            skill_lang: SkillLanguage Protocol implementation for parsing/generation
            main_function: The identified main function dict

        Returns:
            dict: Result dict with program_code, program_name, exec_code, etc.
        """
        # Final validation: check the code (no recursion, no name conflicts, no TDZ issues, etc.)
        validation_result = self._validate_generated_code(
            program_code=program_code,
            exec_code=exec_code,
            main_function_name=normalized_name,
            all_functions=functions_to_include
        )

        if not validation_result["valid"]:
            print(f"\033[33m[Code Validation] warning: {validation_result['reason']}\033[0m")

            # Check whether TDZ issues can be auto-fixed
            if validation_result.get("can_auto_fix", False) and "TDZ" in validation_result.get("reason", ""):
                print(f"\033[36m[TDZ Auto-Fix] TDZ issue detected, attempting auto-fix...\033[0m")
                fixed_code, was_fixed, fix_description = self._fix_tdz_issues(program_code)

                if was_fixed:
                    print(f"\033[32m[TDZ Auto-Fix] {fix_description}\033[0m")
                    program_code = fixed_code

                    # Re-validate the fixed code
                    validation_result = self._validate_generated_code(
                        program_code=program_code,
                        exec_code=exec_code,
                        main_function_name=normalized_name,
                        all_functions=functions_to_include
                    )

                    if not validation_result["valid"]:
                        print(f"\033[33m[TDZ Auto-Fix] issues remain after fix: {validation_result['reason']}\033[0m")
                        if validation_result.get("critical", False):
                            raise ValueError(f"Code validation failed after TDZ fix: {validation_result['reason']}")
                    else:
                        print(f"\033[32m[TDZ Auto-Fix] validation passed; TDZ issue successfully fixed\033[0m")
                else:
                    print(f"\033[33m[TDZ Auto-Fix] auto-fix failed: {fix_description}\033[0m")
                    if validation_result.get("critical", False):
                        raise ValueError(f"Code validation failed: {validation_result['reason']}")
            elif validation_result.get("critical", False):
                # Critical error that can't be auto-fixed — cannot continue
                raise ValueError(f"Code validation failed: {validation_result['reason']}")

        result = {
            "program_code": program_code,
            "program_name": normalized_name,  # Use normalized name
            "exec_code": exec_code,
            "should_save_skill": True,  # Default: save the skill
            "already_normalized": normalized_name != skill_to_save["name"],  # Mark whether normalization was applied
        }

        # === Auto-reparameterize: replace program_code so execution and save use the same version ===
        # The reparameterized version's default parameter values match the original hardcoded values,
        # so execution behavior is identical.
        # Only replace result["program_code"]; do not modify main_function/functions (would break upstream priority ordering).
        # Check skill_to_save (the function actually saved), not main_function.
        skill_to_save_is_parameterized = len(skill_to_save.get("params", [])) > 1
        if not skill_to_save_is_parameterized:
            from skillnet.config.robustness_config import RobustnessConfig
            if RobustnessConfig.ENABLE_AUTO_REPARAMETERIZE:
                # Extract the current function body from program_code via babel (post-normalization)
                current_skill_body = None
                target_start = None
                target_end = None
                try:
                    parse_result = skill_lang.parse(program_code)
                    current_ast = parse_result.raw_ast
                    if current_ast is None:
                        raise RuntimeError("SkillLanguage parse returned no AST")
                    for node in current_ast.program.body:
                        if node.type != "FunctionDeclaration":
                            continue
                        try:
                            if not node["async"]:
                                continue
                            if not (node.id and node.id.name == normalized_name):
                                continue
                        except Exception:
                            continue
                        # Prefer position-based extraction (avoids language roundtrip format differences)
                        node_start = getattr(node, 'start', None)
                        node_end = getattr(node, 'end', None)
                        if node_start is not None and node_end is not None:
                            target_start = int(node_start)
                            target_end = int(node_end)
                            current_skill_body = program_code[target_start:target_end]
                        else:
                            # Fallback: match by function name in parse_result.functions
                            current_skill_body = next(
                                (f.full_code for f in parse_result.functions if f.name == normalized_name),
                                None,
                            )
                        break
                except Exception:
                    current_skill_body = None

                if current_skill_body is None:
                    print(f"\033[33m[Reparameterize] could not extract {normalized_name} from program_code\033[0m")
                else:
                    dk = getattr(self, '_domain_knowledge', None)
                    prims = dk.get_control_primitives() if dk else []
                    if prims:
                        prims_alt = '|'.join(re.escape(p) for p in prims)
                        primitive_pattern = rf'await\s+(?:{prims_alt})\s*\([^)]*,\s*\d+'
                    else:
                        primitive_pattern = None
                    if primitive_pattern and re.search(primitive_pattern, current_skill_body):
                        print(f"\033[33m[Reparameterize] {normalized_name} is not parameterized but has primitive calls; attempting reparameterization...\033[0m")
                        try:
                            reparameterized_code = self._reparameterize_function(current_skill_body)
                            if reparameterized_code:
                                # Re-parse and validate via SkillLanguage
                                reparsed_parse = skill_lang.parse(reparameterized_code)
                                reparsed_ast = reparsed_parse.raw_ast
                                if reparsed_ast is None:
                                    raise RuntimeError("SkillLanguage parse returned no AST")
                                reparsed_main = None
                                llm_func_name = None
                                reparsed_node = None
                                for node in reparsed_ast.program.body:
                                    try:
                                        is_match = (node.type == "FunctionDeclaration" and node["async"]
                                                and node.id and node.id.name)
                                    except Exception:
                                        is_match = False
                                    if is_match:
                                        _dk = getattr(self, '_domain_knowledge', None)
                                        _entry_param = _dk.get_entry_parameter_name() if _dk else "bot"
                                        if len(list(node["params"])) > 1 and node["params"][0].name == _entry_param:
                                            # Match this AST node back to a ParseResult function
                                            # via name; fall back to source-slice on node start/end.
                                            _llm_name = node.id.name
                                            reparsed_main = next(
                                                (f.full_code for f in reparsed_parse.functions if f.name == _llm_name),
                                                None,
                                            )
                                            if reparsed_main is None:
                                                _ns = getattr(node, 'start', None)
                                                _ne = getattr(node, 'end', None)
                                                if _ns is not None and _ne is not None:
                                                    reparsed_main = reparameterized_code[int(_ns):int(_ne)]
                                            llm_func_name = _llm_name
                                            reparsed_node = node
                                            break
                                if reparsed_main:
                                    # Handle a potential rename by the LLM
                                    final_name = normalized_name
                                    if llm_func_name != normalized_name:
                                        # LLM renamed it; route through _normalize_skill_name so normalization isn't bypassed
                                        candidate = self._normalize_skill_name(
                                            llm_func_name,
                                            list(reparsed_node["params"]),  # AST node object, kept consistent with L1892
                                            reparsed_main
                                        )
                                        existing_names = {f["name"] for f in functions if f["name"] != normalized_name}
                                        if candidate not in existing_names:
                                            final_name = candidate
                                        else:
                                            print(f"\033[33m[Reparameterize] renamed {candidate} conflicts with an existing function, keeping original name\033[0m")

                                        # Ensure the function name in reparsed_main equals final_name
                                        if llm_func_name != final_name:
                                            reparsed_main = re.sub(
                                                rf'async\s+function\s+{re.escape(llm_func_name)}\s*\(',
                                                f'async function {final_name}(',
                                                reparsed_main,
                                                count=1
                                            )

                                    # Replace the function body in program_code
                                    if target_start is not None and target_end is not None:
                                        # Position-based replacement (precise, no roundtrip issues)
                                        new_program_code = program_code[:target_start] + reparsed_main + program_code[target_end:]
                                    else:
                                        # Fallback: string replacement + validation
                                        new_program_code = program_code.replace(current_skill_body, reparsed_main)
                                        if new_program_code == program_code:
                                            print(f"\033[33m[Reparameterize] replacement failed (program_code unchanged), keeping original\033[0m")
                                            new_program_code = None

                                    if new_program_code is not None and new_program_code != program_code:
                                        # Re-validate the reparameterized body. The earlier
                                        # _validate_generated_code call at the top of this method
                                        # vetted the original function body, but reparameterize
                                        # replaces that body with LLM-generated code. The LLM has
                                        # been observed to keep a call to the (newly-renamed)
                                        # function inside the body it produces — a self-recursive
                                        # call that exhausts the JS stack a few iterations into
                                        # bot execution. Catch that before saving and fall back to
                                        # the original (already-vetted) program_code.
                                        candidate_exec_code = (
                                            f"await {final_name}({_entry_param});"
                                            if final_name != normalized_name
                                            else exec_code
                                        )
                                        post_validation = self._validate_generated_code(
                                            program_code=new_program_code,
                                            exec_code=candidate_exec_code,
                                            main_function_name=final_name,
                                            all_functions=functions_to_include,
                                        )
                                        if not post_validation["valid"]:
                                            print(
                                                f"\033[33m[Reparameterize] post-validation "
                                                f"rejected: {post_validation['reason']} — "
                                                f"keeping original (non-reparameterized) "
                                                f"version\033[0m"
                                            )
                                        else:
                                            # If the name changed, update call sites in other functions
                                            if final_name != normalized_name:
                                                updated_code, rename_ok = self._safe_replace_function_name(
                                                    code=new_program_code,
                                                    old_func_name=normalized_name,
                                                    new_func_name=final_name,
                                                    main_function_name=normalized_name,
                                                    all_functions=functions
                                                )
                                                if rename_ok:
                                                    new_program_code = updated_code
                                                result["program_name"] = final_name
                                                result["exec_code"] = candidate_exec_code
                                                result["already_normalized"] = True
                                                print(f"\033[36m[Reparameterize] LLM renamed {normalized_name} -> {final_name}\033[0m")

                                            result["program_code"] = new_program_code
                                            print(f"\033[32m[Reparameterize] success; execution and save use the reparameterized version\033[0m")
                                else:
                                    print(f"\033[33m[Reparameterize] LLM-returned code did not pass babel validation\033[0m")
                            else:
                                print(f"\033[33m[Reparameterize] LLM-returned code did not pass basic validation\033[0m")
                        except Exception as e:
                            print(f"\033[33m[Reparameterize] reparameterization failed ({e}), keeping original\033[0m")

        return result
