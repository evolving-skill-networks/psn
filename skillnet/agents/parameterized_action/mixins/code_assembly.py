"""
CodeAssemblyMixin for ParameterizedActionAgent

Methods:
    _assemble_and_normalize: Assemble program_code, normalize skill name, apply replacements
    _safe_replace_function_name: Safely replace function names using AST
    _reparameterize_function: LLM call to add parameters to non-parameterized functions

Cross-mixin dependencies:
    → SkillNamingMixin._normalize_skill_name (via self)
"""

import re
from typing import Tuple

from langchain.schema import SystemMessage

from skillnet.utils.stats_tracker import record_llm_usage



class CodeAssemblyMixin:
    """Code assembly and name replacement helpers for ParameterizedActionAgent."""

    def _safe_replace_function_name(
        self,
        code: str,
        old_func_name: str,
        new_func_name: str,
        main_function_name: str,
        all_functions: list
    ) -> Tuple[str, bool]:
        """
        Safely replace a function name, avoiding recursive calls and name conflicts.

        Uses AST parsing to precisely identify function calls — only the main
        function's call is replaced; helper-function calls are not.

        Args:
            code: code to modify
            old_func_name: old function name
            new_func_name: new function name
            main_function_name: name of the main function (used to identify main calls)
            all_functions: list of all functions, each with name and body

        Returns:
            tuple: (replaced code, success flag)
        """
        # Extract all function names (including helpers)
        function_names = set()
        for func in all_functions:
            if isinstance(func, dict) and "name" in func:
                function_names.add(func["name"])

        # Check whether replacement would cause recursion
        if new_func_name == main_function_name and old_func_name == main_function_name:
            # This would cause recursion
            print(f"\033[31m[Safe Replace] warning: replacement would cause recursion ({old_func_name} -> {new_func_name})\033[0m")
            return code, False

        # Check for name conflicts
        if new_func_name in function_names and new_func_name != main_function_name:
            print(f"\033[31m[Safe Replace] warning: new function name {new_func_name} conflicts with an existing function\033[0m")
            return code, False

        try:
            # Resolve skill language: cache-first, registry-fallback.
            skill_lang = getattr(self, "_skill_language", None)
            if skill_lang is None:
                from skillnet.core.dk_registry import get_domain_knowledge
                dk = get_domain_knowledge()
                if dk is None:
                    raise RuntimeError("No DomainKnowledge registered; cannot parse skill code")
                skill_lang = dk.get_skill_language_impl()

            # Parse via the language; the raw AST is mutated in place below
            # and regenerated via SkillLanguage.regenerate().
            parse_result = skill_lang.parse(code)
            parsed = parse_result.raw_ast
            if parsed is None:
                raise RuntimeError(
                    "SkillLanguage parse returned no AST; in-place AST mutation requires a raw AST"
                )

            # Track whether a replacement occurred
            replaced = False

            # Traverse the AST and replace function calls
            def traverse_and_replace(node):
                """Traverse an AST node and replace function calls."""
                nonlocal replaced
                if node is None:
                    return

                # Check whether this is a CallExpression.
                # Use try/except for all bridge proxy access (JSPyBridge
                # raises JavaScriptError, not AttributeError).
                try:
                    if node.type == "CallExpression":
                        try:
                            if node.callee.type == "Identifier":
                                called_name = node.callee.name
                                if called_name == old_func_name and old_func_name == main_function_name:
                                    if new_func_name != main_function_name:
                                        node.callee.name = new_func_name
                                        replaced = True
                        except Exception:
                            pass
                except Exception:
                    pass

                # Recursively traverse children
                try:
                    for key, value in node.__dict__.items():
                        if isinstance(value, list):
                            for item in value:
                                try:
                                    if item.type:
                                        traverse_and_replace(item)
                                except Exception:
                                    pass
                        else:
                            try:
                                if value.type:
                                    traverse_and_replace(value)
                            except Exception:
                                pass
                except Exception:
                    pass

            # Run the replacement
            traverse_and_replace(parsed)

            # Generate the replaced code
            try:
                new_code = skill_lang.regenerate(parsed)
            except Exception:
                # If AST regeneration fails, return original code unchanged
                return code, False

            # Check whether replacement would cause recursion
            if new_func_name == main_function_name:
                # Check whether the main function calls itself
                main_func_pattern = rf'async\s+function\s+{re.escape(new_func_name)}\s*\([^)]*\)\s*{{[^}}]*\b{re.escape(new_func_name)}\s*\('
                if re.search(main_func_pattern, new_code, re.DOTALL):
                    print(f"\033[31m[Safe Replace] warning: recursion detected after replacement\033[0m")
                    return code, False

            # Replace the function definition first
            func_def_pattern = rf'async\s+function\s+{re.escape(old_func_name)}\s*\('
            new_code = re.sub(
                func_def_pattern,
                f"async function {new_func_name}(",
                new_code,
                count=1
            )

            return new_code, True

        except Exception as e:
            # AST parsing failed — fall back to regex replacement (with safety checks)
            print(f"\033[33m[Safe Replace] AST parsing failed, falling back to regex replacement: {e}\033[0m")

            # Replace the function definition first
            func_def_pattern = rf'async\s+function\s+{re.escape(old_func_name)}\s*\('
            new_code = re.sub(
                func_def_pattern,
                f"async function {new_func_name}(",
                code,
                count=1
            )

            # Then replace call sites (with a more precise match).
            # Only replace calls that are not part of a function definition.
            func_call_pattern = rf'(?<!async\s+function\s+)(?<!\w){re.escape(old_func_name)}\s*\('
            new_code = re.sub(
                func_call_pattern,
                f"{new_func_name}(",
                new_code
            )

            # Check whether replacement would cause recursion
            if new_func_name == main_function_name:
                # Check whether the main function calls itself
                main_func_pattern = rf'async\s+function\s+{re.escape(new_func_name)}\s*\([^)]*\)\s*{{[^}}]*\b{re.escape(new_func_name)}\s*\('
                if re.search(main_func_pattern, new_code, re.DOTALL):
                    print(f"\033[31m[Safe Replace] warning: recursion detected after replacement\033[0m")
                    return code, False

            return new_code, True

    def _reparameterize_function(self, original_code: str) -> str:
        """Focused LLM call: add appropriate parameters to a non-parameterized function.

        Args:
            original_code: original function code (with only the bot parameter)

        Returns:
            Reparameterized function code, or None on failure.
        """
        # domain-aware reparameterize prompt
        prompt_template = ""
        _dk = getattr(self, '_domain_knowledge', None)
        if _dk:
            prompt_template = _dk.get_reparameterize_prompt_template()
        if not prompt_template:
            prompt_template = ""  # domain-owned
        prompt = prompt_template.replace("{original_code}", original_code)

        response = self.llm.invoke([
            SystemMessage(content=prompt)
        ])
        record_llm_usage(response, process_type="skill_generation", function_name="code_assembly.reparameterize")

        # Extract code blocks (same pattern as process_ai_message)
        from skillnet.utils.code_block import build_code_block_pattern
        _dk = getattr(self, '_domain_knowledge', None)
        _lang = _dk.get_skill_language() if _dk else "javascript"
        code_pattern = build_code_block_pattern(_lang)
        matches = code_pattern.findall(response.content)
        if matches:
            code = "\n".join(matches).strip()
        else:
            # The LLM may have returned code directly without markdown fences
            code = response.content.strip()

        if not code:
            return None

        # Basic validation: new code should contain "async function" and have parameters after bot
        if "async function" not in code:
            return None
        # Check that the signature has a comma after bot (i.e. additional parameters)
        sig_match = re.search(r'async\s+function\s+\w+\s*\(([^)]*)\)', code)
        if not sig_match:
            return None
        params_str = sig_match.group(1)
        # After "bot" there must be a comma and additional parameters
        if "bot" not in params_str or "," not in params_str.split("bot", 1)[1]:
            return None

        return code

    def _assemble_and_normalize(self, skill_to_save, functions_to_include, functions, main_function, wrapper_removed):
        """Assemble program_code from selected functions, normalize skill name, and
        apply name replacements.

        Args:
            skill_to_save: The function dict selected to be saved as the skill
            functions_to_include: List of function dicts to include in program_code
            functions: All parsed function dicts (for conflict checking)
            main_function: The identified main function dict
            wrapper_removed: Whether the main function wrapper was removed

        Returns:
            tuple: (program_code, exec_code, normalized_name, all_top_level_declarations)
        """
        # Build program_code with only selected functions.
        # First, collect all top-level declarations that the functions depend on (e.g. require statements).
        all_top_level_declarations = []
        seen_declarations = set()  # Avoid duplicates
        for function in functions_to_include:
            for decl in function.get("top_level_declarations", []):
                if decl not in seen_declarations:
                    all_top_level_declarations.append(decl)
                    seen_declarations.add(decl)

        # Merge top-level declarations and function bodies
        code_parts = all_top_level_declarations + [function["body"] for function in functions_to_include]
        program_code = "\n\n".join(code_parts)

        # Fix: when the wrapper is removed, exec_code should call the retained function
        _dk = getattr(self, '_domain_knowledge', None)
        _entry_param = _dk.get_entry_parameter_name() if _dk else "bot"
        # Sync languages (e.g. Python) cannot run top-level `await ...` —
        # `exec("await foo(bot)")` raises SyntaxError. Resolve via the active
        # DomainKnowledge (cache-first, then DK.get_skill_language_impl(),
        # then default to True for safety). Avoid the registry-based
        # resolver — it raises when no DK is registered globally, which
        # would silently fall back to async and re-trigger the bug.
        _skill_lang = getattr(self, "_skill_language", None)
        if _skill_lang is None and _dk is not None:
            try:
                _skill_lang = _dk.get_skill_language_impl()
            except Exception:
                _skill_lang = None
        _requires_async = getattr(_skill_lang, "requires_async_main", True)

        def _invoke(name: str) -> str:
            return (
                f"await {name}({_entry_param});"
                if _requires_async
                else f"{name}({_entry_param})"
            )

        if wrapper_removed and skill_to_save:
            exec_code = _invoke(skill_to_save['name'])
            print(f"\033[36m[Parameterized Mode] exec_code updated to call {skill_to_save['name']} (wrapper removed)\033[0m")
        else:
            exec_code = _invoke(main_function['name'])

        # Normalize skill name if it's too specific for a parameterized function.
        # Defensive check: ensure skill_to_save["name"] is not None.
        skill_name = skill_to_save.get("name")
        if not skill_name or skill_name is None:
            print(f"\033[33m[Warning] skill_to_save's name is None, using default name\033[0m")
            skill_name = "genericAction"

        # Before normalizing: check whether the normalized name would conflict with an existing function.
        # First try normalizing, then check for conflicts.
        normalized_name = self._normalize_skill_name(
            skill_name,
            skill_to_save.get("params", []),
            skill_to_save.get("body", "")
        )

        # Check whether the normalized name conflicts with an existing function
        existing_function_names = {f["name"] for f in functions if f["name"] != skill_name}
        if normalized_name in existing_function_names:
            print(f"\033[33m[Name Normalization] warning: normalized name '{normalized_name}' conflicts with an existing function, keeping original name '{skill_name}'\033[0m")
            normalized_name = skill_name

        # If name was normalized, update the function name in program_code and exec_code
        if normalized_name != skill_to_save["name"]:
            print(f"\033[36m[Name Normalization] function name normalized from '{skill_to_save['name']}' to '{normalized_name}' (parameterized function supports multiple types)\033[0m")
            old_func_name = skill_to_save["name"]

            # Check for name conflicts: does the normalized name conflict with an existing function?
            existing_function_names = {f["name"] for f in functions if f["name"] != old_func_name}
            if normalized_name in existing_function_names:
                print(f"\033[33m[Name Normalization] warning: normalized name '{normalized_name}' conflicts with an existing function, keeping original name '{old_func_name}'\033[0m")
                normalized_name = old_func_name
            else:
                # Use the safe function-name replacement helper
                new_program_code, replace_success = self._safe_replace_function_name(
                    code=program_code,
                    old_func_name=old_func_name,
                    new_func_name=normalized_name,
                    main_function_name=main_function["name"],
                    all_functions=functions
                )

                if replace_success:
                    program_code = new_program_code

                    # Check whether replacement would cause recursion
                    recursive_call_pattern = rf'\b{re.escape(normalized_name)}\s*\('
                    main_func_body_pattern = rf'async\s+function\s+{re.escape(normalized_name)}\s*\([^)]*\)\s*{{[^}}]*\b{re.escape(normalized_name)}\s*\('
                    if re.search(main_func_body_pattern, program_code, re.DOTALL):
                        print(f"\033[31m[Name Normalization] error: recursion detected after normalization, rolling back to original name\033[0m")
                        normalized_name = old_func_name
                        # Roll back program_code (including top-level declarations)
                        code_parts = all_top_level_declarations + [function["body"] for function in functions_to_include]
                        program_code = "\n\n".join(code_parts)
                else:
                    print(f"\033[33m[Name Normalization] warning: function-name replacement failed, keeping original name '{old_func_name}'\033[0m")
                    normalized_name = old_func_name

            # Update exec_code if it references the old function name
            if old_func_name == main_function["name"]:
                exec_code = _invoke(normalized_name)
            elif old_func_name in exec_code:
                # If exec_code calls the old function, replace it
                exec_code = re.sub(
                    rf'\b{re.escape(old_func_name)}\s*\(',
                    f"{normalized_name}(",
                    exec_code
                )

        return program_code, exec_code, normalized_name, all_top_level_declarations
