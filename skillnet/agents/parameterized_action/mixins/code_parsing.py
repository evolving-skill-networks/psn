"""
CodeParsingMixin for ParameterizedActionAgent

Methods:
    _parse_and_identify_main: Parse skill code, extract functions, identify main
    _identify_main_function: Use heuristic scoring to identify the main function
    _extract_code_from_message: Extract code and main function info from AI message
"""

import re
from types import SimpleNamespace


def _strip_strings_and_comments_js(code: str) -> str:
    """Strip JavaScript string literals and comments from ``code``.

    Replaces strings with empty placeholders and removes single-line /
    multi-line comments so callers can scan for tokens (e.g. recursive
    function references) without false positives from inside string
    literals or comments. JS-specific by design — used by JS-coupled
    parsing paths in this mixin.
    """
    # Remove double-quoted strings (handling escape characters)
    code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
    # Remove single-quoted strings
    code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
    # Remove template strings (backticks)
    code = re.sub(r'`[^`\\]*(?:\\.[^`\\]*)*`', '``', code)
    # Remove single-line comments
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code


def _resolve_skill_language(agent):
    """Resolve a SkillLanguage instance for the given agent.

    Cache-first (``agent._skill_language``), registry-fallback. Raises
    ``RuntimeError`` if no DomainKnowledge is registered.
    """
    skill_lang = getattr(agent, "_skill_language", None)
    if skill_lang is not None:
        return skill_lang
    from skillnet.core.dk_registry import get_domain_knowledge
    dk = get_domain_knowledge()
    if dk is None:
        raise RuntimeError(
            "No DomainKnowledge registered; cannot parse skill code"
        )
    return dk.get_skill_language_impl()


class CodeParsingMixin:
    """Code parsing helpers for ParameterizedActionAgent."""

    def _identify_main_function(self, functions: list, task: str = None) -> dict:
        """
        Identify the main function using multiple heuristics.

        Heuristic scoring:
        1. Task relevance (+50): the function name contains task keywords
        2. Complexity (+lines): the main function is usually longer than helper functions
        3. Call relations (+20 per call): the main function typically calls helper functions
        4. Helper pattern (-100): penalize obvious helper-function names

        Args:
            functions: list of functions; each element contains name, type, body, params
            task: task description, used for the task-relevance score

        Returns:
            The most likely main function, or None
        """
        # Filter to async functions only when the active language requires it.
        skill_lang = _resolve_skill_language(self)
        requires_async = getattr(skill_lang, "requires_async_main", True)
        if requires_async:
            candidates = [f for f in functions if f.get("type") == "AsyncFunctionDeclaration"]
        else:
            candidates = list(functions)
        if not candidates:
            return None

        # If there is only one candidate, return it directly
        if len(candidates) == 1:
            return candidates[0]

        # Prefix patterns for helper functions
        HELPER_PREFIXES = ('safe', 'get', 'find', 'count', 'is', 'has', 'check', '_', 'normalize', 'handle', 'log', 'debug')

        # Score each function
        scored = []
        for func in candidates:
            name = func.get("name", "")
            body = func.get("body", "")
            score = 0
            reasons = []

            # 1. Penalize helper-pattern function names
            name_lower = name.lower()
            if any(name_lower.startswith(p) for p in HELPER_PREFIXES):
                score -= 100
                reasons.append(f"helper prefix (-100)")

            # 2. Task-relevance score
            if task:
                # Extract keywords from the task
                task_lower = task.lower().replace('_', ' ').replace('-', ' ')
                task_words = set(task_lower.split())
                # Remove common meaningless words
                task_words -= {'a', 'an', 'the', '1', '2', '3', '4', '5', 'one', 'two', 'three'}

                # Extract words from the function name (camelCase)
                name_words = set(w.lower() for w in re.findall(r'[A-Z][a-z]*|[a-z]+', name))

                # Compute matches
                matches = task_words & name_words
                if matches:
                    score += 50
                    reasons.append(f"task match: {matches} (+50)")

            # 3. Complexity score (line count)
            line_count = len(body.split('\n'))
            score += line_count
            reasons.append(f"lines: {line_count}")

            # 4. Call-relation score
            calls_count = 0
            for other in candidates:
                other_name = other.get("name", "")
                if other_name and other_name != name:
                    # Check whether it calls other functions
                    if re.search(rf'\b{re.escape(other_name)}\s*\(', body):
                        calls_count += 1
                        score += 20
            if calls_count > 0:
                reasons.append(f"calls {calls_count} functions (+{calls_count * 20})")

            # 5. Extra penalty: very short functions (<= 10 lines) containing only simple operations
            if line_count <= 10:
                # Check whether it's just a simple wrapper (only bot.chat or a single await)
                if 'bot.chat' in body and body.count('await') <= 2:
                    score -= 50
                    reasons.append("simple wrapper (-50)")

            scored.append({
                "function": func,
                "score": score,
                "reasons": reasons
            })

        # Sort by score
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Print debug info
        if len(scored) > 1:
            print(f"\033[36m[Main Function Identification] Candidate function scores:\033[0m")
            for item in scored[:3]:  # Only show the top 3
                func_name = item["function"].get("name", "unknown")
                print(f"\033[36m  - {func_name}: score={item['score']} ({', '.join(item['reasons'])})\033[0m")

        # Return the highest-scoring function
        return scored[0]["function"] if scored else None

    def _functions_from_parse_result(self, parse_result) -> list:
        """Convert a SkillLanguage ParseResult into the dict shape used by
        _identify_main_function and the rest of the parameterized-action
        pipeline.
        """
        functions = []
        top_level_declarations = list(parse_result.top_level_declarations or [])
        for f in parse_result.functions:
            functions.append({
                "name": f.name,
                "type": "AsyncFunctionDeclaration" if f.is_async else "FunctionDeclaration",
                "body": f.full_code or f.body,
                "params": [SimpleNamespace(name=p) for p in f.params],
                "top_level_declarations": list(f.declarations) if f.declarations else top_level_declarations.copy(),
            })
        return functions

    def _extract_code_from_message(self, message, task: str = None):
        """
        Extract code and main-function info from an AI message (helper)

        Args:
            message: AI message to extract code from
            task: task description, used to help identify the main function

        Returns:
            tuple: (main_function_name, main_function_code, all_code) or (None, None, None) if failed
        """
        try:
            # Extract code block from message first
            from skillnet.utils.code_block import build_code_block_pattern
            _dk = getattr(self, '_domain_knowledge', None)
            _lang = _dk.get_skill_language() if _dk else "javascript"
            code_pattern = build_code_block_pattern(_lang)
            code = "\n".join(code_pattern.findall(message.content))
            if not code:
                # Fallback 1: LLM may use wrong language tag (python, java, etc.)
                # Try any fenced code block containing a function declaration.
                import re as _re
                any_blocks = _re.findall(r'```\w*\n(.*?)```', message.content, _re.DOTALL)
                for block in any_blocks:
                    if ('async function' in block or 'function ' in block
                            or _re.search(r'^\s*(async\s+)?def\s+\w+', block, _re.MULTILINE)):
                        code = block
                        print(f"\033[33m[Code Extraction] Fallback: extracted code from non-canonical code block\033[0m")
                        break
            if not code:
                # Fallback 2: LLM emitted unfenced raw code (common with Python
                # responses where gpt-5-mini writes prose as `#` comments and
                # code inline). If the message contains a top-level function
                # declaration, treat the whole message as code.
                import re as _re
                if _re.search(r'^\s*(async\s+)?def\s+\w+', message.content, _re.MULTILINE) \
                        or _re.search(r'^\s*(async\s+)?function\s+\w+', message.content, _re.MULTILINE):
                    code = message.content
                    print(f"\033[33m[Code Extraction] Fallback: LLM omitted fences; using raw message content\033[0m")
            if not code:
                return None, None, None

            # Resolve skill language (cache-first, registry-fallback)
            skill_lang = _resolve_skill_language(self)

            # Sanitize LLM output before parsing
            if hasattr(skill_lang, 'sanitize_llm_output'):
                from skillnet.core.model_profile import detect_model_profile
                _model_name = getattr(self, 'model_name', '') or ''
                _profile = detect_model_profile(_model_name)
                try:
                    code = skill_lang.sanitize_llm_output(
                        code, fix_template_literals=_profile.enable_sanitizer
                    )
                except TypeError:
                    # Implementations may not accept fix_template_literals kwarg
                    code = skill_lang.sanitize_llm_output(code)

            parse_result = skill_lang.parse(code)
            if not parse_result.success:
                print(f"\033[33m[Code Extraction] parse failed: {parse_result.error}\033[0m")
                return None, None, None

            functions = self._functions_from_parse_result(parse_result)
            if not functions:
                return None, None, None

            # Use the smart identification method to find the main function
            main_function = self._identify_main_function(functions, task=task)

            if main_function is None:
                # Fallback: last function of the appropriate type for this language.
                requires_async = getattr(skill_lang, "requires_async_main", True)
                for function in reversed(functions):
                    if requires_async:
                        if function["type"] == "AsyncFunctionDeclaration":
                            main_function = function
                            break
                    else:
                        main_function = function
                        break

            if main_function is None:
                return None, None, None

            return main_function["name"], main_function["body"], code
        except Exception as e:
            print(f"\033[33m[Code Extraction] Failed to extract code: {e}\033[0m")
            return None, None, None

    def _parse_and_identify_main(self, message_content, skill_lang=None):
        """Parse skill code from message content, extract functions, and identify
        the main function. Also validates the main function and checks for recursive calls.

        Args:
            message_content: The raw message content string
            skill_lang: Optional SkillLanguage instance; resolved from
                ``self._skill_language`` (cache) or the dk_registry when omitted.

        Returns:
            tuple: (functions, main_function)

        Raises:
            AssertionError: If no functions found or main function is invalid
            ValueError: If recursive call is detected in main function
        """
        if skill_lang is None:
            skill_lang = _resolve_skill_language(self)

        from skillnet.utils.code_block import build_code_block_pattern
        _dk = getattr(self, '_domain_knowledge', None)
        _lang = _dk.get_skill_language() if _dk else "javascript"
        code_pattern = build_code_block_pattern(_lang)
        code = "\n".join(code_pattern.findall(message_content))

        if not code:
            # Fallback: LLM emitted unfenced raw code. If the message contains
            # a top-level function declaration, treat the whole message as code.
            import re as _re
            if _re.search(r'^\s*(async\s+)?def\s+\w+', message_content, _re.MULTILINE) \
                    or _re.search(r'^\s*(async\s+)?function\s+\w+', message_content, _re.MULTILINE):
                code = message_content
                print(f"\033[33m[Code Extraction] Fallback: LLM omitted fences; using raw message content\033[0m")

        assert code, "No code block found in message content"

        parse_result = skill_lang.parse(code)
        if not parse_result.success:
            raise ValueError(f"Failed to parse skill code: {parse_result.error}")

        functions = self._functions_from_parse_result(parse_result)
        assert functions, "No functions found"

        # find the last function of the appropriate type for this language.
        requires_async = getattr(skill_lang, "requires_async_main", True)
        main_function = None
        for function in reversed(functions):
            if requires_async:
                if function["type"] == "AsyncFunctionDeclaration":
                    main_function = function
                    break
            else:
                main_function = function
                break
        expected = "async function" if requires_async else "function"
        assert main_function is not None, (
            f"No {expected} found. Your main function must be {expected}."
        )

        # Defensive check: ensure main_function has a name
        assert (
            main_function.get("name") is not None and main_function["name"]
        ), "Main function must have a name"

        # In parameterized mode, the main function can be parameterized
        # It must have the entry parameter as the first argument, but can have additional parameters with defaults
        _entry_param = _dk.get_entry_parameter_name() if _dk else "bot"
        assert (
            len(main_function["params"]) >= 1
            and main_function["params"][0].name == _entry_param
        ), f"Main function {main_function['name']} must have '{_entry_param}' as the first argument"

        main_function_body = main_function["body"]
        main_function_name = main_function["name"]

        # Detect recursive calls — prevent a function from calling itself
        # Remove strings and comments to avoid false positives (function names in
        # strings/comments should not count as recursive calls). The local
        # JS-specific helper correctly handles escape characters.
        body_without_comments = _strip_strings_and_comments_js(main_function_body)

        # Check whether the main function calls itself (recursive call)
        recursive_call_pattern = rf'\b{re.escape(main_function_name)}\s*\('
        if re.search(recursive_call_pattern, body_without_comments):
            # Check whether it is a real recursive call (not a typeof check)
            # A check like typeof mineLogs === "function" does not count as a recursive call
            body_without_typeof = re.sub(r'typeof\s+\w+\s*===\s*["\']function["\']', '', body_without_comments, flags=re.IGNORECASE)
            if re.search(recursive_call_pattern, body_without_typeof):
                # Further check: is the call inside the main-function definition
                # (excluding the definition itself)? Match the function-declaration
                # syntax for either JavaScript (`[async] function name(...) {`)
                # or Python (`[async] def name(...):`) so the body-extraction
                # below works for both languages.
                main_func_def_pattern = (
                    rf'(?:async\s+function\s+{re.escape(main_function_name)}\s*\([^)]*\)\s*\{{'
                    rf'|(?:async\s+)?def\s+{re.escape(main_function_name)}\s*\([^)]*\)\s*[^:]*:)'
                )
                match = re.search(main_func_def_pattern, body_without_comments)
                if match:
                    # Extract the body (excluding the function definition)
                    func_body_only = body_without_comments[match.end():]
                    # Check whether the body contains a recursive call
                    if re.search(recursive_call_pattern, func_body_only):
                        error_msg = (
                            f"Detected that main function {main_function_name} calls itself (recursive call), which will cause an infinite loop.\n"
                            f"Suggestions:\n"
                            f"  1. If this is a mistake, remove the recursive call\n"
                            f"  2. If you need to call another skill, use the correct function name\n"
                            f"  3. If you need to check whether a function exists, use a typeof check rather than calling it directly"
                        )
                        print(f"\033[31m[Recursive Call Error] {error_msg}\033[0m")
                        # Raise an exception to stop code execution
                        raise ValueError(f"Recursive call detected in function {main_function_name}: {error_msg}")
                else:
                    # If we couldn't find the function-definition pattern, still check the entire body
                    error_msg = (
                        f"Detected that main function {main_function_name} calls itself (recursive call), which will cause an infinite loop.\n"
                        f"Suggestions:\n"
                        f"  1. If this is a mistake, remove the recursive call\n"
                        f"  2. If you need to call another skill, use the correct function name\n"
                        f"  3. If you need to check whether a function exists, use a typeof check rather than calling it directly"
                    )
                    print(f"\033[31m[Recursive Call Error] {error_msg}\033[0m")
                    # Raise an exception to stop code execution
                    raise ValueError(f"Recursive call detected in function {main_function_name}: {error_msg}")

        return functions, main_function
