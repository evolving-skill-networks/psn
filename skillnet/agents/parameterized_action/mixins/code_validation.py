"""
CodeValidationMixin for ParameterizedActionAgent

Methods:
    _validate_generated_code: Validate generated code for correctness
    _validate_bracket_matching: Check bracket matching in code
    _validate_tdz_issues: Delegate to validators.validate_tdz_issues
    _fix_tdz_issues: Delegate to validators.fix_tdz_issues
"""

import re

from skillnet.agents.optimizer.validators import (
    validate_tdz_issues as _validate_tdz_issues_impl,
    fix_tdz_issues as _fix_tdz_issues_impl,
    validate_function_implementation,
)


class CodeValidationMixin:
    """Code validation helpers for ParameterizedActionAgent."""

    def _validate_generated_code(
        self,
        program_code: str,
        exec_code: str,
        main_function_name: str,
        all_functions: list
    ) -> dict:
        """
        Validate whether the generated code is correct.

        Checks:
        1. Whether there are recursive calls
        2. Whether there are name conflicts
        3. Whether the code structure is correct

        Args:
            program_code: Program code
            exec_code: Execution code
            main_function_name: Name of the main function
            all_functions: List of all functions

        Returns:
            dict: {"valid": bool, "reason": str, "critical": bool}
        """
        # Extract all function names
        function_names = set()
        for func in all_functions:
            if isinstance(func, dict) and "name" in func:
                function_names.add(func["name"])

        # Check 1: detect recursive calls
        recursive_call_pattern = rf'\b{re.escape(main_function_name)}\s*\('
        main_func_def_pattern = rf'async\s+function\s+{re.escape(main_function_name)}\s*\([^)]*\)\s*{{'
        match = re.search(main_func_def_pattern, program_code)
        if match:
            # Extract the main function body
            func_body_start = match.end()
            # Find the end of the main function (simple matching)
            brace_count = 1
            func_body_end = func_body_start
            for i in range(func_body_start, len(program_code)):
                if program_code[i] == '{':
                    brace_count += 1
                elif program_code[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        func_body_end = i
                        break

            func_body = program_code[func_body_start:func_body_end]
            # Check whether the body contains a recursive call
            if re.search(recursive_call_pattern, func_body):
                return {
                    "valid": False,
                    "reason": f"Detected main function {main_function_name} calling itself (recursive call)",
                    "critical": True
                }

        # Check 2: detect name conflicts
        # Verify that the function called in exec_code exists
        exec_call_match = re.search(r'await\s+(\w+)\s*\(', exec_code)
        if exec_call_match:
            called_func_name = exec_call_match.group(1)
            if called_func_name not in function_names and called_func_name != main_function_name:
                return {
                    "valid": False,
                    "reason": f"Function {called_func_name} called by exec_code does not exist",
                    "critical": True
                }

        # Check 3: verify code structure is correct (basic checks)
        if not program_code.strip():
            return {
                "valid": False,
                "reason": "Generated code is empty",
                "critical": True
            }

        if not exec_code.strip():
            return {
                "valid": False,
                "reason": "Generated exec_code is empty",
                "critical": True
            }

        # Check 4: bracket matching (prevents truncation or syntax errors)
        bracket_validation = self._validate_bracket_matching(program_code)
        if not bracket_validation["valid"]:
            return {
                "valid": False,
                "reason": f"Code brackets do not match: {bracket_validation['error']}",
                "critical": True
            }

        # Check 5: TDZ (Temporal Dead Zone) issue detection
        tdz_validation = self._validate_tdz_issues(program_code)
        if not tdz_validation["valid"]:
            return {
                "valid": False,
                "reason": f"TDZ issue: {tdz_validation['error']}",
                "critical": True,
                "tdz_details": tdz_validation.get("tdz_details"),
                "fix_suggestion": tdz_validation.get("fix_suggestion"),
                "can_auto_fix": True  # Mark this issue as auto-fixable
            }

        # Check 6: semantic consistency (function name vs implementation logic)
        # Non-strict mode: only log warnings; do not reject generation
        _dk = getattr(self, '_domain_knowledge', None)
        _dk_fns = _dk.get_known_functions() if _dk else None
        _op_patterns = _dk_fns.get("operation_patterns") if _dk_fns else None
        semantic_validation = validate_function_implementation(
            function_name=main_function_name,
            code=program_code,
            strict=False,
            operation_patterns=_op_patterns,
        )
        if not semantic_validation.get("semantic_match", True):
            warning_msg = semantic_validation.get("warning", "")
            print(f"\033[33m[Code Validation] Semantic warning: {warning_msg}\033[0m")
            if semantic_validation.get("delegate_only"):
                delegate_target = semantic_validation.get("delegate_target", "unknown")
                print(f"\033[33m[Code Validation] Detected pure-delegation pattern: only calls '{delegate_target}'; consider adding real logic\033[0m")

        # Check 7: function-reference validation — detect non-existent function calls hallucinated by the LLM
        from skillnet.agents.optimizer.validators.code_validator._references import (
            validate_function_references,
        )
        available_skills = set()
        if hasattr(self, '_skill_manager_ref') and self._skill_manager_ref:
            available_skills = set(self._skill_manager_ref.get_all_skill_names())
        # Get domain function sets for validation (if domain configured)
        _dk = getattr(self, '_domain_knowledge', None)
        _domain_fns = _dk.get_known_functions() if _dk else None
        _domain_fns = _domain_fns or None  # Normalize empty dict to None
        ref_result = validate_function_references(
            program_code,
            available_skills=available_skills,
            domain_functions=_domain_fns,
            strict_mode=True,
        )
        if not ref_result["valid"]:
            undefined_names = [f["name"] for f in ref_result["undefined_functions"]]
            suggestions = ref_result.get("suggestions", {})
            reason = f"References unknown functions: {undefined_names}"
            if suggestions:
                reason += f" (suggestions: {suggestions})"
            return {
                "valid": False,
                "reason": reason,
                "critical": False,  # Allow the action agent to retry
            }

        return {
            "valid": True,
            "reason": "Code validation passed",
            "critical": False
        }

    def _validate_bracket_matching(self, code: str) -> dict:
        """
        Validate that the code's brackets match.

        Args:
            code: Code to validate

        Returns:
            dict: {"valid": bool, "error": str or None}
        """
        # Use skill_language if available
        _skill_lang = getattr(self, '_skill_language', None)
        if _skill_lang:
            result = _skill_lang.check_bracket_matching(code)
            return {"valid": result.valid, "error": result.errors[0] if result.errors else None}

        # Remove brackets inside strings and comments (avoid false positives)
        # Simple approach: strip single-line and multi-line comments
        code_clean = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)

        # Remove strings (simple handling; not perfect but sufficient to detect most issues)
        code_clean = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code_clean)
        code_clean = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code_clean)
        code_clean = re.sub(r'`[^`\\]*(?:\\.[^`\\]*)*`', '``', code_clean)

        # Check bracket counts
        open_brackets = code_clean.count('(') + code_clean.count('[') + code_clean.count('{')
        close_brackets = code_clean.count(')') + code_clean.count(']') + code_clean.count('}')

        if open_brackets != close_brackets:
            bracket_diff = open_brackets - close_brackets
            if bracket_diff > 0:
                return {
                    "valid": False,
                    "error": f"Missing {bracket_diff} closing bracket(s)"
                }
            else:
                return {
                    "valid": False,
                    "error": f"{abs(bracket_diff)} extra closing bracket(s)"
                }

        # Check pairing for each bracket type (stricter check)
        for open_b, close_b, name in [('(', ')', 'parentheses'), ('[', ']', 'square brackets'), ('{', '}', 'curly braces')]:
            open_count = code_clean.count(open_b)
            close_count = code_clean.count(close_b)
            if open_count != close_count:
                diff = open_count - close_count
                if diff > 0:
                    return {
                        "valid": False,
                        "error": f"{name} mismatch: missing {diff} '{close_b}'"
                    }
                else:
                    return {
                        "valid": False,
                        "error": f"{name} mismatch: {abs(diff)} extra '{close_b}'"
                    }

        return {"valid": True, "error": None}

    def _validate_tdz_issues(self, code: str) -> dict:
        """delegated to the validators.validate_tdz_issues pure function."""
        return _validate_tdz_issues_impl(code)

    def _fix_tdz_issues(self, code: str) -> tuple:
        """delegated to the validators.fix_tdz_issues pure function."""
        return _fix_tdz_issues_impl(code)
