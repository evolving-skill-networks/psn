"""JavaScript/Mineflayer language implementation for PSN skills.

Implements SkillLanguage by delegating to existing pure functions throughout
the codebase. This module centralizes all JavaScript-specific code operations
without reimplementing existing logic.
"""
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple

def _fix_template_literal_quotes(code: str) -> str:
    """Fix template literals opened with backtick but closed with " or '.

    Handles two patterns Qwen3 frequently generates:
    1. bot.chat(`text ${var} more");  →  bot.chat(`text ${var} more`);
    2. bot.chat(`Error: " + err);     →  bot.chat("Error: " + err);
    """
    # Pattern 1: backtick ... ${} ... closing with " or ' before );
    # CRITICAL: [^`\\\n] excludes newlines to prevent cross-line matching.
    # Without \n exclusion, the regex matches from line N's opening backtick
    # to line N+M's closing quote, corrupting valid code on other lines.
    fixed = re.sub(
        r'(`(?:[^`\\\n]|\\.|\$\{[^}]*\})*?)(["\'])(\s*\)\s*;)',
        lambda m: m.group(1) + '`' + m.group(3),
        code,
    )
    # Pattern 2 removed: was converting `text" + to "text" +
    # but caused false positives that corrupted valid template literals.
    if fixed != code:
        count = sum(1 for a, b in zip(code.split('\n'), fixed.split('\n')) if a != b)
        print(f"\033[33m[JS Sanitize] Fixed {count} mismatched template literal(s)\033[0m")
    return fixed


from skillnet.core.skill_language import (
    FunctionInfo,
    ParseResult,
    SkillLanguage,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# Project root for subprocess cwd (Prettier needs package.json)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _safe_get_babel_generator(module):
    """Extract generate function from @babel/generator, handling bridge errors.

    The JSPyBridge raises JavaScriptError (not AttributeError) on property
    access failures, so hasattr() doesn't work as a guard. Use try/except.
    """
    # Try .default (standard CJS/ESM default export)
    try:
        gen = module.default
        if gen is not None and callable(gen):
            return gen
    except Exception:
        pass
    # Try .generate (named export in @babel/generator)
    try:
        gen = module.generate
        if gen is not None and callable(gen):
            return gen
    except Exception:
        pass
    # Do NOT fall back to callable(module) — JSPyBridge Proxy objects
    # always report callable=True (Proxy class defines __call__), but
    # module objects aren't functions. Returning the module itself causes
    # babel_generator(node) to fail silently later.
    return None


class JavaScriptLanguage:
    """JavaScript skill language implementation using Babel and Prettier."""

    def __init__(self):
        self._babel = None
        self._babel_generator = None

    @property
    def name(self) -> str:
        return "javascript"

    @property
    def file_extension(self) -> str:
        return ".js"

    @property
    def requires_async_main(self) -> bool:
        # Mineflayer requires async event handling.
        return True

    # --- Babel Management ---

    def _ensure_babel(self):
        """Lazy-load Babel and Babel generator."""
        if self._babel is not None:
            return
        try:
            from javascript import require
            self._babel = require("@babel/core")
            babel_gen_mod = require("@babel/generator")
            self._babel_generator = _safe_get_babel_generator(babel_gen_mod)
        except Exception as e:
            logger.warning(f"Failed to load Babel: {e}")
            self._babel = None
            self._babel_generator = None

    def _generate_code(self, node, source: str) -> str:
        """Generate code from AST node.

        Tries babel_generator first; falls back to slicing the original source
        using node.start / node.end (character offsets set by @babel/parser).
        """
        if self._babel_generator is not None:
            try:
                return self._babel_generator(node).code
            except Exception:
                pass
        # Fallback: slice from original source using AST node positions
        try:
            return source[int(node.start):int(node.end)]
        except Exception:
            return ""

    # --- Parsing ---

    def parse(self, code: str) -> ParseResult:
        """Parse JavaScript code and extract function information.

        Uses Babel to parse the code into an AST, then walks the AST to
        extract function declarations and top-level variable declarations.
        The main function is identified as the last async function declaration.

        Based on the logic in parameterized_action/mixins/code_parsing.py
        ``_parse_and_identify_main``.
        """
        if not code or not code.strip():
            return ParseResult(
                functions=[], main_function=None,
                success=False, error="Empty code",
            )

        self._ensure_babel()
        if self._babel is None:
            return ParseResult(
                functions=[], main_function=None,
                success=False, error="Babel not available",
            )

        try:
            parsed = self._babel.parse(code)
        except Exception as e:
            return ParseResult(
                functions=[], main_function=None,
                success=False, error=str(e),
            )

        functions: List[FunctionInfo] = []
        top_level_declarations: List[str] = []

        # First pass: collect all top-level declarations
        for node in parsed.program.body:
            try:
                if node.type == "VariableDeclaration":
                    decl_code = self._generate_code(node, code)
                    if decl_code:
                        top_level_declarations.append(decl_code)
                elif node.type == "ExpressionStatement":
                    expr_code = self._generate_code(node, code)
                    # Only keep potentially useful expressions (e.g., require() calls)
                    if expr_code and "require(" in expr_code:
                        top_level_declarations.append(expr_code)
            except Exception:
                continue

        # Second pass: extract all function declarations
        for node in parsed.program.body:
            if node.type != "FunctionDeclaration":
                continue

            # Defensive check: ensure node.id and node.id.name exist
            # Use try/except because JSPyBridge raises JavaScriptError
            # (not AttributeError) on property access failures
            try:
                node_name = node.id and node.id.name
                if not node_name:
                    continue
            except Exception:
                continue

            try:
                is_async = bool(node["async"])
            except Exception:
                is_async = False

            func_code = self._generate_code(node, code)

            # Extract parameter names
            params = []
            try:
                node_params = node["params"]
            except Exception:
                node_params = []
            for param in node_params:
                try:
                    if param.name:
                        params.append(param.name)
                        continue
                except Exception:
                    pass
                try:
                    if param.left and param.left.name:
                        # Default parameter: param = defaultValue
                        params.append(param.left.name)
                except Exception:
                    pass

            try:
                fn_name = node.id.name
            except Exception:
                fn_name = node_name

            functions.append(FunctionInfo(
                name=fn_name,
                params=params,
                body=func_code,
                is_async=is_async,
                full_code=func_code,
                declarations=top_level_declarations.copy(),
            ))

        # Find the last async function (main function)
        main_function = None
        for func in reversed(functions):
            if func.is_async:
                main_function = func
                break

        return ParseResult(
            functions=functions,
            main_function=main_function,
            top_level_declarations=top_level_declarations,
            raw_ast=parsed,
            success=True,
        )

    def extract_main_function(self, code: str) -> Optional[FunctionInfo]:
        """Extract the main async function (last async function declaration)."""
        result = self.parse(code)
        return result.main_function

    def regenerate(self, ast_node: Any) -> str:
        """Regenerate JavaScript source from a Babel AST node.

        Uses ``@babel/generator``. Caller must have previously triggered
        Babel loading (e.g., via :meth:`parse`); if Babel is unavailable
        or the node cannot be regenerated, raises ``RuntimeError``.
        """
        self._ensure_babel()
        if self._babel_generator is None:
            raise RuntimeError(
                "Babel generator not available; cannot regenerate AST node"
            )
        try:
            return self._babel_generator(ast_node).code
        except Exception as e:
            raise RuntimeError(f"Failed to regenerate AST node: {e}") from e

    def extract_function_calls(self, code: str) -> Set[str]:
        """Delegate to existing implementation."""
        from skillnet.agents.optimizer.validators.code_validator._references import (
            extract_function_call_names,
        )
        return extract_function_call_names(code)

    def extract_local_definitions(self, code: str) -> Set[str]:
        """Delegate to existing implementation."""
        from skillnet.agents.optimizer.validators.code_validator._references import (
            _extract_local_definitions,
        )
        return _extract_local_definitions(code)

    # --- Validation ---

    def validate_syntax(self, code: str) -> ValidationResult:
        """Use Babel for full syntax validation, fallback to bracket matching.

        Based on optimizer/_impl/mixins/validation.py
        ``_validate_javascript_syntax``.
        """
        if not code or not code.strip():
            return ValidationResult(valid=False, errors=["Empty code"])

        self._ensure_babel()
        if self._babel is None:
            # Babel not available, fall back to bracket matching
            return self.check_bracket_matching(code)

        try:
            self._babel.parse(code)
            return ValidationResult(valid=True)
        except Exception as e:
            error_str = str(e)

            # Extract meaningful error from Babel parse error
            if "SyntaxError" in error_str:
                syntax_match = re.search(
                    r"SyntaxError: [^:]+: (.+?)(?:\n|$)", error_str
                )
                if syntax_match:
                    return ValidationResult(
                        valid=False, errors=[syntax_match.group(1).strip()]
                    )

                token_match = re.search(
                    r"(Unexpected token[^\n]+)", error_str
                )
                if token_match:
                    return ValidationResult(
                        valid=False, errors=[token_match.group(1).strip()]
                    )

            # Fall back to basic bracket check for additional info
            bracket_result = self.check_bracket_matching(code)
            if not bracket_result.valid:
                return bracket_result

            # Generic error fallback
            return ValidationResult(
                valid=False, errors=[f"Parse error: {error_str[:200]}"]
            )

    def check_bracket_matching(self, code: str) -> ValidationResult:
        """Delegate to existing implementation."""
        from skillnet.agents.optimizer.transforms.diff_engine import basic_syntax_check
        valid, error = basic_syntax_check(code)
        return ValidationResult(valid=valid, errors=[error] if error else [])

    # --- Transformation ---

    def format_code(self, code: str) -> str:
        """Format using Prettier.

        Based on optimizer/_impl/mixins/code_edit.py
        ``_format_generated_code``.
        """
        if not code or not code.strip():
            return code

        # Remove potential markdown code block markers
        from skillnet.utils.code_block import strip_code_block_markers
        code = strip_code_block_markers(code, "javascript")

        try:
            result = subprocess.run(
                ["npx", "prettier", "--parser", "babel", "--print-width", "120"],
                input=code,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=_PROJECT_ROOT,
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                if result.stderr:
                    logger.debug(
                        f"[JS Format] Prettier warning: {result.stderr[:200]}"
                    )
                return code

        except subprocess.TimeoutExpired:
            logger.warning("[JS Format] Prettier timeout")
            return code
        except FileNotFoundError:
            logger.debug("[JS Format] npx/prettier not found, skipping format")
            return code
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"[JS Format] Format exception: {e}")
            return code

    def sanitize_llm_output(self, code: str, fix_template_literals: bool = False) -> str:
        """Fix common LLM code generation errors.

        Args:
            code: Raw LLM-generated JavaScript code.
            fix_template_literals: If True, fix backtick/quote mismatch.
        """
        from skillnet.agents.skill_graph.utils.code_analysis import sanitize_python_to_js
        code = sanitize_python_to_js(code)
        if fix_template_literals:
            code = _fix_template_literal_quotes(code)
        return code

    def ensure_entry_parameter(self, code: str, param_name: str = "bot") -> str:
        """Ensure main function has bot parameter. Delegate to existing impl."""
        from skillnet.agents.refactor.code_utils import ensure_bot_parameter
        return ensure_bot_parameter(code)

    def remove_unused_helpers(
        self, code: str, main_func_name: str
    ) -> Tuple[str, List[str]]:
        """Delegate to existing implementation."""
        from skillnet.agents.optimizer.transforms.code_edit_ops import (
            remove_unused_helper_functions,
        )
        return remove_unused_helper_functions(code, main_func_name)

    # --- Generation ---

    def generate_wrapper_call(self, func_name: str, args: Dict[str, str]) -> str:
        """Generate: await funcName(bot, arg1, arg2)"""
        arg_list = ["bot"] + [v for v in args.values()]
        return f"await {func_name}({', '.join(arg_list)})"

    def generate_function_skeleton(
        self,
        name: str,
        params: List[str],
        is_async: bool = True,
        body: str = "",
    ) -> str:
        """Generate: async function name(bot, param1, param2) { body }"""
        all_params = ["bot"] + params
        prefix = "async " if is_async else ""
        if body:
            indent_body = "\n".join(f"  {line}" for line in body.split("\n"))
        else:
            indent_body = "  // TODO"
        return f"{prefix}function {name}({', '.join(all_params)}) {{\n{indent_body}\n}}"
