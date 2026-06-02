"""Python language implementation of the SkillLanguage Protocol.

Backed by stdlib `ast` for parsing and `compile()` for syntax validation.
Optional `black` dependency for formatting; gracefully degrades if absent.
"""
from __future__ import annotations

import ast
import re
import textwrap
from typing import Dict, List, Optional, Set, Tuple

from skillnet.core.skill_language import (
    FunctionInfo,
    ParseResult,
    ValidationResult,
)


class PythonLanguage:
    """Python implementation of the SkillLanguage Protocol."""

    _black_warned = False

    @property
    def name(self) -> str:
        return "python"

    @property
    def file_extension(self) -> str:
        return ".py"

    @property
    def requires_async_main(self) -> bool:
        # A dict-event domain's bot API is synchronous; skill main functions
        # can be plain def or async def.
        return False

    # ----- Parsing --------------------------------------------------------

    def parse(self, code: str) -> ParseResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ParseResult(
                functions=[],
                main_function=None,
                success=False,
                error=f"SyntaxError: {e.msg} (line {e.lineno})",
            )

        functions: List[FunctionInfo] = []
        top_level_declarations: List[str] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [a.arg for a in node.args.args]
                full_code = ast.get_source_segment(code, node) or ""
                body = textwrap.dedent("\n".join(
                    ast.get_source_segment(code, stmt) or ""
                    for stmt in node.body
                ))
                functions.append(FunctionInfo(
                    name=node.name,
                    params=params,
                    body=body,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    full_code=full_code,
                ))
            elif isinstance(node, ast.ClassDef):
                top_level_declarations.append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = ast.get_source_segment(code, node)
                if segment:
                    top_level_declarations.append(segment)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        top_level_declarations.append(tgt.id)

        main_function = functions[-1] if functions else None

        return ParseResult(
            functions=functions,
            main_function=main_function,
            top_level_declarations=top_level_declarations,
            raw_ast=tree,
            success=True,
        )

    def extract_main_function(self, code: str) -> Optional[FunctionInfo]:
        return self.parse(code).main_function

    def regenerate(self, ast_node) -> str:
        """Regenerate Python source from an AST node via ``ast.unparse``."""
        try:
            return ast.unparse(ast_node)
        except Exception as e:
            raise RuntimeError(f"Failed to regenerate AST node: {e}") from e

    def extract_function_calls(self, code: str) -> Set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        calls: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    calls.add(func.id)
                elif isinstance(func, ast.Attribute):
                    calls.add(func.attr)
        return calls

    def extract_local_definitions(self, code: str) -> Set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        names: Set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        return names

    # ----- Validation -----------------------------------------------------

    def validate_syntax(self, code: str) -> ValidationResult:
        try:
            compile(code, "<skill>", "exec")
            return ValidationResult(valid=True)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})"],
            )

    def check_bracket_matching(self, code: str) -> ValidationResult:
        pairs = {")": "(", "]": "[", "}": "{"}
        stack: List[Tuple[str, int]] = []
        errors: List[str] = []
        in_string: Optional[str] = None  # tracks ', ", ''', \"\"\"
        in_comment = False  # tracks `# ...\n` line comments
        escape = False

        for i, ch in enumerate(code):
            if in_comment:
                if ch == "\n":
                    in_comment = False
                continue
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif code[i:i + len(in_string)] == in_string:
                    in_string = None
                continue
            if ch == "#":
                in_comment = True
                continue
            if ch in ('"', "'"):
                if code[i:i + 3] in ('"""', "'''"):
                    in_string = code[i:i + 3]
                else:
                    in_string = ch
                continue
            if ch in "([{":
                stack.append((ch, i))
            elif ch in ")]}":
                if not stack or stack[-1][0] != pairs[ch]:
                    errors.append(f"Unmatched '{ch}' at offset {i}")
                else:
                    stack.pop()

        for opener, pos in stack:
            errors.append(f"Unclosed '{opener}' at offset {pos}")

        return ValidationResult(valid=not errors, errors=errors)

    # ----- Transformation -------------------------------------------------

    def format_code(self, code: str) -> str:
        try:
            import black
        except ImportError:
            if not PythonLanguage._black_warned:
                print("[PythonLanguage] `black` not installed; "
                      "code formatting will be a no-op. "
                      "Install with: pip install black")
                PythonLanguage._black_warned = True
            return code
        try:
            return black.format_str(code, mode=black.Mode())
        except Exception:
            return code  # never fail a skill on a formatter error

    def sanitize_llm_output(self, code: str) -> str:
        # Strip ``` fences (with optional ```python label)
        code = re.sub(r"^\s*```(?:python)?\s*\n", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n```\s*$", "", code, flags=re.MULTILINE)
        # Map JS literals an LLM might emit by mistake
        code = re.sub(r"\btrue\b", "True", code)
        code = re.sub(r"\bfalse\b", "False", code)
        code = re.sub(r"\bnull\b", "None", code)
        return code.strip() + "\n"

    def ensure_entry_parameter(self, code: str, param_name: str = "bot") -> str:
        """Ensure the MAIN (last top-level) function has param_name as its first arg.

        Mirrors JS behavior of modifying only one function. Skills are conventionally
        structured with helpers first and the main entry function last.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        # Find the last top-level function — convention is main-last
        main = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                main = node
        if main is None:
            return code
        params = [a.arg for a in main.args.args]
        if param_name in params:
            return code  # already has it
        main.args.args.insert(0, ast.arg(arg=param_name, annotation=None))
        try:
            return ast.unparse(tree)
        except Exception:
            return code

    def remove_unused_helpers(
        self,
        code: str,
        main_func_name: str,
    ) -> Tuple[str, List[str]]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, []

        func_nodes: Dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_nodes[node.name] = node

        if main_func_name not in func_nodes:
            return code, []

        # BFS over call graph
        reachable: Set[str] = {main_func_name}
        frontier = [main_func_name]
        while frontier:
            name = frontier.pop()
            node = func_nodes[name]
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    callee = sub.func.id
                    if callee in func_nodes and callee not in reachable:
                        reachable.add(callee)
                        frontier.append(callee)

        removed: List[str] = []
        new_body = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in reachable:
                    new_body.append(node)
                else:
                    removed.append(node.name)
            else:
                new_body.append(node)
        tree.body = new_body
        try:
            return ast.unparse(tree), removed
        except Exception:
            return code, []

    # ----- Generation -----------------------------------------------------

    def generate_wrapper_call(self, func_name: str, args: Dict[str, str]) -> str:
        """Generate: await func_name(bot, arg1, arg2)"""
        arg_list = ["bot"] + [v for v in args.values()]
        return f"await {func_name}({', '.join(arg_list)})"

    def generate_function_skeleton(
        self,
        name: str,
        params: List[str],
        is_async: bool = True,
        body: str = "pass",
    ) -> str:
        """Generate: async def name(bot, param1, param2): body"""
        all_params = ["bot"] + params
        prefix = "async def" if is_async else "def"
        param_str = ", ".join(all_params)
        indented_body = textwrap.indent(body, "    ")
        return f"{prefix} {name}({param_str}):\n{indented_body}\n"
