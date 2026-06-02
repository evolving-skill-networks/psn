"""Language abstraction for skill code operations.

Different domains may use different programming languages for skills
(e.g. JavaScript for Minecraft/Mineflayer). This module defines the
Protocol that language implementations must satisfy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable


@dataclass
class FunctionInfo:
    """Information about an extracted function."""
    name: str
    params: List[str]
    body: str
    is_async: bool = False
    full_code: str = ""  # Complete function source including signature
    declarations: List[str] = field(default_factory=list)  # Top-level declarations before function


@dataclass
class ParseResult:
    """Result of parsing skill code."""
    functions: List[FunctionInfo]
    main_function: Optional[FunctionInfo]
    top_level_declarations: List[str] = field(default_factory=list)
    raw_ast: Any = None  # Language-specific AST (Babel AST, Python ast.Module, etc.)
    success: bool = True
    error: str = ""


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@runtime_checkable
class SkillLanguage(Protocol):
    """Protocol for language-specific skill code operations.

    Implementations handle parsing, validation, formatting, and transformation
    of skill code for a specific programming language.
    """

    @property
    def name(self) -> str:
        """Language identifier (e.g., 'javascript', 'python')."""
        ...

    @property
    def file_extension(self) -> str:
        """File extension for skill code files (e.g., '.js', '.py')."""
        ...

    @property
    def requires_async_main(self) -> bool:
        """Whether main skill functions MUST be declared async.

        True for languages where the runtime requires async (e.g.
        JavaScript with mineflayer). False for languages where sync
        functions are valid skill entry points (e.g. Python with
        a synchronous bot API). PSN's action-parsers consult this to
        decide which function declarations to accept as main.
        """
        ...

    # --- Parsing ---

    def parse(self, code: str) -> ParseResult:
        """Parse code and extract function information."""
        ...

    def extract_main_function(self, code: str) -> Optional[FunctionInfo]:
        """Extract the main entry-point function from code."""
        ...

    def regenerate(self, ast_node: Any) -> str:
        """Regenerate source code from a (possibly mutated) AST node.

        Implementations should map this to their language's standard code
        generator (e.g. ``@babel/generator`` for JS, ``ast.unparse`` for
        Python). This is intended for callers that need to re-emit a
        sub-AST extracted from ``ParseResult.raw_ast``.
        """
        ...

    def extract_function_calls(self, code: str) -> Set[str]:
        """Extract all function call names from code."""
        ...

    def extract_local_definitions(self, code: str) -> Set[str]:
        """Extract all locally defined function/class names."""
        ...

    # --- Validation ---

    def validate_syntax(self, code: str) -> ValidationResult:
        """Full syntax validation using language-specific parser."""
        ...

    def check_bracket_matching(self, code: str) -> ValidationResult:
        """Quick bracket/brace/paren matching check."""
        ...

    # --- Transformation ---

    def format_code(self, code: str) -> str:
        """Format code using language-specific formatter (Prettier/Black)."""
        ...

    def sanitize_llm_output(self, code: str) -> str:
        """Clean up common LLM output issues (e.g., Python True -> JS true)."""
        ...

    def ensure_entry_parameter(self, code: str, param_name: str) -> str:
        """Ensure the main function has the required entry parameter."""
        ...

    def remove_unused_helpers(self, code: str, main_func_name: str) -> Tuple[str, List[str]]:
        """Remove helper functions not called by the main function.
        Returns (cleaned_code, removed_function_names)."""
        ...

    # --- Generation ---

    def generate_wrapper_call(self, func_name: str, args: Dict[str, str]) -> str:
        """Generate a function call expression (e.g., 'await funcName(bot, arg1, arg2)')."""
        ...

    def generate_function_skeleton(self, name: str, params: List[str],
                                    is_async: bool = True, body: str = "") -> str:
        """Generate a function definition skeleton."""
        ...
