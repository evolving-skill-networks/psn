"""
Naming Conflict Checker (Fix 12)

Detect whether optimized code conflicts with existing skill names.

Background:
Optimizer-generated code may add aliases like
`const setupCraftingTable = setupCraftingTableBlock;`. If
`setupCraftingTable` already exists as a separate skill, the runtime raises:
`SyntaxError: Identifier 'setupCraftingTable' has already been declared`.

This checker detects such conflicts before applying an optimization and
rejects problematic optimizations.
"""

import re
from dataclasses import dataclass
from typing import List, Set, Tuple


@dataclass
class NamingConflict:
    """Naming-conflict info."""
    conflict_type: str  # 'variable_declaration' | 'function_declaration' | 'destructuring_variable'
    name: str  # Conflicting name
    line_content: str  # Source line containing the conflict
    reason: str  # Description of the conflict


@dataclass
class NamingConflictResult:
    """Naming-conflict detection result."""
    has_conflicts: bool
    conflicts: List[NamingConflict]

    @property
    def is_valid(self) -> bool:
        """Whether the code is safe to apply (no naming conflicts)."""
        return not self.has_conflicts

    def get_summary(self) -> str:
        """Return a conflict summary."""
        if not self.has_conflicts:
            return "no naming conflicts"

        lines = [f"found {len(self.conflicts)} naming conflict(s):"]
        for conflict in self.conflicts:
            lines.append(f"  - [{conflict.conflict_type}] {conflict.name}: {conflict.reason}")
        return "\n".join(lines)


class NamingConflictChecker:
    """
    Naming-conflict detector.

    Detects whether optimized code conflicts with existing skill names.
    """

    def check(
        self,
        optimized_code: str,
        existing_skill_names: Set[str],
    ) -> NamingConflictResult:
        """
        Check whether optimized code conflicts with existing skill names.

        Args:
            optimized_code: optimized code
            existing_skill_names: set of existing skill names

        Returns:
            NamingConflictResult: detection result.
        """
        conflicts = []

        # 1. Check variable declarations (const/let/var X = ...)
        conflicts.extend(
            self._check_variable_declarations(optimized_code, existing_skill_names)
        )

        # 2. Check non-async function declarations (function X() {...}).
        # Note: async function declarations are normal skill definitions and not conflicts.
        conflicts.extend(
            self._check_sync_function_declarations(optimized_code, existing_skill_names)
        )

        # 3. check destructuring declarations (const { a, b } = ...)
        conflicts.extend(
            self._check_destructuring_declarations(optimized_code, existing_skill_names)
        )

        return NamingConflictResult(
            has_conflicts=len(conflicts) > 0,
            conflicts=conflicts,
        )

    def _check_variable_declarations(
        self,
        code: str,
        existing_names: Set[str],
    ) -> List[NamingConflict]:
        """Check whether variable declarations conflict with existing skills."""
        conflicts = []

        # Match const/let/var name = ...
        # Excludes default values in function parameters (e.g. function foo(x = 1))
        var_decl_pattern = r'^[^\n]*\b(const|let|var)\s+(\w+)\s*='

        for match in re.finditer(var_decl_pattern, code, re.MULTILINE):
            var_name = match.group(2)
            if var_name in existing_names:
                # Get the full line containing the declaration
                line_start = code.rfind('\n', 0, match.start()) + 1
                line_end = code.find('\n', match.end())
                if line_end == -1:
                    line_end = len(code)
                line_content = code[line_start:line_end].strip()

                conflicts.append(NamingConflict(
                    conflict_type='variable_declaration',
                    name=var_name,
                    line_content=line_content,
                    reason=f"variable declaration '{var_name}' shares the name of an existing skill; runtime will throw 'Identifier has already been declared'",
                ))

        return conflicts

    def _check_sync_function_declarations(
        self,
        code: str,
        existing_names: Set[str],
    ) -> List[NamingConflict]:
        """Check whether sync function declarations conflict with existing skills."""
        conflicts = []

        # Match function name(...) but not async function name(...)
        # Use a negative lookbehind to exclude async
        sync_func_pattern = r'(?<!async\s)\bfunction\s+(\w+)\s*\('

        for match in re.finditer(sync_func_pattern, code):
            func_name = match.group(1)
            if func_name in existing_names:
                # Verify this isn't a helper nested inside an async function.
                # If it's an inner helper, it's typically safe (scope isolation),
                # but top-level declarations are conflict-prone.

                # Simple check: if there's an open async function before this
                # and braces aren't fully closed, it's a nested helper — skip.
                prefix = code[:match.start()]
                # Count async functions and matching closing braces
                async_func_count = len(re.findall(r'async\s+function\s+\w+\s*\([^)]*\)\s*\{', prefix))
                close_brace_count = prefix.count('}')

                # If async function count exceeds closing-brace count, this is nested
                if async_func_count > close_brace_count:
                    continue  # Skip nested helper

                # Get the full line containing the declaration
                line_start = code.rfind('\n', 0, match.start()) + 1
                line_end = code.find('\n', match.end())
                if line_end == -1:
                    line_end = len(code)
                line_content = code[line_start:line_end].strip()

                conflicts.append(NamingConflict(
                    conflict_type='function_declaration',
                    name=func_name,
                    line_content=line_content,
                    reason=f"function declaration '{func_name}' shares the name of an existing skill; may cause namespace conflicts",
                ))

        return conflicts

    def _check_destructuring_declarations(
        self,
        code: str,
        existing_names: Set[str],
    ) -> List[NamingConflict]:
        """
        v7.3 Phase 2: detect identifier conflicts in destructuring assignments.

        Patterns detected:
        - const { a, b } = ...
        - const { nested: { x } } = ...
        - let { x = default } = ...

        Args:
            code: JavaScript code
            existing_names: set of existing skill names

        Returns:
            List[NamingConflict]: conflict list.
        """
        conflicts = []

        # Match destructuring assignments: const/let/var { ... } = ...
        # Note: nested destructuring needs handling
        destructure_pattern = r'(const|let|var)\s*\{\s*([^}]+)\s*\}\s*='

        for match in re.finditer(destructure_pattern, code, re.MULTILINE):
            destructure_content = match.group(2)

            # Extract all identifiers (handle nesting, defaults, renaming, etc.).
            # Match: identifier, identifier: alias, identifier = default.
            # Excludes common non-variable names (e.g. goals, require).
            identifiers = re.findall(
                r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?:[:,=}]|$)',
                destructure_content
            )

            for identifier in identifiers:
                # Exclude common namespace names and keywords
                if identifier in ['goals', 'require', 'default', 'as']:
                    continue

                if identifier in existing_names:
                    # Get the full line containing the declaration
                    line_start = code.rfind('\n', 0, match.start()) + 1
                    line_end = code.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(code)
                    line_content = code[line_start:line_end].strip()

                    conflicts.append(NamingConflict(
                        conflict_type='destructuring_variable',
                        name=identifier,
                        line_content=line_content,
                        reason=f"destructuring identifier '{identifier}' shares the name of an existing skill; runtime will throw 'Identifier has already been declared'",
                    ))

        return conflicts


def check_naming_conflicts(
    optimized_code: str,
    existing_skill_names: Set[str],
) -> Tuple[bool, List[str]]:
    """
    Check whether optimized code conflicts with existing skill names.

    Simple function-style interface for use in optimizer_impl.py.

    Args:
        optimized_code: optimized code
        existing_skill_names: set of existing skill names

    Returns:
        Tuple[bool, List[str]]: (is_valid, conflict_messages)
            - is_valid: True if no conflicts, False otherwise.
            - conflict_messages: list of conflict descriptions.
    """
    checker = NamingConflictChecker()
    result = checker.check(optimized_code, existing_skill_names)

    if result.is_valid:
        return True, []

    messages = []
    for conflict in result.conflicts:
        messages.append(conflict.reason)

    return False, messages
