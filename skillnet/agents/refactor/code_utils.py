"""
Refactor Code Utilities

JavaScript dependency-analysis tools for analyzing and injecting the external dependencies skill code needs.

Contains:
- GLOBAL_PROVIDED_DEPENDENCIES: set of dependencies provided at runtime
- INJECTABLE_DEPENDENCIES: map of dependencies that need to be injected
- STANDARD_DEPENDENCIES: backward-compatible alias
- analyze_code_dependencies(): analyze external dependencies in code
- inject_dependencies(): inject dependency declarations at the top of a function body
- ensure_bot_parameter(): ensure the function signature includes the bot parameter

Extracted from base.py for modularity.
"""

import re
from typing import List


# ========== Dependency categorization ==========
#
# Domain knowledge provides two sets of globally-available variables:
# 1. Data library dependencies (via get_global_dependencies()) — need require statements
# 2. Environment/runtime globals (via get_environment_globals()) — already in scope

# Empty defaults — domain knowledge provides actual values at runtime.
GLOBAL_PROVIDED_DEPENDENCIES = set()

_ENVIRONMENT_GLOBALS = set()


def get_global_provided_dependencies(domain_knowledge=None):
    """Get global provided dependencies from domain knowledge.

    Returns the union of domain-provided data dependencies and
    environment globals. Returns empty set when no domain is available.
    """
    if domain_knowledge:
        deps = domain_knowledge.get_global_dependencies()
        env_globals = domain_knowledge.get_environment_globals()
        if deps or env_globals:
            return set(deps.keys()) | env_globals
    return set()

# Map of dependencies that need to be injected: dependency name -> require statement
# Goal* and Movements have been moved to GLOBAL_PROVIDED_DEPENDENCIES (provided by the /step scope)
INJECTABLE_DEPENDENCIES = {}

# Backward compatibility: keep STANDARD_DEPENDENCIES (deprecated, used by older code).
# New code should use INJECTABLE_DEPENDENCIES.
STANDARD_DEPENDENCIES = INJECTABLE_DEPENDENCIES


def analyze_code_dependencies(code: str, include_global_provided: bool = False) -> List[str]:
    """
    Analyze the external dependencies used by JavaScript skill code.

    Scans the code for dependency references and checks whether they are already declared.
    By default only returns dependencies that need injection (INJECTABLE_DEPENDENCIES);
    dependencies already provided by globalDepsCode (mcData, Vec3, checkRecipe) are excluded.

    Args:
        code: JavaScript skill code
        include_global_provided: whether to include dependencies already provided by globalDepsCode (default False)

    Returns:
        List[str]: list of dependency names that need to be injected (used in code but not declared)
    """
    if not code:
        return []

    used_deps = []

    # Decide which dependency set to check
    deps_to_check = INJECTABLE_DEPENDENCIES
    if include_global_provided:
        # Merge the two sets (for diagnostic purposes)
        deps_to_check = {**INJECTABLE_DEPENDENCIES}
        for dep in GLOBAL_PROVIDED_DEPENDENCIES:
            if dep not in deps_to_check:
                deps_to_check[dep] = f'// {dep} provided by globalDepsCode'

    for dep_name in deps_to_check:
        # Check whether the dependency is used in the code (with a word boundary to avoid partial matches)
        if re.search(rf'\b{dep_name}\b', code):
            # Check whether the dependency is already declared in the code
            # Match const/let/var declarations, including destructuring
            has_declaration = re.search(
                rf'(const|let|var)\s+.*\b{dep_name}\b',
                code
            )
            # Also check whether it appears inside a require statement
            has_require = re.search(
                rf'require\s*\([^)]*\).*{dep_name}',
                code
            )
            if not has_declaration and not has_require:
                # Only add dependencies that need injection (exclude those already provided by globalDepsCode)
                if dep_name in INJECTABLE_DEPENDENCIES:
                    used_deps.append(dep_name)

    return used_deps


def inject_dependencies(code: str, deps: List[str]) -> str:
    """
    Inject dependency declarations at the top of a JavaScript skill function body.

    Inserts the necessary require statements right after the opening { of the async function.

    v7.3 Phase 2: add deduplication to avoid injecting dependency declarations that already exist.

    Args:
        code: JavaScript skill code
        deps: list of dependency names that need to be injected

    Returns:
        str: code with the dependencies injected
    """
    if not deps or not code:
        return code

    # Phase 2: check for dependencies already declared in the code to avoid duplicate injection
    deps_to_inject = []
    for dep in deps:
        # Check whether the dependency is already declared in the code (including destructured forms)
        # Pattern 1: const GoalPlaceBlock = ...
        # Pattern 2: const { GoalPlaceBlock } = ...
        # Pattern 3: const { goals: { GoalPlaceBlock } } = ...
        declaration_patterns = [
            rf'(const|let|var)\s+{dep}\s*=',           # direct declaration
            rf'(const|let|var)\s*\{{[^}}]*\b{dep}\b',  # destructured declaration (single or nested level)
        ]
        already_declared = any(
            re.search(pattern, code) for pattern in declaration_patterns
        )
        if already_declared:
            # Record skipped dependencies (for debugging)
            # logger.debug(f"[inject_dependencies] Skipping already-declared dependency: {dep}")
            continue
        deps_to_inject.append(dep)

    if not deps_to_inject:
        return code  # all dependencies already declared; nothing to inject

    # Build dependency-declaration statements (only inject undeclared dependencies)
    dep_statements = '\n    '.join(STANDARD_DEPENDENCIES[d] for d in deps_to_inject if d in STANDARD_DEPENDENCIES)

    if not dep_statements:
        return code

    # Insert after the opening { of the async function
    # Match the pattern async function name(...) {
    match = re.search(r'(async\s+function\s+\w+\s*\([^)]*\)\s*\{)', code)
    if match:
        insert_pos = match.end()
        return (
            code[:insert_pos] +
            f'\n    // Auto-injected dependencies\n    {dep_statements}\n' +
            code[insert_pos:]
        )

    return code


def ensure_bot_parameter(code: str) -> str:
    """
    Ensure the function signature includes the bot parameter.

    If the bot parameter is missing from the function signature, add it automatically.

    Args:
        code: JavaScript skill code

    Returns:
        str: code with the bot parameter ensured
    """
    if not code:
        return code

    # Check whether the bot parameter is already present
    match = re.search(r'async\s+function\s+(\w+)\s*\(([^)]*)\)', code)
    if not match:
        return code

    func_name = match.group(1)
    params = match.group(2).strip()

    # If the bot parameter already exists, do not modify
    if re.search(r'\bbot\b', params):
        return code

    # Add bot as the first parameter
    if params:
        new_params = f'bot, {params}'
    else:
        new_params = 'bot'

    new_signature = f'async function {func_name}({new_params})'
    return re.sub(
        r'async\s+function\s+\w+\s*\([^)]*\)',
        new_signature,
        code,
        count=1
    )
