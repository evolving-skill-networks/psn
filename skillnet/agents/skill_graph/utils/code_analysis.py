"""
Code Analysis - code-analysis utilities

Pure functions extracted from graph_manager_impl.py for code analysis and transformation.
Supports naming conversion, function extraction, serialization, etc.

v4.0 modular refactor
"""

import re
import logging
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import SkillPrecondition, SkillEffect

logger = logging.getLogger(__name__)


def camel_to_snake(name: str) -> str:
    """
    Convert CamelCase to snake_case

    Args:
        name: CamelCase name

    Returns:
        str: snake_case name
    """
    # Handle consecutive uppercase letters (e.g. IronOre -> iron_ore)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def sanitize_python_to_js(code: str) -> str:
    """
    Convert Python syntax to JavaScript syntax.

    Fixes cases where LLM or code generation may conflate Python and JavaScript syntax:
    - Python: False, True, None
    - JavaScript: false, true, null

    Args:
        code: code that may contain Python syntax

    Returns:
        str: converted JavaScript code
    """
    original_code = code

    # Convert booleans and None (uses word boundaries so variables like "isFalse" are not matched)
    code = re.sub(r'\bFalse\b', 'false', code)
    code = re.sub(r'\bTrue\b', 'true', code)
    code = re.sub(r'\bNone\b', 'null', code)

    # If any conversion occurred, log it
    if code != original_code:
        false_count = len(re.findall(r'\bFalse\b', original_code))
        true_count = len(re.findall(r'\bTrue\b', original_code))
        none_count = len(re.findall(r'\bNone\b', original_code))

        if false_count or true_count or none_count:
            logger.info("[Skill Graph] Python syntax detected and converted to JavaScript:")
            if false_count:
                logger.info(f"[Skill Graph]   - False -> false: {false_count} occurrences")
            if true_count:
                logger.info(f"[Skill Graph]   - True -> true: {true_count} occurrences")
            if none_count:
                logger.info(f"[Skill Graph]   - None -> null: {none_count} occurrences")

    return code


def extract_function_calls(code: str) -> List[str]:
    """
    Extract function calls from JavaScript code (parses await functionName() calls).

    Args:
        code: JavaScript code

    Returns:
        List[str]: list of called function names (in code order, deduplicated, preserving first-occurrence order)
    """
    # Remove strings and comments to avoid matching their contents
    # Order: strings first, then comments (so // inside strings is not mistaken for a comment)
    code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
    code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
    code = re.sub(r'`[^`\\]*(?:\\.[^`\\]*)*`', '``', code)
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    called_functions = []
    seen = set()  # for dedup while preserving order

    # Match await functionName(...) pattern
    # Also match functionName(...) (non-await)
    patterns = [
        r'await\s+(\w+)\s*\(',
        r'(?:^|[^\.\w])(\w+)\s*\(',  # function call, not a method call (e.g. bot.findBlock)
    ]

    # JavaScript built-ins and keywords
    js_builtins = {
        'if', 'for', 'while', 'switch', 'catch', 'console', 'Math',
        'Array', 'Object', 'String', 'Number', 'Boolean', 'Date',
        'Promise', 'setTimeout', 'setInterval',
        # Keywords that can be followed by (
        'return', 'throw', 'new', 'typeof', 'instanceof', 'delete', 'void', 'await',
    }

    for pattern in patterns:
        matches = re.finditer(pattern, code, re.MULTILINE)
        for match in matches:
            func_name = match.group(1)
            # Filter out common JavaScript built-ins and bot methods
            if func_name not in js_builtins:
                # Check whether it is a bot method call (e.g. bot.findBlock)
                # Skip if preceded by bot. or this.
                start_pos = match.start()
                if start_pos > 0:
                    before = code[max(0, start_pos-10):start_pos]
                    if 'bot.' in before or 'this.' in before:
                        continue
                # Order-preserving dedup: only add the first occurrence of each function name
                if func_name not in seen:
                    called_functions.append(func_name)
                    seen.add(func_name)

    return called_functions


def _is_in_comment(code: str, pos: int) -> bool:
    """
    Check whether a position is inside a comment.

    Args:
        code: JavaScript code
        pos: position

    Returns:
        bool: whether the position is inside a comment
    """
    i = 0
    in_string = None  # None, '"', "'", '`'
    in_single_comment = False
    in_multi_comment = False

    while i < pos:
        if i >= len(code):
            break

        char = code[i]

        # Handle multi-line comments
        if in_multi_comment:
            if i + 1 < len(code) and code[i:i+2] == '*/':
                in_multi_comment = False
                i += 2
                continue
            i += 1
            continue

        # Handle single-line comments
        if in_single_comment:
            if char == '\n':
                in_single_comment = False
            i += 1
            continue

        # Handle strings
        if in_string:
            if char == '\\' and i + 1 < len(code):
                i += 2
                continue
            if char == in_string:
                in_string = None
            i += 1
            continue

        # Check for comment start
        if i + 1 < len(code):
            two_chars = code[i:i+2]
            if two_chars == '//':
                in_single_comment = True
                i += 2
                continue
            if two_chars == '/*':
                in_multi_comment = True
                i += 2
                continue

        # Check for string start
        if char in '"\'`':
            in_string = char
            i += 1
            continue

        i += 1

    # Return whether we are still inside a comment
    return in_single_comment or in_multi_comment


def _calculate_brace_depth_before(code: str, pos: int) -> int:
    """
    Compute the effective brace depth before pos (ignoring braces inside strings and comments).

    Args:
        code: JavaScript code
        pos: position

    Returns:
        int: brace depth (count of { minus count of })
    """
    depth = 0
    i = 0
    in_string = None  # None, '"', "'", '`'
    in_single_comment = False
    in_multi_comment = False

    while i < pos:
        # Check whether we reached the position
        if i >= len(code):
            break

        char = code[i]

        # Handle multi-line comment end
        if in_multi_comment:
            if i + 1 < len(code) and code[i:i+2] == '*/':
                in_multi_comment = False
                i += 2
                continue
            i += 1
            continue

        # Handle single-line comment end
        if in_single_comment:
            if char == '\n':
                in_single_comment = False
            i += 1
            continue

        # Handle string end
        if in_string:
            if char == '\\' and i + 1 < len(code):
                # Skip escape character
                i += 2
                continue
            if char == in_string:
                in_string = None
            i += 1
            continue

        # Check for comment start
        if i + 1 < len(code):
            two_chars = code[i:i+2]
            if two_chars == '//':
                in_single_comment = True
                i += 2
                continue
            if two_chars == '/*':
                in_multi_comment = True
                i += 2
                continue

        # Check for string start
        if char in '"\'`':
            in_string = char
            i += 1
            continue

        # Compute brace depth
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1

        i += 1

    return depth


def extract_all_function_definitions(code: str) -> List[Dict[str, str]]:
    """
    Extract all **top-level** async function definitions from JavaScript code.

    Used for Eager Skill Registration: extract all function definitions before executing code,
    so they can be pre-registered into the skill graph for correct Skill Tracking.

    Fix (2026-01-16): use brace-depth counting to only extract functions declared at depth=0,
    avoiding incorrect extraction of helper functions nested inside others (these helpers
    typically capture outer variables via closures and cannot stand alone as a skill).

    Args:
        code: JavaScript code

    Returns:
        List[Dict[str, str]]: list of function definitions; each element contains:
            - name: function name
            - code: complete function code (from `async function` to the matching `}`)
            - params: parameter-list string

    Example:
        >>> code = '''
        ... async function ensureLogs(bot) {
        ...     async function mineNearbyLogs() { }  // nested function; not extracted
        ...     await mineBlock(bot, "oak_log", 4);
        ... }
        ... async function craftTable(bot) {
        ...     await ensureLogs(bot);
        ... }
        ... '''
        >>> defs = extract_all_function_definitions(code)
        >>> [d["name"] for d in defs]
        ['ensureLogs', 'craftTable']  # does not include mineNearbyLogs
    """
    definitions = []

    # Match async function name(params) { ... }
    # Use non-greedy matching with brace balancing to extract the full function body
    pattern = r'async\s+function\s+(\w+)\s*\(([^)]*)\)\s*\{'

    for match in re.finditer(pattern, code):
        func_name = match.group(1)
        params = match.group(2).strip()
        start_pos = match.start()

        # Check whether the match is in a comment
        if _is_in_comment(code, start_pos):
            logger.debug(f"[Code Analysis] Skipping commented-out function definition '{func_name}'")
            continue

        # Check whether at top level (effective brace depth before is 0)
        depth_before = _calculate_brace_depth_before(code, start_pos)
        if depth_before != 0:
            # Nested function; skip
            logger.debug(f"[Code Analysis] Skipping nested function '{func_name}' (depth {depth_before})")
            continue

        # Find the function body end position (brace-balanced)
        brace_count = 1
        pos = match.end()  # start after the opening {
        while pos < len(code) and brace_count > 0:
            char = code[pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            # Found the matching }
            func_code = code[start_pos:pos]
            definitions.append({
                "name": func_name,
                "code": func_code,
                "params": params,
            })
        else:
            # Braces unbalanced; warn but continue
            logger.warning(f"[Code Analysis] Function '{func_name}' has unbalanced braces; skipping")

    return definitions


def serialize_effects(effects: List["SkillEffect"]) -> List[Dict[str, Any]]:
    """
    Serialize a list of SkillEffect into a JSON-friendly form.

    Args:
        effects: list of SkillEffect

    Returns:
        List[Dict]: list of serialized dicts
    """
    result = []
    for effect in effects:
        effect_dict = {
            "description": effect.description,
            "code": effect.code if hasattr(effect, 'code') else "",
        }
        if effect.state_representation:
            effect_dict["state_representation"] = effect.state_representation
        result.append(effect_dict)
    return result


# Keyword patterns for filtering preconditions
PRECONDITION_FILTER_PATTERNS = [
    # Nearby blocks
    r'exists within \d+ blocks',
    r'within \d+ blocks',
    r'nearby',
    r'findblock',
    r'findblocks',
    # Empty inventory slot
    r'empty inventory slot',
    r'inventory slot available',
    r'inventory\.slots',
    # Other soft requirements
    r'blockat',
    r'position',
    # Optimization hints disguised as hard requirements.
    # Cold-start LLMs sometimes generate "X required to do Y efficiently" where
    # X is actually optional (e.g., "pickaxe required to mine logs efficiently" —
    # axes/pickaxes speed up log mining but are NOT required).
    r'\befficient(ly)?\b',
    r'\bfor best results?\b',
    r'\bpreferably\b',
    r'\bto avoid\b',
    r'\bto make .{1,30} faster\b',
    r'\bto ensure .{1,30} success\b',
    r'\bmore reliable\b',
    r'\brecommended\b',
    r'\boptional(ly)?\b',
    r'\bfaster\b',
    # Bogus tool-requirement patterns for items that have no tool requirement.
    # Logs/wood/planks are mineable bare-handed; extracting "pickaxe required for oak_log"
    # is a factual error about Minecraft mechanics. These patterns catch the description text
    # when pickaxe and log/wood keywords co-occur (up to 100 chars apart — some LLMs emit
    # long parenthetical descriptions between them).
    r'pickaxe.{0,100}(?:log|oak|birch|spruce|jungle|acacia|dark_oak|cherry|mangrove)',
    r'(?:log|oak|birch|spruce|jungle|acacia|dark_oak|cherry|mangrove).{0,100}pickaxe',
    r'tool capable of mining.{0,50}(?:log|wood|plank)',
    r'mining tool.{0,50}for.{0,30}(?:log|wood|plank)',
    r'tool.{0,30}(?:breaking|mining|chopping).{0,30}(?:log|wood)',
    # Non-checkable "world-dependent" assertions the LLM sometimes emits
    # when it can't decide between a hard requirement and background context.
    r'world-dependent',
    r'not checkable',
]


def filter_preconditions(
    preconditions: List["SkillPrecondition"]
) -> List["SkillPrecondition"]:
    """
    Filter out non-essential preconditions.

    Filters out:
    - Nearby blocks (e.g. "At least one log block exists within 32 blocks")
    - Empty inventory slot (e.g. "At least 1 empty inventory slot available")
    - Other soft requirements

    Keeps:
    - Required tools (e.g. "Has a stone_pickaxe or better")
    - Required materials (e.g. "At least 3 oak_planks in inventory")
    - Other hard requirements

    Args:
        preconditions: list of preconditions

    Returns:
        List[SkillPrecondition]: filtered list of preconditions
    """
    filtered = []

    for precondition in preconditions:
        # Null-safe: LLM responses occasionally set "code": null or
        # "description": null in the JSON. The SkillPrecondition
        # dataclass does not runtime-enforce str, so those None values
        # propagate here. ``hasattr`` returns True even when the value
        # is None, which used to crash on ``.lower()``.
        description = (getattr(precondition, 'description', None) or "").lower()
        code = (getattr(precondition, 'code', None) or "").lower()

        # Check whether it matches a filter pattern
        should_filter = False
        for pattern in PRECONDITION_FILTER_PATTERNS:
            if re.search(pattern, description) or re.search(pattern, code):
                should_filter = True
                break

        # If not in the filter list, keep it
        if not should_filter:
            filtered.append(precondition)
        else:
            logger.debug(f"[Precondition Filter] Filtered out non-essential precondition: {precondition.description[:80]}...")

    return filtered


def is_task_specific_skill(func_name: str, func_code: str) -> bool:
    """
    Check whether this is a task-specific skill (a composition wrapper from the graph planner).

    Only composition wrappers generated by graph planner should be task-specific.
    These have names matching the task description pattern (e.g., mine_1_birch_log).
    Original generated skills (e.g., craftOakCraftingTable, placeCraftingTable)
    should NOT be marked task-specific even if they only have a bot parameter.

    Args:
        func_name: function name
        func_code: function code

    Returns:
        bool: whether it is a task-specific skill
    """
    # Check whether the function name contains a number and specific item (composition-wrapper trait)
    # E.g.: mine_1_birch_log, craft_4_oak_planks, smelt_4_raw_iron
    task_specific_pattern = r'\d+'
    item_pattern = r'_(logs?|planks?|pickaxe|sword|axe|ores?|stone|dirt|wood|iron|gold|copper|ingots?|raw_)'

    if re.search(task_specific_pattern, func_name) and re.search(item_pattern, func_name):
        return True

    # Removed "only bot parameter → task-specific" heuristic.
    # That heuristic was too broad — it incorrectly marked original generated
    # skills like craftOakCraftingTable(bot) and placeCraftingTable(bot) as
    # task-specific, preventing them from being reused in the skill graph.

    return False


def is_general_skill(func_name: str, func_code: str) -> bool:
    """
    Check whether this is a general skill.

    Traits of a general skill:
    1. The function has parameters that support multiple types (e.g. allowedLogTypes, count, etc.)
    2. The function name does not contain a specific number and item
    3. The function body uses parameters rather than hardcoded values

    Args:
        func_name: function name
        func_code: function code

    Returns:
        bool: whether it is a general skill
    """
    # Check function name: if it contains a number and a specific item, it is likely not general
    task_specific_pattern = r'\d+'
    item_pattern = r'_(logs?|planks?|pickaxe|sword|axe|ores?|stone|dirt|wood|iron|gold|copper|ingots?|raw_)'

    if re.search(task_specific_pattern, func_name) and re.search(item_pattern, func_name):
        return False

    # Check function parameters: if it has type parameters (allowedLogTypes, count, etc.), it may be general
    func_match = re.search(r'async\s+function\s+\w+\s*\(([^)]*)\)', func_code)
    if func_match:
        params_str = func_match.group(1)
        params = [p.strip().split('=')[0].strip() for p in params_str.split(',') if p.strip() and p.strip() != 'bot']

        # Check for type-related parameters
        type_param_patterns = [
            r'allowedLogTypes?', r'allowedPlanks?', r'logTypes?', r'plankTypes?',
            r'count', r'num', r'quantity', r'amount',
            r'type', r'types', r'itemType', r'blockType',
        ]

        for param in params:
            for pattern in type_param_patterns:
                if re.search(pattern, param, re.IGNORECASE):
                    return True

    return False


# ============================================================
# Code Completeness Validation (Fix 10b)
# ============================================================

# JavaScript built-in objects and functions (need not be defined in code)
JS_BUILTINS = {
    # Global objects
    'bot', 'mcData', 'Vec3', 'require', 'console', 'Promise',
    'Object', 'Array', 'Math', 'JSON', 'Error', 'String',
    'Number', 'Boolean', 'Date', 'RegExp', 'Map', 'Set',
    # Control-flow keywords (often mis-matched as function calls)
    'if', 'for', 'while', 'switch', 'catch', 'function', 'async',
    'return', 'throw', 'new', 'typeof', 'instanceof', 'delete', 'void', 'await',
    # Timers
    'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
    # Other common ones
    'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURI', 'decodeURI',
    # mineflayer-pathfinder Goal classes (injected at runtime in the /step route scope; see index.js:441-455)
    'Goal', 'GoalBlock', 'GoalNear', 'GoalXZ', 'GoalNearXZ', 'GoalY',
    'GoalGetToBlock', 'GoalLookAtBlock', 'GoalBreakBlock',
    'GoalCompositeAny', 'GoalCompositeAll', 'GoalInvert',
    'GoalFollow', 'GoalPlaceBlock',
}


def extract_all_defined_functions(code: str) -> set:
    """
    Extract all functions defined in the code (both async function and plain function).

    Args:
        code: JavaScript code

    Returns:
        set: set of all function names defined in code
    """
    defined = set()

    # Match async function name(...) and function name(...)
    # Also match arrow functions: const name = async (...) => or const name = (...) =>
    patterns = [
        r'async\s+function\s+(\w+)\s*\(',      # async function name(
        r'function\s+(\w+)\s*\(',               # function name(
        r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',  # const name = () =>
        r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?function',  # const name = function
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, code):
            defined.add(match.group(1))

    return defined


def validate_code_completeness(
    code: str,
    graph_nodes: set,
    control_primitives: set,
) -> tuple:
    """
    Validate code completeness: detect undefined function calls.

    This is the static validation layer of Fix 10b, a backstop for Fix 10 (prompt constraints).
    Called before env.step() to surface missing-dependency issues in LLM-generated code early.

    Args:
        code: JavaScript code to validate
        graph_nodes: set of registered skill names
        control_primitives: set of control-primitive names

    Returns:
        tuple: (is_valid: bool, issues: List[str])
            - is_valid: True if code is complete, False if any function calls are undefined
            - issues: list of issue descriptions

    Example:
        >>> code = '''
        ... async function craftWoodenPickaxe(bot) {
        ...     const table = findNearbyCraftingTable();  // undefined!
        ...     await craftItem(bot, "wooden_pickaxe", 1);
        ... }
        ... '''
        >>> graph_nodes = {'craftItem', 'mineBlock'}
        >>> is_valid, issues = validate_code_completeness(code, graph_nodes, set())
        >>> is_valid
        False
        >>> 'findNearbyCraftingTable' in issues[0]
        True
    """
    issues = []

    # 1. Extract all functions defined in the code
    defined_funcs = extract_all_defined_functions(code)

    # 2. Extract all function calls in the code
    called_funcs = extract_function_calls(code)

    # 3. Check each call
    for func in called_funcs:
        # Skip known definitions
        if func in defined_funcs:
            continue
        # Skip registered skills
        if func in graph_nodes:
            continue
        # Skip control primitives
        if func in control_primitives:
            continue
        # Skip JavaScript built-ins
        if func in JS_BUILTINS:
            continue

        # Undefined function call
        issues.append(f"Undefined function call: {func} (not in local definitions, skill graph, or control primitives)")

    return len(issues) == 0, issues
