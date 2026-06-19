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


def _blank_noncode(code: str) -> str:
    """Return a same-length copy of `code` with the *contents* of string
    literals and comments replaced by spaces, so brace/identifier scanning is
    not fooled by braces or words inside strings/comments. Code inside template
    `${ ... }` interpolations is KEPT (it is real code that may reference free
    variables); the surrounding template text is blanked.
    """
    out = list(code)
    n = len(code)
    i = 0
    state = None  # None | "'" | '"' | '`' | '//' | '/*'
    while i < n:
        c = code[i]
        if state is None:
            if c in ("'", '"', '`'):
                state = c
            elif c == '/' and i + 1 < n and code[i + 1] == '/':
                state = '//'; out[i] = ' '
            elif c == '/' and i + 1 < n and code[i + 1] == '*':
                state = '/*'; out[i] = ' '
            i += 1
        elif state in ("'", '"'):
            if c == '\\':
                out[i] = ' '
                if i + 1 < n:
                    out[i + 1] = ' '
                i += 2
                continue
            if c == state:
                state = None
            else:
                out[i] = ' '
            i += 1
        elif state == '`':
            if c == '\\':
                out[i] = ' '
                if i + 1 < n:
                    out[i + 1] = ' '
                i += 2
                continue
            if c == '`':
                state = None
                i += 1
                continue
            if c == '$' and i + 1 < n and code[i + 1] == '{':
                # keep the interpolated expression as code; skip to matching }
                depth = 1
                j = i + 2
                while j < n and depth > 0:
                    if code[j] == '{':
                        depth += 1
                    elif code[j] == '}':
                        depth -= 1
                    j += 1
                i = j
                continue
            out[i] = ' '
            i += 1
        elif state == '//':
            if c == '\n':
                state = None
            else:
                out[i] = ' '
            i += 1
        elif state == '/*':
            if c == '*' and i + 1 < n and code[i + 1] == '/':
                out[i] = ' '; out[i + 1] = ' '
                state = None
                i += 2
                continue
            if c != '\n':
                out[i] = ' '
            i += 1
    return ''.join(out)


def _balanced_block_end(blanked: str, open_brace_idx: int) -> int:
    """Given the index of an opening '{', return the index just AFTER its
    matching '}' (brace-balanced over the blanked source). -1 if unbalanced."""
    depth = 0
    i = open_brace_idx
    n = len(blanked)
    while i < n:
        ch = blanked[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


# Reserved words / literals that must never be treated as free variables.
_JS_RESERVED = {
    'true', 'false', 'null', 'undefined', 'this', 'super', 'arguments',
    'let', 'const', 'var', 'else', 'do', 'of', 'in', 'case', 'default',
    'break', 'continue', 'yield', 'try', 'finally', 'class', 'extends',
    'export', 'import', 'from', 'as', 'static', 'get', 'set',
}


def _split_param_names(param_str: str) -> set:
    """Extract bound names from a JS parameter list, handling defaults and
    simple object/array destructuring: `(a, b = 1, {x, y}, [z])` -> {a,b,x,y,z}."""
    names = set()
    for tok in re.findall(r'[A-Za-z_$][\w$]*', param_str or ''):
        names.add(tok)
    # drop obvious default-value identifiers? keep conservative: destructured /
    # default RHS identifiers being marked "bound" only ever REMOVES a name from
    # free vars, which is safe (won't cause a false closure-capture skip here).
    return names


def _find_function_defs(code: str):
    """Locate every function/arrow definition in `code` with span + metadata.

    Returns a list of dicts: {name, kind, is_async, params, def_start,
    body_start, body_end} where kind is 'decl' | 'funcexpr' | 'arrow_block' |
    'arrow_expr'. body_start..body_end is the source slice of the body
    (including braces for blocks; the bare expression for arrow_expr).
    """
    blanked = _blank_noncode(code)
    defs = []
    seen_spans = set()

    def add(name, kind, is_async, params, def_start, body_start, body_end):
        key = (def_start, body_end)
        if body_end <= body_start or key in seen_spans:
            return
        seen_spans.add(key)
        defs.append({
            "name": name, "kind": kind, "is_async": is_async, "params": params,
            "def_start": def_start, "body_start": body_start, "body_end": body_end,
        })

    # 1) function declarations:  [async] function NAME(params) { ... }
    for m in re.finditer(r'(\basync\s+)?\bfunction\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{', blanked):
        body_start = blanked.index('{', m.end() - 1)
        end = _balanced_block_end(blanked, body_start)
        if end != -1:
            add(m.group(2), 'decl', bool(m.group(1)), m.group(3), m.start(), body_start, end)

    # 2) named function expressions:  const|let|var NAME = [async] function(params) { ... }
    for m in re.finditer(r'\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(async\s+)?function\s*\(([^)]*)\)\s*\{', blanked):
        body_start = blanked.index('{', m.end() - 1)
        end = _balanced_block_end(blanked, body_start)
        if end != -1:
            add(m.group(1), 'funcexpr', bool(m.group(2)), m.group(3), m.start(), body_start, end)

    # 3) arrow functions:  const|let|var NAME = [async] (params) => ...   or   x => ...
    for m in re.finditer(
        r'\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(async\s+)?'
        r'(?:\(([^)]*)\)|([A-Za-z_$][\w$]*))\s*=>\s*', blanked):
        name = m.group(1)
        is_async = bool(m.group(2))
        params = m.group(3) if m.group(3) is not None else (m.group(4) or '')
        after = m.end()
        if after < len(blanked) and blanked[after] == '{':
            end = _balanced_block_end(blanked, after)
            if end != -1:
                add(name, 'arrow_block', is_async, params, m.start(), after, end)
        else:
            # expression body: scan to a top-level ';' (or end), tracking nesting
            depth = 0
            j = after
            end = -1
            while j < len(blanked):
                ch = blanked[j]
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    if depth == 0:
                        end = j
                        break
                    depth -= 1
                elif ch == ';' and depth == 0:
                    end = j
                    break
                j += 1
            if end == -1:
                end = len(blanked)
            add(name, 'arrow_expr', is_async, params, m.start(), after, end)

    return defs


def _bound_names_in(blanked_region: str) -> set:
    """All identifiers BOUND inside a code region: params of any nested
    function/arrow, and var/let/const declarations (incl. simple destructuring),
    plus nested function declaration names."""
    bound = set()
    for m in re.finditer(r'\bfunction\s*([A-Za-z_$][\w$]*)?\s*\(([^)]*)\)', blanked_region):
        if m.group(1):
            bound.add(m.group(1))
        bound |= _split_param_names(m.group(2))
    for m in re.finditer(r'\(([^)]*)\)\s*=>', blanked_region):
        bound |= _split_param_names(m.group(1))
    for m in re.finditer(r'(?<![.\w$])([A-Za-z_$][\w$]*)\s*=>', blanked_region):
        bound.add(m.group(1))
    for m in re.finditer(r'\b(?:var|let|const)\s+(\{[^}]*\}|\[[^\]]*\]|[A-Za-z_$][\w$]*)', blanked_region):
        bound |= _split_param_names(m.group(1))
    return bound


def _referenced_idents(blanked_region: str) -> set:
    """Identifiers referenced as values (not property accesses) in a region."""
    return set(re.findall(r'(?<![.\w$])([A-Za-z_$][\w$]*)', blanked_region))


def _enclosing_scope_bindings(blanked: str, defs: list, ancestor: dict) -> set:
    """Identifiers bound DIRECTLY in ``ancestor``'s own function scope: its
    params, the var/let/const it declares (at any block depth but NOT inside a
    nested function), and the names of functions it directly declares. The
    params/locals of NESTED functions (e.g. a sibling helper's parameters) are
    deliberately EXCLUDED -- they are not in scope for another nested helper, so
    counting them would wrongly flag a free variable as a closure capture when an
    undefined-variable typo happens to share a name with an unrelated sibling
    param."""
    binds = set(_split_param_names(ancestor["params"]))
    a_s, a_e = ancestor["body_start"], ancestor["body_end"]
    region = list(blanked[a_s:a_e])
    for f in defs:
        if f is ancestor:
            continue
        if not (a_s < f["def_start"] and f["body_end"] <= a_e):
            continue
        # immediate parent of f = the deepest def that contains it
        parents = [g for g in defs if g is not f
                   and g["body_start"] < f["def_start"] and f["body_end"] <= g["body_end"]]
        immediate = max(parents, key=lambda g: g["body_start"]) if parents else None
        if immediate is ancestor:
            binds.add(f["name"])  # a function declared directly in ancestor's scope
            # blank f's whole span so its params/locals do not leak into ancestor scope
            for i in range(f["def_start"] - a_s, f["body_end"] - a_s):
                region[i] = ' '
    region_str = ''.join(region)
    for m in re.finditer(r'\b(?:var|let|const)\s+(\{[^}]*\}|\[[^\]]*\]|[A-Za-z_$][\w$]*)', region_str):
        binds |= _split_param_names(m.group(1))
    return binds


def extract_self_contained_helpers(code: str) -> List[Dict[str, Any]]:
    """Detect helper functions that the top-level-async path
    (`extract_all_function_definitions`) MISSES -- nested (any), top-level sync
    declarations, named function expressions, and arrow functions -- and
    classify each as self-contained (hoistable to a standalone async skill node)
    or not.

    A helper is NOT extractable iff it references a free variable that is bound
    in an ENCLOSING FUNCTION scope (a real closure capture): hoisting it would
    produce a broken `X is not defined` node. A free variable bound in NO
    enclosing scope (an undefined-variable typo) is still extractable: extraction
    preserves the bug for the optimizer to find and fix on the now-isolated skill
    instead of mis-blaming the caller. `bot`/`mcData`/`Vec3` are re-injectable and
    never count as blocking captures.

    Returns a list of dicts:
      name, form, raw_code, standalone_code (None if not extractable),
      params, takes_bot, needs_bot_injection, extractable, skip_reason,
      free_vars, closure_captures
    """
    if not code or not isinstance(code, str):
        return []

    blanked = _blank_noncode(code)
    defs = _find_function_defs(code)

    results = []
    for d in defs:
        # ancestors: function defs whose body strictly contains this def
        ancestors = [
            a for a in defs
            if a is not d and a["body_start"] < d["def_start"] and d["body_end"] <= a["body_end"]
        ]
        is_top_level = len(ancestors) == 0

        # SKIP forms already handled by extract_all_function_definitions:
        # top-level `async function NAME` declarations.
        if is_top_level and d["kind"] == "decl" and d["is_async"]:
            continue
        # Never treat a non-async TOP-LEVEL sibling that is actually the "main"
        # is impossible to know here; we surface all candidates and let the
        # caller decide. (A top-level sync decl that is the task wrapper is async
        # in practice, so it is excluded above.)

        body_blanked = blanked[d["body_start"]:d["body_end"]]
        body_raw = code[d["def_start"]:d["body_end"]]

        own_params = _split_param_names(d["params"])
        bound_in = own_params | _bound_names_in(body_blanked) | {d["name"]}
        refs = _referenced_idents(body_blanked)
        free = {r for r in refs if r not in bound_in and r not in JS_BUILTINS and r not in _JS_RESERVED}

        # enclosing-function scope bindings (the only thing that blocks hoisting):
        # each ancestor's OWN params + direct locals + direct child-function names,
        # NOT the params/locals of sibling nested functions.
        ancestor_locals = set()
        for a in ancestors:
            ancestor_locals |= _enclosing_scope_bindings(blanked, defs, a)

        reinjectable = {'bot', 'mcData', 'Vec3'}
        closure_captures = sorted((free & ancestor_locals) - reinjectable)
        takes_bot = bool(own_params) and list_first_is_bot(d["params"])
        # `bot` is a JS_BUILTINS entry so it is filtered out of `free`; check the
        # raw refs instead so a helper that uses bot from closure (no bot param)
        # is flagged for bot-injection when hoisted to a standalone skill.
        needs_bot_injection = ('bot' in refs) and not takes_bot
        extractable = len(closure_captures) == 0

        form = {
            'decl': 'nested' if not is_top_level else 'toplevel_sync',
            'funcexpr': 'funcexpr',
            'arrow_block': 'arrow',
            'arrow_expr': 'arrow',
        }[d["kind"]]
        if not is_top_level:
            form = 'nested'

        standalone_code = None
        if extractable:
            params = d["params"].strip()
            if needs_bot_injection:
                params = 'bot' if not params else 'bot, ' + params
            if d["kind"] == 'arrow_expr':
                expr = code[d["body_start"]:d["body_end"]].strip()
                standalone_code = f"async function {d['name']}({params}) {{\n  return {expr};\n}}"
            else:
                block = code[d["body_start"]:d["body_end"]]
                standalone_code = f"async function {d['name']}({params}) {block}"

        results.append({
            "name": d["name"],
            "form": form,
            "raw_code": body_raw,
            "standalone_code": standalone_code,
            "params": d["params"].strip(),
            "takes_bot": takes_bot,
            "needs_bot_injection": needs_bot_injection,
            "extractable": extractable,
            "skip_reason": (None if extractable
                            else f"captures enclosing var(s): {closure_captures}"),
            "free_vars": sorted(free),
            "closure_captures": closure_captures,
            "is_top_level": is_top_level,
        })

    return results


def remove_inline_function_def(code: str, name: str):
    """Strip an inline ``[async] function name(...) { ... }`` declaration (and its
    brace-balanced body) from ``code``. Returns ``(new_code, removed)``.
    ``removed`` is False (and code unchanged) if the declaration is not found or
    its braces are unbalanced (never risk a corrupt edit)."""
    blanked = _blank_noncode(code)
    pat = re.compile(rf'(?:\basync\s+)?\bfunction\s+{re.escape(name)}\s*\([^)]*\)\s*\{{')
    m = pat.search(blanked)
    if not m:
        return code, False
    start = m.start()
    body_open = blanked.index('{', m.end() - 1)
    end = _balanced_block_end(blanked, body_open)
    if end == -1:
        return code, False
    updated = code[:start] + code[end:]
    updated = re.sub(r'\n{3,}', '\n\n', updated)
    return updated, True


def rewrite_calls_await(code: str, name: str, inject_bot: bool):
    """Rewrite bare ``name(args)`` call sites to ``await name(bot, args)`` (when
    ``inject_bot``) or ``await name(args)`` (when the helper already takes bot).
    Property calls (``.name(``) and already-``await``ed calls are left untouched.
    Returns ``(new_code, ok)``; ok=False on a malformed/unbalanced call so the
    caller rejects the whole migration rather than risk a bad edit."""
    blanked = _blank_noncode(code)
    pat = re.compile(r'(?<![.\w$])' + re.escape(name) + r'\s*\(')
    out = []
    last = 0
    for m in pat.finditer(blanked):
        open_idx = m.end() - 1
        depth = 0
        j = open_idx
        while j < len(blanked):
            ch = blanked[j]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return code, False  # unbalanced -> reject migration
        close_idx = j
        already_awaited = code[:m.start()].rstrip().endswith('await')
        out.append(code[last:m.start()])
        if already_awaited:
            out.append(code[m.start():close_idx + 1])
        else:
            args = code[open_idx + 1:close_idx].strip()
            if inject_bot:
                new_args = 'bot' if args == '' else 'bot, ' + args
            else:
                new_args = args
            out.append(f"await {name}({new_args})")
        last = close_idx + 1
    out.append(code[last:])
    return ''.join(out), True


def migrate_parent_extract_helper(parent_code: str, name: str, inject_bot: bool):
    """Make ``parent_code`` safe after extracting helper ``name`` to a standalone
    skill: remove its inline definition AND rewrite its call sites to the
    async/bot contract. Returns ``(new_code, ok)``. ``ok=False`` leaves the parent
    unchanged (caller must skip the extraction). Does NOT validate JS syntax --
    the caller verifies with the real parser (all-or-nothing)."""
    without_def, removed = remove_inline_function_def(parent_code, name)
    if not removed:
        return parent_code, False
    migrated, ok = rewrite_calls_await(without_def, name, inject_bot)
    if not ok:
        return parent_code, False
    # completeness: no old-contract (non-awaited) bare call may remain
    blanked = _blank_noncode(migrated)
    for m in re.finditer(r'(?<![.\w$])' + re.escape(name) + r'\s*\(', blanked):
        if not migrated[:m.start()].rstrip().endswith('await'):
            return parent_code, False
    return migrated, True


def list_first_is_bot(param_str: str) -> bool:
    """True iff the first declared parameter is literally `bot`."""
    parts = [p.strip() for p in (param_str or '').split(',') if p.strip()]
    if not parts:
        return False
    first = re.match(r'[A-Za-z_$][\w$]*', parts[0])
    return bool(first) and first.group(0) == 'bot'


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
