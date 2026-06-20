import os
import skillnet.utils as U


_PRIMITIVES_DIR = os.path.dirname(os.path.abspath(__file__))


def load_control_primitive_names():
    """Return the list of control primitive names (without `.js` suffix)."""
    return [
        f[:-3]
        for f in os.listdir(_PRIMITIVES_DIR)
        if f.endswith(".js") and not f.startswith(".")
    ]


_PRIMITIVE_FUNCTION_NAMES = None


def load_control_primitive_function_names():
    """Return the set of TOP-LEVEL function names defined across all control
    primitive .js files (e.g. ``depositItemIntoChest``, ``moveToChest`` defined
    inside useChest.js).

    The primitive sources are concatenated verbatim into the runtime `programs`
    blob, so every top-level ``[async] function NAME`` they declare is hoisted
    into skill scope and is genuinely callable from skill code. The static
    validators only knew the file STEMS (``useChest``), so calls to these inner
    functions were wrongly flagged as undefined, forcing skills to reinvent the
    primitive instead of reusing it. Only column-0 declarations are returned
    (nested helpers are indented and not in skill scope). Cached per process.
    """
    global _PRIMITIVE_FUNCTION_NAMES
    if _PRIMITIVE_FUNCTION_NAMES is None:
        import re
        pat = re.compile(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", re.M)
        names = set()
        for fname in os.listdir(_PRIMITIVES_DIR):
            if not fname.endswith(".js") or fname.startswith("."):
                continue
            try:
                src = U.load_text(f"{_PRIMITIVES_DIR}/{fname}")
            except Exception:
                continue
            names.update(pat.findall(src))
        _PRIMITIVE_FUNCTION_NAMES = names
    return set(_PRIMITIVE_FUNCTION_NAMES)


def load_control_primitives(primitive_names=None):
    """Load control primitive JavaScript source code.

    Args:
        primitive_names: Specific primitives to load (auto-discover if None).

    Returns:
        List[str]: JavaScript source code strings.
    """
    if primitive_names is None:
        primitive_names = load_control_primitive_names()
    return [
        U.load_text(f"{_PRIMITIVES_DIR}/{name}.js")
        for name in primitive_names
    ]
