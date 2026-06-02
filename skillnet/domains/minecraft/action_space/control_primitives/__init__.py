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
