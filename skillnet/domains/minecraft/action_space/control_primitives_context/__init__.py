import os
import skillnet.utils as U


_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CTX_DIR = f"{_PKG_DIR}/control_primitives_context"
_CTX_DIR_OPTIMIZER = f"{_PKG_DIR}/control_primitives_context_optimizer"
_CTX_DIR_QWEN3 = f"{_PKG_DIR}/control_primitives_context_qwen3"


def _resolve_file(name: str, model_profile, for_optimizer: bool) -> str:
    """Resolve one primitive name to a .js file path with per-file overlay.

    Priority: optimizer override → model-profile override → default.
    Falls back per-file, so overlay dirs only need files that actually differ.
    """
    if for_optimizer:
        candidate = f"{_CTX_DIR_OPTIMIZER}/{name}.js"
        if os.path.isfile(candidate):
            return candidate
    if model_profile == "qwen3-coder":
        candidate = f"{_CTX_DIR_QWEN3}/{name}.js"
        if os.path.isfile(candidate):
            return candidate
    return f"{_CTX_DIR}/{name}.js"


def load_control_primitives_context(primitive_names=None, model_profile=None,
                                    for_optimizer=False):
    """Load control primitive context descriptions for LLM prompts.

    Args:
        primitive_names: Specific primitives to load (auto-discover if None).
        model_profile: Optional model profile name (e.g., "qwen3-coder").
            Per-file overlay: a name present in the model-specific dir wins;
            otherwise falls back to the default.
        for_optimizer: If True, prefer full-implementation versions from
            ``control_primitives_context_optimizer/`` (for Phase 1/2 analysis).

    Returns:
        List[str]: Context description strings.
    """
    if primitive_names is None:
        # Canonical name list is sourced from the default dir, not the overlays.
        primitive_names = [
            f[:-3]
            for f in os.listdir(_CTX_DIR)
            if f.endswith(".js")
        ]

    return [
        U.load_text(_resolve_file(name, model_profile, for_optimizer))
        for name in primitive_names
    ]
