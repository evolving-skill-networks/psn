"""Domain-agnostic event helpers."""
from typing import Any, Optional, Tuple


def unpack_event(ev: Any) -> Tuple[Optional[str], Any]:
    """Unpack an event into (event_type, event_payload).

    Tolerates three shapes:
      - Minecraft tuple-events ``(name, payload)`` from the mineflayer
        JSPyBridge.
      - Minecraft list-events ``[name, payload]`` — what the bridge
        emits after a JSON round-trip (e.g. when events are pickled or
        recorded through a JSON encoder), since JSON has no tuples.
      - dict-events ``{"type": name, ...}``.
    Returns (None, None) for unknown shapes — callers should `continue`
    on that.
    """
    if isinstance(ev, (tuple, list)) and len(ev) >= 2:
        return ev[0], ev[1]
    if isinstance(ev, dict):
        return ev.get("type", ""), ev
    return None, None
