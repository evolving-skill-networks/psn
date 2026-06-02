"""
Event format utilities — centralize event validation logic.

Events may arrive in two shapes:
  - Minecraft / Mineflayer: List[Tuple[str, dict]] — `(event_type, payload)`
  - dict-event domains: List[Dict] — `{"type": str, ...}`

These helpers provide safe access patterns used throughout PSNAgent mixins.
They delegate to ``skillnet.utils.event_utils.unpack_event`` so both shapes
are tolerated uniformly; without that, dict-events get silently dropped on
the floor (or crash a raw tuple-unpack loop, as
``ResourceTracker.update_inventory`` did on a dict-event domain live run).

Complements _ensure_events_list() which handles JSON→list normalization
at the boundary; these helpers operate on already-normalized event lists.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from skillnet.utils.event_utils import unpack_event as _canonical_unpack_event

# Type aliases for event data
Event = Tuple[str, Dict[str, Any]]
Events = List[Event]


def unpack_event(e: Any) -> Optional[Tuple[str, Any]]:
    """Safely unpack event into (type, data). Returns None if malformed.

    Delegates to the canonical helper in ``skillnet.utils.event_utils`` so
    both Minecraft tuple-events and dict-events are recognized.
    The local ``Optional[Tuple]`` return contract is preserved for
    backward compatibility with the many ``if unpacked is None: continue``
    call sites in ``_psn_impl``.
    """
    event_type, event_data = _canonical_unpack_event(e)
    if event_type is None:
        return None
    return event_type, event_data


def get_event_data_dict(e: Any) -> Optional[dict]:
    """Extract event data as dict, or None if event is malformed or data is not dict."""
    unpacked = unpack_event(e)
    if unpacked is None:
        return None
    _, data = unpacked
    return data if isinstance(data, dict) else None


def iter_events(events: Any) -> Iterator[Tuple[str, Any]]:
    """Iterate over valid events, silently skipping malformed entries.

    Yields:
        (event_type, event_data) tuples for each valid event.
    """
    if not isinstance(events, list):
        return
    for e in events:
        result = unpack_event(e)
        if result is not None:
            yield result


def find_last_observe(events: Any) -> Optional[dict]:
    """Find the last 'observe' event's data dict.

    Searches backwards through events for the most recent observe event
    with dict-typed data.

    Returns:
        The observe event's data dict, or None if no valid observe found.
    """
    if not events:
        return None
    items = events if isinstance(events, list) else []
    for e in reversed(items):
        unpacked = unpack_event(e)
        if unpacked and unpacked[0] == "observe" and isinstance(unpacked[1], dict):
            return unpacked[1]
    return None
