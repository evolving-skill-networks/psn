"""Refactor Layer 3 — generic retry-on-validation-failure + forensics.

Observed at v3.H+ live iter 4: SiblingRefactor LLM emitted a JSON whose
`general_skill_code` field contained JS that Babel rejected with
"Unexpected token, expected ',' (70:20)". The refactor framework correctly
aborted but the sibling-merge opportunity was lost, and the raw LLM response
was not saved — making post-mortem diagnosis impossible.

This module supplies two pieces:

1. **Forensics dump** — `save_refactor_failure_forensics()` writes the raw
   LLM response, the Babel error, and the prompt that produced it to a
   stable location inside the ckpt's `refactor_failures/` directory.
   Triggered on any validation failure across all 5 refactor types so future
   occurrences can be diagnosed from disk.

2. **Single-shot retry** — `JSON_CODE_RETRY_HINT` is the augmentation
   appended to the LLM prompt on retry; mirrors the optimizer's Phase 2
   retry hint introduced in commit 2773451. Each refactor's
   `_create_*_plan_with_llm` accepts a `retry_hint` parameter and threads
   it into the LLM message stack when set.

Design decision: this is intentionally a thin module — no abstract retry
loop class, no decorator magic. Each refactor's `apply()` orchestrates its
own retry locally so the control flow stays readable. The shared bits are:
the hint text, the forensics writer, and an LLM-response-capture context
manager.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

# Standard retry hint appended to refactor LLM prompts when the first
# attempt's general_skill_code fails Babel syntax validation. Wording mirrors
# optimizer/_impl/mixins/quick_optimization.py:1036-1041 (commit 2773451).
JSON_CODE_RETRY_HINT = (
    "Your previous response produced INVALID JavaScript code. Babel rejected "
    "it with the error message below. Respond with ONLY a valid JSON object "
    "whose code field(s) contain syntactically valid JavaScript. "
    "CRITICAL: when embedding JS code inside JSON string values, escape every "
    'backslash as \\\\, every double quote as \\", and every newline as \\n. '
    "DO NOT prepend a backslash to backticks or to ${...} interpolations — "
    "JSON does not require those to be escaped. Keep the JSON compact, "
    "no markdown fences, no prose outside the object."
)


def build_retry_message(babel_error: str) -> str:
    """Compose the retry hint + the specific Babel error preview that
    gets appended to the LLM prompt on the second attempt."""
    trimmed = (babel_error or "")[:600]
    return f"{JSON_CODE_RETRY_HINT}\n\nBabel error from your previous code:\n{trimmed}"


# ---------------------------------------------------------------------------
# Forensics: persist raw LLM response + error to disk for later diagnosis
# ---------------------------------------------------------------------------


def _resolve_ckpt_dir(skill_graph_manager: Any) -> Optional[Path]:
    """Best-effort extraction of the ckpt directory from a SkillGraphManager.

    Returns None if the manager is a mock or the field is missing — the
    caller then skips the disk dump (no-op safety)."""
    if not skill_graph_manager:
        return None
    for attr in ("ckpt_dir", "_ckpt_dir", "checkpoint_dir"):
        val = getattr(skill_graph_manager, attr, None)
        if val:
            try:
                return Path(val)
            except (TypeError, ValueError):
                return None
    return None


def save_refactor_failure_forensics(
    *,
    skill_graph_manager: Any,
    refactor_type: str,
    skill_names: Sequence[str],
    attempt: int,
    raw_response: str,
    babel_error: str,
    prompt_messages: Optional[Iterable[Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Persist a single failure event to <ckpt>/refactor_failures/.

    Returns the base path (without suffix) on success, None on no-op or
    write failure. Never raises.
    """
    ckpt_dir = _resolve_ckpt_dir(skill_graph_manager)
    if ckpt_dir is None:
        return None
    try:
        forensics_dir = ckpt_dir / "refactor_failures"
        forensics_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        names_part = "_".join(list(skill_names)[:3]) or "unknown"
        base = forensics_dir / f"{ts}__{refactor_type}__{names_part}__attempt{attempt}"

        (base.with_suffix(".raw_response.txt")).write_text(
            raw_response or "<empty>", encoding="utf-8"
        )
        (base.with_suffix(".babel_error.txt")).write_text(
            babel_error or "<empty>", encoding="utf-8"
        )

        meta = {
            "timestamp": ts,
            "refactor_type": refactor_type,
            "skill_names": list(skill_names),
            "attempt": attempt,
            "raw_response_bytes": len(raw_response or ""),
            "babel_error_first_line": (babel_error or "").splitlines()[0][:200]
            if babel_error
            else "",
        }
        (base.with_suffix(".meta.json")).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        if prompt_messages is not None:
            try:
                prompt_dump = []
                for msg in prompt_messages:
                    role = type(msg).__name__
                    content = getattr(msg, "content", None) or str(msg)
                    prompt_dump.append({"role": role, "content": content})
                (base.with_suffix(".prompt.json")).write_text(
                    json.dumps(prompt_dump, indent=2), encoding="utf-8"
                )
            except Exception:
                # prompt dump is best-effort; don't fail the forensics save
                pass

        if logger is not None:
            try:
                logger.warning(
                    f"[Refactor Forensics] {refactor_type} attempt {attempt} "
                    f"failed; raw response + Babel error saved to {base}.*"
                )
            except Exception:
                pass
        return base
    except Exception as exc:
        if logger is not None:
            try:
                logger.warning(
                    f"[Refactor Forensics] failed to persist failure data: {exc!r}"
                )
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# LLM response capture — lets refactor code see the raw bytes the LLM
# emitted in the SAME call that built the plan, without changing
# `*_plan_with_llm` return signatures.
# ---------------------------------------------------------------------------


class _LLMResponseCapture:
    """Transparent wrapper around an LLM client that records each
    `invoke()` response's raw content.

    Used by refactor's apply() to grab the raw bytes the LLM emitted in
    the SAME call that built the plan, without changing the existing
    `_create_plan_with_llm` return signature.

    Stores ALL captured responses in `raw_responses` (newest at index -1).
    """

    def __init__(self, inner_llm):
        self._inner = inner_llm
        self.raw_responses: List[str] = []
        self.last_messages = None

    def invoke(self, messages, *args, **kwargs):
        self.last_messages = messages
        resp = self._inner.invoke(messages, *args, **kwargs)
        content = resp.content if hasattr(resp, "content") else str(resp)
        self.raw_responses.append(content)
        return resp

    def __getattr__(self, name):
        # Transparent delegate for any other LLM attribute
        return getattr(self._inner, name)


@contextmanager
def capture_llm_responses(refactor_instance):
    """Context manager that temporarily replaces `refactor_instance.llm`
    with a capturing wrapper. Yields the wrapper so the caller can read
    `wrapper.raw_responses` and `wrapper.last_messages` afterward.

    No-op fall-through when `refactor_instance.llm` is missing or None
    (e.g. tests that stub __init__ may leave .llm unset).

    Usage:
        with capture_llm_responses(self) as cap:
            plan = self._create_plan_with_llm(group, skill_nodes)
        raw = cap.raw_responses[-1] if cap.raw_responses else ""
        msgs = cap.last_messages
    """
    original = getattr(refactor_instance, "llm", None)
    if original is None:
        # No LLM wired — nothing to capture; yield a no-op stub
        class _Stub:
            raw_responses = []
            last_messages = None

        yield _Stub()
        return

    cap = _LLMResponseCapture(original)
    refactor_instance.llm = cap
    try:
        yield cap
    finally:
        refactor_instance.llm = original
