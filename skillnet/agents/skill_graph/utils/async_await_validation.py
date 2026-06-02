"""Static validator for concurrency-risky async patterns in skill code.

Detects unawaited async function calls inside sync callbacks. Such patterns
cause unbounded concurrent async work that crashes mineflayer's Node.js
event loop.

Primary consumer: step_execute_phase.py — runs before env.step() to block
crash-risky skill code before it reaches mineflayer.

Usage:
    from skillnet.agents.skill_graph.utils.async_await_validation import (
        detect_concurrency_risk, Violation,
    )

    violations = detect_concurrency_risk(js_code)
    if violations:
        for v in violations:
            print(f"{v.severity} @ L{v.line}: {v.message}")
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_DETECTOR_JS = Path(__file__).parent / "_async_validator.js"


@dataclass(frozen=True)
class Violation:
    """One problem flagged by the detector."""
    severity: str          # "crash_risk" or "warning"
    kind: str              # "sync_cb_fire_forget_raw" / "sync_cb_fire_forget_then_catch" / "unhandled_rejection"
    name: str              # the async function name
    line: Optional[int]
    col: Optional[int]
    message: str


def _find_node_and_babel_root() -> Optional[Path]:
    """Return directory containing node_modules/@babel/parser + traverse."""
    # Repo root (where node_modules lives) — 5 levels up from this file:
    # skillnet/agents/skill_graph/utils/async_await_validation.py → repo/
    candidate = Path(__file__).resolve().parents[4]
    parser_pkg = candidate / "node_modules" / "@babel" / "parser" / "package.json"
    traverse_pkg = candidate / "node_modules" / "@babel" / "traverse" / "package.json"
    if parser_pkg.exists() and traverse_pkg.exists():
        return candidate
    # Fallback: mineflayer env dir (has its own node_modules with babel helpers)
    # — but typically missing parser/traverse. Fail closed.
    return None


class _DetectorUnavailable(RuntimeError):
    """Raised when the JS detector can't be invoked (missing node or babel)."""


def _run_detector(code: str, timeout_s: float = 20.0) -> dict:
    """Invoke the Babel detector subprocess; return parsed JSON."""
    workdir = _find_node_and_babel_root()
    if workdir is None:
        raise _DetectorUnavailable(
            "Cannot find node_modules/@babel/parser+traverse. "
            "Run `npm install @babel/parser @babel/traverse` at repo root."
        )

    # Defensive: coerce non-string input (e.g. MagicMock in tests) to str.
    if not isinstance(code, str):
        code = str(code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(code)
        code_path = f.name
    try:
        result = subprocess.run(
            ["node", str(_DETECTOR_JS), code_path],
            capture_output=True, text=True,
            timeout=timeout_s, cwd=str(workdir),
        )
        if result.returncode != 0:
            raise _DetectorUnavailable(
                f"detector exited {result.returncode}: {result.stderr[:400]}"
            )
        return json.loads(result.stdout.strip())
    finally:
        try: os.unlink(code_path)
        except OSError: pass


def analyze_code(code: str, timeout_s: float = 20.0) -> dict:
    """Low-level: return raw detector output (parseError / asyncDecls / violations).

    On detector unavailability, returns {"parseError": "detector_unavailable:<reason>",
                                        "violations": [], ...} — caller decides policy.
    """
    try:
        return _run_detector(code, timeout_s=timeout_s)
    except _DetectorUnavailable as e:
        return {"parseError": f"detector_unavailable:{e}",
                "asyncDecls": [], "violations": [], "has_crash_risk": False}
    except subprocess.TimeoutExpired:
        return {"parseError": "detector_timeout",
                "asyncDecls": [], "violations": [], "has_crash_risk": False}
    except (json.JSONDecodeError, OSError) as e:
        return {"parseError": f"detector_runtime:{e}",
                "asyncDecls": [], "violations": [], "has_crash_risk": False}


def detect_concurrency_risk(code: str, timeout_s: float = 20.0) -> List[Violation]:
    """Return list of crash_risk violations (empty if code is safe).

    This is the main production entry point. Callers should block execution
    of code when this returns a non-empty list, and feed the violation
    messages back to the agent as error signal.

    Only "crash_risk" violations are returned here; "warning" severity
    (plain unhandled-rejection without sync-callback context) is filtered
    out because it doesn't crash the process.
    """
    result = analyze_code(code, timeout_s=timeout_s)
    crash_risks = []
    for v in result.get("violations", []):
        if v.get("severity") != "crash_risk":
            continue
        crash_risks.append(Violation(
            severity=v["severity"],
            kind=v.get("kind", ""),
            name=v.get("name", ""),
            line=v.get("line"),
            col=v.get("col"),
            message=v.get("message", ""),
        ))
    return crash_risks


def format_violations_message(violations: List[Violation]) -> str:
    """Human-readable summary for LLM feedback."""
    if not violations:
        return ""
    lines = [
        "ConcurrencyRiskError: skill code contains unawaited async calls inside "
        "synchronous callbacks. These cause unbounded concurrent async work that "
        "crashes mineflayer's event loop. The code was NOT executed."
    ]
    for v in violations:
        loc = f"line {v.line}" if v.line else "?"
        lines.append(f"  - [{loc}] {v.message}")
    lines.append(
        "Fix guidance: Do not call async functions inside the callback argument "
        "of exploreUntil / setInterval / bot.on(...) / etc. Instead, structure "
        "the code as an async outer loop that awaits each async call in sequence; "
        "use the callback only to report sync conditions (e.g. inventory counts, "
        "nearby-block checks)."
    )
    return "\n".join(lines)
