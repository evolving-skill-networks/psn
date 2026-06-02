"""
Semantic Analysis - Parameter semantic mismatch detection

Extracted from optimizer_impl.py for better modularity.
Contains functions for detecting parameter semantic mismatches.
"""

import re
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from skillnet.agents.skill_graph import SkillGraphManager

logger = logging.getLogger(__name__)


def detect_parameter_semantic_mismatch(
    error_text: str,
    call_context: Optional[Dict[str, Any]] = None,
    skill_graph: Optional["SkillGraphManager"] = None
) -> Optional[Dict[str, Any]]:
    """
    Enhanced semantic mismatch detection.

    Detection patterns (by priority):
    1. Regex pattern matching (fast)
    2. Context analysis (if call_context provided)
    3. LLM fallback (when pattern matching fails)

    Common issue patterns:
    - Caller expects count to mean "additional quantity to get" (delta)
    - Callee understands count as "ensure at least total quantity" (target_total)
    - Result: Callee sees "already have X >= count" and returns early,
              but caller needs more

    Args:
        error_text: Error text
        call_context: Call context (optional)
        skill_graph: SkillGraphManager reference (optional, for metadata)

    Returns:
        Dict with mismatch info or None
    """
    if not error_text:
        return None

    # ========== Pattern 1: "already have X (need Y)" format ==========
    already_have_pattern = r"[Aa]lready have (\d+).*\(need (\d+)\)"
    match = re.search(already_have_pattern, error_text)
    if match:
        have = int(match.group(1))
        need = int(match.group(2))
        if have >= need:
            print(f"\033[33m[Semantic Warning] Detected parameter semantic mismatch: "
                  f"have {have} >= need {need}, but still failed\033[0m")
            return {
                "type": "semantic_mismatch",
                "pattern": "have_need_format",
                "have": have,
                "need": need,
                "description": (
                    f"Child skill thinks 'have {have} >= need {need}' means success, "
                    f"but parent actually needs additional {need}. "
                    f"Need to fix count parameter semantics."
                ),
            }

    # ========== Pattern 2: "I already have X" + subsequent failure ==========
    already_have_then_fail = r"I already have \d+ \w+"
    fail_patterns = r"(?:not enough|Aborting|failed|insufficient)"
    if re.search(already_have_then_fail, error_text, re.IGNORECASE):
        if re.search(fail_patterns, error_text, re.IGNORECASE):
            print(f"\033[33m[Semantic Warning] Detected pattern: "
                  f"'I already have' + subsequent failure\033[0m")
            return {
                "type": "semantic_mismatch",
                "pattern": "already_have_then_fail",
                "description": (
                    "Child skill returned 'I already have X', but parent still "
                    "failed. Likely parameter semantic mismatch (target_total vs delta)"
                ),
            }

    # ========== Pattern 3: "Even after crafting/mining, not enough" ==========
    even_after_pattern = r"[Ee]ven after (?:crafting|mining|collecting).*not enough"
    if re.search(even_after_pattern, error_text, re.IGNORECASE):
        print(f"\033[33m[Semantic Warning] Detected pattern: "
              f"'Even after crafting, not enough'\033[0m")
        return {
            "type": "semantic_mismatch",
            "pattern": "even_after_fail",
            "description": (
                "Still not enough after crafting/mining. Child skill's count "
                "parameter may be understood as target_total"
            ),
        }

    # ========== Pattern 4: "Nothing to do" followed by failure ==========
    nothing_to_do_pattern = r"[Nn]othing to do"
    if re.search(nothing_to_do_pattern, error_text, re.IGNORECASE):
        if re.search(fail_patterns, error_text, re.IGNORECASE):
            print(f"\033[33m[Semantic Warning] Detected pattern: "
                  f"'Nothing to do' + subsequent failure\033[0m")
            return {
                "type": "semantic_mismatch",
                "pattern": "nothing_to_do_fail",
                "description": (
                    "Child skill returned 'Nothing to do' (thinks condition met), "
                    "but parent still failed"
                ),
            }

    # ========== Pattern 5: "did not produce required additional" ==========
    if re.search(
        r"did not produce.*required additional|did not produce.*additional needed",
        error_text,
        re.IGNORECASE
    ):
        return {
            "type": "semantic_mismatch",
            "pattern": "did_not_produce",
            "description": (
                "Child skill did not produce the additional resources "
                "expected by parent"
            ),
        }

    # ========== Context analysis (if available) ==========
    if call_context and skill_graph:
        context_result = analyze_semantic_from_context(call_context, skill_graph)
        if context_result:
            return context_result

    return None


def analyze_semantic_from_context(
    call_context: Dict[str, Any],
    skill_graph: "SkillGraphManager"
) -> Optional[Dict[str, Any]]:
    """
    Analyze semantic mismatch based on call context.

    call_context format:
    {
        "caller_skill": "craftWoodenPickaxe",
        "callee_skill": "craftBirchPlanks",
        "passed_params": {"count": 1},
        "chat_log": "...",
    }

    Args:
        call_context: Call context dictionary
        skill_graph: SkillGraphManager reference

    Returns:
        Dict with mismatch info or None
    """
    if not call_context:
        return None

    callee_skill = call_context.get("callee_skill")
    passed_params = call_context.get("passed_params", {})

    if not callee_skill or not passed_params:
        return None

    # Try to get callee skill's parameter semantics
    try:
        callee_node = skill_graph.get_node(callee_skill)
        if not callee_node or not callee_node.parameters:
            return None

        for param_name, param_info in callee_node.parameters.items():
            if param_name in passed_params:
                semantic = param_info.get("semantic")
                if semantic == "target_total":
                    passed_value = passed_params[param_name]
                    print(
                        f"\033[33m[Semantic Warning] {callee_skill}'s "
                        f"{param_name} uses target_total semantics, "
                        f"passed value is {passed_value}\033[0m"
                    )
                    return {
                        "type": "semantic_mismatch",
                        "pattern": "context_analysis",
                        "callee_skill": callee_skill,
                        "param_name": param_name,
                        "passed_value": passed_value,
                        "semantic": semantic,
                        "description": (
                            f"{callee_skill}'s {param_name} parameter uses "
                            f"target_total semantics (ensure total). "
                            f"Passed value {passed_value} may be interpreted as "
                            f"'ensure have {passed_value}' instead of "
                            f"'get additional {passed_value}'"
                        ),
                    }
    except Exception as e:
        print(f"\033[33m[Semantic] Context analysis failed: {e}\033[0m")

    return None


def detect_semantic_backprop_candidate(
    parent_skill: str,
    child_skills: List[str],
    chat_log: str,
    parent_failed: bool
) -> Optional[Dict[str, Any]]:
    """
    Detect cases where child skill "succeeds" but parent skill fails
    (semantic mismatch candidate).

    Pattern features:
    1. Parent skill failed
    2. Chat log has child skill's "I already have" or "Nothing to do" message
    3. Parent skill subsequently reports "not enough" or "Aborting"

    Args:
        parent_skill: Parent skill name
        child_skills: List of child skill names
        chat_log: Chat log during execution
        parent_failed: Whether parent skill failed

    Returns:
        {"child_skill": str, "suspected_issue": "parameter_semantic_mismatch", ...}
        or None
    """
    if not parent_failed or not chat_log:
        return None

    # Detection pattern: child says "already have enough" + parent says "not enough"
    child_success_patterns = [
        r"I already have \d+",
        r"Nothing to do",
        r"already have enough",
        r"Already satisfied",
    ]

    parent_fail_patterns = [
        r"not enough",
        r"Aborting",
        r"failed",
        r"insufficient",
    ]

    # Check for child skill's "success" pattern
    child_success_match = None
    for pattern in child_success_patterns:
        match = re.search(pattern, chat_log, re.IGNORECASE)
        if match:
            child_success_match = match.group()
            break

    # Check for parent skill's "failure" pattern
    parent_fail_match = None
    for pattern in parent_fail_patterns:
        match = re.search(pattern, chat_log, re.IGNORECASE)
        if match:
            parent_fail_match = match.group()
            break

    if child_success_match and parent_fail_match:
        # Try to identify which child skill
        suspected_child = None
        for child in child_skills:
            # Check if child skill name appears in chat log
            if child.lower() in chat_log.lower():
                suspected_child = child
                break
            # Also try matching keywords in skill name
            for keyword in ["craft", "mine", "ensure", "planks", "logs"]:
                if keyword in child.lower() and keyword in chat_log.lower():
                    suspected_child = child
                    break
            if suspected_child:
                break

        # If no specific child found, use first as candidate
        if not suspected_child and child_skills:
            suspected_child = child_skills[0]

        if suspected_child:
            return {
                "child_skill": suspected_child,
                "suspected_issue": "parameter_semantic_mismatch",
                "evidence": (
                    f"Child skill returned '{child_success_match}' "
                    f"but parent failed with '{parent_fail_match}'"
                ),
                "recommendation": (
                    "Check if count parameter was interpreted as "
                    "target_total instead of delta"
                )
            }

    return None
