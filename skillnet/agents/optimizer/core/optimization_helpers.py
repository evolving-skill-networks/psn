"""
Optimization Helpers - pure-function utilities

Stateless helper functions extracted from quick_optimize_skill.

v3.4 P2 Phase A
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..feedback.types import SkillFeedback


def build_current_state_info(
    current_task: Optional[str] = None,
    current_context: Optional[str] = None,
    current_state: Optional[Dict[str, Any]] = None,
    current_error: Optional[str] = None,
    current_critique: Optional[str] = None,
    environmental_feedback: Optional[Any] = None,  # EnvironmentalFeedback support
    quality_metrics: Optional[Dict[str, Any]] = None,  # Critic quality metrics
    domain_knowledge: Optional[Any] = None,
) -> str:
    """
    Build the current-state info string.

    A pure function extracted from quick_optimize_skill.

    Args:
        current_task: current task
        current_context: current context
        current_state: current environment state (inventory, position, biome, equipment)
        current_error: current execution error
        current_critique: current critique
        environmental_feedback: v7.5 environmental feedback (from precondition_checker)
        quality_metrics: v7.5.4 Critic quality metrics (robustness_score, dependency_safety, environment_awareness)

    Returns:
        str: the formatted state-info string
    """
    info = ""

    if current_task:
        info += f"Current Task: {current_task}\n\n"

    if current_context:
        info += f"Current Context: {current_context}\n\n"

    if current_state:
        if current_state.get("inventory"):
            info += f"Current Inventory: {current_state['inventory']}\n\n"

        if current_state.get("position"):
            pos = current_state['position']
            info += f"Current Position: x={pos.get('x', 0):.1f}, y={pos.get('y', 0):.1f}, z={pos.get('z', 0):.1f}\n\n"

        env_context = ""
        if domain_knowledge and hasattr(domain_knowledge, 'get_environment_context'):
            env_context = domain_knowledge.get_environment_context(current_state)
        if env_context:
            info += env_context.lstrip("\n") + "\n\n"
        elif current_state.get("biome"):
            info += f"Current Biome: {current_state['biome']}\n\n"

        if current_state.get("equipment"):
            info += f"Current Equipment: {current_state['equipment']}\n\n"

    if current_error:
        info += f"Current Execution Error: {current_error}\n\n"

    if current_critique:
        info += f"Current Critique: {current_critique}\n\n"

    # add environmental-feedback context (guides the LLM to add fallback logic)
    if environmental_feedback:
        # Supports either an EnvironmentalFeedback object or a dict
        if hasattr(environmental_feedback, 'to_prompt_context'):
            info += environmental_feedback.to_prompt_context() + "\n\n"
        elif isinstance(environmental_feedback, dict):
            info += "## Environmental Context (ENVIRONMENT_MISMATCH)\n"
            if environmental_feedback.get("current_biome"):
                info += f"- Current biome: {environmental_feedback['current_biome']}\n"
            if environmental_feedback.get("missing_resources"):
                info += f"- Missing resources in this biome: {', '.join(environmental_feedback['missing_resources'])}\n"
            info += "\n"

    # add Critic quality metrics
    if quality_metrics:
        info += "## Critic Quality Assessment:\n"
        info += f"- robustness_score: {quality_metrics.get('robustness_score', 'N/A')}/100\n"
        info += f"- dependency_safety: {quality_metrics.get('dependency_safety', 'N/A')}\n"
        info += f"- environment_awareness: {quality_metrics.get('environment_awareness', 'N/A')}\n\n"

    return info


def extract_issues_from_feedbacks(
    feedbacks: List["SkillFeedback"],
    extract_code_fn: Optional[callable] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract issues and suggested fixes from feedbacks.

    A function extracted from quick_optimize_skill. Supports the caller_error and llm_analysis types.

    Args:
        feedbacks: list of SkillFeedback objects
        extract_code_fn: function that extracts code examples from feedback (optional)
        logger: logger (optional)

    Returns:
        Tuple[List[Dict], List[Dict]]: (issues, suggested_fixes)
    """
    # Lazy import to avoid circular dependency
    if extract_code_fn is None:
        from ..feedback.parser import extract_code_example_from_feedback
        extract_code_fn = extract_code_example_from_feedback

    issues = []
    suggested_fixes = []

    for fb in feedbacks:
        # ========== Handle the caller_error type ==========
        if fb.feedback_type == "caller_error" or "CALLER FIX NEEDED" in (fb.content or ""):
            issues.append({
                "type": "caller_error",
                "description": fb.content[:500] if fb.content else "Unknown caller error",
                "severity": "critical",
                "source": fb.source or "backpropagation"
            })

            # Try to extract a code example (if any)
            code_example = extract_code_fn(fb.content) if fb.content else None
            if code_example:
                suggested_fixes.append({
                    "edit_type": "fix_parameter_passing",
                    "code_example": code_example,
                    "source": "extracted_from_caller_error"
                })
            # Full feedback content is already in issues[].description;
            # no prescriptive text hint needed.

            if logger:
                logger.info(
                    f"\033[36m[Quick Optimize] Extracted caller_error issue: "
                    f"{fb.content[:100] if fb.content else 'N/A'}...\033[0m"
                )

        # ========== Handle the llm_analysis type ==========
        elif fb.feedback_type == "llm_analysis" and hasattr(fb, 'content') and fb.content:
            # Extract a "Root cause: X" preamble if the LLM emitted one
            if "Root cause:" in fb.content:
                match = re.search(r'Root cause:\s*(\w+)', fb.content)
                if match:
                    issues.append({
                        "type": "llm_analysis",
                        "root_cause": match.group(1),
                        "description": fb.content[:300],
                        "severity": "high",
                        "source": "llm_analyzer"
                    })

            # Extract the code example
            code_example = extract_code_fn(fb.content)
            if code_example:
                suggested_fixes.append({
                    "edit_type": "llm_suggested",
                    "code_example": code_example,
                    "source": "llm_analysis"
                })

            if logger:
                logger.info(f"\033[36m[Quick Optimize] Extracted llm_analysis issue\033[0m")

    return issues, suggested_fixes


def check_requirements_addressed(
    requirements_addressed: List[Dict[str, Any]],
    feedback_requirements: List[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """
    Check whether the requirements have been addressed.

    A function extracted from quick_optimize_skill.

    Args:
        requirements_addressed: list of requirements the LLM reported as addressed
        feedback_requirements: original list of requirements
        logger: logger (optional)

    Returns:
        Tuple[bool, List[str], List[Dict]]:
            - should_reject: whether the optimization should be rejected (more than half of requirements unaddressed)
            - missing_requirements: list of unaddressed requirements
            - addressed_list: details of addressed requirements
    """
    if not feedback_requirements or len(feedback_requirements) == 0:
        return False, [], requirements_addressed

    # Collect the indices of addressed requirements
    addressed_indices = set()
    for addr in requirements_addressed:
        if isinstance(addr, dict) and "requirement_index" in addr:
            addressed_indices.add(addr["requirement_index"])

    # Find the unaddressed requirements
    missing_requirements = []
    for i, req in enumerate(feedback_requirements, 1):
        if i not in addressed_indices:
            missing_requirements.append(f"Requirement {i}: {req}")

    # Decide whether to reject
    should_reject = len(missing_requirements) > len(feedback_requirements) / 2

    # Logging
    if logger:
        if missing_requirements:
            logger.warning(f"\033[33m[Quick Optimize] Warning: the following requirements were not addressed:\033[0m")
            for mr in missing_requirements:
                logger.warning(f"\033[33m  - {mr}\033[0m")

            if should_reject:
                logger.error(f"\033[31m[Quick Optimize] Rejecting optimization: more than half of the requirements were not addressed\033[0m")
        else:
            logger.info(
                f"\033[32m[Quick Optimize] ✓ All {len(feedback_requirements)} requirements were addressed\033[0m"
            )
            for addr in requirements_addressed:
                if isinstance(addr, dict):
                    logger.info(
                        f"\033[32m  - Requirement {addr.get('requirement_index', '?')}: "
                        f"{addr.get('how_addressed', 'N/A')[:100]}...\033[0m"
                    )

    return should_reject, missing_requirements, requirements_addressed
