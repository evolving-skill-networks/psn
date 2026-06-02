"""
Feedback Parser

Feedback-content parsing module: provides pure-function helpers to extract and format feedback.
Extracts code examples, issue types, and similar info from feedback content.
"""

import re
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import SkillFeedback
    from skillnet.agents.skill_graph.models import SkillExecutionTrace


def extract_code_example_from_feedback(
    content: str, language: str = "javascript"
) -> Optional[str]:
    """
    Extract a code example from feedback content.

    Supports several code-block formats, including:
    - ```<language> ... ```
    - Example: ... format
    - Suggested fix: ... format

    Args:
        content: feedback content
        language: skill language (e.g., "javascript", "python"). Used to match code-block markers.

    Returns:
        Optional[str]: the extracted code example, or None if none was found
    """
    if not content:
        return None

    from skillnet.utils.code_block import get_language_aliases
    aliases = get_language_aliases(language)
    lang_alt = "|".join(re.escape(a) for a in aliases)

    patterns = [
        rf'Example[:\s]*\n?```(?:{lang_alt})?\n?(.*?)```',
        rf'Suggested fix[:\s]*\n?```(?:{lang_alt})?\n?(.*?)```',
        rf'Suggested code[:\s]*\n?```(?:{lang_alt})?\n?(.*?)```',
        rf'Fix[:\s]*\n?```(?:{lang_alt})?\n?(.*?)```',
        rf'```(?:{lang_alt})\n?(.*?)```',  # language-specific code block
        r'```\n?(.*?)```',  # generic code block
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            if len(code) > 10:  # ensure it isn't an empty code block
                return code
    return None


def extract_issue_type_from_content(content: str) -> str:
    """
    Extract the issue type from feedback content.

    For example: "[Internal Analysis] logic_error: ..." -> "logic_error"

    Args:
        content: feedback content

    Returns:
        str: issue type, or "unknown" if not found
    """
    if not content:
        return "unknown"

    match = re.search(r'\[Internal Analysis\]\s*(\w+):', content)
    return match.group(1) if match else "unknown"


def format_feedbacks_for_llm(feedbacks: List["SkillFeedback"]) -> str:
    """
    Format feedback info for LLM consumption.

    Args:
        feedbacks: list of SkillFeedback objects

    Returns:
        str: formatted feedback string
    """
    if not feedbacks:
        return "No feedback available."

    formatted = []
    for i, feedback in enumerate(feedbacks, 1):
        formatted.append(
            f"{i}. [{feedback.feedback_type.upper()}] {feedback.content}\n"
            f"   Source: {feedback.source}, Severity: {feedback.severity}"
        )

    return "\n".join(formatted)


def extract_constraint_feedback_from_critique(
    skill_name: str,
    current_critique: str,
    children: List[str],
    current_task: str = None,
    logger=None,
) -> Optional["SkillFeedback"]:
    """
    Extract physical-constraint info from current_critique and produce a SkillFeedback.

    This function checks whether the critique contains a physical-constraint violation (e.g., CONSTRAINT_VIOLATION),
    and if so converts it into a context-rich SkillFeedback for the backpropagation LLM to analyze.

    Args:
        skill_name: current skill name
        current_critique: critique content
        children: list of child skills
        current_task: current task identifier
        logger: optional logger

    Returns:
        Optional[SkillFeedback]: constraint-violation feedback, or None if not found
    """
    if not current_critique:
        return None

    # Lazy import to avoid circular dependency
    from .types import SkillFeedback

    # Check whether a constraint violation is present
    constraint_match = re.search(
        r'CONSTRAINT_VIOLATION\s*\((\w+)\):\s*([^.]+)\.\s*Current=([^,]+),\s*Required=([^.]+)',
        current_critique,
        re.IGNORECASE
    )

    if not constraint_match:
        return None

    constraint_type = constraint_match.group(1)
    description = constraint_match.group(2).strip()
    current_value = constraint_match.group(3).strip()
    required_value = constraint_match.group(4).strip()

    # Extract suggested_fix (if any)
    suggested_fix_match = re.search(
        r'Suggested fix:\s*(.+?)(?=\n|$)',
        current_critique,
        re.IGNORECASE
    )
    suggested_fix = suggested_fix_match.group(1) if suggested_fix_match else ""

    # Build a context-rich feedback content
    content = f"""PHYSICAL CONSTRAINT VIOLATION:
Type: {constraint_type}
Resource: {description.split(':')[0] if ':' in description else 'unknown'}
Current: Bot is at Y={current_value}
Required: {required_value}

Impact: The bot cannot find the target resource because it's at the wrong position/depth.
This is a Minecraft world constraint - certain resources only exist at specific Y coordinates.

Constraint facts:
- The target resource only spawns at specific Y-coordinate ranges in Minecraft
- The bot's current Y position is outside the valid spawn range for this resource
- Searching at the wrong depth will never find the resource regardless of horizontal range

Child skills that may be responsible: {children}

{f'Suggested fix: {suggested_fix}' if suggested_fix else ''}
"""

    if logger:
        logger.info(
            f"\033[33m[Constraint→Feedback] Extracted constraint violation from critique: "
            f"{constraint_type}, Current={current_value}\033[0m"
        )

    return SkillFeedback(
        skill_name=skill_name,
        feedback_type="constraint_violation",
        content=content,
        source="constraint_detection",
        severity="high",
        related_skills=children,
        priority="internal",  # not filtered by relevance
        task=current_task,
    )


# ========== migrated from optimizer_impl.py ==========

