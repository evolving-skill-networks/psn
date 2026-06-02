"""
Feedback Persistence - Feedback history save/load operations

Extracted from optimizer_impl.py for better modularity.
Contains pure I/O functions for feedback persistence.
"""

import os
import logging
from typing import List, Optional, TYPE_CHECKING

import skillnet.utils as U
from skillnet.agents.optimizer.feedback import SkillFeedback

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer

logger = logging.getLogger(__name__)


def save_feedback_history(
    feedback_history: List[SkillFeedback],
    ckpt_dir: str,
    max_entries: int = 1000
) -> None:
    """
    Save feedback history to file.

    Args:
        feedback_history: List of SkillFeedback objects
        ckpt_dir: Checkpoint directory path
        max_entries: Maximum number of entries to save (default 1000)
    """
    feedback_file = f"{ckpt_dir}/skill_graph/optimizer/feedback_history.json"
    U.f_mkdir(os.path.dirname(feedback_file))

    # Convert to serializable format
    feedbacks_data = []
    for feedback in feedback_history[-max_entries:]:  # Only save recent entries
        feedbacks_data.append({
            "skill_name": feedback.skill_name,
            "feedback_type": feedback.feedback_type,
            "content": feedback.content,
            "source": feedback.source,
            "timestamp": feedback.timestamp,
            "execution_id": feedback.execution_id,
            "trajectory_segment_id": feedback.trajectory_segment_id,
            "trajectory_file": feedback.trajectory_file,
            "task": feedback.task,
            "severity": feedback.severity,
            "related_skills": feedback.related_skills,
            "priority": getattr(feedback, 'priority', 'normal'),
            "resolved": getattr(feedback, 'resolved', False),
        })

    U.dump_json(feedbacks_data, feedback_file)


def load_feedback_history(ckpt_dir: str) -> List[SkillFeedback]:
    """
    Load feedback history from file.

    Args:
        ckpt_dir: Checkpoint directory path

    Returns:
        List of SkillFeedback objects
    """
    feedback_file = f"{ckpt_dir}/skill_graph/optimizer/feedback_history.json"

    if os.path.exists(feedback_file):
        feedbacks_data = U.load_json(feedback_file)
        feedback_history = []
        for data in feedbacks_data:
            # Compatible with old format (may lack new fields)
            feedback = SkillFeedback(
                skill_name=data.get("skill_name", ""),
                feedback_type=data.get("feedback_type", ""),
                content=data.get("content", ""),
                source=data.get("source", ""),
                timestamp=data.get("timestamp"),
                execution_id=data.get("execution_id"),
                trajectory_segment_id=data.get("trajectory_segment_id"),
                trajectory_file=data.get("trajectory_file"),
                task=data.get("task"),
                severity=data.get("severity", "medium"),
                related_skills=data.get("related_skills", []),
                priority=data.get("priority", "normal"),
                resolved=data.get("resolved", False),
            )
            feedback_history.append(feedback)
        return feedback_history
    else:
        return []


def mark_feedbacks_resolved(
    feedback_history: List[SkillFeedback],
    skill_name: str,
    source: Optional[str] = None,
    log_callback: Optional[callable] = None,
    save_callback: Optional[callable] = None
) -> int:
    """
    Mark internal analysis feedbacks for a skill as resolved.

    Called after optimization is successfully applied, to avoid
    the same issue triggering optimization repeatedly.

    Args:
        feedback_history: List of SkillFeedback objects to update
        skill_name: Skill name
        source: Optional, only mark feedbacks from specific source
                (e.g., 'backpropagation_analysis')
        log_callback: Optional callback for logging (e.g., logger.info)
        save_callback: Optional callback to save after marking

    Returns:
        Number of feedbacks marked as resolved
    """
    marked_count = 0
    for fb in feedback_history:
        if fb.skill_name != skill_name:
            continue
        if source and fb.source != source:
            continue
        # Only mark internal analysis related feedbacks
        if fb.source in ('backpropagation_analysis', 'semantic_detection'):
            if not getattr(fb, 'resolved', False):
                fb.resolved = True
                marked_count += 1

    if marked_count > 0:
        if log_callback:
            log_callback(
                f"\033[32m[Feedback Lifecycle] Marked {marked_count} "
                f"feedback(s) for '{skill_name}' as resolved\033[0m"
            )
        if save_callback:
            save_callback()

    return marked_count
