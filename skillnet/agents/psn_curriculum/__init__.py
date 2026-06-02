"""
PSN Curriculum - Planning-aware, Symbolic, Neurally-guided Curriculum Agent

This module provides an enhanced curriculum agent that combines:
- Planning-aware: Long-term goal planning with milestone tracking
- Symbolic: Minecraft recipe/dependency knowledge for feasibility checking
- Neurally-guided: LLM-based final decision making
- Adaptive Learning: Task decomposition and skill gap analysis

Usage:
    from skillnet.agents.psn_curriculum import PSNCurriculumAgent

    agent = PSNCurriculumAgent(
        model_name="gpt-5-mini",
        knowledge_base_path="path/to/minecraft-data",
        ...
    )
"""

# Import resource_tracker first (no external dependencies)
from skillnet.agents.psn_curriculum.resource_tracker import ResourceTracker

# GoalPlanner depends on knowledge_base and resource_tracker
from skillnet.agents.psn_curriculum.goal_planner import GoalPlanner

# Adaptive Learning components
from skillnet.agents.psn_curriculum.decomposition_filter import DecompositionFilter
from skillnet.agents.psn_curriculum.task_decomposer import TaskDecomposer, SubTask, DecomposedTask
from skillnet.agents.psn_curriculum.skill_gap_analyzer import SkillGapAnalyzer, SkillGap
from skillnet.agents.psn_curriculum.adaptive_planner import AdaptiveLearningPlanner, LearningPath, TaskProgress


def get_psn_curriculum_agent():
    """Lazy import of PSNCurriculumAgent to avoid circular imports with CurriculumAgent"""
    from skillnet.agents.psn_curriculum.agent import PSNCurriculumAgent
    return PSNCurriculumAgent


# For backwards compatibility and explicit import
# PSNCurriculumAgent is imported lazily when needed
__all__ = [
    # Main agent
    "PSNCurriculumAgent",
    "get_psn_curriculum_agent",
    # Core components
    "ResourceTracker",
    "GoalPlanner",
    # Adaptive Learning components
    "DecompositionFilter",
    "TaskDecomposer",
    "SubTask",
    "DecomposedTask",
    "SkillGapAnalyzer",
    "SkillGap",
    "AdaptiveLearningPlanner",
    "LearningPath",
    "TaskProgress",
]


# Lazy attribute for PSNCurriculumAgent
def __getattr__(name):
    if name == "PSNCurriculumAgent":
        from skillnet.agents.psn_curriculum.agent import PSNCurriculumAgent
        return PSNCurriculumAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
