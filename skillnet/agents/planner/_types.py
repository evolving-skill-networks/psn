"""
Shared types and constants for the Planner module.

Contains dataclasses and base classes that are referenced by multiple mixins.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from skillnet.agents.constants.task_semantics import TaskWithSemantic
from skillnet.agents.planning.inference.config_validator import (
    TRANSFORM_TO_PATTERN,
)


@dataclass
class SkillCallContext:
    """Skill call context, containing the reason for the call and the required output."""
    skill_name: str
    precondition_context: Optional[Dict[str, Any]] = None
    # precondition_context structure:
    # {
    # "required_item": "raw_iron",      # Item required to be produced
    # "required_count": 3,              # Required quantity
    # "caller_skill": "smeltRawIron",   # Caller skill
    # "precondition_desc": "..."        # Precondition description
    # }


@dataclass
class PlanningResult:
    """Planning result."""
    success: bool  # Whether plan generation succeeded
    plan_type: str  # "llm" or "graph"
    code: Optional[str] = None  # Generated code (if successful)
    plan: Optional[str] = None  # Plan description (optional)
    skill_sequence: Optional[List['SkillCallContext']] = None  # Skill sequence returned by graph planner (with context)
    error: Optional[str] = None  # Error message (if failed)
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


class BasePlanner(ABC):
    """Planner base class."""

    def __init__(self, **kwargs):
        """Initialize the planner."""
        pass

    @abstractmethod
    def plan(
        self,
        task: TaskWithSemantic,
        context: str,
        current_state: Dict[str, Any],
        available_skills: List[str],
        skill_metadata: Optional[Dict[str, Any]] = None,
        previous_code: str = "",
        critique: str = "",
    ) -> PlanningResult:
        """
        Generate an execution plan.

        Args:
            task: TaskWithSemantic object containing the task description and semantic info
            context: Context information
            current_state: Current state (inventory, position, equipment, etc.)
            available_skills: List of available skill code
            skill_metadata: Skill metadata (parameter info, etc.)
            previous_code: Previous-round code (used for iterative optimization)
            critique: Critic feedback

        Returns:
            PlanningResult: Planning result
        """
        pass
