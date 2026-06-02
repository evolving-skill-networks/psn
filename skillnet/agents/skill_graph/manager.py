"""
Skill Graph Manager

This is the public API entry point for SkillGraphManager.

SkillGraphManager is a graph-based skill manager that provides:
- Graph-structured skill dependency management
- Skill precondition and effect extraction
- Skill version control
- Skill retrieval and search

Usage:
    from skillnet.agents.skill_graph import SkillGraphManager

    manager = SkillGraphManager(
        model_name="gpt-5-mini",
        ckpt_dir="ckpt",
        resume=True,
    )

    # Add new skill
    manager.add_new_skill(skill_code, skill_description)

    # Retrieve related skills
    skills = manager.retrieve_skills(query="mine diamond ore")
"""

# Import from internal implementation
from skillnet.agents.skill_graph._impl import (
    SkillGraphManager,
    _strip_comments_and_strings,
)

__all__ = [
    "SkillGraphManager",
    "_strip_comments_and_strings",
]
