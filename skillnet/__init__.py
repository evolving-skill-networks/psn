"""
SkillNet package - Evolving Programmatic Skill Networks (PSN) for Open-Ended Embodied Agents.

Some runtime components (e.g., Minecraft launcher integration) are optional and may not
be available in all environments. Keep imports lightweight so tooling can run without
full runtime dependencies.
"""

try:
    from .psn import PSNAgent  # type: ignore
except Exception:  # pragma: no cover
    PSNAgent = None  # type: ignore
