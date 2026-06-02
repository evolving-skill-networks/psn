"""
Agents utils — proxy module.

Re-exports the single helper used through the proxy path
``from ..utils import validate_code_syntax`` (refactor/ submodules).
Other utilities should be imported directly from
``skillnet.agents.skill_graph.utils``.
"""

from skillnet.agents.skill_graph.utils import validate_code_syntax

__all__ = ["validate_code_syntax"]
