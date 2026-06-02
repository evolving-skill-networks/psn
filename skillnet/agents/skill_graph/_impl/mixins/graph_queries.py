"""
Graph Queries Mixin for SkillGraphManager

Thin wrapper methods that delegate to `self.graph` (SkillGraph) for
graph structure queries, node access, traversal, and public aliases
for internal methods.

Extracted from graph_manager_impl.py for better modularity.
"""

from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import SkillGraph, SkillNode
    from ..graph_manager_impl import SkillGraphManager


class GraphQueriesMixin:
    """Graph Queries Mixin - thin wrappers around self.graph.

    Methods grouped by category:

    Graph structure:
        add_edge, remove_edge, has_edge, would_create_cycle

    Node access:
        get_node, has_node, add_skill_node, remove_skill_node, get_subgraph

    Iteration / query:
        iter_skills, skill_count (property), get_all_skill_names

    Traversal:
        get_parents, get_children

    Public aliases (Phase 6 unified interface):
        save, update_skill_dependencies, is_task_specific,
        expand_with_dependencies, extract_effects

    Attributes (from SkillGraphManager, resolved via MRO):
        graph: SkillGraph instance
    """

    # ========== Graph structure ==========

    def add_edge(self, parent_name: str, child_name: str) -> bool:
        """Add a skill-call relationship."""
        return self.graph.add_edge(parent_name, child_name)

    def remove_edge(self, parent_name: str, child_name: str) -> bool:
        """Remove a skill-call relationship."""
        return self.graph.remove_edge(parent_name, child_name)

    def has_edge(self, parent: str, child: str) -> bool:
        """
        Check whether an edge exists.

        Args:
            parent: parent node name
            child: child node name

        Returns:
            bool: whether the edge exists
        """
        return self.graph.has_edge(parent, child)

    def would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """Check whether adding the edge would create a cycle."""
        return self.graph.would_create_cycle(from_node, to_node)

    # ========== Node access ==========

    def get_node(self, name: str) -> Optional["SkillNode"]:
        """Get a skill node."""
        return self.graph.get_node(name)

    def has_node(self, name: str) -> bool:
        """
        Check whether a node exists.

        Args:
            name: skill name

        Returns:
            bool: whether the node exists
        """
        return self.graph.has_node(name)

    def add_skill_node(self, node: "SkillNode") -> bool:
        """
        Add a skill node to the graph.

        Args:
            node: SkillNode instance

        Returns:
            bool: whether added successfully (returns False if it already exists)
        """
        if self.graph.has_node(node.name):
            return False
        self.graph.add_node(node)
        return True

    def remove_skill_node(self, name: str) -> bool:
        """
        Remove a skill node from the graph.

        Args:
            name: skill name

        Returns:
            bool: whether removed successfully (returns False if it does not exist)
        """
        if not self.graph.has_node(name):
            return False
        self.graph.remove_node(name)
        return True

    def get_subgraph(self, root_name: str, max_depth: int = None) -> "SkillGraph":
        """Get a subgraph."""
        return self.graph.get_subgraph(root_name, max_depth)

    # ========== Iteration / query ==========

    def get_all_skill_names(self, include_task_specific: bool = False) -> List[str]:
        """
        Get all skill names.

        Args:
            include_task_specific: whether to include task-specific skills

        Returns:
            List[str]: list of skill names
        """
        names = list(self.graph.nodes.keys())
        if not include_task_specific:
            names = [n for n in names if not getattr(self.graph.get_node(n), 'is_task_specific', False)]
        return names

    def iter_skills(self, include_task_specific: bool = False) -> Iterator[Tuple[str, "SkillNode"]]:
        """
        Iterate over all skills.

        Args:
            include_task_specific: whether to include task-specific skills

        Yields:
            Tuple[str, SkillNode]: (skill name, SkillNode instance)
        """
        for name, node in self.graph.nodes.items():
            if include_task_specific or not getattr(node, 'is_task_specific', False):
                yield name, node

    @property
    def skill_count(self) -> int:
        """Get the number of skills."""
        return len(self.graph.nodes)

    # ========== Traversal ==========

    def get_parents(self, name: str) -> List[str]:
        """
        Get the list of parent nodes for a node (callers).

        Args:
            name: skill name

        Returns:
            List[str]: list of parent node names
        """
        return self.graph.get_parents(name)

    def get_children(self, name: str) -> List[str]:
        """
        Get the list of child nodes for a node (callees).

        Args:
            name: skill name

        Returns:
            List[str]: list of child node names
        """
        return self.graph.get_children(name)

    # ========== Phase 6b: Public aliases for externally-called private methods ==========

    def save(self) -> None:
        """Public alias for checkpoint persistence."""
        self._save_to_checkpoint()

    def update_skill_dependencies(self, skill_name: str) -> None:
        """Public alias: rebuild call-graph edges from skill code."""
        self._update_graph_from_code(skill_name)

    def is_task_specific(self, func_name: str, func_code: str, node: "SkillNode" = None) -> bool:
        """Public alias: check if skill is task-specific (non-reusable)."""
        return self._is_task_specific_skill(func_name, func_code, node)

    def expand_with_dependencies(self, skills: List[str], max_depth: int = 5) -> List[str]:
        """Public alias: expand skill list with full dependency chain."""
        return self._expand_skills_with_call_chain(skills, max_depth=max_depth)

    def extract_effects(self, code, description, task=None, context=None,
                        skill_name=None, execution_traces=None):
        """Public API: Extract expected effects from skill code."""
        return self._extract_effects(code, description, task=task, context=context,
                                      skill_name=skill_name, execution_traces=execution_traces)
