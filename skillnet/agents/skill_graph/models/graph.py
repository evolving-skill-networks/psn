"""
SkillGraph dataclass

Skill Graph: manages call relationships between skills.
"""

import copy
from collections import defaultdict, deque
from typing import Dict, List, Set, Optional, Any

from .node import SkillNode
from .version import GraphVersion
from .coverage import CoverageType


def should_skip_covered_skill(node) -> bool:
    """
    Decide whether to skip a covered skill during retrieve.

    Purpose: avoid returning simple wrapper skills; return the more valuable general skill instead.
    Example: when the user asks "how to mine logs", return mineLogs rather than mineOakLog.

    Skip conditions:
    - skills with is_covered=True and coverage_type.is_wrapper=True

    Cases that are not skipped:
    1. Skills that are not covered (is_covered=False)
    2. Skills that are independently valuable (coverage_type.is_independent=True)
       - COMMON_SUBSKILL: e.g. setupCraftingTable
       - BEHAVIORAL: independent behavior skills
    3. Skills with is_general_skill=True

    Args:
        node: SkillNode instance

    Returns:
        bool: whether the skill should be skipped during retrieve
    """
    if not getattr(node, 'is_covered', False):
        return False

    # Use CoverageType to decide
    coverage_type = getattr(node, 'coverage_type', None)
    if coverage_type is not None:
        # Independently valuable skills are not skipped
        if coverage_type.is_independent:
            return False
        # Wrapper types are skipped
        if coverage_type.is_wrapper:
            return True
        # Unknown type: conservatively do not skip
        return False

    # Skills flagged is_general_skill are not skipped
    if getattr(node, 'is_general_skill', False):
        return False

    # Default: skip (is_covered=True with no other protection)
    return True


class SkillGraph:
    """
    Skill Graph: manages call relationships between skills.

    Nodes: SkillNode
    Edges: call relationships between skills (from caller to callee)
    """

    def __init__(self):
        self.nodes: Dict[str, SkillNode] = {}  # nodes dict: name -> SkillNode
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # edges: parent -> {children}
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # reverse edges: child -> {parents}

        # Graph version history
        self.versions: List[GraphVersion] = []
        self.current_version: Optional[str] = None  # current version number

    def add_node(self, node: SkillNode) -> bool:
        """
        Add a node.

        Returns:
            bool: False if the node already exists, otherwise True
        """
        if node.name in self.nodes:
            return False
        self.nodes[node.name] = node
        return True

    def remove_node(self, name: str) -> bool:
        """
        Delete a node (also deletes related edges).

        Returns:
            bool: False if the node does not exist, otherwise True
        """
        if name not in self.nodes:
            return False

        # Delete all outgoing edges (nodes called by this one)
        if name in self.edges:
            for child in self.edges[name]:
                if child in self.reverse_edges:
                    self.reverse_edges[child].discard(name)
                if child in self.nodes:
                    # Safety check: only remove when name is in parents
                    # (the parent of a task-specific skill is not added to child.parents)
                    if name in self.nodes[child].parents:
                        self.nodes[child].parents.remove(name)
                        self.nodes[child].in_degree -= 1
            del self.edges[name]

        # Delete all incoming edges (nodes that call this one)
        if name in self.reverse_edges:
            for parent in self.reverse_edges[name]:
                if parent in self.edges:
                    self.edges[parent].discard(name)
                if parent in self.nodes:
                    self.nodes[parent].children.remove(name)
                    self.nodes[parent].out_degree -= 1
            del self.reverse_edges[name]

        # Delete the node
        del self.nodes[name]
        return True

    def add_edge(self, parent_name: str, child_name: str) -> bool:
        """
        Add an edge (parent calls child).

        Returns:
            bool: False if the node does not exist, otherwise True
        """
        if parent_name not in self.nodes or child_name not in self.nodes:
            return False

        parent_node = self.nodes[parent_name]
        child_node = self.nodes[child_name]

        # If parent is a task-specific skill, only add to edges/reverse_edges
        # do not modify child.parents (to avoid polluting normal skill parents lists during serialization)
        if getattr(parent_node, 'is_task_specific', False):
            self.edges[parent_name].add(child_name)
            self.reverse_edges[child_name].add(parent_name)
            if child_name not in parent_node.children:
                parent_node.children.append(child_name)
                parent_node.out_degree += 1
            return True

        # Normal skill: full edge-addition logic
        self.edges[parent_name].add(child_name)
        self.reverse_edges[child_name].add(parent_name)

        if child_name not in parent_node.children:
            parent_node.children.append(child_name)
            parent_node.out_degree += 1

        if parent_name not in child_node.parents:
            child_node.parents.append(parent_name)
            child_node.in_degree += 1

        return True

    def remove_edge(self, parent_name: str, child_name: str) -> bool:
        """
        Delete an edge.

        Returns:
            bool: False if the edge does not exist, otherwise True
        """
        if parent_name not in self.edges or child_name not in self.edges[parent_name]:
            return False

        # Delete the edge
        self.edges[parent_name].discard(child_name)
        self.reverse_edges[child_name].discard(parent_name)

        # Update the graph structural info for the node
        if parent_name in self.nodes:
            parent_node = self.nodes[parent_name]
            if child_name in parent_node.children:
                parent_node.children.remove(child_name)
                parent_node.out_degree -= 1

        if child_name in self.nodes:
            child_node = self.nodes[child_name]
            if parent_name in child_node.parents:
                child_node.parents.remove(parent_name)
                child_node.in_degree -= 1

        return True

    def has_edge(self, parent_name: str, child_name: str) -> bool:
        """
        Check whether an edge exists.

        Args:
            parent_name: parent node name
            child_name: child node name

        Returns:
            bool: True if the edge exists, otherwise False
        """
        if parent_name not in self.edges:
            return False
        return child_name in self.edges[parent_name]

    def get_node(self, name: str) -> Optional[SkillNode]:
        """Get a node."""
        return self.nodes.get(name)

    def has_node(self, name: str) -> bool:
        """Check whether a node exists."""
        return name in self.nodes

    def get_children(self, name: str) -> List[str]:
        """Get a node's children (in call order)."""
        if name not in self.nodes:
            return []
        return self.nodes[name].children.copy()

    def get_parents(self, name: str) -> List[str]:
        """Get a node's parents."""
        if name not in self.nodes:
            return []
        return self.nodes[name].parents.copy()

    def get_subgraph(self, root_name: str, max_depth: int = None) -> "SkillGraph":
        """
        Get the subgraph (the dependency subgraph starting from the root node).

        Args:
            root_name: root node name
            max_depth: maximum depth (None means unbounded)

        Returns:
            SkillGraph: the subgraph
        """
        if root_name not in self.nodes:
            return SkillGraph()

        subgraph = SkillGraph()
        visited = set()
        queue = deque([(root_name, 0)])  # (node_name, depth)

        while queue:
            node_name, depth = queue.popleft()

            if node_name in visited:
                continue

            if max_depth is not None and depth > max_depth:
                continue

            visited.add(node_name)

            # Add node to the subgraph
            node = self.nodes[node_name]
            subgraph.add_node(copy.deepcopy(node))

            # Add children to the queue
            for child_name in self.get_children(node_name):
                if child_name not in visited:
                    queue.append((child_name, depth + 1))

        # Add edges within the subgraph
        for node_name in subgraph.nodes:
            for child_name in self.get_children(node_name):
                if child_name in subgraph.nodes:
                    subgraph.add_edge(node_name, child_name)

        return subgraph

    def would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """
        Check whether adding an edge from from_node to to_node would create a circular dependency.

        Args:
            from_node: source node name
            to_node: target node name

        Returns:
            bool: True if a cycle would be created, otherwise False
        """
        # If a node does not exist, no cycle is created
        if from_node not in self.nodes or to_node not in self.nodes:
            return False

        # If from_node == to_node, adding the edge creates a self-loop
        if from_node == to_node:
            return True

        # Check via DFS: if to_node can reach from_node, then adding from_node -> to_node creates a cycle
        visited = set()
        stack = [to_node]

        while stack:
            current = stack.pop()

            if current == from_node:
                # Path found from to_node to from_node; adding the edge would create a cycle
                return True

            if current in visited:
                continue

            visited.add(current)

            # Iterate over all children of the current node (called skills)
            for child in self.get_children(current):
                if child not in visited:
                    stack.append(child)

        # No path from to_node to from_node; adding the edge will not create a cycle
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for serialization)."""

        def should_persist(node) -> bool:
            """Decide whether a node should be persisted."""
            # Phase 7: do not persist task-specific or experimental skills
            if getattr(node, 'is_task_specific', False):
                return False
            if getattr(node, 'is_experimental', False):
                return False
            return True

        return {
            # Exclude task-specific and experimental skills
            "nodes": {
                name: node.to_dict()
                for name, node in self.nodes.items()
                if should_persist(node)
            },
            # Exclude edges whose parent or child is a task-specific or experimental skill
            "edges": {
                parent: filtered_children
                for parent, children in self.edges.items()
                if parent not in self.nodes or should_persist(self.nodes[parent])
                for filtered_children in [[
                    child for child in children
                    if child not in self.nodes or should_persist(self.nodes[child])
                ]]
                if filtered_children
            },
            "versions": [
                {
                    "version": v.version,
                    "created_at": v.created_at,
                    "change_log": v.change_log,
                    "node_versions": v.node_versions,
                    "edges": v.edges,
                    "update_source": v.update_source,
                    "update_reason": v.update_reason,
                    "changes": v.changes,
                }
                for v in self.versions
            ],
            "current_version": self.current_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillGraph":
        """Build a SkillGraph from a dict (for deserialization)."""
        graph = cls()

        # Restore nodes
        for name, node_data in data.get("nodes", {}).items():
            node = SkillNode.from_dict(node_data)
            graph.add_node(node)

        # Restore edges
        for parent, children in data.get("edges", {}).items():
            for child in children:
                graph.add_edge(parent, child)

        # Restore version history
        for v_data in data.get("versions", []):
            graph.versions.append(GraphVersion(
                version=v_data.get("version", "1.0.0"),
                created_at=v_data.get("created_at"),
                change_log=v_data.get("change_log", ""),
                node_versions=v_data.get("node_versions", {}),
                edges=v_data.get("edges", {}),
                update_source=v_data.get("update_source", "unknown"),
                update_reason=v_data.get("update_reason", ""),
                changes=v_data.get("changes", {}),
            ))

        graph.current_version = data.get("current_version")

        return graph
