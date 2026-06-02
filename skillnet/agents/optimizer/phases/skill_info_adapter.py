"""
Skill Info Adapter - retrieves skill info from the SkillGraphManager.

This module provides the skill_info_getter callback required by ReflectionChain.build().

Signature of skill_info_getter:
    (skill_name: str) -> Dict[str, Any]

The returned dict should contain:
    - code: str - skill code
    - description: str - skill description
    - children: List[str] - list of child skills
    - children_info: Dict[str, Dict] - child-skill info
    - execution_traces: List[Dict] - recent execution traces
    - pre_state: Optional[Dict] - pre-state of the most recent execution
    - post_state: Optional[Dict] - post-state of the most recent execution
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from skillnet.agents.skill_graph import SkillGraphManager, SkillNode
    from skillnet.agents.skill_graph.models.execution import SkillExecutionTrace


@dataclass
class SkillInfo:
    """Structured representation of skill info"""
    code: str
    description: str
    children: List[str]
    children_info: Dict[str, Dict[str, Any]]
    execution_traces: List[Dict[str, Any]]
    pre_state: Optional[Dict[str, Any]]
    post_state: Optional[Dict[str, Any]]
    preconditions: List[Dict[str, Any]]
    expected_effects: List[Dict[str, Any]]
    parameters: Dict[str, Dict[str, Any]]
    statistics: Dict[str, Any]
    is_task_specific: bool = False
    # Plan v3-rev Fix 1.A: refactor-derived wrapper flags
    is_covered: bool = False
    covered_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict"""
        return {
            "code": self.code,
            "description": self.description,
            "children": self.children,
            "children_info": self.children_info,
            "execution_traces": self.execution_traces,
            "pre_state": self.pre_state,
            "post_state": self.post_state,
            "preconditions": self.preconditions,
            "expected_effects": self.expected_effects,
            "parameters": self.parameters,
            "statistics": self.statistics,
            "is_task_specific": self.is_task_specific,
            "is_covered": self.is_covered,
            "covered_by": self.covered_by,
        }


class SkillInfoGetter:
    """
    Adapter that retrieves skill info from the SkillGraphManager.

    Usage:
        manager = SkillGraphManager(...)
        getter = SkillInfoGetter(manager)

        # Pass it as a callback
        chain = ReflectionChain(...)
        result = chain.build(
            root_skill_name="mySkill",
            root_feedback_content="...",
            skill_info_getter=getter  # pass the instance directly
        )

        # Or obtain the bound callable
        result = chain.build(
            ...,
            skill_info_getter=getter.get_skill_info
        )
    """

    def __init__(
        self,
        skill_graph_manager: 'SkillGraphManager',
        max_traces: int = 5,
        include_statistics: bool = True,
        logger=None,
    ):
        """
        Initialize the SkillInfoGetter

        Args:
            skill_graph_manager: SkillGraphManager instance
            max_traces: maximum number of execution traces to return
            include_statistics: whether to include statistics
            logger: logger
        """
        self.manager = skill_graph_manager
        self.max_traces = max_traces
        self.include_statistics = include_statistics
        self.logger = logger

    def __call__(self, skill_name: str) -> Dict[str, Any]:
        """
        Make the instance callable

        Args:
            skill_name: skill name

        Returns:
            Dict[str, Any]: skill info dict
        """
        return self.get_skill_info(skill_name)

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)

    def get_skill_info(self, skill_name: str) -> Dict[str, Any]:
        """
        Get info for the given skill

        Args:
            skill_name: skill name

        Returns:
            Dict[str, Any]: skill info dict; returns an empty dict if the skill does not exist
        """
        # Get the node
        node = self._get_node(skill_name)
        if not node:
            self._log(f"Skill not found: {skill_name}", "warning")
            return {}

        # Phase 3: use skill_name to correlate execution records with the child skill
        # Note: the JS-side call_stack contains skill names, not execution_ids,
        # so we pass skill_name instead of execution_id

        # Extract info
        skill_info = SkillInfo(
            code=node.code or "",
            description=node.description or "",
            children=list(node.children) if node.children else [],
            children_info=self._get_children_info(node, parent_skill_name=skill_name),
            execution_traces=self._get_execution_traces(node),
            pre_state=self._get_latest_pre_state(node),
            post_state=self._get_latest_post_state(node),
            preconditions=self._get_preconditions(node),
            expected_effects=self._get_expected_effects(node),
            parameters=node.parameters if hasattr(node, 'parameters') else {},
            statistics=self._get_statistics(node) if self.include_statistics else {},
            is_task_specific=getattr(node, 'is_task_specific', False),
            # Plan v3-rev Fix 1.A: surface refactor-derived wrapper state to Phase 1
            is_covered=bool(getattr(node, 'is_covered', False)),
            covered_by=getattr(node, 'covered_by', None),
        )

        return skill_info.to_dict()

    def _get_node(self, skill_name: str) -> Optional['SkillNode']:
        """Get the SkillNode"""
        if not self.manager:
            return None
        return self.manager.get_node(skill_name)

    def _get_children_info(
        self,
        node: 'SkillNode',
        parent_skill_name: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get child-skill info, including the most recent execution state.

        Phase 3 improvement: correlate executions on the same call chain via parent_skill_name.
        Note: the JS-side call_stack contains skill names, not execution_ids.

        Args:
            node: parent SkillNode instance
            parent_skill_name: name of the parent skill, used to look up the child's call_stack

        Each returned child info contains:
        - description: str - child-skill description
        - has_code: bool - whether it has code
        - children: List[str] - the child's own children
        - success_rate: float - success rate
        - last_execution: Dict - most recent execution info
            - success: bool - whether it succeeded
            - error_message: str - error message (if it failed)
            - inventory_changes: Dict[str, int] - inventory changes
            - from_same_call_chain: bool - whether it comes from the same call chain (Phase 3 addition)
        """
        children_info = {}

        if not node.children:
            return children_info

        for child_name in node.children:
            child_node = self._get_node(child_name)
            if child_node:
                child_info = {
                    "description": child_node.description or "",
                    "has_code": bool(child_node.code),
                    "children": list(child_node.children) if child_node.children else [],
                    "success_rate": child_node.statistics.success_rate if hasattr(child_node, 'statistics') and child_node.statistics else 0.0,
                    "parameters": child_node.parameters if hasattr(child_node, 'parameters') else {},
                }

                # Phase 3: pass parent_skill_name to correlate the same call chain
                last_execution = self._get_child_last_execution(
                    child_node,
                    parent_skill_name=parent_skill_name
                )
                if last_execution:
                    child_info["last_execution"] = last_execution

                children_info[child_name] = child_info
            else:
                children_info[child_name] = {
                    "description": f"Unknown skill: {child_name}",
                    "has_code": False,
                    "children": [],
                    "success_rate": 0.0,
                }

        return children_info

    def _get_child_last_execution(
        self,
        child_node: 'SkillNode',
        parent_skill_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the child skill's execution info.

        Phase 3 improvement: prefer execution records on the same call chain as the parent skill.
        Note: the JS-side call_stack contains skill names
        (e.g., ["craftWoodenPickaxe", "ensureCraftingTable"]).

        Args:
            child_node: child skill node
            parent_skill_name: parent skill name, used to match against call_stack

        Returns:
            Dict containing:
            - success: bool - whether it succeeded
            - error_message: str - error message (if it failed)
            - inventory_changes: Dict[str, int] - inventory changes
            - from_same_call_chain: bool - whether it comes from the same call chain (Phase 3)
            Returns None if there is no execution record
        """
        if not hasattr(child_node, 'statistics') or not child_node.statistics:
            return None

        traces = child_node.statistics.execution_traces
        if not traces:
            return None

        # Phase 3: if parent_skill_name is provided, try to find an execution on the same call chain
        target_trace = None

        if parent_skill_name:
            # Search backwards from the most recent trace for one whose call_stack contains parent_skill_name
            # call_stack format: ["grandparent_skill", "parent_skill", "current_skill"]
            for trace in reversed(traces):
                call_stack = getattr(trace, 'call_stack', None) or []
                if parent_skill_name in call_stack:
                    target_trace = trace
                    break

            # If parent_skill_name was provided but no match was found, do not fall back.
            # This avoids the "historical execution pollution" problem:
            # - incorrectly marking a fixed bug as unfixed
            # - mixing up execution results from different call chains
            if target_trace is None:
                return None
        else:
            # If parent_skill_name is not provided, use the most recent execution
            # This is the backward-compatible behavior
            target_trace = traces[-1]

        result = {
            "success": getattr(target_trace, 'success', False),
            "from_same_call_chain": parent_skill_name is not None,  # parent_skill_name was given and a match was found
        }

        # Add the error message (if any)
        error_msg = getattr(target_trace, 'error_message', None)
        if error_msg:
            result["error_message"] = error_msg

        # Compute inventory changes
        pre_state = getattr(target_trace, 'pre_state', None)
        post_state = getattr(target_trace, 'post_state', None)
        inventory_changes = self._compute_inventory_changes(pre_state, post_state)
        if inventory_changes:
            result["inventory_changes"] = inventory_changes

        return result

    def _compute_inventory_changes(
        self,
        pre_state: Optional[Dict[str, Any]],
        post_state: Optional[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Compute inventory changes

        Args:
            pre_state: pre-execution state {"inventory": {...}, ...}
            post_state: post-execution state {"inventory": {...}, ...}

        Returns:
            Dict[str, int]: per-item delta {item_name: delta}
        """
        changes = {}

        if not pre_state and not post_state:
            return changes

        # Extract inventory
        pre_inv = {}
        post_inv = {}

        if pre_state and isinstance(pre_state, dict):
            pre_inv = pre_state.get("inventory", {})
            if not isinstance(pre_inv, dict):
                pre_inv = {}

        if post_state and isinstance(post_state, dict):
            post_inv = post_state.get("inventory", {})
            if not isinstance(post_inv, dict):
                post_inv = {}

        # Compute changes
        all_items = set(pre_inv.keys()) | set(post_inv.keys())
        for item in all_items:
            before = pre_inv.get(item, 0)
            after = post_inv.get(item, 0)

            # Ensure numeric
            if not isinstance(before, (int, float)):
                before = 0
            if not isinstance(after, (int, float)):
                after = 0

            delta = int(after - before)
            if delta != 0:
                changes[item] = delta

        return changes

    def _get_execution_traces(self, node: 'SkillNode') -> List[Dict[str, Any]]:
        """Get execution traces"""
        traces = []

        if not hasattr(node, 'statistics') or not node.statistics:
            return traces

        execution_traces = node.statistics.execution_traces
        if not execution_traces:
            return traces

        # Take the most recent N
        recent_traces = execution_traces[-self.max_traces:]

        for trace in recent_traces:
            trace_dict = {
                "execution_id": getattr(trace, 'execution_id', ''),
                "timestamp": getattr(trace, 'timestamp', ''),
                "task": getattr(trace, 'task', ''),
                "success": getattr(trace, 'success', False),
                "error_message": getattr(trace, 'error_message', None),
                "critique": getattr(trace, 'critique', None),
                "call_args": getattr(trace, 'call_args', {}),
                "call_depth": getattr(trace, 'call_depth', 1),
            }
            traces.append(trace_dict)

        return traces

    def _get_latest_pre_state(self, node: 'SkillNode') -> Optional[Dict[str, Any]]:
        """Get the pre-state of the most recent execution"""
        if not hasattr(node, 'statistics') or not node.statistics:
            return None

        traces = node.statistics.execution_traces
        if not traces:
            return None

        latest = traces[-1]
        return getattr(latest, 'pre_state', None)

    def _get_latest_post_state(self, node: 'SkillNode') -> Optional[Dict[str, Any]]:
        """Get the post-state of the most recent execution"""
        if not hasattr(node, 'statistics') or not node.statistics:
            return None

        traces = node.statistics.execution_traces
        if not traces:
            return None

        latest = traces[-1]
        return getattr(latest, 'post_state', None)

    def _get_preconditions(self, node: 'SkillNode') -> List[Dict[str, Any]]:
        """Get preconditions"""
        preconditions = []

        if not hasattr(node, 'preconditions') or not node.preconditions:
            return preconditions

        for pc in node.preconditions:
            pc_dict = {
                "condition": getattr(pc, 'condition', ''),
                "description": getattr(pc, 'description', ''),
                "is_optional": getattr(pc, 'is_optional', False),
            }
            preconditions.append(pc_dict)

        return preconditions

    def _get_expected_effects(self, node: 'SkillNode') -> List[Dict[str, Any]]:
        """Get expected effects"""
        effects = []

        if not hasattr(node, 'expected_effects') or not node.expected_effects:
            return effects

        for effect in node.expected_effects:
            effect_dict = {
                "effect": getattr(effect, 'effect', ''),
                "description": getattr(effect, 'description', ''),
                "is_guaranteed": getattr(effect, 'is_guaranteed', True),
            }
            effects.append(effect_dict)

        return effects

    def _get_statistics(self, node: 'SkillNode') -> Dict[str, Any]:
        """Get statistics"""
        if not hasattr(node, 'statistics') or not node.statistics:
            return {}

        stats = node.statistics
        return {
            "total_executions": stats.total_executions,
            "successful_executions": stats.successful_executions,
            "failed_executions": stats.failed_executions,
            "success_rate": stats.success_rate,
            "value_function": node.value_function,
        }


def create_skill_info_getter(
    skill_graph_manager: 'SkillGraphManager',
    max_traces: int = 5,
    include_statistics: bool = True,
    logger=None,
) -> Callable[[str], Dict[str, Any]]:
    """
    Create the skill_info_getter callback

    Args:
        skill_graph_manager: SkillGraphManager instance
        max_traces: maximum number of execution traces to return
        include_statistics: whether to include statistics
        logger: logger

    Returns:
        Callable[[str], Dict[str, Any]]: the callable getter
    """
    getter = SkillInfoGetter(
        skill_graph_manager=skill_graph_manager,
        max_traces=max_traces,
        include_statistics=include_statistics,
        logger=logger,
    )
    return getter.get_skill_info


# Exports
__all__ = [
    'SkillInfo',
    'SkillInfoGetter',
    'create_skill_info_getter',
]
