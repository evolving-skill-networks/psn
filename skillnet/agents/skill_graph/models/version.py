"""
Version-related dataclasses.

Contains SkillVersion and GraphVersion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .precondition import SkillPrecondition, SkillEffect


def _check_js_skill_contracts(code: str) -> tuple:
    """
    Check whether the JavaScript skill code satisfies basic contracts.

    Contracts include:
    1. Cannot directly pass `undefined` as an argument
    2. Brackets must be balanced

    Args:
        code: JavaScript code

    Returns:
        Tuple[bool, List[Dict]]: (passed, list of violations)
    """
    if not code or not code.strip():
        return (True, [])

    violations = []

    # Detect patterns that explicitly pass `undefined`
    import re
    undefined_patterns = [
        (r'\(\s*undefined\s*[,)]', "Direct 'undefined' passed as first argument"),
        (r',\s*undefined\s*[,)]', "Direct 'undefined' passed as argument"),
        (r'=\s*undefined\s*[;,)]', "Direct assignment of 'undefined'"),
    ]

    for pattern, message in undefined_patterns:
        if re.search(pattern, code):
            violations.append({
                "type": "undefined_passing",
                "pattern": pattern,
                "message": "Detected `undefined` passed into a skill call. This often indicates wrapper parameter mapping failure; avoid padding args with undefined.",
            })

    return (len(violations) == 0), violations


@dataclass
class SkillVersion:
    """Skill version information."""
    version: str  # Version number, e.g. "1.0.0", "1.1.0"
    created_at: str  # Creation timestamp
    change_log: str  # Change description
    code: str  # Code of this version
    description: str  # Description of this version

    # Version snapshot (all attributes of this version)
    preconditions: List[SkillPrecondition] = field(default_factory=list)  # Preconditions of this version
    effects: List[SkillEffect] = field(default_factory=list)  # Expected effects of this version
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Parameter metadata of this version
    value_function: float = 0.0  # Value function of this version
    statistics_snapshot: Dict[str, Any] = field(default_factory=dict)  # Statistics snapshot of this version
    # Format: {"total_executions": int, "successful_executions": int, "failed_executions": int, "success_rate": float}

    # Update-source tracking
    update_source: str = "unknown"  # Update source: "optimizer", "refactor", "manual", "backpropagation", "unknown"
    update_reason: str = ""  # Update reason (detailed)
    optimization_id: Optional[str] = None  # Associated optimization-flow ID (if from optimizer)

    # Used feedback/gradient information
    used_feedbacks: List[Dict[str, Any]] = field(default_factory=list)  # List of feedbacks used
    # Format: [{
    # "content": "feedback content",
    # "source_skill": "s1",  # Source skill name
    # "propagation_path": ["s1", "s0"],  # Propagation path
    # "optimization_id": "opt_12345",  # Associated optimization ID
    # "timestamp": "2024-01-01T00:00:00"
    # }]

    # Backpropagation info — populated when an optimization is triggered by
    # backpropagation up a skill-call chain (parent skill drove the fix).
    backpropagation_info: Optional[Dict[str, Any]] = None

    # Fields related to the rollback mechanism
    affected_subgraph: Dict[str, Any] = field(default_factory=dict)  # Information about the affected subgraph
    # Format: {
    # "directly_affected": ["skill1", "skill2"],  # Directly affected skills (parents that call this skill)
    # "indirectly_affected": ["skill3", "skill4"],  # Indirectly affected skills (through the call chain)
    # "graph_snapshot": {...},  # Graph snapshot at version creation
    # "previous_version": "1.0.0",  # Previous version number
    # }

    # Sliding window tracking
    sliding_window_size: int = 20  # Sliding-window size; default 20 executions
    execution_window: List[Dict[str, Any]] = field(default_factory=list)  # Execution-window records
    # Format: [{"execution_id": "...", "success": bool, "timestamp": "...", "version": "..."}, ...]

    # Contract gate (generation/interface safety)
    is_contract_valid: bool = True
    contract_violations: List[Dict[str, Any]] = field(default_factory=list)

    def __init__(
        self,
        version: str = "1.0.0",
        created_at: str = None,
        change_log: str = "",
        code: str = "",
        description: str = "",
        preconditions: List[SkillPrecondition] = None,
        effects: List[SkillEffect] = None,
        parameters: Dict[str, Dict[str, Any]] = None,
        value_function: float = 0.0,
        statistics_snapshot: Dict[str, Any] = None,
        update_source: str = "unknown",
        update_reason: str = "",
        optimization_id: str = None,
        used_feedbacks: List[Dict[str, Any]] = None,
        backpropagation_info: Dict[str, Any] = None,
        affected_subgraph: Dict[str, Any] = None,
        sliding_window_size: int = 20,
        execution_window: List[Dict[str, Any]] = None,
        is_contract_valid: Optional[bool] = None,
        contract_violations: Optional[List[Dict[str, Any]]] = None,
    ):
        self.version = version
        self.created_at = created_at or datetime.now().isoformat()
        self.change_log = change_log
        self.code = code
        self.description = description
        self.preconditions = preconditions or []
        self.effects = effects or []
        self.parameters = parameters or {}
        self.value_function = value_function
        self.statistics_snapshot = statistics_snapshot or {}
        self.update_source = update_source
        self.update_reason = update_reason
        self.optimization_id = optimization_id
        self.used_feedbacks = used_feedbacks or []
        self.backpropagation_info = backpropagation_info
        self.affected_subgraph = affected_subgraph or {}
        self.sliding_window_size = sliding_window_size
        self.execution_window = execution_window or []

        # Contract validity: compute by default from code unless explicitly provided
        if is_contract_valid is None or contract_violations is None:
            ok, violations = _check_js_skill_contracts(self.code)
            self.is_contract_valid = ok if is_contract_valid is None else bool(is_contract_valid)
            self.contract_violations = violations if contract_violations is None else (contract_violations or [])
        else:
            self.is_contract_valid = bool(is_contract_valid)
            self.contract_violations = contract_violations or []


@dataclass
class GraphVersion:
    """Skill graph version information."""
    version: str  # Version number, e.g. "1.0.0", "2.0.0"
    created_at: str  # Creation timestamp
    change_log: str  # Change description

    # Graph snapshot
    node_versions: Dict[str, str] = field(default_factory=dict)  # {node_name: version} versions of all nodes
    edges: Dict[str, List[str]] = field(default_factory=dict)  # {parent: [children]} snapshot of all edges

    # Update tracking
    update_source: str = "unknown"  # Update source: "refactor", "add_skill", "remove_skill", "optimization", "manual", "unknown"
    update_reason: str = ""  # Update reason (detailed)
    changes: Dict[str, Any] = field(default_factory=dict)  # Change details
    # Format: {
    # "added_nodes": [...],  # List of newly added nodes
    # "removed_nodes": [...],  # List of removed nodes
    # "added_edges": [...],  # List of newly added edges [(parent, child), ...]
    # "removed_edges": [...],  # List of removed edges [(parent, child), ...]
    # "updated_nodes": [...],  # List of updated nodes (node version changed)
    # }

    def __init__(
        self,
        version: str = "1.0.0",
        created_at: str = None,
        change_log: str = "",
        node_versions: Dict[str, str] = None,
        edges: Dict[str, List[str]] = None,
        update_source: str = "unknown",
        update_reason: str = "",
        changes: Dict[str, Any] = None,
    ):
        self.version = version
        self.created_at = created_at or datetime.now().isoformat()
        self.change_log = change_log
        self.node_versions = node_versions or {}
        self.edges = edges or {}
        self.update_source = update_source
        self.update_reason = update_reason
        self.changes = changes or {}
