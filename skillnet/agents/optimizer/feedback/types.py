"""
Feedback type definitions.

Includes the SkillFeedback and OptimizationRecord dataclasses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..analysis.error_classifier import ErrorCategory, FixTargetType


@dataclass
class OptimizationRecord:
    """
    Optimization record - enhanced.

    Stores enough information to reproduce what happened:
    - the error/feedback that triggered the optimization
    - code before and after
    - strategy used and LLM interactions
    - results and diagnostic info
    """
    skill_name: str
    timestamp: str
    error_category: ErrorCategory
    error_pattern: str                          # Error-pattern signature (used for duplicate detection)
    strategy_used: str                          # Optimization strategy used
    fix_target: FixTargetType
    successful: bool
    root_cause_analysis: Optional[Any] = None
    version_before: Optional[int] = None
    version_after: Optional[int] = None

    # ========== Trigger info ==========
    task: Optional[str] = None                  # Task that triggered the optimization
    error_message: Optional[str] = None         # Specific error message
    error_stack: Optional[str] = None           # Error stack trace
    feedback_content: Optional[str] = None      # feedback/critique content

    # ========== Code change ==========
    code_before: Optional[str] = None           # Code before optimization
    code_after: Optional[str] = None            # Code after optimization
    code_diff: Optional[str] = None             # Code-change diff

    # ========== Execution context ==========
    execution_context: Optional[Dict[str, Any]] = None  # inventory, position, etc.
    called_skills: Optional[List[str]] = None   # Which child skills were called

    # ========== LLM interaction ==========
    llm_prompt: Optional[str] = None            # Prompt sent to the LLM (summary)
    llm_response: Optional[str] = None          # LLM response (summary)

    # ========== Consistency check ==========
    consistency_check_passed: Optional[bool] = None     # Whether the consistency check passed
    consistency_conflicts: Optional[List[str]] = None   # List of conflicts
    consistency_score: Optional[float] = None           # Consistency score

    # ========== Failure reason ==========
    failure_reason: Optional[str] = None        # If it failed, what was the reason

    # ========== Value function snapshot ==========
    value_at_optimization: Optional[float] = None  # V(s) when optimization was attempted

    # ========== P1: separated storage logs ==========
    optimization_id: Optional[str] = None       # Unique optimization ID
    detailed_log_path: Optional[str] = None     # Detailed log file path (relative to detailed_logs_dir)


@dataclass
class SkillFeedback:
    """Structured skill-feedback info."""
    skill_name: str
    feedback_type: str  # "error", "critique", "performance", "suggestion", "internal_analysis"
    content: str
    source: str  # "execution_trace", "critic_agent", "user", "backpropagation", "backpropagation_analysis", "semantic_detection", etc.
    timestamp: str
    execution_id: Optional[str] = None  # Associated execution trace ID
    trajectory_segment_id: Optional[str] = None  # Associated trajectory segment ID
    trajectory_file: Optional[str] = None  # Associated trajectory file path
    task: Optional[str] = None  # Associated task
    severity: str = "medium"  # "low", "medium", "high"
    related_skills: List[str] = field(default_factory=list)  # Other related skills
    priority: str = "normal"  # "normal", "high", "internal" - "internal" means it comes from internal system analysis and should not be filtered by relevance
    resolved: bool = False  # Whether resolved — resolved feedback is not retrieved by default

    def __init__(
        self,
        skill_name: str,
        feedback_type: str,
        content: str,
        source: str,
        timestamp: str = None,
        execution_id: str = None,
        trajectory_segment_id: str = None,
        trajectory_file: str = None,
        task: str = None,
        severity: str = "medium",
        related_skills: List[str] = None,
        priority: str = "normal",
        resolved: bool = False,
    ):
        self.skill_name = skill_name
        self.feedback_type = feedback_type
        self.content = content
        self.source = source
        self.timestamp = timestamp or datetime.now().isoformat()
        self.execution_id = execution_id
        self.trajectory_segment_id = trajectory_segment_id
        self.trajectory_file = trajectory_file
        self.task = task
        self.severity = severity
        self.related_skills = related_skills or []
        self.priority = priority
        self.resolved = resolved


@dataclass
class CodeEdit:
    """Structured code-edit operation."""
    edit_type: str  # Edit type
    target: str  # Edit target (function name, variable name, line number, etc.)
    content: Optional[str] = None  # New content
    params: Dict[str, Any] = field(default_factory=dict)  # Extra parameters
    description: str = ""  # Edit description

    # Supported edit types
    EDIT_TYPES = {
        # Declaration-movement category
        "move_declaration_to_top": "move declaration to the start of the function",
        "move_require_to_top": "move require statements to the top",

        # Error-handling category
        "add_try_catch": "add try-catch wrapping",
        "add_null_check": "add null checks",
        "add_type_check": "add type checks",
        "add_error_throw": "add error throw",

        # Function-modification category
        "add_parameter": "add a function parameter",
        "modify_parameter_default": "modify a parameter default",
        "add_return_check": "add a return-value check",

        # Code-block category
        "replace_line_range": "replace a line range",
        "insert_before_line": "insert before a line",
        "insert_after_line": "insert after a line",
        "delete_line_range": "delete a line range",

        # Structural category
        "wrap_in_function": "wrap code in a function",
        "extract_helper": "extract a helper function",
        "inline_helper": "inline a helper function",

        # Validation category
        "add_precondition_check": "add a precondition check",
        "add_postcondition_check": "add a postcondition check",

        # Minecraft resource/name-fix category
        "fix_resource_logic": "fix resource-name logic (e.g. item-name -> block-name conversion)",
        "add_name_mapping": "add a name-mapping helper function",
        "add_safe_block_lookup": "add safe block-lookup logic",

        # Interface/parameter-fix category
        "fix_parameter_passing": "fix parameter-passing issues in function calls",
        "fix_interface_mismatch": "fix parent/child skill interface mismatch",
        "add_parameter_validation": "add parameter-validation logic",

        # Item-collection-fix category
        "add_collection_wait": "add item-collection wait logic",
        "fix_inventory_check": "fix inventory-check logic",

        # Child-skill return-value-check category
        "check_child_skill_return_value": "add child-skill return-value check (capture and abort on failure)",
        "add_inventory_verification": "verify inventory after a child-skill call",
    }


@dataclass
class ModificationAnalysis:
    """Modification analysis result."""
    modification_type: str  # "structural", "logic", "error_handling", "small_fix", "refactor"
    complexity: str  # "trivial", "simple", "moderate", "complex"
    affected_lines: int  # Estimated number of affected lines
    requires_rewrite: bool  # Whether a full rewrite is needed
    suggested_edits: List[CodeEdit] = field(default_factory=list)  # Suggested edits
    reasoning: str = ""  # Analysis reasoning
    confidence: float = 0.0  # Confidence (0-1)
