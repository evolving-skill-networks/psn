"""
EditAnalysisMixin - Feedback-to-edit analysis for optimization.

Extracted from optimizer_impl.py for better modularity.

Methods included:
- _analyze_feedback_for_edits: Pattern-match feedback to suggested code edits
- _build_call_context_from_events: Build call context from skill execution events

Cross-mixin dependencies (resolved via MRO):
    OptimizationMixin: _detect_parameter_semantic_mismatch
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from skillnet.agents.optimizer.feedback.types import CodeEdit, SkillFeedback

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer


class EditAnalysisMixin:
    """Edit Analysis Mixin - Analyze feedback to produce edit suggestions.

    Attributes (from SkillGraphOptimizer):
        skill_graph_manager: SkillGraphManager instance (via getattr fallback)
    """

    def _build_call_context_from_events(
        self: "SkillGraphOptimizer",
        skill_name: str,
        skill_executions: Dict[str, Any],
        chat_log: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Build call_context from skill execution events

        Used by parameter semantic mismatch detection, which needs to know:
        - Which skill called the current skill (caller_skill)
        - What parameters were passed (passed_params)

        Args:
            skill_name: name of the skill currently being optimized
            skill_executions: skill execution info parsed by psn.py
            chat_log: execution log (used for context analysis)

        Returns:
            call_context dict or None
        """
        if not skill_executions:
            return None

        # Look up the execution info for skill_name
        exec_info = skill_executions.get(skill_name)
        if not exec_info:
            return None

        call_stack = exec_info.get("call_stack", [])
        js_args = exec_info.get("js_args")  # JSON string or parsed object

        # Only nested calls have a caller
        if len(call_stack) <= 1:
            return None

        # Parse js_args
        passed_params = {}
        if js_args:
            try:
                if isinstance(js_args, str):
                    import json
                    passed_params = json.loads(js_args) if js_args else {}
                elif isinstance(js_args, (dict, list)):
                    passed_params = js_args
            except (json.JSONDecodeError, TypeError):
                passed_params = {}

        # Locate skill_name in call_stack
        try:
            skill_index = call_stack.index(skill_name)
            if skill_index > 0:
                return {
                    "caller_skill": call_stack[skill_index - 1],
                    "callee_skill": skill_name,
                    "passed_params": passed_params,
                    "chat_log": chat_log,
                    "call_stack": call_stack,
                }
        except ValueError:
            # skill_name not in call_stack, use the last two entries
            if len(call_stack) >= 2:
                return {
                    "caller_skill": call_stack[-2],
                    "callee_skill": call_stack[-1],
                    "passed_params": passed_params,
                    "chat_log": chat_log,
                    "call_stack": call_stack,
                }

        return None

    def _analyze_feedback_for_edits(
        self: "SkillGraphOptimizer",
        feedback: List[SkillFeedback] = None,
        current_error: str = None,
        current_critique: str = None,
        skill_name: str = None,              # current skill name
        skill_executions: Dict = None,       # skill execution info (includes call_stack)
    ) -> Dict[str, Any]:
        """Analyze feedback content and extract suggested edit operations (supports call-context analysis)."""
        result = {"suggested_edits": [], "issues": []}

        error_text = ""
        if current_error:
            error_text += current_error + "\n"
        if current_critique:
            error_text += current_critique + "\n"
        if feedback:
            for fb in feedback:
                error_text += fb.content + "\n"

        if not error_text:
            return result

        # Pattern matching for common issues
        patterns = [
            # TDZ-related
            (r"Cannot access.*before initialization", "structural", CodeEdit(
                edit_type="move_declaration_to_top",
                target="declarations",
                description="Fix variable declaration order"
            )),
            # Type errors
            (r"is not a function|is undefined|is null", "error_handling", CodeEdit(
                edit_type="add_null_check",
                target="function_calls",
                description="Add function existence check"
            )),
            # Argument errors
            (r"expected \d+ arguments|missing.*argument", "logic", CodeEdit(
                edit_type="add_parameter",
                target="function_params",
                description="Fix function parameters"
            )),
            # Insufficient resources - requires deeper analysis
            (r"not enough|insufficient|need more|Unable to obtain", "logic", CodeEdit(
                edit_type="fix_resource_logic",
                target="resource_logic",
                description="Check whether the resource-acquisition logic is correct"
            )),
            # Parameter semantic mismatch: child skill returns success but parent still fails
            (r"already have.*\(need|did not produce.*required|returned success.*but.*failed", "semantic_mismatch", CodeEdit(
                edit_type="fix_parameter_semantics",
                target="parameter_semantics",
                description="Fix parameter semantic mismatch: check whether the count parameter means 'additional quantity' or 'total quantity'"
            )),

            # ========== Resource/name related error patterns ==========
            # Block/item not found
            (r"Cannot resolve block.*|block.*not found|No.*block.*found", "resource_name", CodeEdit(
                edit_type="add_name_mapping",
                target="block_lookup",
                description="Add a name-mapping helper to automatically handle item/block name conversion"
            )),
            # Name/ID resolution error
            (r"name.*resolution.*failed|id.*mismatch|using.*instead of", "interface", CodeEdit(
                edit_type="fix_resource_logic",
                target="name_resolution",
                description="Fix name resolution: ensure the correct name type is used"
            )),

            # ========== Added: interface/argument related error patterns ==========
            # Child skill interface call problems
            (r"child.*failed.*interface|interface.*mismatch|wrong.*parameter", "interface", CodeEdit(
                edit_type="fix_interface_mismatch",
                target="child_call",
                description="Fix parent-child skill interface mismatch"
            )),
            # Extraneous argument passing
            (r"too many arguments|unexpected.*parameter|extra.*argument", "interface", CodeEdit(
                edit_type="fix_parameter_passing",
                target="function_call",
                description="Fix function call: remove redundant arguments"
            )),

            # ========== Added: item-collection related error patterns ==========
            # Item collection failed though digging succeeded
            (r"mining.*but.*inventory.*0|dug.*but.*not.*collected|Collect finish.*but.*0", "collection", CodeEdit(
                edit_type="add_collection_wait",
                target="dig_operation",
                description="Add wait-for-collection logic to ensure items are properly picked up"
            )),
            # Inventory check logic problem
            (r"inventory.*count.*wrong|count.*not.*updated|inventory.*not.*increase", "collection", CodeEdit(
                edit_type="fix_inventory_check",
                target="inventory_check",
                description="Fix inventory-check logic"
            )),

            # ========== Added: child-skill return-value check error patterns ==========
            # Failure caused by not checking child skill return value (extended match patterns)
            (r"does not check.*return value|ignores.*return|does not capture.*return|blindly awaits|"
             r"check the.*return value|should check.*return|not.*checking.*return|"
             r"return.*value.*not.*checked|need.*check.*return|must.*check.*return", "return_value", CodeEdit(
                edit_type="check_child_skill_return_value",
                target="",  # Auto-detect
                description="Add a child-skill return-value check; abort execution when the child skill fails"
            )),
            # Child skill failed but parent kept executing
            (r"child.*failed.*but.*continued|failed.*still.*proceeded|assumes.*success|"
             r"abort.*if.*false|abort.*if.*return.*false|"
             r"returned false.*but|returns false|if.*returns? false|"
             r"proceeds after.*fail|proceed.*after.*false", "return_value", CodeEdit(
                edit_type="check_child_skill_return_value",
                target="",
                description="Add a child-skill return-value check so execution does not continue when the child skill fails"
            )),
            # Need to verify the result of an operation
            (r"verify.*result|re-?check.*after|should.*validate.*before.*proceeding|"
             r"re-?verify.*inventory|check.*inventory.*after|verify.*after.*call", "verification", CodeEdit(
                edit_type="add_inventory_verification",
                target="",
                description="Add verification logic after calling a child skill"
            )),
        ]

        for pattern, issue_type, edit in patterns:
            if re.search(pattern, error_text, re.IGNORECASE):
                result["issues"].append((issue_type, pattern, {}))
                result["suggested_edits"].append(edit)

        # Domain-specific error patterns
        _dk = getattr(self, '_domain_knowledge', None)
        if _dk:
            for d_pattern, d_issue_type, d_description in _dk.get_resource_error_patterns():
                if re.search(d_pattern, error_text, re.IGNORECASE):
                    result["issues"].append((d_issue_type, d_pattern, {}))
                    result["suggested_edits"].append(CodeEdit(
                        edit_type="fix_resource_logic",
                        target="block_name",
                        description=d_description,
                    ))

        # Deep analysis: detect parameter semantic mismatch issues
        # Signature: feedback mentions "already have X (need Y)" with X >= Y but still ultimately fails
        # Pass in skill_graph_manager to enable deeper context analysis
        skill_graph = getattr(self, 'skill_graph_manager', None)

        # Build call_context from skill_executions
        call_context = None
        if skill_name and skill_executions:
            call_context = self._build_call_context_from_events(
                skill_name, skill_executions, chat_log=error_text
            )

        semantic_issue = self._detect_parameter_semantic_mismatch(
            error_text,
            feedback,
            call_context=call_context,  # pass the actual call context
            skill_graph=skill_graph
        )
        if semantic_issue:
            result["issues"].append(("semantic_mismatch", "parameter semantic mismatch", semantic_issue))
            result["suggested_edits"].append(CodeEdit(
                edit_type="rewrite_parameter_logic",
                target="count_parameter",
                description=f"Parameter semantic issue: {semantic_issue.get('description', 'requires LLM rewrite')}"
            ))

        return result
