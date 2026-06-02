"""
Pure helper functions and data classes for skill execution recording.

Extracted from SkillRecordingMixin._auto_record_skill_executions to enable
independent unit testing. All functions are near-pure: no self access, side
effects limited to diagnostic print() calls.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from skillnet._psn_impl.event_helpers import unpack_event


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SkillEventParseResult:
    """Structured output from parsing skill execution events."""
    skill_executions: Dict[str, dict] = field(default_factory=dict)
    error_message: Optional[str] = None
    environment_events: List[dict] = field(default_factory=list)
    skill_event_count: int = 0


@dataclass
class ExecutionClassification:
    """Classification of a single skill's execution outcome."""
    skill_success: bool
    skill_execution_status: str  # "success" | "failed" | "not_executed" | "interrupted"
    skill_error_message: Optional[str]
    skill_error_stack: Optional[str]
    is_primary_skill: bool
    effect_verification: Optional[dict] = None


# ---------------------------------------------------------------------------
# parse_skill_events: Section 3 (L85-189) of _auto_record_skill_executions
# ---------------------------------------------------------------------------

def parse_skill_events(
    events: list,
    find_nearby_blocks_fn: Callable[[list, int], list],
) -> SkillEventParseResult:
    """Parse raw events into structured skill execution data.

    Extracts skillStart/skillEnd/skillError events along with
    error messages and environment state.

    Args:
        events: Raw event list from environment execution.
        find_nearby_blocks_fn: Callback to extract nearby blocks from events
            (typically EventProcessingMixin._find_nearby_blocks_from_events,
            which is a @staticmethod).

    Returns:
        SkillEventParseResult with parsed data.
    """
    skill_executions = {}
    error_message = None
    environment_events = []
    skill_event_count = 0

    for i, event in enumerate(events):
        unpacked = unpack_event(event)
        if unpacked is None:
            continue
        event_type, event_data = unpacked

        if event_type == "skillStart":
            skill_name = event_data.get("skillName")
            if skill_name:
                skill_event_count += 1
                pre_state = None
                if isinstance(event_data, dict) and "state" in event_data:
                    state_data = event_data["state"]
                    nearby_blocks = find_nearby_blocks_fn(events, i)
                    pre_state = {
                        "inventory": state_data.get("inventory", {}),
                        "position": state_data.get("position", {}),
                        "equipment": state_data.get("equipment", {}),
                        "nearby_blocks": nearby_blocks,
                    }
                skill_executions[skill_name] = {
                    "start_time": event_data.get("timestamp"),
                    "call_stack": event_data.get("callStack", []),
                    "js_args": event_data.get("args"),
                    "status": "running",
                    "pre_state": pre_state,
                }
        elif event_type == "skillEnd":
            skill_name = event_data.get("skillName")
            if skill_name:
                skill_event_count += 1
                if skill_name not in skill_executions:
                    skill_executions[skill_name] = {}
                post_state = None
                if isinstance(event_data, dict) and "state" in event_data:
                    state_data = event_data["state"]
                    nearby_blocks = find_nearby_blocks_fn(events, i)
                    post_state = {
                        "inventory": state_data.get("inventory", {}),
                        "position": state_data.get("position", {}),
                        "equipment": state_data.get("equipment", {}),
                        "nearby_blocks": nearby_blocks,
                    }
                skill_executions[skill_name].update({
                    "end_time": event_data.get("timestamp"),
                    "duration": event_data.get("duration", 0),
                    "status": "success",
                    "success": True,
                    "post_state": post_state,
                })
        elif event_type == "skillError":
            skill_name = event_data.get("skillName")
            if skill_name:
                skill_event_count += 1
                if skill_name not in skill_executions:
                    skill_executions[skill_name] = {}
                post_state = None
                if isinstance(event_data, dict) and "state" in event_data:
                    state_data = event_data["state"]
                    nearby_blocks = find_nearby_blocks_fn(events, i)
                    post_state = {
                        "inventory": state_data.get("inventory", {}),
                        "position": state_data.get("position", {}),
                        "equipment": state_data.get("equipment", {}),
                        "nearby_blocks": nearby_blocks,
                    }
                skill_executions[skill_name].update({
                    "end_time": event_data.get("timestamp"),
                    "duration": event_data.get("duration", 0),
                    "status": "error",
                    "success": False,
                    "error_message": event_data.get("error", ""),
                    "error_stack": event_data.get("stack"),
                    "post_state": post_state,
                })
        elif event_type == "error":
            error_message = str(event_data.get("error", "")) if isinstance(event_data, dict) else str(event_data)
        elif event_type in ["onSave", "onChat", "onItemDrop", "onItemPickUp", "onBlockBreak", "onBlockPlace"]:
            environment_events.append({"type": event_type, "data": event_data})

    return SkillEventParseResult(
        skill_executions=skill_executions,
        error_message=error_message,
        environment_events=environment_events,
        skill_event_count=skill_event_count,
    )


# ---------------------------------------------------------------------------
# classify_execution_status: Section 4 (L237-387)
# ---------------------------------------------------------------------------

def classify_execution_status(
    skill_name: str,
    skill_exec_info: dict,
    main_function_name: Optional[str],
    overall_success: bool,
    error_message: Optional[str],
    critique: Optional[str],
    not_executed_skills: set,
    called_skills: list,
    check_effect_fn: Callable[[str, Optional[dict], Optional[dict]], Tuple[bool, str, dict]],
) -> ExecutionClassification:
    """Classify a single skill's execution outcome.

    Handles primary vs nested skills, code_success vs effect_achieved,
    syntax error detection, and not_executed classification.

    Note: This function does NOT mutate skill_exec_info. The effect_verification
    result is returned in the ExecutionClassification dataclass; the caller is
    responsible for writing it back if needed.

    Args:
        skill_name: Name of the skill being classified.
        skill_exec_info: Execution info dict from parse_skill_events (read-only).
        main_function_name: Name of the main/primary function.
        overall_success: Whether the overall task succeeded (from Critic).
        error_message: Global error message (from error events).
        critique: Critic feedback text.
        not_executed_skills: Set of skills in code but not executed at runtime.
        called_skills: List of skills detected by static analysis.
        check_effect_fn: Callback for effect verification
            (typically self._check_skill_effect_achieved).

    Returns:
        ExecutionClassification with all classification results.
    """
    is_primary_skill = (skill_name == main_function_name)

    if is_primary_skill:
        # Primary function: use Critic judgment
        skill_success = overall_success
        skill_execution_status = "success" if overall_success else "failed"
        skill_error_stack = skill_exec_info.get("error_stack") if "error_stack" in skill_exec_info else None

        if not overall_success:
            if skill_exec_info.get("error_message"):
                skill_error_message = skill_exec_info["error_message"]
                if skill_exec_info.get("error_stack"):
                    skill_error_message = f"{skill_error_message}\nStack: {skill_exec_info['error_stack']}"
            elif error_message:
                skill_error_message = error_message
            elif critique:
                skill_error_message = f"Task failed: {critique}"
            else:
                skill_error_message = "Task failed (no specific error message)"
        else:
            skill_error_message = None

        return ExecutionClassification(
            skill_success=skill_success,
            skill_execution_status=skill_execution_status,
            skill_error_message=skill_error_message,
            skill_error_stack=skill_error_stack,
            is_primary_skill=True,
        )

    elif "success" in skill_exec_info:
        # Has precise execution info (from runtime skill events)
        code_execution_success = skill_exec_info["success"]
        skill_error_message = skill_exec_info.get("error_message")
        skill_error_stack = skill_exec_info.get("error_stack")
        effect_verification = None

        if not code_execution_success:
            skill_success = False
            skill_execution_status = "failed"
        elif overall_success:
            skill_success = True
            skill_execution_status = "success"
        else:
            # Code succeeded but task failed — verify effect
            pre_state = skill_exec_info.get("pre_state")
            post_state = skill_exec_info.get("post_state")

            effect_achieved, effect_reason, state_changes = check_effect_fn(
                skill_name, pre_state, post_state
            )

            effect_verification = {
                "achieved": effect_achieved,
                "reason": effect_reason,
                "state_changes": state_changes,
            }

            if effect_achieved:
                skill_success = True
                skill_execution_status = "success"
                skill_error_message = None
                print(f"\033[36m[Skill Effect] {skill_name}: effects achieved ({effect_reason})), but overall task failed\033[0m")
            else:
                skill_success = False
                skill_execution_status = "failed"
                if not skill_error_message:
                    skill_error_message = f"Skill effect not achieved: {effect_reason}"
                print(f"\033[33m[Skill Effect] {skill_name}: effects not achieved ({effect_reason})\033[0m")

        return ExecutionClassification(
            skill_success=skill_success,
            skill_execution_status=skill_execution_status,
            skill_error_message=skill_error_message,
            skill_error_stack=skill_error_stack,
            is_primary_skill=False,
            effect_verification=effect_verification,
        )

    else:
        # No precise execution info — use heuristics
        skill_error_stack = None

        if skill_name in not_executed_skills:
            syntax_error_keywords = [
                "SyntaxError", "Unexpected token", "Unexpected end of input",
                "Unexpected identifier", "missing )", "missing }",
                "Invalid or unexpected token", "Uncaught SyntaxError"
            ]
            error_text = str(error_message) if error_message else ""
            critique_text = str(critique) if critique else ""
            combined_error = error_text + " " + critique_text

            has_syntax_error = any(keyword in combined_error for keyword in syntax_error_keywords)

            if has_syntax_error:
                skill_success = False
                skill_execution_status = "failed"
                skill_error_message = f"Code has syntax error: {error_message or critique}"
                print(f"\033[31m[Skill Tracking] {skill_name}: syntax error prevented execution\033[0m")
            else:
                skill_success = False
                skill_error_message = None
                skill_execution_status = "not_executed"
                print(f"\033[36m[Skill Tracking] {skill_name}: not executed (call exists in code but not triggered at runtime)\033[0m")
        elif "start_time" in skill_exec_info and "end_time" not in skill_exec_info:
            skill_success = False
            skill_error_message = "Skill execution was interrupted or did not complete"
            skill_execution_status = "interrupted"
        elif overall_success:
            skill_success = True
            skill_error_message = None
            skill_execution_status = "success"
        else:
            skill_success = False
            skill_execution_status = "failed"
            if error_message:
                if skill_name.lower() in error_message.lower():
                    skill_error_message = error_message
                else:
                    skill_error_message = f"Task failed (may be caused by other skills): {error_message}"
            else:
                skill_error_message = "Task failed (no specific error message)"

        return ExecutionClassification(
            skill_success=skill_success,
            skill_execution_status=skill_execution_status,
            skill_error_message=skill_error_message,
            skill_error_stack=skill_error_stack,
            is_primary_skill=False,
        )


# ---------------------------------------------------------------------------
# extract_trajectory_info: Section 2 (L71-83)
# ---------------------------------------------------------------------------

def extract_trajectory_info(
    trajectory_recorder: Any,
    task: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Extract trajectory file path and segment ID from recorder state.

    Returns:
        (trajectory_file, trajectory_segment_id) or (None, None).
    """
    trajectory_file = None
    trajectory_segment_id = None
    if trajectory_recorder.current_trajectory:
        if hasattr(trajectory_recorder, 'trajectory_id') and trajectory_recorder.trajectory_id:
            task_name = re.sub(r'[\\/:"*?<>| ]', "_", task or "unknown")
            task_name = task_name.replace(" ", "_")
            timestamp = trajectory_recorder.trajectory_id.split("_")[-1] if "_" in trajectory_recorder.trajectory_id else ""
            trajectory_file = f"{trajectory_recorder.ckpt_dir}/trajectories/{task_name}_{timestamp}.json"
        trajectory_segment_id = f"step_{trajectory_recorder.step_counter}"
    return trajectory_file, trajectory_segment_id


# ---------------------------------------------------------------------------
# build_action_info: Section 5 (L388-424)
# ---------------------------------------------------------------------------

def build_action_info(
    skill_name: str,
    is_nested_skill: bool,
    skill_exec_info: dict,
    full_code: str,
    parsed_result: dict,
    skill_manager: Any,
) -> dict:
    """Build the action_info dict for a skill execution record.

    Args:
        skill_name: Name of the skill.
        is_nested_skill: True if skill was detected via events but not in
            the static call list (nested/indirect call).
        skill_exec_info: Execution info from parse_skill_events.
        full_code: Full code string (program_code + exec_code).
        parsed_result: Parsed result dict with program_code, program_name.
        skill_manager: SkillGraphManager (read-only: has_node, get_node).

    Returns:
        action_info dict suitable for record_execution().
    """
    if is_nested_skill and skill_manager.has_node(skill_name):
        nested_node = skill_manager.get_node(skill_name)
        nested_code = nested_node.code if nested_node else ""
        action_info = {
            "code": nested_code,
            "program_name": skill_name,
            "program_code": nested_code,
        }
    else:
        action_info = {
            "code": full_code,
            "program_name": parsed_result.get("program_name", ""),
            "program_code": parsed_result.get("program_code", ""),
        }

    if "call_stack" in skill_exec_info:
        action_info["call_stack"] = skill_exec_info["call_stack"]
    if "js_args" in skill_exec_info:
        action_info["js_args"] = skill_exec_info["js_args"]
    if "duration" in skill_exec_info:
        action_info["execution_duration_ms"] = skill_exec_info["duration"]

    return action_info
