"""
Interface Analyzer

Interface analysis module: provides pure-function-style function-signature parsing and
interface-change analysis. Used to detect interface changes after a skill's code is
optimized, helping decide whether callers need to be updated.
"""

import re
from enum import Enum
from typing import Dict, List, Any, Tuple, Optional, TypedDict


class InterfaceChangeType(str, Enum):
    """Interface change type"""
    NONE = "none"                       # No change
    RENAME = "rename"                   # Only parameter rename
    COUNT_CHANGE = "count_change"       # Parameter count changed (only added optional parameters)
    STRUCTURE_CHANGE = "structure_change"  # Structural change (callers must be updated)
    COMPLEX = "complex"                 # Complex change


class ParamInfo(TypedDict):
    """Parsed parameter info"""
    name: str                   # parameter name
    default: Optional[str]      # default value (if any)
    is_optional: bool           # whether optional (has a default)
    is_destructured: bool       # whether this is a destructured parameter { ... }
    raw: str                    # raw string


class ParamChanges(TypedDict):
    """Parameter-change analysis result"""
    added: List[str]            # added parameter names
    removed: List[str]          # removed parameter names
    renamed: List[Dict[str, Any]]  # rename info [{"old": ..., "new": ..., "position": ...}]
    count_changed: bool         # whether parameter count changed
    destructured_change: bool   # whether the destructured parameter changed
    positional_to_options: bool # whether positional parameters became an options object
    old_count: int              # old parameter count
    new_count: int              # new parameter count


class OptionsObjectChange(TypedDict, total=False):
    """Options-object change analysis result"""
    has_change: bool
    added_fields: List[str]
    removed_fields: List[str]
    old_fields: List[str]
    new_fields: List[str]
    param_destructure_changed: Dict[str, str]


class FunctionSignature(TypedDict):
    """Function-signature info"""
    name: str
    params: List[str]           # simplified parameter-name list
    params_detailed: List[ParamInfo]  # detailed parameter info


def parse_function_params(params_str: str) -> List[ParamInfo]:
    """
    Parse a function-parameter string into a structured format.

    Args:
        params_str: parameter string, e.g. "bot, count = 1, options = {}"

    Returns:
        List[ParamInfo]: parameter list where each entry contains name, default, is_optional, is_destructured, raw

    Example:
        >>> parse_function_params("bot, count = 1, { blocks, radius }")
        [
            {"name": "bot", "default": None, "is_optional": False, "is_destructured": False, "raw": "bot"},
            {"name": "count", "default": "1", "is_optional": True, "is_destructured": False, "raw": "count = 1"},
            {"name": "{ blocks, radius }", "default": None, "is_optional": False, "is_destructured": True, "raw": "{ blocks, radius }"},
        ]
    """
    params: List[ParamInfo] = []
    if not params_str or not params_str.strip():
        return params

    # Handle destructured parameters such as { blocks, count, radius }
    # Simple splitting; nested complex cases are not handled
    parts = []
    depth = 0
    current = ""
    for char in params_str:
        if char in "({[":
            depth += 1
            current += char
        elif char in ")}]":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        param_info: ParamInfo = {
            "name": "",
            "default": None,
            "is_optional": False,
            "is_destructured": False,
            "raw": part,
        }

        # Check for a default value
        if "=" in part:
            name_part, default_part = part.split("=", 1)
            param_info["name"] = name_part.strip()
            param_info["default"] = default_part.strip()
            param_info["is_optional"] = True
        else:
            param_info["name"] = part

        # Check whether this is a destructured parameter
        if param_info["name"].startswith("{") or param_info["name"].startswith("["):
            param_info["is_destructured"] = True

        params.append(param_info)

    return params


def analyze_param_changes(
    old_params: List[ParamInfo],
    new_params: List[ParamInfo],
) -> ParamChanges:
    """
    Analyze the differences between old and new parameters.

    Args:
        old_params: old parameter list (from parse_function_params)
        new_params: new parameter list

    Returns:
        ParamChanges: contains added, removed, renamed, count_changed, etc.
    """
    old_names = [p["name"] for p in old_params]
    new_names = [p["name"] for p in new_params]

    # Exclude the bot parameter (typically the first and unchanged)
    old_names_no_bot = [n for n in old_names if n != "bot"]
    new_names_no_bot = [n for n in new_names if n != "bot"]

    added = [n for n in new_names_no_bot if n not in old_names_no_bot]
    removed = [n for n in old_names_no_bot if n not in new_names_no_bot]

    # Detect possible renames (based on position)
    renamed: List[Dict[str, Any]] = []
    if len(added) == len(removed) == 1:
        # Likely a rename
        renamed.append({
            "old": removed[0],
            "new": added[0],
            "position": old_names_no_bot.index(removed[0]) if removed[0] in old_names_no_bot else -1,
        })

    # Detect changes in destructured parameters
    old_destructured = [p for p in old_params if p.get("is_destructured")]
    new_destructured = [p for p in new_params if p.get("is_destructured")]

    destructured_change = len(old_destructured) != len(new_destructured)

    # Detect transition from positional parameters to an options object
    positional_to_options = False
    if len(old_params) > 2 and len(new_params) == 2:  # bot + multiple params -> bot + options
        # Check whether the new parameter looks like an options object
        if new_params and not new_params[-1].get("is_destructured"):
            last_new = new_params[-1]["name"]
            if last_new in ["options", "opts", "config", "params"] or last_new.startswith("{"):
                positional_to_options = True

    return {
        "added": added,
        "removed": removed,
        "renamed": renamed,
        "count_changed": len(old_params) != len(new_params),
        "destructured_change": destructured_change,
        "positional_to_options": positional_to_options,
        "old_count": len(old_params),
        "new_count": len(new_params),
    }


def classify_interface_change(
    old_params: List[ParamInfo],
    new_params: List[ParamInfo],
    param_changes: ParamChanges,
) -> InterfaceChangeType:
    """
    Classify the interface change type.

    Args:
        old_params: old parameter list
        new_params: new parameter list
        param_changes: parameter-change analysis result

    Returns:
        InterfaceChangeType: change-type enum value
    """
    # No change
    if not param_changes.get("added") and not param_changes.get("removed") and not param_changes.get("count_changed"):
        return InterfaceChangeType.NONE

    # Positional -> options object
    if param_changes.get("positional_to_options"):
        return InterfaceChangeType.STRUCTURE_CHANGE

    # Destructured-parameter change
    if param_changes.get("destructured_change"):
        return InterfaceChangeType.STRUCTURE_CHANGE

    # Rename only
    if param_changes.get("renamed") and not param_changes.get("count_changed"):
        return InterfaceChangeType.RENAME

    # Parameter count changed
    if param_changes.get("count_changed"):
        # If only optional parameters were added
        new_optional_only = all(
            any(p["name"] == added and p.get("is_optional") for p in new_params)
            for added in param_changes.get("added", [])
        )
        if new_optional_only and not param_changes.get("removed"):
            return InterfaceChangeType.COUNT_CHANGE  # relatively safe change
        return InterfaceChangeType.STRUCTURE_CHANGE  # may require caller updates

    # Complex change
    if len(param_changes.get("added", [])) > 1 or len(param_changes.get("removed", [])) > 1:
        return InterfaceChangeType.COMPLEX

    return InterfaceChangeType.NONE


def detect_options_object_change(
    old_code: str,
    new_code: str,
    old_params: List[ParamInfo],
    new_params: List[ParamInfo],
) -> OptionsObjectChange:
    """
    Detect changes inside the options-object structure.

    Even when the function signature is the same (e.g., function foo(bot, options)),
    the way the options object is destructured may have changed.

    Args:
        old_code: old code
        new_code: new code
        old_params: old parameter list
        new_params: new parameter list

    Returns:
        OptionsObjectChange: contains has_change, added_fields, removed_fields, etc.
    """
    result: OptionsObjectChange = {"has_change": False}

    # Look for patterns that destructure options
    # e.g.: const { blocks, count } = options;
    # or:   const { blocks, count, radius } = options;
    old_destructure_match = re.search(
        r'const\s*\{([^}]+)\}\s*=\s*(?:options|opts|config|params)\s*;?',
        old_code
    )
    new_destructure_match = re.search(
        r'const\s*\{([^}]+)\}\s*=\s*(?:options|opts|config|params)\s*;?',
        new_code
    )

    if old_destructure_match and new_destructure_match:
        old_fields = set(f.strip().split("=")[0].strip() for f in old_destructure_match.group(1).split(","))
        new_fields = set(f.strip().split("=")[0].strip() for f in new_destructure_match.group(1).split(","))

        added_fields = new_fields - old_fields
        removed_fields = old_fields - new_fields

        if added_fields or removed_fields:
            result["has_change"] = True
            result["added_fields"] = list(added_fields)
            result["removed_fields"] = list(removed_fields)
            result["old_fields"] = list(old_fields)
            result["new_fields"] = list(new_fields)

    # Check destructuring within the function parameters
    # e.g., async function foo(bot, { blocks, count })
    for old_p, new_p in zip(old_params, new_params):
        if old_p.get("is_destructured") and new_p.get("is_destructured"):
            old_raw = old_p.get("raw", "")
            new_raw = new_p.get("raw", "")
            if old_raw != new_raw:
                result["has_change"] = True
                result["param_destructure_changed"] = {
                    "old": old_raw,
                    "new": new_raw,
                }

    return result


def determine_impact_and_strategy(
    change_type: InterfaceChangeType,
    param_changes: ParamChanges,
) -> Tuple[str, str]:
    """
    Determine impact level and update strategy

    Args:
        change_type: interface change type
        param_changes: parameter-change analysis result

    Returns:
        Tuple[str, str]: (impact_level, update_strategy)
            - impact_level: "low", "medium", "high"
            - update_strategy: "no_update_needed", "simple_rename", "check_defaults",
                               "add_missing_params", "convert_to_options_object",
                               "llm_rewrite", "manual_review"
    """
    # Handle both string and enum inputs
    change_type_str = change_type.value if isinstance(change_type, InterfaceChangeType) else change_type

    if change_type_str == InterfaceChangeType.NONE.value:
        return "low", "no_update_needed"

    if change_type_str == InterfaceChangeType.RENAME.value:
        return "low", "simple_rename"

    if change_type_str == InterfaceChangeType.COUNT_CHANGE.value:
        if not param_changes.get("removed"):
            # Only parameters were added; callers likely do not need changes
            return "low", "check_defaults"
        return "medium", "add_missing_params"

    if change_type_str == InterfaceChangeType.STRUCTURE_CHANGE.value:
        if param_changes.get("positional_to_options"):
            return "high", "convert_to_options_object"
        return "high", "llm_rewrite"

    if change_type_str == InterfaceChangeType.COMPLEX.value:
        return "high", "llm_rewrite"

    return "medium", "manual_review"

