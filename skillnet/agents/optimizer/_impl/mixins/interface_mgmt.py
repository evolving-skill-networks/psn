"""
InterfaceManagementMixin - Interface change analysis and caller updates.

Extracted from optimizer_impl.py for better modularity.
Contains methods for analyzing interface changes and updating callers.

Methods included:
- _analyze_interface_impact: Analyze interface impact on callers
- _parse_function_params: Parse function parameters
- _analyze_param_changes: Analyze parameter changes
- _classify_interface_change: Classify interface change type
- _detect_options_object_change: Detect options object changes
- _determine_impact_and_strategy: Determine impact and update strategy
- _llm_analyze_interface_change: LLM analyze interface change
- on_skill_code_changed: Code change callback
- _generate_update_notification: Generate update notification
- _batch_update_callers_for_interface_change: Batch update callers
- _update_caller_for_interface_change: Update single caller
- _topological_sort_callers: Topological sort callers
- _get_call_depth: Get call depth
"""

import re
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from langchain.schema import HumanMessage, SystemMessage

# Delegate imports
from skillnet.agents.optimizer.analysis import (
    parse_function_params as _parse_function_params_impl,
    analyze_param_changes as _analyze_param_changes_impl,
    classify_interface_change as _classify_interface_change_impl,
    detect_options_object_change as _detect_options_object_change_impl,
    determine_impact_and_strategy as _determine_impact_and_strategy_impl,
    detect_semantic_changes as _detect_semantic_changes_impl,
    SemanticChangeResult,
)

from skillnet.agents.optimizer.core import robust_json_parse

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer

logger = logging.getLogger(__name__)


class InterfaceManagementMixin:
    """
    Interface Management Mixin - Interface change analysis and caller updates.

    Requires self attributes:
    - self.skill_graph_manager: SkillGraphManager instance
    - self.llm: LLM instance
    - self.logger: Logger instance
    - self._llm_invoker: LLMInvoker instance
    """

    def _analyze_interface_impact(
        self: "SkillGraphOptimizer",
        old_code: str,
        new_code: str,
        skill_name: str = None,
        use_llm_for_complex: bool = True,
        old_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        new_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze interface impact (impact on callers).

        Enhanced version: not only compares signature strings, but also analyzes parameter semantic changes.

        Args:
            old_code: Old code
            new_code: New code
            skill_name: Skill name (for logging)
            use_llm_for_complex: Whether to use LLM for complex cases
            old_metadata: Old parameter metadata {param_name: {type, semantic, ...}}
            new_metadata: New parameter metadata {param_name: {type, semantic, ...}}

        Returns:
            Dict[str, Any]: Interface impact analysis
        """
        # 1. Extract function signature
        old_signature_match = re.search(r'async\s+function\s+(\w+)\s*\(([^)]*)\)', old_code)
        new_signature_match = re.search(r'async\s+function\s+(\w+)\s*\(([^)]*)\)', new_code)

        old_signature = old_signature_match.group(0) if old_signature_match else ""
        new_signature = new_signature_match.group(0) if new_signature_match else ""

        # 2. Parse parameter list
        old_params = self._parse_function_params(old_signature_match.group(2) if old_signature_match else "")
        new_params = self._parse_function_params(new_signature_match.group(2) if new_signature_match else "")

        # 3. Analyze parameter changes
        param_changes = self._analyze_param_changes(old_params, new_params)

        # 4. Determine change type
        change_type = self._classify_interface_change(old_params, new_params, param_changes)

        # 5. Detect options object structure changes (even if signature is the same)
        options_change = self._detect_options_object_change(old_code, new_code, old_params, new_params)
        if options_change.get("has_change") and change_type == "none":
            change_type = "structure_change"
            param_changes["options_object_change"] = options_change

        # 6. Detect semantic changes (e.g., TARGET_TOTAL ↔ DELTA)
        semantic_changes: Optional[SemanticChangeResult] = None
        if old_metadata is not None and new_metadata is not None:
            semantic_changes = _detect_semantic_changes_impl(old_metadata, new_metadata)
            if semantic_changes.has_breaking_change and change_type == "none":
                change_type = "semantic_change"

        # 7. Determine if there is interface change
        has_interface_change = change_type != "none"

        # 8. Determine impact level and update strategy
        impact_level, update_strategy = self._determine_impact_and_strategy(change_type, param_changes)

        # If semantic breaking change, escalate impact level
        if semantic_changes and semantic_changes.has_breaking_change:
            impact_level = "high"
            update_strategy = "semi_auto"  # Require LLM to understand semantic changes

        # 9. Use LLM for complex cases
        llm_analysis = None
        if use_llm_for_complex and change_type == "complex":
            llm_analysis = self._llm_analyze_interface_change(old_code, new_code, skill_name)
            if llm_analysis:
                if llm_analysis.get("change_type"):
                    change_type = llm_analysis["change_type"]
                if llm_analysis.get("update_strategy"):
                    update_strategy = llm_analysis["update_strategy"]

        result = {
            "has_interface_change": has_interface_change,
            "old_signature": old_signature,
            "new_signature": new_signature,
            "change_type": change_type,
            "old_params": old_params,
            "new_params": new_params,
            "param_changes": param_changes,
            "impact_level": impact_level,
            "update_strategy": update_strategy,
        }

        if llm_analysis:
            result["llm_analysis"] = llm_analysis

        # Add semantic changes to result
        if semantic_changes:
            result["semantic_changes"] = semantic_changes

        # Log output
        if has_interface_change and skill_name:
            print(f"\033[33m[Interface Analysis] {skill_name}: Interface change detected\033[0m")
            print(f"  Change type: {change_type}")
            print(f"  Impact level: {impact_level}")
            print(f"  Update strategy: {update_strategy}")
            if semantic_changes and semantic_changes.has_breaking_change:
                print(f"  \033[31mSemantic breaking changes: {len(semantic_changes.breaking_changes)}\033[0m")
                for sc in semantic_changes.breaking_changes:
                    print(f"    - {sc}")

        return result

    def _parse_function_params(self: "SkillGraphOptimizer", params_str: str) -> List[Dict[str, Any]]:
        """Delegate to analysis.interface_analyzer.parse_function_params"""
        return _parse_function_params_impl(params_str)

    def _analyze_param_changes(
        self: "SkillGraphOptimizer",
        old_params: List[Dict[str, Any]],
        new_params: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Delegate to analysis.interface_analyzer.analyze_param_changes"""
        return _analyze_param_changes_impl(old_params, new_params)

    def _classify_interface_change(
        self: "SkillGraphOptimizer",
        old_params: List[Dict[str, Any]],
        new_params: List[Dict[str, Any]],
        param_changes: Dict[str, Any],
    ) -> str:
        """Delegate to analysis.interface_analyzer.classify_interface_change"""
        return _classify_interface_change_impl(old_params, new_params, param_changes)

    def _detect_options_object_change(
        self: "SkillGraphOptimizer",
        old_code: str,
        new_code: str,
        old_params: List[Dict[str, Any]],
        new_params: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Delegate to analysis.interface_analyzer.detect_options_object_change"""
        return _detect_options_object_change_impl(old_code, new_code, old_params, new_params)

    def _determine_impact_and_strategy(
        self: "SkillGraphOptimizer",
        change_type: str,
        param_changes: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Delegate to analysis.interface_analyzer.determine_impact_and_strategy"""
        return _determine_impact_and_strategy_impl(change_type, param_changes)

    def _llm_analyze_interface_change(
        self: "SkillGraphOptimizer",
        old_code: str,
        new_code: str,
        skill_name: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to deeply analyze complex interface changes.
        """
        try:
            from skillnet.agents.optimizer.prompts import OptimizerPromptLoader
            template = OptimizerPromptLoader.load("interface_change_analysis")
            system_prompt, human_prompt = template.format(
                old_code=old_code[:2000],
                new_code=new_code[:2000],
                skill_name=skill_name or "unknown",
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]

            response = self._invoke_llm_with_stats(
                messages=messages,
                process_type="analyze",
                function_name="_llm_analyze_interface_change",
                task=None,
                skill_name=skill_name,
            )

            return robust_json_parse(response.content)

        except Exception as e:
            self.logger.warning(f"[Interface Analysis] LLM analysis failed: {e}")
            return None

    def on_skill_code_changed(
        self: "SkillGraphOptimizer",
        skill_name: str,
        old_code: str,
        new_code: str,
        change_source: str = "unknown",
        auto_update_callers: bool = True,
        old_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        new_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Callback when skill code changes.

        Analyzes interface impact and updates callers if necessary.

        Args:
            skill_name: Skill name
            old_code: Old code
            new_code: New code
            change_source: Change source (optimize, refactor, manual, etc.)
            auto_update_callers: Whether to auto-update callers when interface changes
            old_metadata: Old parameter metadata for semantic change detection
            new_metadata: New parameter metadata for semantic change detection

        Returns:
            Dict[str, Any]: Update result summary
        """
        result = {
            "skill_name": skill_name,
            "change_source": change_source,
            "interface_changed": False,
            "callers_updated": [],
            "callers_failed": [],
            "notifications": [],
        }

        # 1. Analyze interface impact (including semantic changes if metadata provided)
        interface_impact = self._analyze_interface_impact(
            old_code, new_code, skill_name,
            old_metadata=old_metadata,
            new_metadata=new_metadata,
        )

        if not interface_impact.get("has_interface_change"):
            return result

        result["interface_changed"] = True
        result["interface_impact"] = interface_impact

        # 2. Get all callers
        callers = self.skill_graph_manager.get_parents(skill_name)
        if not callers:
            return result

        # 3. Decide update strategy
        update_strategy = interface_impact.get("update_strategy", "manual")
        impact_level = interface_impact.get("impact_level", "low")

        # 4. Generate notifications
        for caller in callers:
            notification = self._generate_update_notification(
                caller_skill=caller,
                callee_skill=skill_name,
                interface_impact=interface_impact,
            )
            result["notifications"].append(notification)

        # 5. Auto-update if strategy allows and auto_update_callers is enabled
        if auto_update_callers and update_strategy in ["auto", "semi_auto"] and impact_level != "high":
            updated, failed = self._batch_update_callers_for_interface_change(
                callers=callers,
                callee_skill=skill_name,
                interface_impact=interface_impact,
            )
            result["callers_updated"] = updated
            result["callers_failed"] = failed

        return result

    def _generate_update_notification(
        self: "SkillGraphOptimizer",
        caller_skill: str,
        callee_skill: str,
        interface_impact: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate update notification for a caller.

        Args:
            caller_skill: Caller skill name
            callee_skill: Callee skill name (interface changed)
            interface_impact: Interface impact analysis

        Returns:
            Dict[str, Any]: Notification info
        """
        change_type = interface_impact.get("change_type", "unknown")
        old_signature = interface_impact.get("old_signature", "")
        new_signature = interface_impact.get("new_signature", "")
        param_changes = interface_impact.get("param_changes", {})

        message = f"Interface of '{callee_skill}' has changed."

        if change_type == "count_change":
            added = param_changes.get("added", [])
            removed = param_changes.get("removed", [])
            if added:
                message += f" Added parameters: {', '.join(added)}."
            if removed:
                message += f" Removed parameters: {', '.join(removed)}."

        elif change_type == "rename":
            renamed = param_changes.get("renamed", [])
            if renamed:
                renamed_strs = [f"{r['old']} -> {r['new']}" for r in renamed]
                message += f" Renamed parameters: {', '.join(renamed_strs)}."

        elif change_type == "type_change":
            type_changes = param_changes.get("type_changed", [])
            if type_changes:
                message += f" Type changes: {len(type_changes)} parameter(s)."

        elif change_type == "semantic_change":
            semantic_changes = interface_impact.get("semantic_changes")
            if semantic_changes and semantic_changes.has_breaking_change:
                message += " BREAKING SEMANTIC CHANGES:"
                for sc in semantic_changes.breaking_changes:
                    message += f" {sc}."

        notification = {
            "caller_skill": caller_skill,
            "callee_skill": callee_skill,
            "change_type": change_type,
            "old_signature": old_signature,
            "new_signature": new_signature,
            "message": message,
            "impact_level": interface_impact.get("impact_level", "unknown"),
            "update_strategy": interface_impact.get("update_strategy", "manual"),
        }

        # Include semantic changes in notification for downstream processing
        semantic_changes = interface_impact.get("semantic_changes")
        if semantic_changes:
            notification["semantic_changes"] = semantic_changes.to_dict()

        return notification

    def _batch_update_callers_for_interface_change(
        self: "SkillGraphOptimizer",
        callers: List[str],
        callee_skill: str,
        interface_impact: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        """
        Batch update callers for interface change.

        Args:
            callers: List of caller skill names
            callee_skill: Callee skill name
            interface_impact: Interface impact analysis

        Returns:
            Tuple[List[str], List[str]]: (updated callers, failed callers)
        """
        updated = []
        failed = []

        # Sort callers by call depth (update closest callers first)
        sorted_callers = self._topological_sort_callers(callers, callee_skill)

        old_signature = interface_impact.get("old_signature", "")
        new_signature = interface_impact.get("new_signature", "")
        semantic_changes = interface_impact.get("semantic_changes")

        callee_node = self.skill_graph_manager.get_node(callee_skill)
        old_code = callee_node.code if callee_node else ""
        new_code = old_code  # Current code is the new code

        for caller in sorted_callers:
            success = self._update_caller_for_interface_change(
                caller_skill_name=caller,
                callee_skill_name=callee_skill,
                old_signature=old_signature,
                new_signature=new_signature,
                old_code=old_code,
                new_code=new_code,
                semantic_changes=semantic_changes,
            )

            if success:
                updated.append(caller)
            else:
                failed.append(caller)

        return updated, failed

    def _topological_sort_callers(
        self: "SkillGraphOptimizer",
        callers: List[str],
        target: str,
    ) -> List[str]:
        """
        Topological sort callers by call depth.

        Update closest callers first.

        Args:
            callers: List of caller skill names
            target: Target skill name

        Returns:
            List[str]: Sorted callers (closest first)
        """
        # Calculate depth for each caller
        caller_depths = []
        for caller in callers:
            depth = self._get_call_depth(caller, target)
            caller_depths.append((caller, depth))

        # Sort by depth (ascending)
        sorted_callers = [c[0] for c in sorted(caller_depths, key=lambda x: x[1])]

        return sorted_callers

    def _get_call_depth(
        self: "SkillGraphOptimizer",
        caller: str,
        target: str,
        max_depth: int = 10,
    ) -> int:
        """
        Get call depth from caller to target skill.

        Uses BFS to find shortest path.

        Args:
            caller: Caller skill name
            target: Target skill name
            max_depth: Maximum search depth

        Returns:
            int: Call depth (1 = direct call, 2 = one indirect level, etc.)
                 Returns max_depth + 1 if no call relationship found
        """
        # Direct call case
        children = self.skill_graph_manager.get_children(caller)
        if target in children:
            return 1

        # BFS for shortest path
        visited = {caller}
        queue = deque([(caller, 0)])

        while queue:
            current, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for child in self.skill_graph_manager.get_children(current):
                if child == target:
                    return depth + 1

                if child not in visited:
                    visited.add(child)
                    queue.append((child, depth + 1))

        # No call relationship found
        return max_depth + 1

    def _validate_caller_update(
        self: "SkillGraphOptimizer",
        updated_code: str,
        callee_name: str,
        new_signature: str,
    ) -> Optional[str]:
        """
        Validate that the generated caller code has correct parameter order.

        This catches a common LLM error where parameters are mapped by position
        instead of by semantic meaning, leading to issues like passing an object
        where a string is expected.

        Args:
            updated_code: The LLM-generated updated caller code
            callee_name: The callee function name
            new_signature: The new function signature (e.g., "(bot, resourceName, options)")

        Returns:
            Optional[str]: Error message if validation fails, None if OK
        """
        import re

        # Parse the new signature to understand expected parameter types
        # Example: "(bot, resourceName, options)" -> we expect resourceName to be a string
        sig_match = re.search(r'\(([^)]+)\)', new_signature)
        if not sig_match:
            return None  # Can't parse signature, skip validation

        params = [p.strip() for p in sig_match.group(1).split(',')]

        # Look for common patterns indicating parameter order errors
        # Pattern: callee_name(arg1, arg2, arg3)
        call_pattern = rf'{re.escape(callee_name)}\s*\(\s*([^)]+)\)'

        for match in re.finditer(call_pattern, updated_code):
            args_str = match.group(1)
            # Simple split by comma (this is a heuristic, not a full parser)
            args = [a.strip() for a in args_str.split(',')]

            # Check if we have enough args
            if len(args) < 2:
                continue

            # If new signature has a string-like parameter (e.g., "resourceName")
            # at position 1 (0-indexed), check if the second arg looks like an object
            if len(params) >= 2:
                param_name = params[1].lower()
                second_arg = args[1] if len(args) > 1 else ""

                # Heuristic: if param name suggests a string (name, type, resource, etc.)
                # but arg starts with { or is a known object variable, flag as error
                string_param_hints = ['name', 'type', 'resource', 'item', 'id', 'key']
                is_string_param = any(hint in param_name for hint in string_param_hints)

                if is_string_param:
                    # Check if the argument is an object literal or object-like
                    if second_arg.startswith('{'):
                        return (
                            f"Parameter order error detected: '{second_arg[:30]}...' passed as "
                            f"'{params[1]}' which expects a string. The options object should be "
                            f"the third parameter."
                        )
                    # Check for common object variable names
                    object_hints = ['options', 'opts', 'config', 'params', 'args', 'settings']
                    if any(hint in second_arg.lower() for hint in object_hints):
                        if not (second_arg.startswith('"') or second_arg.startswith("'")):
                            return (
                                f"Possible parameter order error: '{second_arg}' passed as "
                                f"'{params[1]}' which typically expects a string, not an object variable."
                            )

        return None  # Validation passed

    def _update_caller_for_interface_change(
        self: "SkillGraphOptimizer",
        caller_skill_name: str,
        callee_skill_name: str,
        old_signature: str,
        new_signature: str,
        old_code: str,
        new_code: str,
        semantic_changes: Optional[SemanticChangeResult] = None,
    ) -> bool:
        """
        Update caller skill code to adapt to callee's interface change.

        Args:
            caller_skill_name: Caller skill name
            callee_skill_name: Callee skill name (interface changed)
            old_signature: Old function signature
            new_signature: New function signature
            old_code: Callee's old code
            new_code: Callee's new code
            semantic_changes: Semantic change detection result (TARGET_TOTAL ↔ DELTA, etc.)

        Returns:
            bool: Whether update was successful
        """
        if not self.skill_graph_manager.has_node(caller_skill_name):
            return False

        caller_node = self.skill_graph_manager.get_node(caller_skill_name)
        if not caller_node:
            return False

        caller_code = caller_node.code

        # Check if caller actually calls the callee
        if callee_skill_name not in caller_code:
            # Caller doesn't directly call callee, skip
            return True

        print(f"\033[36m[Interface Update] Updating {caller_skill_name} to adapt to {callee_skill_name} interface change\033[0m")

        # Build semantic change info for LLM prompt
        semantic_info = ""
        if semantic_changes and semantic_changes.has_breaking_change:
            semantic_info = "\n\nCRITICAL SEMANTIC CHANGES (MUST UPDATE CALLING LOGIC):\n"
            for change in semantic_changes.breaking_changes:
                if change.old_quantity and change.new_quantity:
                    if change.old_quantity.value == "target_total" and change.new_quantity.value == "delta":
                        semantic_info += f"""
- Parameter '{change.param_name}' meaning changed:
  - BEFORE: "{change.old_quantity.value}" - ensures bot has N items total
  - AFTER: "{change.new_quantity.value}" - produces N additional items
  - ACTION: Update the calling logic. If you previously passed a target count,
    you now need to calculate how many MORE items to produce.
"""
                    elif change.old_quantity.value == "delta" and change.new_quantity.value == "target_total":
                        semantic_info += f"""
- Parameter '{change.param_name}' meaning changed:
  - BEFORE: "{change.old_quantity.value}" - produces N additional items
  - AFTER: "{change.new_quantity.value}" - ensures bot has N items total
  - ACTION: Update the calling logic. If you previously passed an increment,
    you now need to pass the desired total count.
"""
                    else:
                        semantic_info += f"- Parameter '{change.param_name}': {change.old_quantity.value} → {change.new_quantity.value}\n"
                if change.old_direction and change.new_direction:
                    semantic_info += f"- Parameter '{change.param_name}' direction: {change.old_direction.value} → {change.new_direction.value}\n"

        # Use externalized prompt
        from skillnet.agents.optimizer.prompts import OptimizerPromptLoader
        template = OptimizerPromptLoader.load("caller_update")
        system_prompt, human_prompt = template.format(
            caller_skill_name=caller_skill_name,
            caller_code=caller_code,
            callee_skill_name=callee_skill_name,
            old_signature=old_signature,
            new_signature=new_signature,
            old_code_snippet=old_code[:500] + "...",
            new_code_snippet=new_code[:500] + "...",
            semantic_changes=semantic_info,  # Pass semantic info to prompt
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            response = self._invoke_llm_with_stats(
                messages=messages,
                process_type="refactor",
                function_name="_update_caller_for_interface_change",
                task=None,
                skill_name=caller_skill_name,
                metadata={"callee_skill": callee_skill_name},
            )
            response_content = response.content

            # Extract updated code
            code_match = re.search(r'```(?:javascript|js)?\s*\n(.*?)\n```', response_content, re.DOTALL)
            if code_match:
                updated_code = code_match.group(1).strip()
            else:
                # If no code block, try extracting entire response
                updated_code = response_content.strip()

            if not updated_code or updated_code == caller_code:
                print(f"\033[33m[Interface Update] Code unchanged or extraction failed, skipping update\033[0m")
                return False

            # Validate the generated call - check for common errors like wrong parameter order
            validation_error = self._validate_caller_update(
                updated_code, callee_skill_name, new_signature
            )
            if validation_error:
                print(f"\033[33m[Interface Update] Validation failed for {caller_skill_name}: {validation_error}\033[0m")
                print(f"\033[33m[Interface Update] Keeping original code to avoid introducing errors\033[0m")
                return False

            # Apply update (create new version)
            success = self.skill_graph_manager.create_new_version(
                skill_name=caller_skill_name,
                new_code=updated_code,
                change_log=f"Updated to adapt to interface change in {callee_skill_name}",
            )

            if success:
                # Update programs code (for subsequent execution)
                self.skill_graph_manager.programs[caller_skill_name] = updated_code
                print(f"\033[32m[Interface Update] Successfully updated {caller_skill_name}\033[0m")

            return success

        except Exception as e:
            print(f"\033[31m[Interface Update] Update failed: {e}\033[0m")
            return False
