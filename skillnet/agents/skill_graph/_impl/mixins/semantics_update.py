"""
Semantics Update Mixin for SkillGraphManager

Runtime semantics learning engine: updates effects and preconditions
from execution data using Value Function evaluation.

Extracted from skill_execution.py for modularity.
Contains 14 methods:
  - _update_semantics_from_executions: Entry point for semantics update
  - _update_effects_from_executions: Update effects from execution data
  - _analyze_parameterized_effects: Analyze parameter-effect correlations
  - _calculate_correlation: Delegate to execution_analysis
  - _classify_effect_importance: Delegate to code_verifier
  - _parse_js_args_to_dict: Parse JS-serialized arguments
  - _parse_call_args_from_exec_code: Parse call args from exec code
  - _parse_js_arguments: Delegate to execution_analysis
  - _parse_js_value: Delegate to execution_analysis
  - _categorize_failure: Delegate to execution_analysis
  - _calculate_precondition_value: Delegate to execution module
  - _calculate_effect_value: Delegate to execution module
  - _verify_precondition_in_code: Delegate to code_verifier
  - _update_preconditions_from_executions: Update preconditions from execution data
"""

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from skillnet.agents.skill_graph.models import (
    SkillNode,
    SkillExecutionTrace,
    SkillEffect,
    SkillPrecondition,
    FailureCategory,
)
from skillnet.agents.skill_graph.execution import (
    calculate_precondition_value,
    calculate_effect_value,
)
from skillnet.agents.skill_graph.utils import (
    calculate_correlation,
    categorize_failure,
    parse_js_value,
    parse_js_arguments,
    verify_precondition_in_code,
    classify_effect_importance,
)
from skillnet.agents.skill_graph.models.config import (
    EffectValueFunctionConfig,
    PreconditionValueFunctionConfig,
)

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class SemanticsUpdateMixin:
    """Semantics Update Mixin - Runtime semantics learning from execution data.

    Methods:
        _update_semantics_from_executions: Entry point for semantics update
        _update_effects_from_executions: Update effects from execution data
        _analyze_parameterized_effects: Analyze parameter-effect correlations
        _calculate_correlation: Delegate to execution_analysis
        _classify_effect_importance: Delegate to code_verifier
        _parse_js_args_to_dict: Parse JS-serialized arguments
        _parse_call_args_from_exec_code: Parse call args from exec code
        _parse_js_arguments: Delegate to execution_analysis
        _parse_js_value: Delegate to execution_analysis
        _categorize_failure: Delegate to execution_analysis
        _calculate_precondition_value: Delegate to execution module
        _calculate_effect_value: Delegate to execution module
        _verify_precondition_in_code: Delegate to code_verifier
        _update_preconditions_from_executions: Update preconditions from execution data

    Attributes (from SkillGraphManager):
        graph: SkillGraph instance
        logger: Logger instance
    """

    def _update_semantics_from_executions(self: "SkillGraphManager", skill_name: str) -> None:
        """
        Online update of preconditions and effects based on execution data.

        Strategy:
        1. Find "frequent stable" effects by tracking net changes (s_after - s_before)
        2. Find preconditions by tracking which conditions in s_before are almost always satisfied in successful samples
        3. Update semantics_confidence (grows with the number of samples)

        Args:
            skill_name: skill name
        """
        node = self.graph.get_node(skill_name)
        if not node:
            return

        execution_traces = node.statistics.execution_traces
        if len(execution_traces) < 3:  # need at least 3 samples to start stats
            return

        # Separate successful and failed executions
        successful_traces = [t for t in execution_traces if t.success and t.pre_state and t.post_state]
        failed_traces = [t for t in execution_traces if not t.success and t.pre_state]

        if len(successful_traces) < 2:  # need at least 2 successful samples
            return

        # 1. Update effects: tally frequent stable state changes
        self._update_effects_from_executions(node, successful_traces)

        # 2. Update preconditions: tally conditions almost always satisfied in successful samples
        self._update_preconditions_from_executions(node, successful_traces, failed_traces)

        # 3. Update semantics_confidence based on sample count
        # Use the logistic function: confidence = 1 - exp(-sample_count / threshold)
        # When sample_count=10, confidence≈0.63; sample_count=30, confidence≈0.95
        sample_count = len(successful_traces)
        threshold = 20.0  # 20 samples yield ~0.63 confidence
        node.semantics_confidence = min(1.0, 1.0 - (2.718 ** (-sample_count / threshold)))

        print(f"\033[36m[Semantics Update] Skill '{skill_name}': {len(successful_traces)} successful samples, confidence={node.semantics_confidence:.2f}\033[0m")

    def _update_effects_from_executions(
        self: "SkillGraphManager",
        node: SkillNode,
        successful_traces: List[SkillExecutionTrace]
    ) -> None:
        """
        Update effects based on successful executions (Value Function approach).

        Strategy:
        1. Count occurrence frequency of each state change
        2. Use the Value Function to compute confidence for each candidate effect
        3. Distinguish core effects from incidental effects
        4. Decide whether to add based on confidence; record the confidence level

        Value Function formula:
        V(effect) = p_s - λ * uncertainty

        Args:
            node: skill node
            successful_traces: list of successful execution records
        """
        if not successful_traces:
            return

        # Get Value Function configuration
        config = EffectValueFunctionConfig()
        n_total = len(successful_traces)

        # Tally occurrence frequency of each state change
        effect_frequency: Dict[str, Dict[str, Any]] = {}

        for trace in successful_traces:
            if not trace.actual_effects:
                continue

            for actual_effect in trace.actual_effects:
                # Tally inventory changes
                if actual_effect.inventory_changes:
                    for item, change in actual_effect.inventory_changes.items():
                        if change != 0:
                            key = f"inventory:{item}:{change}"
                            if key not in effect_frequency:
                                effect_frequency[key] = {
                                    "count": 0,
                                    "item": item,
                                    "change": change,
                                    "type": "inventory",
                                    "total_change": 0,
                                }
                            effect_frequency[key]["count"] += 1
                            effect_frequency[key]["total_change"] += change

                # Tally position changes (if significant)
                if actual_effect.position_changes:
                    for axis, change in actual_effect.position_changes.items():
                        if abs(change) > 0.5:
                            key = f"position:{axis}"
                            if key not in effect_frequency:
                                effect_frequency[key] = {
                                    "count": 0,
                                    "axis": axis,
                                    "type": "position",
                                }
                            effect_frequency[key]["count"] += 1

                # Tally equipment changes
                if actual_effect.equipment_changes:
                    for slot, change_info in actual_effect.equipment_changes.items():
                        if change_info.get("after"):
                            key = f"equipment:{slot}:{change_info['after']}"
                            if key not in effect_frequency:
                                effect_frequency[key] = {
                                    "count": 0,
                                    "slot": slot,
                                    "item": change_info["after"],
                                    "type": "equipment",
                                }
                            effect_frequency[key]["count"] += 1

                # Tally block changes
                if actual_effect.block_changes:
                    for block_change in actual_effect.block_changes:
                        change_type = block_change.get("type", "")
                        if change_type:
                            key = f"block:{change_type}"
                            if key not in effect_frequency:
                                effect_frequency[key] = {
                                    "count": 0,
                                    "type": "block",
                                    "action": change_type,
                                }
                            effect_frequency[key]["count"] += 1

        print(f"\033[36m[Effect Inference] Evaluating {len(effect_frequency)} candidate effects via Value Function\033[0m")
        print(f"\033[36m[Effect Inference] Sample size: {n_total}\033[0m")

        # Use the Value Function to evaluate each candidate effect
        stable_effects = []

        for key, freq_info in effect_frequency.items():
            n_occurrences = freq_info["count"]

            # Classify effect importance
            importance = self._classify_effect_importance(key, freq_info, node.name, node.code)

            # Skip incidental effects
            if importance == "incidental":
                continue

            # Bug R1' fix: skip secondary inventory:add effects whose item is
            # already owned by a child skill. The runtime trace counts inventory
            # deltas at skill boundaries, including those caused by sub-skill
            # calls — without this filter, parents inherit children's produces
            # and the planner then treats the parent as a producer of items it
            # only handed off to children (e.g. craftPickaxe inheriting
            # ensureWoodLogs's "Adds oak_log to inventory").
            if importance == "secondary" and freq_info.get("type") == "inventory":
                item = freq_info.get("item")
                if (
                    item
                    and freq_info.get("total_change", 0) > 0
                    and self._item_produced_by_child(node.name, item)
                ):
                    print(
                        f"\033[90m[Effect Filter] Skipping '{key}': item '{item}' "
                        f"is owned by a child skill of '{node.name}' — not "
                        f"attributing to parent\033[0m"
                    )
                    continue

            # Compute Value Function
            value_result = self._calculate_effect_value(
                n_occurrences=n_occurrences,
                n_total=n_total,
                config=config,
            )

            # Print details
            if value_result["value"] > -0.3:
                print(f"\033[90m[Effect Value] {key}: value={value_result['value']:.3f} "
                      f"({value_result['confidence_level']}) "
                      f"[p_s={value_result['details']['p_s']:.2f}, "
                      f"u={value_result['details']['uncertainty']:.2f}]\033[0m")

            # Decide whether to add based on the Value Function
            if value_result["should_add"]:
                # Generate description and state_representation
                state_repr = {}
                avg_change = 0

                if freq_info["type"] == "inventory":
                    item = freq_info["item"]
                    avg_change = freq_info["total_change"] / freq_info["count"] if freq_info["count"] > 0 else 0
                    if avg_change > 0:
                        description = f"Adds {abs(avg_change):.1f} {item} to inventory"
                        state_repr = {
                            "type": "inventory",
                            "item": item,
                            "count": int(abs(avg_change)),
                            "operation": "add"
                        }
                    else:
                        description = f"Consumes {abs(avg_change):.1f} {item} from inventory"
                        state_repr = {
                            "type": "inventory",
                            "item": item,
                            "count": int(abs(avg_change)),
                            "operation": "consume"
                        }
                elif freq_info["type"] == "position":
                    description = f"Changes position on {freq_info['axis']} axis"
                    state_repr = {
                        "type": "position",
                        "axis": freq_info["axis"],
                        "operation": "change"
                    }
                elif freq_info["type"] == "equipment":
                    description = f"Equips {freq_info['item']} in {freq_info['slot']} slot"
                    state_repr = {
                        "type": "equipment",
                        "slot": freq_info["slot"],
                        "item": freq_info["item"],
                        "operation": "equip"
                    }
                elif freq_info["type"] == "block":
                    description = f"{freq_info['action'].capitalize()}s blocks"
                    state_repr = {
                        "type": "block",
                        "action": freq_info["action"],
                        "operation": freq_info["action"]
                    }
                else:
                    description = f"Effect: {key}"

                stable_effects.append({
                    "description": description,
                    "type": freq_info["type"],
                    "key": key,
                    "importance": importance,
                    "state_representation": state_repr,
                    "confidence_value": value_result["value"],
                    "confidence_level": value_result["confidence_level"],
                    "inference_stats": {
                        "n_occurrences": n_occurrences,
                        "n_total": n_total,
                        "avg_change": avg_change if freq_info["type"] == "inventory" else None,
                        "last_updated": datetime.now().isoformat(),
                    },
                })

        # Update expected_effects
        existing_effect_descriptions = {e.description.lower() for e in node.expected_effects}

        for stable_effect in stable_effects:
            effect_desc_lower = stable_effect["description"].lower()
            if not any(desc in effect_desc_lower or effect_desc_lower in desc for desc in existing_effect_descriptions):
                # Add new effect (with confidence info)
                # is_primary=False: effects learned at runtime are not primary by default
                # Primary effects should be marked by the LLM during code analysis
                new_effect = SkillEffect(
                    description=stable_effect["description"],
                    code="",
                    state_representation=stable_effect.get("state_representation", {}),
                    confidence_value=stable_effect["confidence_value"],
                    confidence_level=stable_effect["confidence_level"],
                    importance=stable_effect["importance"],
                    is_primary=False,
                    inference_stats=stable_effect["inference_stats"],
                )
                node.expected_effects.append(new_effect)
                existing_effect_descriptions.add(effect_desc_lower)

                # Print addition info including confidence
                confidence_color = {
                    "high": "\033[32m",
                    "medium": "\033[33m",
                    "low": "\033[90m",
                }.get(stable_effect["confidence_level"], "\033[0m")

                importance_tag = f"[{stable_effect['importance']}]"
                print(f"{confidence_color}[Effect Update] Added {importance_tag} [{stable_effect['confidence_level']}] effect to '{node.name}': "
                      f"{stable_effect['description']} (value={stable_effect['confidence_value']:.3f})\033[0m")

        # Scheme 6: analyze parameterized effects
        self._analyze_parameterized_effects(node, successful_traces, effect_frequency)

    def _analyze_parameterized_effects(
        self: "SkillGraphManager",
        node,
        successful_traces: List,
        effect_frequency: Dict[str, Dict[str, Any]]
    ):
        """
        Scheme 6: analyze parameter-effect correlations.

        Detect whether effect values are linearly correlated with call parameters, e.g.:
        - mineLogs(bot, count=3) -> obtains ~3 logs
        - mineLogs(bot, count=5) -> obtains ~5 logs
        => infer that the effect correlates with the count parameter

        Args:
            node: SkillNode object
            successful_traces: list of successful execution records
            effect_frequency: effect-frequency statistics
        """
        if not node.parameters:
            return  # no parameters; skip

        # Collect traces that have call_args
        traces_with_args = [t for t in successful_traces if t.call_args]

        if len(traces_with_args) < 3:
            return  # insufficient samples

        # Analyze correlation between each parameter and each effect
        param_effect_correlations = {}

        for param_name in node.parameters:
            # Collect parameter-value to effect-value pairs
            param_effect_pairs = {}  # {effect_key: [(param_value, effect_value), ...]}

            for trace in traces_with_args:
                if param_name not in trace.call_args:
                    continue

                param_value = trace.call_args[param_name]

                # Only handle numeric parameters
                if not isinstance(param_value, (int, float)):
                    continue

                # Collect effects from this trace
                if trace.actual_effects:
                    for effect in trace.actual_effects:
                        if effect.inventory_changes:
                            for item, change in effect.inventory_changes.items():
                                effect_key = f"inventory:{item}"
                                if effect_key not in param_effect_pairs:
                                    param_effect_pairs[effect_key] = []
                                param_effect_pairs[effect_key].append((param_value, change))

            # Analyze correlation
            for effect_key, pairs in param_effect_pairs.items():
                if len(pairs) < 3:
                    continue

                # Compute Pearson correlation (simplified)
                correlation = self._calculate_correlation(pairs)

                if abs(correlation) > 0.8:  # strong correlation
                    key = f"{param_name}:{effect_key}"
                    param_effect_correlations[key] = {
                        "param": param_name,
                        "effect_key": effect_key,
                        "correlation": correlation,
                        "pairs": pairs,
                    }
                    print(f"\033[35m[Parameterized Effect] Detected parameter correlation: {param_name} -> {effect_key} (r={correlation:.2f})\033[0m")

        # Save parameterized-effect info onto the node
        if param_effect_correlations:
            if not hasattr(node, 'parameterized_effects') or node.parameterized_effects is None:
                node.parameterized_effects = {}
            node.parameterized_effects.update(param_effect_correlations)

            # Update effect description to a parameterized form
            for key, corr_info in param_effect_correlations.items():
                param_name = corr_info["param"]
                effect_key = corr_info["effect_key"]

                if effect_key.startswith("inventory:"):
                    item = effect_key.split(":")[1]
                    # Compute average ratio
                    pairs = corr_info["pairs"]
                    avg_ratio = sum(p[1] / p[0] for p in pairs if p[0] != 0) / len([p for p in pairs if p[0] != 0]) if pairs else 1

                    if abs(avg_ratio - 1) < 0.2:  # close to a 1:1 relationship
                        param_desc = f"Adds {{{param_name}}} {item} to inventory"
                    elif avg_ratio > 1:
                        param_desc = f"Adds ~{avg_ratio:.1f}*{{{param_name}}} {item} to inventory"
                    else:
                        param_desc = f"Adds ~{avg_ratio:.1f}*{{{param_name}}} {item} to inventory"

                    print(f"\033[35m[Parameterized Effect] Parameterized effect description: {param_desc}\033[0m")

    def _calculate_correlation(self: "SkillGraphManager", pairs: List[Tuple[float, float]]) -> float:
        """Delegate to execution_analysis module (v4.0 refactor)."""
        return calculate_correlation(pairs)

    def _classify_effect_importance(
        self: "SkillGraphManager",
        key: str,
        freq_info: Dict[str, Any],
        skill_name: str,
        skill_code: str
    ) -> str:
        """Delegate to code_verifier module (v4.0 refactor)."""
        return classify_effect_importance(key, freq_info, skill_name, skill_code)

    def _item_produced_by_child(
        self: "SkillGraphManager",
        skill_name: str,
        item: str,
    ) -> bool:
        """Whether any child skill of ``skill_name`` already claims, via its
        expected_effects, to produce ``item`` with ``operation=='add'``.

        Used by :meth:`_update_effects_from_executions` to avoid crediting a
        parent skill with inventory produces that originate inside its
        sub-skill calls. The runtime trace captures inventory deltas at the
        skill boundary — meaning the parent's actual_effects naturally include
        anything its children produced during execution. Without this check
        the parent is misattributed with the child's effects, which then
        misleads precondition-based skill discovery (e.g. planner treating
        craftPickaxe as an oak_log producer because it internally calls
        ensureWoodLogs).
        """
        if not hasattr(self, "get_children") or not hasattr(self, "get_node"):
            return False
        try:
            children = self.get_children(skill_name) or []
        except Exception:
            return False
        for child_name in children:
            child_node = self.get_node(child_name)
            if not child_node or not getattr(child_node, "expected_effects", None):
                continue
            for eff in child_node.expected_effects:
                sr = getattr(eff, "state_representation", None)
                if not isinstance(sr, dict):
                    continue
                if sr.get("item") == item and sr.get("operation") == "add":
                    return True
        return False

    def _parse_js_args_to_dict(
        self: "SkillGraphManager",
        js_args: str,
        skill_name: str,
        node
    ) -> Dict[str, Any]:
        """
        Parse JS-side serialized arguments (recorded args from the skill wrapper).

        Args:
            js_args: JSON.stringify-serialized argument string from the JS side, e.g. '["[bot]", 17, "copper_ingot"]'
            skill_name: skill name
            node: SkillNode object, used to fetch parameter names

        Returns:
            Dict[str, Any]: mapping of parameter name to value, e.g. {"count": 17, "fuelName": "copper_ingot"}
        """
        call_args = {}

        try:
            # Check for invalid input
            if not js_args or js_args == "[serialization failed]":
                return call_args

            # Parse JSON
            args_list = json.loads(js_args)

            if not isinstance(args_list, list):
                return call_args

            # Get parameter names
            param_names = []
            if node and node.parameters:
                param_names = list(node.parameters.keys())
            else:
                # Extract parameter names from code
                if node and node.code:
                    func_pattern = rf'async\s+function\s+{re.escape(skill_name)}\s*\(([^)]*)\)'
                    func_match = re.search(func_pattern, node.code)
                    if func_match:
                        params_str = func_match.group(1)
                        for param in params_str.split(','):
                            param = param.strip()
                            if param and param != 'bot':
                                param_name = param.split('=')[0].strip()
                                param_names.append(param_name)

            # Map parameters (skip bot)
            value_index = 0
            for i, value in enumerate(args_list):
                # Skip bot-object marker
                if value == "[bot]":
                    continue

                if value_index < len(param_names):
                    call_args[param_names[value_index]] = value
                    value_index += 1

            if call_args:
                print(f"\033[36m[JS Args] Parsed JS-side parameters for {skill_name}: {call_args}\033[0m")

        except json.JSONDecodeError as e:
            print(f"\033[33m[JS Args] JSON parse failed: {e}\033[0m")
        except Exception as e:
            print(f"\033[33m[JS Args] Failed to parse JS-side parameters: {e}\033[0m")

        return call_args

    def _parse_call_args_from_exec_code(
        self: "SkillGraphManager",
        exec_code: str,
        skill_name: str,
        node
    ) -> Dict[str, Any]:
        """
        Parse call arguments from exec_code.

        For example:
        - exec_code: "await mineLogs(bot, 3, ["oak_log"]);"
        - skill_name: "mineLogs"
        - Returns: {"count": 3, "logTypes": ["oak_log"]}

        Args:
            exec_code: execution code, e.g. "await mineLogs(bot, 3, ["oak_log"]);"
            skill_name: skill name
            node: SkillNode object, used to fetch parameter names

        Returns:
            Dict[str, Any]: mapping of parameter name to value
        """
        call_args = {}

        try:
            # Match function call: await funcName(arg1, arg2, ...)
            # Use greedy matching to capture all arguments (including nested parens and arrays)
            pattern = rf'await\s+{re.escape(skill_name)}\s*\((.+)\)\s*;?'
            match = re.search(pattern, exec_code, re.DOTALL)

            if not match:
                return call_args

            args_str = match.group(1).strip()

            # Parse arguments (handle arrays, objects, and other complex types)
            arg_values = self._parse_js_arguments(args_str)

            # Get parameter names from node.parameters or from code
            param_names = []
            if node and node.parameters:
                param_names = list(node.parameters.keys())
            else:
                # Extract parameter names from code
                func_pattern = rf'async\s+function\s+{re.escape(skill_name)}\s*\(([^)]*)\)'
                func_match = re.search(func_pattern, node.code if node else "")
                if func_match:
                    params_str = func_match.group(1)
                    for param in params_str.split(','):
                        param = param.strip()
                        if param and param != 'bot':
                            param_name = param.split('=')[0].strip()
                            param_names.append(param_name)

            # Associate argument values with parameter names (skip the bot parameter)
            value_index = 0
            for i, value in enumerate(arg_values):
                if i == 0 and value == "bot":
                    continue  # skip the bot parameter

                if value_index < len(param_names):
                    param_name = param_names[value_index]
                    call_args[param_name] = value
                    value_index += 1

            if call_args:
                print(f"\033[36m[Call Args] Parsed call arguments for {skill_name}: {call_args}\033[0m")

        except Exception as e:
            print(f"\033[33m[Call Args] Failed to parse call arguments: {e}\033[0m")

        return call_args

    def _parse_js_arguments(self: "SkillGraphManager", args_str: str) -> List[Any]:
        """Delegate to execution_analysis module (v4.0 refactor)."""
        return parse_js_arguments(args_str)

    def _categorize_failure(self: "SkillGraphManager", trace) -> FailureCategory:
        """Delegate to execution_analysis module (v4.0 refactor)."""
        error_msg = getattr(trace, 'error_message', None) or ""
        category_str = categorize_failure(error_msg)
        return FailureCategory[category_str]

    def _calculate_precondition_value(
        self: "SkillGraphManager",
        n_success_with: int,
        n_success_total: int,
        n_failure_missing: int,
        n_failure_total: int,
        code_verified: bool,
        config: PreconditionValueFunctionConfig = None,
    ) -> Dict[str, Any]:
        """
        delegate to execution.calculate_precondition_value.

        Compute the Value Function value for a Precondition.

        Formula: V(precond) = p_combined - λ * uncertainty - γ * (1 - code_verified)
        """
        return calculate_precondition_value(
            n_success_with=n_success_with,
            n_success_total=n_success_total,
            n_failure_missing=n_failure_missing,
            n_failure_total=n_failure_total,
            code_verified=code_verified,
            config=config,
        )

    def _calculate_effect_value(
        self: "SkillGraphManager",
        n_occurrences: int,
        n_total: int,
        config: EffectValueFunctionConfig = None,
    ) -> Dict[str, Any]:
        """
        delegate to execution.calculate_effect_value.

        Compute the Value Function value for an Effect.

        Formula: V(effect) = p_s - λ * uncertainty
        """
        return calculate_effect_value(
            n_occurrences=n_occurrences,
            n_total=n_total,
            config=config,
        )

    def _verify_precondition_in_code(
        self: "SkillGraphManager",
        code: str,
        key: str,
        precond_info: Dict[str, Any]
    ) -> bool:
        """Delegate to code_verifier module (v4.0 refactor)."""
        return verify_precondition_in_code(code, key, precond_info)

    def _update_preconditions_from_executions(
        self: "SkillGraphManager",
        node: SkillNode,
        successful_traces: List[SkillExecutionTrace],
        failed_traces: List[SkillExecutionTrace]
    ) -> None:
        """
        Update preconditions based on execution data (Value Function approach).

        Strategy:
        1. Tally state features in successful samples
        2. Tally missing conditions in failed samples (only relevant failures)
        3. Use the Value Function to compute confidence for each candidate precondition
        4. Decide whether to add based on confidence; record the confidence level

        Value Function formula:
        V(precond) = p_combined - λ * uncertainty - γ * (1 - code_verified)

        Args:
            node: skill node
            successful_traces: list of successful execution records
            failed_traces: list of failed execution records
        """
        if not successful_traces:
            return

        # Get Value Function configuration
        config = PreconditionValueFunctionConfig()

        # Tally state features in successful samples
        success_preconditions: Dict[str, Dict[str, Any]] = {}

        for trace in successful_traces:
            if not trace.pre_state:
                continue

            # Tally inventory items
            inventory = trace.pre_state.get("inventory", {})
            if isinstance(inventory, dict):
                for item, count in inventory.items():
                    if count > 0:
                        key = f"inventory:{item}"
                        if key not in success_preconditions:
                            success_preconditions[key] = {
                                "type": "inventory",
                                "item": item,
                                "count": 0,
                                "total_count": 0,
                            }
                        success_preconditions[key]["count"] += 1
                        success_preconditions[key]["total_count"] += count

            # Tally equipment
            equipment = trace.pre_state.get("equipment", {})
            if isinstance(equipment, dict):
                for slot, item in equipment.items():
                    if item:
                        key = f"equipment:{slot}:{item}"
                        if key not in success_preconditions:
                            success_preconditions[key] = {
                                "type": "equipment",
                                "slot": slot,
                                "item": item,
                                "count": 0,
                            }
                        success_preconditions[key]["count"] += 1

        n_success_total = len(successful_traces)

        # Tally missing conditions in failed samples (only relevant failures)
        relevant_failures = []
        for trace in failed_traces:
            # Use categorize_failure to determine failure type
            category = self._categorize_failure(trace)
            # Only use PRECONDITION_NOT_MET and EXECUTION_ERROR (potentially relevant)
            if category in [FailureCategory.PRECONDITION_NOT_MET, FailureCategory.EXECUTION_ERROR]:
                relevant_failures.append(trace)

        n_failure_total = len(relevant_failures)

        # Analyze items missing in failed samples
        failure_missing: Dict[str, int] = {}

        for trace in relevant_failures:
            if not trace.pre_state:
                continue

            inventory = trace.pre_state.get("inventory", {})

            # Check whether each item present on success is missing on failure
            for key, precond_info in success_preconditions.items():
                if precond_info["type"] == "inventory":
                    item = precond_info["item"]
                    item_count = inventory.get(item, 0) if isinstance(inventory, dict) else 0

                    # If item is absent or count is 0 on failure
                    if item_count == 0:
                        if key not in failure_missing:
                            failure_missing[key] = 0
                        failure_missing[key] += 1

        print(f"\033[36m[Precondition Inference] Evaluating {len(success_preconditions)} candidate conditions via Value Function\033[0m")
        print(f"\033[36m[Precondition Inference] Successful samples: {n_success_total}, relevant failed samples: {n_failure_total}\033[0m")

        # Use the Value Function to evaluate each candidate precondition
        stable_preconditions = []

        for key, precond_info in success_preconditions.items():
            n_success_with = precond_info["count"]
            n_failure_missing = failure_missing.get(key, 0)

            # Verify in code
            code_verified = self._verify_precondition_in_code(node.code, key, precond_info)

            # Compute Value Function
            value_result = self._calculate_precondition_value(
                n_success_with=n_success_with,
                n_success_total=n_success_total,
                n_failure_missing=n_failure_missing,
                n_failure_total=n_failure_total,
                code_verified=code_verified,
                config=config,
            )

            # Print details
            # Bug 3 fix: rename keys p_s → p_success, p_f → p_failure
            if value_result["value"] > -0.3:
                print(f"\033[90m[Precondition Value] {key}: value={value_result['value']:.3f} "
                      f"({value_result['confidence_level']}) "
                      f"[p_s={value_result['details']['p_success']:.2f}, "
                      f"p_f={value_result['details']['p_failure']:.2f}, "
                      f"code_verified={code_verified}]\033[0m")

            # Decide whether to add based on the Value Function
            if value_result["should_add"]:
                # Generate description
                if precond_info["type"] == "inventory":
                    item = precond_info["item"]
                    avg_count = precond_info["total_count"] / precond_info["count"] if precond_info["count"] > 0 else 1
                    if avg_count > 1.5:
                        description = f"Need {int(avg_count)} {item} in inventory"
                        state_repr = {
                            "type": "inventory",
                            "item": item,
                            "count": int(avg_count),
                            "operation": "have"
                        }
                    else:
                        description = f"Need {item} in inventory"
                        state_repr = {
                            "type": "inventory",
                            "item": item,
                            "count": 1,
                            "operation": "have"
                        }
                elif precond_info["type"] == "equipment":
                    description = f"Need {precond_info['item']} equipped in {precond_info['slot']}"
                    state_repr = {
                        "type": "equipment",
                        "slot": precond_info["slot"],
                        "item": precond_info["item"],
                        "operation": "equipped"
                    }
                else:
                    description = f"Precondition: {key}"
                    state_repr = {}

                stable_preconditions.append({
                    "description": description,
                    "type": precond_info["type"],
                    "key": key,
                    "state_representation": state_repr,
                    "confidence_value": value_result["value"],
                    "confidence_level": value_result["confidence_level"],
                    "inference_stats": {
                        "n_success_with": n_success_with,
                        "n_success_total": n_success_total,
                        "n_failure_missing": n_failure_missing,
                        "n_failure_total": n_failure_total,
                        "code_verified": code_verified,  # moved into inference_stats (aligned with precondition.py design)
                        "last_updated": datetime.now().isoformat(),
                    },
                })

        # Update preconditions
        existing_precond_descriptions = {p.description.lower() for p in node.preconditions}

        for stable_precond in stable_preconditions:
            precond_desc_lower = stable_precond["description"].lower()
            if not any(desc in precond_desc_lower or precond_desc_lower in desc for desc in existing_precond_descriptions):
                # Add a new precondition (with confidence info)
                new_precond = SkillPrecondition(
                    description=stable_precond["description"],
                    code="",
                    state_representation=stable_precond.get("state_representation", {}),
                    confidence_value=stable_precond["confidence_value"],
                    confidence_level=stable_precond["confidence_level"],
                    inference_stats=stable_precond["inference_stats"],  # code_verified is included
                )
                node.preconditions.append(new_precond)
                existing_precond_descriptions.add(precond_desc_lower)

                # Print addition info including confidence
                confidence_color = {
                    "high": "\033[32m",
                    "medium": "\033[33m",
                    "low": "\033[90m",
                }.get(stable_precond["confidence_level"], "\033[0m")

                print(f"{confidence_color}[Precondition Update] Added [{stable_precond['confidence_level']}] precondition to '{node.name}': "
                      f"{stable_precond['description']} (value={stable_precond['confidence_value']:.3f})\033[0m")
