"""Parameter Resolution Mixin - Skill parameter value inference and extraction."""

from __future__ import annotations
import json
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from langchain.schema import HumanMessage, SystemMessage
from skillnet.agents.planning import (
    extract_base_type,
    map_output_to_input_material,
    item_matches_config_context,
)
from skillnet.agents.planning.inference import (
    ParameterInferenceEngine,
    InferenceContext,
    ParameterSemantic,
)
from skillnet.agents.constants.task_semantics import detect_task_semantic, TaskSemanticType
from skillnet.utils.stats_tracker import record_llm_usage
from .._types import TRANSFORM_TO_PATTERN

if TYPE_CHECKING:
    from ..graph_planner import GraphPlanner


class ParameterResolutionMixin:
    """Parameter Resolution Mixin - Skill parameter value inference and extraction.

    Methods:
        _try_inference_engine: Try using inference engine
        _get_parameter_value: Get parameter value - core method, coordinates multiple inference strategies
        _build_object_parameter_value: Build object-type parameter value
        _extract_field_value_from_effects: Extract field value from effects
        _extract_base_type: Extract base type from item name
        _map_output_to_input_material: Map output to input material
        _is_item_compatible_with_parameter: Check item compatibility with parameter
        _format_inventory: Format inventory for LLM prompt
        _get_parameter_value_with_llm: Get parameter value with LLM - fallback
        _apply_count_parameter_fallback: Apply count parameter fallback
        _find_semantic_match: Find semantic match for parameter name

    Note:
        Rule learning methods have been moved to RuleLearningMixin.
        Core inference logic has been extracted to skillnet.agents.planning.inference module.
    """

    def _try_inference_engine(
        self,
        param_name: str,
        skill_name: str,
        target_effects: Optional[List[Dict[str, Any]]],
        skill_metadata: Optional[Dict[str, Any]],
        precondition_context: Optional[Dict[str, Any]] = None,
        expected_inventory: Optional[Dict[str, int]] = None,
    ) -> Optional[str]:
        """
        Try inferring a parameter value using the new parameter inference engine.

        bridge method — converts legacy interface args to InferenceContext
        and invokes the new engine.

        Args:
            param_name: parameter name
            skill_name: skill name
            target_effects: list of target effects
            skill_metadata: skill metadata
            precondition_context: precondition context
            expected_inventory: expected inventory

        Returns:
            Parameter value string (JS format), or None if inference fails or
            confidence is too low.
        """
        # Get parameter info
        param_info = {}
        if skill_metadata and skill_name in skill_metadata:
            params = skill_metadata[skill_name].get("parameters", {})
            param_info = params.get(param_name, {})

        if not param_info and self.skill_graph_manager.has_node(skill_name):
            node = self.skill_graph_manager.get_node(skill_name)
            if node.parameters:
                param_info = node.parameters.get(param_name, {})

        # Layer 1.5: even when param_info is empty (e.g. refactor-born skills
        # without extracted parameter metadata), the SPECIALIZED_WHEN strategy
        # can still resolve the value from propagated effect annotations.
        # Probe whether the skill has any effects carrying specialized_when
        # for this param_name — if so, let the inference engine run.
        has_specialized_when = False
        if not param_info and self.skill_graph_manager.has_node(skill_name):
            sk_node = self.skill_graph_manager.get_node(skill_name)
            for eff in (sk_node.expected_effects or []):
                sw = getattr(eff, 'specialized_when', None)
                if sw and param_name in sw and getattr(eff, 'is_primary', False):
                    has_specialized_when = True
                    break

        # No parameter info — let legacy logic handle it (but still run inference
        # when specialized_when is present)
        if not param_info and not has_specialized_when:
            return None

        # Parse semantic info
        semantic = None
        semantic_data = param_info.get("semantic")
        if semantic_data:
            semantic = ParameterSemantic.from_dict(semantic_data)

        # Query task-scoped parameter corrections from optimizer feedback
        parameter_corrections = None
        if hasattr(self, 'skill_graph_manager') and hasattr(
            self.skill_graph_manager, 'get_task_parameter_corrections'
        ):
            corrections = self.skill_graph_manager.get_task_parameter_corrections(
                skill_name=skill_name,
                param_name=param_name,
            )
            if corrections:
                parameter_corrections = corrections

        # Layer 1.5: collect skill's expected_effects for the SPECIALIZED_WHEN
        # strategy. Carries any specialized_when annotations propagated by
        # sibling refactor (e.g. craftTool's wooden_axe effect carries
        # specialized_when={toolType: 'axe'}).
        skill_expected_effects = []
        if self.skill_graph_manager.has_node(skill_name):
            sk_node = self.skill_graph_manager.get_node(skill_name)
            skill_expected_effects = list(sk_node.expected_effects or [])

        # Build InferenceContext
        cs = self._current_state or {}
        context = InferenceContext(
            skill_name=skill_name,
            param_name=param_name,
            param_info=param_info,
            target_effects=target_effects or [],
            current_inventory=cs.get("inventory", {}),
            current_task=self._current_task,
            precondition_context=precondition_context,
            semantic=semantic,
            expected_inventory=expected_inventory,
            caller_skill=precondition_context.get("caller_skill") if precondition_context else None,
            # Environment context
            nearby_blocks=cs.get("nearby_blocks"),
            biome=cs.get("biome"),
            position=cs.get("position"),
            equipment=cs.get("equipment"),
            # Optimizer corrections (task-scoped)
            parameter_corrections=parameter_corrections,
            # Layer 1.5: expose skill effects to SPECIALIZED_WHEN strategy
            skill_expected_effects=skill_expected_effects,
        )

        # Invoke the inference engine
        result = self.param_inference_engine.infer_parameter_value(context)

        # Phase 3.5: examine result, including validation_passed status
        if result.is_successful() and result.confidence >= 0.6:
            # hard type-safety check — number params reject non-numeric strings
            param_type = param_info.get("type")
            if param_type == "number" and result.value is not None:
                value_str = str(result.value).strip('"').strip("'")
                try:
                    float(value_str)
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"[Type Safety] {skill_name}.{param_name} expects number, "
                        f"got '{result.value}' — rejecting, using fallback"
                    )
                    return None  # Let legacy logic handle (use default value)

            # Check whether type validation passed
            # Bug 7 fix: reject the inference result when validation fails.
            # The prior "accepting result anyway" path let LLM-emitted bad
            # values through — log:11193 showed craftPickaxe.targetItem =
            # "oak_log" (validation failed, accepted anyway). Returning None
            # here forces the fallback chain (semantic match → default value)
            # to take over, mirroring the number-type rejection above.
            if not result.validation_passed:
                self.logger.warning(
                    f"[v7.3 TypeValidation] {skill_name}.{param_name} = {result.value} "
                    f"type validation failed (confidence={result.confidence:.2f}) — "
                    f"rejecting, falling back to default/semantic match"
                )
                return None
            self.logger.info(
                f"[v6.0 Inference] {skill_name}.{param_name} = {result.value} "
                f"(strategy={result.strategy_used.value}, confidence={result.confidence:.2f})"
            )
            return result.value

        # Confidence too low — return None and let legacy logic handle it
        self.logger.debug(
            f"[v6.0 Inference] {skill_name}.{param_name} confidence too low "
            f"({result.confidence:.2f}), falling back to legacy logic"
        )
        return None

    def _get_parameter_value(
        self,
        param_name: str,
        skill_name: str,
        target_effects: Optional[List[Dict[str, Any]]],
        skill_metadata: Optional[Dict[str, Any]],
        use_llm: bool = True, # Whether to use LLM (optional, default True)
        precondition_context: Optional[Dict[str, Any]] = None,  # Precondition context (call-chain info)
        expected_inventory: Optional[Dict[str, int]] = None  # Expected inventory (from upstream skills' outputs)
    ) -> str:
        """
        Get parameter value.

        Uses the unified parameter inference engine, supporting
        multi-dimensional semantic inference and strategy-chain fallback.

        Prefers skill_metadata; if unavailable, falls back to
        skill_graph_manager. Uses metadata (type, default, supported values,
        etc.) to validate and match parameters.

        Args:
            param_name: parameter name (e.g. "count", "preferredLogTypes")
            skill_name: skill name
            target_effects: target effects (extracted from the task)
            skill_metadata: skill metadata (optional; falls back to the graph
                if unavailable)
            precondition_context: precondition context containing required_item,
                is_flexible, caller_skill, etc.
            expected_inventory: expected inventory (from upstream skills'
                outputs), used for inference of fuel and similar parameters

        Returns:
            str: string representation of the parameter value (JavaScript code format)
        """
        # ========== try the new parameter inference engine ==========
        try:
            result = self._try_inference_engine(
                param_name=param_name,
                skill_name=skill_name,
                target_effects=target_effects,
                skill_metadata=skill_metadata,
                precondition_context=precondition_context,
                expected_inventory=expected_inventory,
            )
            if result is not None:
                return result
        except Exception as e:
            self.logger.warning(f"[v6.0 Inference] new engine failed, falling back to legacy logic: {e}")

        # ========== Legacy logic (used as fallback) ==========
        # Get parameter info: prefer skill_metadata, else fetch from the graph
        param_info = None
        all_parameters = {}

        # Collect all available parameter info
        if skill_metadata and skill_name in skill_metadata:
            all_parameters = skill_metadata[skill_name].get("parameters", {})
            param_info = all_parameters.get(param_name)

        # If not in skill_metadata, try the graph
        if not param_info and self.skill_graph_manager.has_node(skill_name):
            node = self.skill_graph_manager.get_node(skill_name)
            if node.parameters:
                all_parameters = node.parameters
                param_info = all_parameters.get(param_name)

        # If we can't find the parameter info, emit detailed debug info
        if not param_info:
            print(f"\033[33m[Parameter Mismatch] parameter name mismatch:\033[0m")
            print(f"  Skill: {skill_name}")
            print(f"  Looking up parameter: '{param_name}'")
            print(f"  Parameters in metadata: {list(all_parameters.keys()) if all_parameters else 'None'}")

            # Try semantic matching as a fallback (with a warning)
            if all_parameters:
                semantic_match = self._find_semantic_match(param_name, all_parameters)
                if semantic_match:
                    matched_param = None
                    for name, info in all_parameters.items():
                        if info == semantic_match:
                            matched_param = name
                            break
                    if matched_param:
                        print(f"\033[33m[Parameter Mismatch]   using semantic match: '{param_name}' -> '{matched_param}'\033[0m")
                        print(f"\033[33m[Parameter Mismatch]   suggestion: check that code and metadata are in sync\033[0m")
                        param_info = semantic_match
                    else:
                        print(f"\033[31m[Parameter Mismatch]   error: could not find parameter '{param_name}', will use default value\033[0m")
                else:
                    print(f"\033[31m[Parameter Mismatch]   error: could not find parameter '{param_name}', will use default value\033[0m")
            else:
                print(f"\033[31m[Parameter Mismatch]   error: no parameter metadata available\033[0m")

        if not param_info:
            # No parameter info found — return undefined (let the function use its default)
            return "undefined"

        param_type = param_info.get("type")
        default_value = param_info.get("default")
        supported_values = param_info.get("supported_values")  # List of supported values (optional)
        schema = param_info.get("schema")  # Inner structure for object-typed params (optional)

        # ===== Fallback handling for unknown/nullable types =====
        # When type inference fails, do a secondary inference based on the parameter name semantics
        if param_type in ("unknown", "nullable"):
            param_lower = param_name.lower()
            # Array-typed parameter name patterns
            if any(hint in param_lower for hint in ["types", "list", "items", "names", "options", "priority"]):
                self.logger.warning(
                    f"\033[33m[Type Fallback] {skill_name}.{param_name}: "
                    f"type {param_type} -> inferred as array (based on parameter name)\033[0m"
                )
                param_type = "array"
            # String-typed parameter name patterns
            elif any(hint in param_lower for hint in ["name", "type", "kind", "mode"]):
                self.logger.warning(
                    f"\033[33m[Type Fallback] {skill_name}.{param_name}: "
                    f"type {param_type} -> inferred as string (based on parameter name)\033[0m"
                )
                param_type = "string"
            # Number-typed parameter name patterns
            elif any(hint in param_lower for hint in ["count", "num", "amount", "target", "max", "min"]):
                self.logger.warning(
                    f"\033[33m[Type Fallback] {skill_name}.{param_name}: "
                    f"type {param_type} -> inferred as number (based on parameter name)\033[0m"
                )
                param_type = "number"
            # Vec3/direction/position parameter name patterns — return null (skills check `if (dir)` etc.)
            elif any(hint in param_lower for hint in ["dir", "direction", "pos", "position", "vec3", "offset"]):
                self.logger.warning(
                    f"\033[33m[Type Fallback] {skill_name}.{param_name}: "
                    f"type {param_type} -> direction/vec3 parameter, using null\033[0m"
                )
                return "null"
            else:
                # Cannot infer — use undefined so the skill falls back to its internal default
                self.logger.warning(
                    f"\033[33m[Type Fallback] {skill_name}.{param_name}: "
                    f"type {param_type} could not be inferred, using undefined\033[0m"
                )
                return "undefined"

        # ===== Check parameter error history: avoid repeating the same mistake =====
        if self._has_parameter_error_history(skill_name, param_name):
            self.logger.info(
                f"\033[36m[Parameter History] parameter '{param_name}' has an error record; using default to avoid repeating mistake\033[0m"
            )
            if default_value is not None:
                if isinstance(default_value, str):
                    return f'"{default_value}"'
                return str(default_value)
            # No default value — return undefined so the skill falls back to its internal default
            return "undefined"

        # Special handling: fuel parameters should not be extracted from target_effects' item
        # because the target product (e.g. copper_ingot) is not a valid fuel.
        # Enhancement: infer the best fuel from expected_inventory and current inventory.
        param_lower = param_name.lower()
        if "fuel" in param_lower:
            self.logger.info(f"\033[36m[Parameter Special] parameter '{param_name}' is a fuel parameter\033[0m")

            # Check whether expected_inventory contains fuel-typed outputs.
            # Note: keys may be generic types ("planks", "log") or concrete types ("oak_planks").
            expected_fuel_types = []
            if expected_inventory:
                fuel_patterns = ["planks", "log", "coal", "charcoal", "wood"]
                for item in expected_inventory:
                    if any(pattern in item.lower() for pattern in fuel_patterns):
                        expected_fuel_types.append(item)

            # If expected outputs include fuel types, use the default so the
            # skill decides at runtime — we may not know the concrete type
            # (e.g. oak_planks vs birch_planks).
            if expected_fuel_types:
                self.logger.info(f"\033[36m[Parameter Special] expected outputs include fuel types: {expected_fuel_types}, using default\033[0m")
                if default_value is not None:
                    if isinstance(default_value, str):
                        return f'"{default_value}"'
                    return str(default_value)
                return "null"

            # Pick a fuel from the current inventory
            inventory = self._current_state.get('inventory', {}) if self._current_state else {}
            # Fuel priority: coal > charcoal > planks > logs
            fuel_priority = ['coal', 'charcoal', 'oak_planks', 'birch_planks', 'spruce_planks',
                             'jungle_planks', 'acacia_planks', 'dark_oak_planks', 'oak_log',
                             'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log']
            for fuel in fuel_priority:
                if inventory.get(fuel, 0) > 0:
                    self.logger.info(f"\033[36m[Parameter Special] inferred fuel from current inventory: {fuel}\033[0m")
                    return f'"{fuel}"'

            # No usable fuel — use default or return null
            if default_value is not None:
                if isinstance(default_value, str):
                    return f'"{default_value}"'
                return str(default_value)
            return "null"

        # 1. First try learned rules
        if target_effects:
            learned_value = self._try_learned_rules(param_name, skill_name, target_effects, param_type)
            if learned_value:
                return learned_value

        # 2. Handle object-typed parameters (e.g. options = {})
        if param_type == "object" and schema and target_effects:
            obj_value = self._build_object_parameter_value(param_name, schema, target_effects, skill_name)
            if obj_value != "{}":
                # Learn the successfully built rule
                if self.learn_parameter_rules:
                    self._learn_object_parameter_rule(param_name, skill_name, target_effects, obj_value, schema)
                return obj_value
            # If construction fails, keep going with original logic / default value

        # 3. Try extracting the value from target_effects (rule matching)
        if target_effects:
            for effect in target_effects:
                param_lower = param_name.lower()

                # Match count/amount-style parameters
                if "count" in param_lower or "amount" in param_lower or "quantity" in param_lower:
                    if "count" in effect:
                        count_value = effect["count"]
                        # Validate the type
                        if param_type == "number" and isinstance(count_value, (int, float)):
                            # ========== Semantic conversion logic ==========
                            # Detect whether task semantic and skill parameter semantic match
                            task_semantic = None
                            skill_semantic = param_info.get("semantic", "unknown") if param_info else "unknown"

                            # ========== Plan B: conservative handling when semantic inference fails ==========
                            # When skill_semantic == "unknown", infer a safe default from the function name
                            if skill_semantic == "unknown" and skill_name:
                                skill_name_lower = skill_name.lower()
                                if skill_name_lower.startswith("ensure"):
                                    skill_semantic = "target_total"
                                    self.logger.warning(
                                        f"\033[33m[Semantic Fallback] {skill_name}.{param_name} semantic unknown -> target_total (based on function name ensure*)\033[0m"
                                    )
                                elif any(skill_name_lower.startswith(p) for p in ["craft", "mine", "collect", "harvest", "kill"]):
                                    skill_semantic = "delta"
                                    self.logger.warning(
                                        f"\033[33m[Semantic Fallback] {skill_name}.{param_name} semantic unknown -> delta (based on function name prefix)\033[0m"
                                    )

                            # Detect task semantic
                            if hasattr(self, '_current_task') and self._current_task:
                                task_semantic = detect_task_semantic(self._current_task)

                            # Get target item and current inventory
                            target_item = effect.get("item", "")
                            current_count = 0
                            if hasattr(self, '_current_state') and self._current_state:
                                inventory = self._current_state.get('inventory', {})
                                # Try exact match and fuzzy match
                                current_count = inventory.get(target_item, 0)
                                if current_count == 0:
                                    # Fuzzy match: check for similar items (e.g. oak_log, birch_log both count as log)
                                    for inv_item, inv_count in inventory.items():
                                        if target_item.lower() in inv_item.lower() or inv_item.lower() in target_item.lower():
                                            current_count += inv_count

                            # Semantic conversion
                            final_count = int(count_value)
                            conversion_applied = False

                            if task_semantic and skill_semantic != "unknown":
                                if task_semantic == TaskSemanticType.DELTA and skill_semantic == "target_total":
                                    # Task is a delta, skill ensures a total -> convert: current + delta
                                    final_count = current_count + int(count_value)
                                    conversion_applied = True
                                    self.logger.warning(
                                        f"\033[33m[Semantic Conversion] task semantic=DELTA, skill '{skill_name}' param '{param_name}' semantic=target_total\033[0m"
                                    )
                                    self.logger.warning(
                                        f"\033[33m[Semantic Conversion] conversion: current inventory({current_count}) + task delta({int(count_value)}) = {final_count}\033[0m"
                                    )
                                elif task_semantic == TaskSemanticType.TARGET_TOTAL and skill_semantic == "delta":
                                    # Task ensures a total, skill is delta -> convert: max(0, target - current)
                                    final_count = max(0, int(count_value) - current_count)
                                    conversion_applied = True
                                    self.logger.warning(
                                        f"\033[33m[Semantic Conversion] task semantic=TARGET, skill '{skill_name}' param '{param_name}' semantic=delta\033[0m"
                                    )
                                    self.logger.warning(
                                        f"\033[33m[Semantic Conversion] conversion: max(0, target({int(count_value)}) - current inventory({current_count})) = {final_count}\033[0m"
                                    )

                            # If no conversion was applied, log info (when semantics match or are indeterminate)
                            if not conversion_applied and task_semantic and skill_semantic != "unknown":
                                self.logger.info(
                                    f"\033[36m[Semantic Match] task semantic={task_semantic}, skill '{skill_name}' param '{param_name}' semantic={skill_semantic} (matched, no conversion needed)\033[0m"
                                )

                            return str(final_count)
                        elif param_type == "number":
                            # Try converting
                            try:
                                return str(int(count_value))
                            except (ValueError, TypeError):
                                pass

                # Match item/type-style parameters
                if "item" in effect:
                    item = effect["item"]

                    # Critical fix 0: if item is a generic type (e.g. "log", "planks"),
                    # for material-selection parameters, use the default rather than passing
                    # the generic type name (Minecraft has no block called "log").
                    if self._is_generic_item_type(item):
                        material_indicators = ["type", "log", "ore", "stone", "wood", "material", "fuel",
                                               "preferred", "preference", "priority", "allowed", "accepted"]
                        is_material_param = any(ind in param_lower for ind in material_indicators)

                        if is_material_param:
                            self.logger.info(f"\033[36m[Parameter Generic] '{item}' in target_effects is a generic type; parameter '{param_name}' will use the default\033[0m")
                            # Skip this effect and let downstream logic handle the default
                            continue

                    # Critical fix 1: check whether parameter semantics are compatible with the target product.
                    # E.g. plankPreference expects *_planks; wooden_pickaxe is not planks.
                    # In that case do NOT pass wooden_pickaxe to plankPreference.
                    if not self._is_item_compatible_with_parameter(param_name, item, supported_values):
                        # Parameter and target product are incompatible — skip this effect
                        self.logger.info(f"\033[33m[Parameter Mapping] skipping incompatible mapping: '{item}' -> '{param_name}' (parameter expects a different type of value)\033[0m")
                        continue

                    # Critical fix 2: check whether the parameter expects input material rather than output product.
                    # E.g. craftPlanks' logType expects "birch_log", not "birch_planks";
                    # we need to derive input material from the target product.
                    # Note: if param semantic is "config", do NOT map (parameter configures the target type).
                    param_semantic = param_info.get("semantic") if param_info else None
                    mapped_item = self._map_output_to_input_material(
                        param_name=param_name,
                        output_item=item,
                        skill_name=skill_name,
                        supported_values=supported_values,
                        semantic=param_semantic
                    )

                    # Verify the mapped item is in supported_values (if provided)
                    if supported_values and mapped_item not in supported_values:
                        # If the mapped value isn't supported, try the original
                        if item in supported_values:
                            mapped_item = item
                        else:
                            # Skip unsupported item
                            continue

                    # Enhancement: recognize array vs single-element equivalence.
                    # If the parameter is array-typed but the task only needs a single
                    # element, return an array containing just that element.
                    # If the parameter is string-typed, return the string directly.
                    if param_type == "array":
                        # Array type: return an array containing just the target item.
                        # This ensures the function only processes the target type.
                        # E.g. preferredLogTypes = ["oak_log"] is equivalent to logType = "oak_log".
                        return f'["{mapped_item}"]'
                    elif param_type == "string":
                        # String type: return the string directly.
                        # E.g. logType = "oak_log"
                        return f'"{mapped_item}"'
                    else:
                        # Other types — try matching parameter name
                        if "type" in param_lower or "item" in param_lower or "name" in param_lower:
                            if param_type == "string":
                                return f'"{mapped_item}"'
                            else:
                                return f'"{mapped_item}"'  # Default to string handling

        # 3.5 Flexible-requirement handling: when the requirement is flexible,
        # material-selection parameters use the default.
        # This includes: logType, oreType, stoneType, etc. (not just preferredXxx).
        if precondition_context and precondition_context.get("is_flexible", False):
            # Check whether the parameter is a "material selection" type
            material_indicators = ["type", "log", "ore", "stone", "wood", "material", "fuel",
                                   "preferred", "preference", "priority", "allowed", "accepted"]
            is_material_param = any(ind in param_lower for ind in material_indicators)

            if is_material_param:
                self.logger.info(f"\033[36m[Parameter Flexible] parameter '{param_name}' is a material-selection parameter with a flexible requirement (is_flexible=True); using default value\033[0m")
                if default_value is not None:
                    if isinstance(default_value, list):
                        return json.dumps(default_value)
                    elif isinstance(default_value, str):
                        return f'"{default_value}"'
                    else:
                        return str(default_value)
                return "undefined"

        # 3.6 Generic-type-requirement handling: when required_item is a generic type,
        # material parameters also use the default.
        if precondition_context:
            required_item = precondition_context.get("required_item", "")
            if required_item and self._is_generic_item_type(required_item):
                material_indicators = ["type", "log", "ore", "stone", "wood", "material", "fuel",
                                       "preferred", "preference", "priority", "allowed", "accepted"]
                is_material_param = any(ind in param_lower for ind in material_indicators)

                if is_material_param:
                    self.logger.info(f"\033[36m[Parameter Generic] required_item '{required_item}' is a generic type; parameter '{param_name}' will use the default\033[0m")
                    if default_value is not None:
                        if isinstance(default_value, list):
                            return json.dumps(default_value)
                        elif isinstance(default_value, str):
                            return f'"{default_value}"'
                        else:
                            return str(default_value)
                    return "undefined"

        # 4. If no value was extracted from target_effects, try the LLM (if enabled)
        if use_llm and self.llm and target_effects:
            llm_value = self._get_parameter_value_with_llm(
                param_name, param_info, skill_name, target_effects,
                precondition_context=precondition_context  # Pass context through
            )
            if llm_value:
                # LLM extracted successfully — learn this rule
                if self.learn_parameter_rules:
                    self._learn_parameter_rule(param_name, skill_name, target_effects, llm_value, param_type)
                return llm_value

        # If no value was extracted from target_effects, use the default
        if default_value is not None:
            if isinstance(default_value, list):
                return json.dumps(default_value)
            elif isinstance(default_value, str):
                return f'"{default_value}"'
            else:
                return str(default_value)

        # No default value — return undefined (let the function use JavaScript's default)
        return "undefined"

    def _build_object_parameter_value(
        self,
        param_name: str,
        schema: Dict[str, Any],
        target_effects: List[Dict[str, Any]],
        skill_name: str
    ) -> str:
        """
        Build an object parameter value from a schema.

        Args:
            param_name: parameter name (e.g. "options")
            schema: inner structure of the object parameter, e.g.:
                {
                    "fieldName": {
                        "type": "string",
                        "default": "copper",
                        "source_hint": {"effect_field": "item", "transform": "extract_ore_base"}
                    }
                }
            target_effects: list of target effects
            skill_name: skill name (for logging)

        Returns:
            The constructed object parameter string, e.g. '{oreBase: "iron", count: 6}'
        """
        obj_fields = {}

        for field_name, field_info in schema.items():
            value = self._extract_field_value_from_effects(field_name, field_info, target_effects)
            if value is not None:
                obj_fields[field_name] = value

        if obj_fields:
            props = [f'{k}: {v}' for k, v in obj_fields.items()]
            result = "{" + ", ".join(props) + "}"
            print(f"\033[36m[Object Parameter] Built '{param_name}' for {skill_name}: {result}\033[0m")
            return result

        print(f"\033[33m[Object Parameter] Could not build '{param_name}' for {skill_name}, using empty object\033[0m")
        return "{}"

    def _extract_field_value_from_effects(
        self,
        field_name: str,
        field_info: Dict[str, Any],
        target_effects: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Extract the value of an object field from target_effects.

        Prefers source_hint (field-name independent), falling back to field-name pattern matching.

        Args:
            field_name: field name
            field_info: field info containing type, default, source_hint
            target_effects: list of target effects

        Returns:
            String representation of the field value, or None if it cannot be extracted.
        """
        # 1. Prefer source_hint (field-name independent, more robust)
        source_hint = field_info.get("source_hint")
        if source_hint:
            effect_field = source_hint.get("effect_field")
            transform = source_hint.get("transform")

            for effect in target_effects:
                if effect_field and effect_field in effect:
                    value = effect[effect_field]

                    if transform and transform.startswith("extract_"):
                        # Handle via _extract_base_type
                        pattern_type = TRANSFORM_TO_PATTERN.get(transform, "generic")
                        base = self._extract_base_type(str(value), pattern_type)
                        if base:
                            return f'"{base}"'
                    elif transform is None:
                        # Use the value directly
                        if isinstance(value, str):
                            return f'"{value}"'
                        else:
                            return str(value)

            # source_hint specified but extraction failed — return None
            return None

        # 2. Fallback: field-name pattern matching (backward compatibility)
        field_lower = field_name.lower()

        for effect in target_effects:
            # count/quantity/amount-style fields
            if field_lower in ["count", "quantity", "amount", "num"]:
                if "count" in effect:
                    return str(effect["count"])

            # oreBase/itemBase and other fields containing "base"
            if "base" in field_lower:
                if "item" in effect:
                    base = self._extract_base_type(str(effect["item"]), "ore")
                    if base:
                        return f'"{base}"'

        return None

    def _extract_base_type(self, item_name: str, pattern_type: str = "ore") -> Optional[str]:
        """
        Extract the base type from an item name.

        delegated to the pure function extract_base_type.
        """
        return extract_base_type(item_name, pattern_type)

    def _map_output_to_input_material(
        self,
        param_name: str,
        output_item: str,
        skill_name: str,
        supported_values: Optional[List[str]] = None,
        semantic: Optional[str] = None
    ) -> str:
        """
        Map an output product to its input material.

        delegated to the pure function map_output_to_input_material.
        """
        return map_output_to_input_material(
            param_name=param_name,
            output_item=output_item,
            skill_name=skill_name,
            supported_values=supported_values,
            semantic=semantic,
            custom_logger=self.logger,
        )

    def _is_item_compatible_with_parameter(
        self,
        param_name: str,
        item: str,
        supported_values: Optional[List[str]] = None
    ) -> bool:
        """
        Check whether a target-effect item is semantically compatible with the
        parameter's expected type.

        This method prevents passing unrelated target products to a parameter.
        Examples:
        - plankPreference expects *_planks; wooden_pickaxe is incompatible
        - logType expects *_log; birch_planks is incompatible
        - fuelPriority expects fuel types; iron_ingot is incompatible
        - count parameters don't need item validation

        Args:
            param_name: parameter name (used to infer the expected item type)
            item: item name from the target effect
            supported_values: list of values supported by the parameter
                (preferred when provided)

        Returns:
            bool: True if the item is compatible with the parameter, else False.
        """
        param_lower = param_name.lower()
        item_lower = item.lower()

        # Layer 1: if supported_values exist, check whether item is in the list,
        # or whether it can be mapped to a value in the list.
        if supported_values:
            if item in supported_values:
                return True
            # Check whether the mapped item is compatible
            mapped_item = self._map_output_to_input_material(param_name, item, "", supported_values)
            if mapped_item in supported_values:
                return True
            # If explicit supported_values exist and item isn't in them, consider it incompatible
            return False

        # Layer 2: handle config-type parameters (priority, preference, option, etc.)
        # These should be validated via _item_matches_config_context.
        config_indicators = [
            "priority", "preference", "prefer", "option", "options",
            "config", "timeout", "distance", "fallback", "default",
            "selection", "choices", "allowed"
        ]
        is_config_param = any(ind in param_lower for ind in config_indicators)

        if is_config_param:
            # For config-type parameters, perform a semantic-context check
            if self._item_matches_config_context(param_name, item):
                return True
            # Config param incompatible with item — return False so caller uses the default
            self.logger.info(f"\033[33m[Parameter Compatibility] config parameter '{param_name}' rejects '{item}' (semantic mismatch)\033[0m")
            return False

        # Layer 3: infer the expected item type from parameter-name semantics.
        # Pull parameter-type pattern mapping from domain knowledge.
        dk = getattr(self, '_domain_knowledge', None)
        param_type_patterns = dk.get_param_type_patterns() if dk else {}

        # Identify the expected types for the parameter
        expected_patterns = []
        for pattern_key, patterns in param_type_patterns.items():
            if pattern_key in param_lower:
                expected_patterns.extend(patterns)

        # If no specific type was recognized, consider it compatible (allow generic params)
        if not expected_patterns:
            return True

        # Check whether item matches an expected pattern
        for pattern in expected_patterns:
            if pattern in item_lower:
                return True

        # When expected patterns exist but the item doesn't match, return False
        self.logger.info(f"\033[33m[Parameter Compatibility] parameter '{param_name}' rejects '{item}' (expected patterns: {expected_patterns})\033[0m")
        return False

    def _format_inventory(self, inventory: Dict[str, int]) -> str:
        """
        Format inventory for the LLM prompt.

        Args:
            inventory: mapping of item name to count

        Returns:
            Formatted string, one item per line.
        """
        if not inventory:
            return "Empty"
        items = [f"- {item}: {count}" for item, count in sorted(inventory.items()) if count > 0]
        return "\n".join(items[:20])  # Cap display count to avoid an oversized prompt

    def _get_parameter_value_with_llm(
        self,
        param_name: str,
        param_info: Dict[str, Any],
        skill_name: str,
        target_effects: List[Dict[str, Any]],
        precondition_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Use the LLM to extract a parameter value from target_effects.

        Args:
            param_name: parameter name
            param_info: parameter metadata (includes type, default, supported_values, etc.)
            skill_name: skill name
            target_effects: list of target effects
            precondition_context: precondition context containing required_item,
                caller_skill, etc.

        Returns:
            String representation of the parameter value, or None if it cannot
            be extracted.
        """
        try:
            param_type = param_info.get("type")
            default_value = param_info.get("default")
            supported_values = param_info.get("supported_values")
            description = param_info.get("description", "")
            schema = param_info.get("schema")  # Inner structure for object types

            system_prompt = """You are a parameter mapping expert. Your task is to extract the appropriate parameter value from target effects for a skill function.

Given:
- Parameter name and metadata
- Target effects (what the task aims to achieve)
- Call chain context (why this skill is being called)
- Current inventory

Extract the value that should be passed to this parameter.

Return ONLY a JSON object:
{
  "value": <the extracted value>,
  "reason": "brief explanation"
}

Important:
- For array types, return a JSON array string like '["item1", "item2"]'
- For string types, return a quoted string like '"item_name"'
- For number types, return the number as a string like "3"
- For OBJECT types with schema, return a JavaScript object literal like '{fieldName: value}'
  - Extract base type from item names: "iron_ore" → oreBase: "iron"
  - Use count from effects for quantity fields
- If the value should come from supported_values, ensure it's in that list
- If no suitable value can be extracted, return null

## CONTEXT-AWARE RULES (CRITICAL)
1. acceptSmelted parameter:
   - If caller skill is a smelting skill (smelt*, furnace*), set acceptSmelted: false
   - Smelting requires RAW materials (like raw_iron), not finished products (like iron_ingot)
   - Example: smeltRawIron needs raw_iron, NOT iron_ingot

2. Fuel-related parameters (fuelPreferences, fuel, etc.):
   - Check current inventory for available fuels
   - Prioritize fuels already in inventory: coal > charcoal > planks > logs
   - If inventory has coal, prefer ["coal"]

3. When required_item is specified in call chain context:
   - The skill should produce the required_item, not something else
   - Parameters should be set to ensure the required output

## FLEXIBLE REQUIREMENTS (CRITICAL - NEW)
4. Check the 'is_flexible' field in Call Chain Context:
   - If is_flexible=true, this means ANY item from accepted_items can satisfy the requirement
   - In this case, return null to use the skill's default value
   - DO NOT restrict options when the requirement is flexible

5. For preference/priority parameters (preferredLogs, preferredFuelName, logType, etc.):
   - These parameters often have rich default values covering all valid options
   - ONLY override defaults when:
     a) is_flexible=false AND
     b) required_item is a SPECIFIC type (like "oak_planks", not "planks")
   - Examples:
     - is_flexible=true, required_item="fuel" → return null (use default)
     - is_flexible=false, required_item="oak_planks" → preferredLogs = ["oak_log"]
     - required_item="planks" (generic) → return null (use default)

6. When in doubt, return null to use the skill's default value
   - This is SAFER than guessing a specific value that may not be available"""

            # Build schema description
            schema_desc = ""
            if param_type == "object" and schema:
                schema_fields = []
                for field_name, field_info in schema.items():
                    field_type = field_info.get("type", "unknown")
                    field_default = field_info.get("default")
                    schema_fields.append(f"  - {field_name} ({field_type}): default={field_default}")
                schema_desc = "\nObject Schema:\n" + "\n".join(schema_fields)

            # Build context info
            context_info = ""
            if precondition_context:
                is_flexible = precondition_context.get('is_flexible', False)
                accepted_items = precondition_context.get('accepted_items', [])
                accepted_items_str = str(accepted_items[:5]) + "..." if len(accepted_items) > 5 else str(accepted_items)
                context_info = f"""
## Call Chain Context
- This skill is called to satisfy a precondition
- Required output: {precondition_context.get('required_item', 'unknown')} x {precondition_context.get('required_count', 1)}
- Is flexible requirement: {is_flexible}
- Accepted items: {accepted_items_str}
- Caller skill: {precondition_context.get('caller_skill', 'unknown')}
- Precondition: {precondition_context.get('precondition_desc', '')}
"""

            # Add inventory info
            inventory_info = ""
            if self._current_state and self._current_state.get('inventory'):
                inventory_info = f"""
## Current Inventory
{self._format_inventory(self._current_state['inventory'])}
"""

            # Add task context
            task_info = ""
            if self._current_task:
                task_info = f"""
## Task Context
Current task: {self._current_task}
"""

            human_prompt = f"""Skill: {skill_name}
Parameter: {param_name}
Parameter Type: {param_type}
Parameter Description: {description}
Default Value: {default_value}
Supported Values: {supported_values if supported_values else "Any"}{schema_desc}
{task_info}{context_info}{inventory_info}
Target Effects:
{json.dumps(target_effects, indent=2)}

What value should be passed to parameter '{param_name}' based on the target effects and context?
Return only JSON."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            _llm_resp = self.llm.invoke(messages)
            record_llm_usage(_llm_resp, process_type="planning_param_inference", function_name="planner.parameter_resolution._get_parameter_value_with_llm", skill_name=skill_name)
            response = _llm_resp.content

            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                value = result.get("value")
                reason = result.get("reason", "")

                if value is not None:
                    print(f"\033[36m[LLM Parameter] extracted value for parameter '{param_name}': {value} ({reason})\033[0m")

                    # Verify the value matches the type requirement
                    if param_type == "array":
                        if isinstance(value, list):
                            return json.dumps(value)
                        elif isinstance(value, str):
                            # Try parsing as an array string
                            try:
                                parsed = json.loads(value)
                                if isinstance(parsed, list):
                                    return json.dumps(parsed)
                            except (json.JSONDecodeError, ValueError, TypeError):
                                pass
                    elif param_type == "string":
                        if isinstance(value, str):
                            # Strip any pre-existing quotes to avoid double-quoting
                            clean_value = value.strip('"').strip("'")
                            return f'"{clean_value}"'
                    elif param_type == "number":
                        if isinstance(value, (int, float)):
                            return str(value)

                    # Type doesn't match — try converting
                    return str(value)

        except Exception as e:
            print(f"\033[33m[LLM Parameter] LLM extraction failed: {e}\033[0m")

        return None

    def _apply_count_parameter_fallback(
        self,
        call_statement: str,
        target_effects: List[Dict[str, Any]]
    ) -> str:
        """
        Rule-based fallback: if the LLM produced count=0 but target_effects
        requires producing items, replace it with the count from target_effects.

        This is the last line of defense against the LLM mistakenly generating
        total=0 / count=0.

        Args:
            call_statement: the call statement generated by the LLM
            target_effects: list of target effects

        Returns:
            str: possibly corrected call statement.
        """
        if not target_effects:
            return call_statement

        # Find the first valid target count
        target_count = None
        for target in target_effects:
            count = target.get("count", 0)
            if count > 0:
                target_count = count
                break

        if target_count is None or target_count <= 0:
            return call_statement

        # Check the call statement for patterns like =0, , 0, or (bot, 0
        # Matches: ", 0)", ", 0,", "(bot, 0)", etc.
        zero_patterns = [
            (r',\s*0\s*\)', f', {target_count})'),           # , 0) -> , N)
            (r',\s*0\s*,', f', {target_count},'),            # , 0, -> , N,
            (r'\(bot\s*,\s*0\s*\)', f'(bot, {target_count})'),  # (bot, 0) -> (bot, N)
            (r'\(bot\s*,\s*0\s*,', f'(bot, {target_count},'),   # (bot, 0, -> (bot, N,
        ]

        original = call_statement
        for pattern, replacement in zero_patterns:
            if re.search(pattern, call_statement):
                call_statement = re.sub(pattern, replacement, call_statement, count=1)
                if call_statement != original:
                    self.logger.warning(
                        f"\033[33m[Fallback] detected count=0, replaced with {target_count}: "
                        f"{original} -> {call_statement}\033[0m"
                    )
                    break

        return call_statement

    def _find_semantic_match(
        self,
        param_name: str,
        all_parameters: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the matching parameter info via semantic matching.

        Enhancement: recognizes singular/plural semantic similarity.
        Examples:
        - preferredLogTypes can match logType (singular/plural forms)
        - preferredLogTypes (array) can match logType (string) if the array
          has only one element.

        Args:
            param_name: parameter name to match
            all_parameters: all available parameter info

        Returns:
            Optional[Dict[str, Any]]: matched parameter info, or None.
        """
        param_lower = param_name.lower()

        # Extract key semantic words
        semantic_keywords = []
        if "log" in param_lower:
            semantic_keywords.append("log")
        if "type" in param_lower or "types" in param_lower:
            semantic_keywords.append("type")
        if "preferred" in param_lower or "allowed" in param_lower or "supported" in param_lower:
            semantic_keywords.append("preference")
        if "count" in param_lower or "amount" in param_lower or "quantity" in param_lower:
            semantic_keywords.append("count")
        if "item" in param_lower:
            semantic_keywords.append("item")

        # No key semantic words — return None
        if not semantic_keywords:
            return None

        # Try matching: find a parameter containing all the key semantic words
        best_match = None
        best_score = 0
        best_match_name = None

        for candidate_name, candidate_info in all_parameters.items():
            candidate_lower = candidate_name.lower()

            # Improvement 1: check singular/plural matching
            # E.g. logType vs logTypes, preferredLogTypes vs logType
            param_base = param_lower.rstrip('s')  # Remove trailing 's'
            candidate_base = candidate_lower.rstrip('s')

            # If base forms are identical (ignoring singular/plural), award a high score
            if param_base == candidate_base and param_base:
                score = 10  # High score: singular/plural match
                # Check type compatibility (array and string can be equivalent)
                param_type = candidate_info.get("type")
                if param_type in ["array", "string"]:
                    if score > best_score:
                        best_score = score
                        best_match = candidate_info
                        best_match_name = candidate_name
                        continue

            # Improvement 2: check semantic-keyword matching (singular/plural aware)
            # Score: more shared keywords -> higher score
            score = 0
            matched_keywords = []

            # Extract base keywords (ignoring singular/plural)
            param_keywords = set()
            for word in param_lower.split():
                base_word = word.rstrip('s')
                if base_word:
                    param_keywords.add(base_word)

            candidate_keywords = set()
            for word in candidate_lower.split():
                base_word = word.rstrip('s')
                if base_word:
                    candidate_keywords.add(base_word)

            # Count shared keywords
            common_keywords = param_keywords & candidate_keywords
            if len(common_keywords) >= 2:  # At least 2 shared keywords
                score = len(common_keywords)

            # If all keywords matched and types match, this is a good match
            if score > 0:
                # Check whether the parameter types match (when keywords match)
                param_type = candidate_info.get("type")

                # Improvement 3: array and string can be equivalent if semantically similar
                # E.g. preferredLogTypes (array) vs logType (string)
                if ("log" in semantic_keywords or "type" in semantic_keywords):
                    # If one is array and the other is string but semantics are similar, still a match
                    if param_type in ["array", "string"] and score >= 2:
                        if score > best_score:
                            best_score = score
                            best_match = candidate_info
                            best_match_name = candidate_name
                            continue

                if "count" in semantic_keywords and param_type == "number":
                    if score > best_score:
                        best_score = score
                        best_match = candidate_info
                        best_match_name = candidate_name
                elif score > best_score:
                    best_score = score
                    best_match = candidate_info
                    best_match_name = candidate_name

        # Log the match if found
        if best_match and best_match_name and best_match_name != param_name:
            match_type = "singular/plural form" if param_lower.rstrip('s') == best_match_name.lower().rstrip('s') else "semantically similar"
            print(f"\033[36m[Parameter Match] parameter name '{param_name}' matched ({match_type}) to '{best_match_name}'\033[0m")

        return best_match
