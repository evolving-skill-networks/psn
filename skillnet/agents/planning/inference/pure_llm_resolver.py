"""
PureLLMParameterResolver: Rich-context LLM parameter inference.

Standalone resolver for A/B comparison against STATE_MAPPING strategy.
Unlike the existing _infer_with_llm (parameter_engine.py:688-760) which
uses a minimal prompt (6 fields), this resolver passes ALL InferenceContext
fields for maximum context.

NOT in the production strategy chain — used only for comparison testing.
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from .parameter_engine import (
    InferenceContext,
    InferenceResult,
    InferenceStrategy,
    ParameterValue,
)
from skillnet.utils.stats_tracker import record_llm_usage


logger = logging.getLogger("skillnet.planning.inference.pure_llm_resolver")


SYSTEM_PROMPT = """You are a parameter inference expert for Minecraft skill composition.

Given a skill, its parameter metadata, target effects, current/expected inventory, environment, and call chain context, infer the correct parameter value.

PRIORITY ORDER for inferring values:
1. Target effects (strongest signal — extract directly from effect fields)
2. Semantic metadata and supported_values (constrain the answer space)
3. Default value (use when effects don't provide a clear answer)
4. Task description and environment are CONTEXT for disambiguation — they help choose
   between candidates but should NOT introduce values that aren't grounded in effects,
   supported_values, or the default.

RULES:
1. For count parameters: Extract from effect's count field IF it contains a valid number.
   If the effect has no count, has a non-numeric count (e.g., a variable name like
   "targetCount"), or there are no effects, use the default value.
   Do NOT extract numbers from the task description — the task describes the overall
   goal, which may not correspond 1:1 to this parameter's value.
2. For item type parameters (logType, oreType, material, plankName, etc.):
   - If the parameter represents an INPUT material (precondition), infer from what's needed to produce the target effect.
     Example: craftAxe produces "wooden_axe" → material parameter should be "wooden" (the material prefix).
   - If the parameter represents a DIRECT output item, use the effect's 'item' field directly.
     Example: mineBlock produces "iron_ore" → oreType should be "iron_ore".
   - NEVER return abstract concepts like "INPUT", "OUTPUT", "material".
   - Check supported_values if provided — your answer MUST be one of them.
   - Check expected_inventory — prefer items that will actually be available.
   - Your answer must be an exact item identifier (e.g., "birch_planks", "oak_log",
     "iron_ore"), not an abbreviated form (e.g., "birch", "oak", "iron").
     When an effect contains a specific item name, prefer that exact name.
3. For fuel parameters: Return a valid fuel item like "coal", "charcoal", "oak_planks".
4. For config parameters (timeout, radius, etc.): Prefer the default value. Config parameters control
   skill behavior (timeout, radius, count limits), NOT task targets. Only override the default if the
   task explicitly specifies a different value for this specific parameter.
5. Environment context (nearby_blocks, biome, position) is SUPPLEMENTARY:
   - Use it to disambiguate when multiple values are equally valid.
   - Do NOT let environment override a value already determined from effects or
     supported_values.
   - nearby_blocks lists block names — these are not necessarily the same as
     parameter values (e.g., "birch_log" nearby doesn't mean plankType="birch").
6. If optimizer corrections are shown, they come from analysis of previous execution
   failures within this task. Strongly prefer the suggested value unless your current
   context clearly indicates a different value is needed.

Return ONLY valid JSON: {"value": <inferred_value>, "reasoning": "<brief explanation>", "confidence": <0.0-1.0>}
"""


class PureLLMParameterResolver:
    """Rich-context LLM parameter inference for A/B comparison.

    Uses ALL InferenceContext fields for maximum context, unlike the
    production _infer_with_llm which only uses 6 fields.
    """

    def __init__(self, llm, logger=None, system_prompt=None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self._system_prompt = system_prompt or SYSTEM_PROMPT

    def resolve(self, context: InferenceContext) -> Optional[InferenceResult]:
        """Resolve parameter value using rich-context LLM prompt.

        Args:
            context: Full InferenceContext with all available fields.

        Returns:
            InferenceResult if successful, None otherwise.
        """
        if not self.llm:
            return None

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            user_prompt = self._build_user_prompt(context)

            messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=user_prompt),
            ]

            start_time = time.monotonic()
            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="planning_param_inference", function_name="pure_llm_resolver.resolve", skill_name=context.skill_name)
            latency_ms = (time.monotonic() - start_time) * 1000

            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            parsed = self._parse_response(response_text, context.get_param_type())
            if parsed is None:
                return None

            value, confidence, reasoning = parsed

            # Reject placeholder values
            PLACEHOLDER_VALUES = {"INPUT", "OUTPUT", "material", "item", "type", "value"}
            if isinstance(value, str) and value.upper() in {v.upper() for v in PLACEHOLDER_VALUES}:
                self.logger.warning(
                    f"[PureLLM] Returned placeholder '{value}', rejecting"
                )
                return None

            pv = ParameterValue.from_python(value, context.get_param_type())
            return InferenceResult(
                value=pv.js_literal,
                strategy_used=InferenceStrategy.LLM,
                confidence=confidence,
                explanation=f"PureLLM: {reasoning} (latency: {latency_ms:.0f}ms)",
                raw_value=pv.raw_value,
            )

        except Exception as e:
            self.logger.warning(f"[PureLLM] Inference failed: {e}")
            return None

    def _build_user_prompt(self, context: InferenceContext) -> str:
        """Build rich user prompt from all InferenceContext fields."""
        sections = []

        # === Basic Info ===
        sections.append(f"Skill: {context.skill_name}")
        sections.append(f"Parameter: {context.param_name}")
        sections.append(f"Type: {context.get_param_type()}")
        sections.append(f"Default: {context.get_default_value()}")

        supported = context.get_supported_values()
        if supported:
            sections.append(f"Supported values: {json.dumps(supported)}")

        # === Target Effects ===
        sections.append("")
        sections.append("== Target Effects ==")
        sections.append(json.dumps(context.target_effects, indent=2))

        # === Current Task ===
        sections.append("")
        sections.append(f"== Current Task ==")
        sections.append(context.current_task or "Unknown")

        # === Precondition Context ===
        if context.precondition_context:
            sections.append("")
            sections.append("== Precondition Context ==")
            sections.append(json.dumps(context.precondition_context, indent=2))

        # === Semantic Metadata ===
        if context.semantic:
            sections.append("")
            sections.append("== Parameter Semantic Metadata ==")
            sem = context.semantic
            sem_info = {
                "direction": sem.direction.value if hasattr(sem.direction, 'value') else str(sem.direction),
                "quantity_semantic": sem.quantity_semantic.value if hasattr(sem.quantity_semantic, 'value') else str(sem.quantity_semantic),
            }
            if sem.state_mapping:
                sem_info["state_mapping"] = sem.state_mapping.to_dict()
            if sem.config_context:
                sem_info["config_context"] = sem.config_context
            sections.append(json.dumps(sem_info, indent=2))

        # === Expected Inventory ===
        if context.expected_inventory:
            sections.append("")
            sections.append("== Expected Inventory (from upstream skills) ==")
            sections.append(json.dumps(context.expected_inventory, indent=2))

        # === Current Inventory ===
        if context.current_inventory:
            sections.append("")
            sections.append("== Current Inventory ==")
            sections.append(json.dumps(context.current_inventory, indent=2))

        # === Environment ===
        has_env = context.biome or context.nearby_blocks or context.position or context.equipment
        if has_env:
            sections.append("")
            sections.append("== Environment ==")
            if context.biome:
                sections.append(f"Biome: {context.biome}")
            if context.position:
                sections.append(f"Position: {json.dumps(context.position)}")
            if context.nearby_blocks:
                sections.append(f"Nearby blocks: {json.dumps(context.nearby_blocks)}")
            if context.equipment:
                sections.append(f"Equipment: {json.dumps(context.equipment)}")

        # === Call Chain ===
        if context.upstream_output or context.caller_skill or context.output_to_input_mapping:
            sections.append("")
            sections.append("== Call Chain ==")
            if context.caller_skill:
                sections.append(f"Caller skill: {context.caller_skill}")
            if context.upstream_output:
                sections.append(f"Upstream output: {context.upstream_output}")
            if context.output_to_input_mapping:
                sections.append(f"Output→Input mapping: {json.dumps(context.output_to_input_mapping)}")

        # === Optimizer Corrections ===
        if context.parameter_corrections:
            sections.append("")
            sections.append("== Optimizer Corrections (from previous failure analysis) ==")
            for corr in context.parameter_corrections:
                sections.append(
                    f"- Value {corr['passed_value']!r} was used and caused failure."
                    f" Suggested: {corr['suggested_value']!r}"
                    f" (Reason: {corr.get('reason', 'N/A')})"
                )

        sections.append("")
        sections.append("What value should this parameter have?")

        return "\n".join(sections)

    def _parse_response(self, response: str, param_type: str):
        """Parse LLM response to extract value, confidence, and reasoning.

        Returns:
            Tuple of (value, confidence, reasoning) or None if parsing fails.
        """
        # Try direct JSON parse
        data = self._extract_json(response)
        if data and "value" in data:
            confidence = float(data.get("confidence", 0.7))
            reasoning = data.get("reasoning", "LLM inference")
            return data["value"], confidence, reasoning

        # Fallback: extract value for simple types
        if param_type == "number":
            num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
            if num_match:
                return float(num_match.group(1)), 0.5, "Extracted number from response"

        if param_type == "string":
            str_match = re.search(r'"([^"]+)"', response)
            if str_match:
                return str_match.group(1), 0.5, "Extracted string from response"

        return None

    def _extract_json(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response, handling markdown wrapping."""
        # Direct parse
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Markdown JSON block
        json_match = re.search(r'```json\n(.+?)\n```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

        # Find JSON object with "value" key
        json_match = re.search(r'\{[^{}]*"value"\s*:.+?\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

        return None
