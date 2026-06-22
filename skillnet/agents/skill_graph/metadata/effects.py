"""
Effect Extraction Module

Extracts effects from code and task intent.

v5.0 architecture reorganization — migrated from graph_manager_impl.py
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import HumanMessage, SystemMessage

from skillnet.agents.skill_graph.models import SkillEffect
from skillnet.agents.skill_graph.models.execution import SkillExecutionTrace
from skillnet.agents.skill_graph.utils import (
    code_has_effect_implementation,
    validate_naming_effect_consistency,
)
from .runtime_validator import (
    RuntimeEffectValidator,
    ValidationStatus,
    get_runtime_validator,
)

from skillnet.core.dk_registry import get_domain_knowledge as _get_domain_knowledge

logger = logging.getLogger(__name__)


# ============================================================================
# System Prompts
# ============================================================================

INTENT_EXTRACTION_SYSTEM_PROMPT = """You are a helpful assistant that analyzes Minecraft task descriptions and extracts expected effects for skill planning.

An expected effect is the FINAL GOAL that the skill achieves - what the task aims to ACHIEVE, not intermediate steps or consumed materials.

CRITICAL: Only extract FINAL GOALS (what should be added/created/placed):
- Use "add" operation for items that should be obtained/created (e.g., "Adds 3 oak_log to inventory")
- Use "place" operation for blocks that should be placed (e.g., "Places 1 crafting_table block")
- DO NOT extract "remove" operations for consumed materials - those are intermediate steps, not goals

IMPORTANT - is_primary field:
- Set "is_primary": true for the effect that directly matches the task goal (the main purpose of the skill)
- Set "is_primary": false for intermediate products or side effects
- Usually there should be exactly ONE primary effect per skill

Examples:
- Task: "Mine 3 oak logs" → Effect: "Adds 3 oak_log to inventory" (is_primary: true)
- Task: "Craft 4 oak planks" → Effect: "Adds 4 oak_planks to inventory" (is_primary: true)
- Task: "Craft 1 crafting table" → Effect: "Adds 1 crafting_table to inventory" (is_primary: true)
  Note: Even if oak_planks are created as intermediate, crafting_table is the primary effect

Focus on FINAL GOALS that match what the task description asks for.

IMPORTANT: The state_representation supports AND/OR logic:
- Default format (list, OR logic for effects): [{"type": "inventory", "item": "oak_log", "count": 3, "operation": "add"}, ...]
- OR logic format: {"logic": "OR", "conditions": [...]}
- AND logic format: {"logic": "AND", "conditions": [...]}

Return ONLY a JSON object in this format:
{
  "effects": [
    {
      "description": "Adds 1 wooden_pickaxe to inventory",
      "code": "...",
      "state_representation": {"type": "inventory", "item": "wooden_pickaxe", "count": 1, "operation": "add"},
      "is_primary": true
    }
  ]
}"""

CODE_EXTRACTION_SYSTEM_PROMPT = """You are a helpful assistant that analyzes Minecraft skill code and extracts effects for skill planning.

An expected effect is the outcome/change that the skill produces.

CRITICAL: Extract effects that represent GAME STATE CHANGES:
- Items added to inventory (mining, crafting, picking up)
- Items removed from inventory (consuming materials)
- Blocks placed in the world
- Equipment changes

IMPORTANT - is_primary field (CRITICAL for correct skill matching):
Use these 4 heuristic rules to determine is_primary:

1. FUNCTION NAME MATCH: If the skill function name contains the item name, that effect is likely primary
   - Example: "craftOakCraftingTable" → crafting_table is primary, oak_planks is NOT primary
   - Example: "mineOakLog" → oak_log is primary

2. CALL HIERARCHY: Effects from the main function body (not nested helper calls) are more likely primary
   - Direct bot.craft() or bot.dig() in main function → likely primary
   - Effects from await helperFunction() → likely NOT primary

3. CONDITIONAL NESTING: Effects inside "if (!hasEnough)" or similar conditions are intermediate
   - These are "ensure we have materials" steps, not the final goal

4. PURPOSE ALIGNMENT: The primary effect should match the skill's stated purpose/description
   - If description says "craft a crafting table", crafting_table is primary
   - Intermediate crafting steps (like planks) are NOT primary

Set "is_primary": true for the effect that IS the main goal of the skill
Set "is_primary": false for intermediate products, consumed materials, or side effects

For each effect, provide:
1. A human-readable description (e.g., "Adds 3 oak_log to inventory")
2. Executable code to verify the state change
3. state_representation in the correct format
4. is_primary boolean

IMPORTANT: The state_representation uses this format:
- {"type": "inventory", "item": "item_name", "count": number, "operation": "add"}
- operation MUST be one of: "add", "remove", "place", "equip" (NOT "ensure" or other values)

Return ONLY a JSON object in this format:
{
  "effects": [
    {
      "description": "...",
      "code": "...",
      "state_representation": {...},
      "is_primary": true/false
    }
  ]
}"""

# Shared tail for both extraction prompts: dimension transitions are runtime
# verifiable (status.js emits the dimension), so they are a legal effect type.
_DIMENSION_PROMPT_EXTENSION = """

DIMENSION-CHANGE TASKS: if the task or code requires ENTERING another dimension (e.g. "enter the nether", walking through a nether portal or end portal), express that final goal as:
{"type": "dimension", "dimension": "the_nether", "count": 1, "operation": "enter"}
(use "the_end" for end portals, "overworld" for returning). Building or lighting a portal is still a "place"/"inventory" effect; only the actual dimension transition uses "dimension"."""

INTENT_EXTRACTION_SYSTEM_PROMPT += _DIMENSION_PROMPT_EXTENSION
CODE_EXTRACTION_SYSTEM_PROMPT += _DIMENSION_PROMPT_EXTENSION


MERGE_SYSTEM_PROMPT = """You are an expert at analyzing and merging Minecraft skill effects.

Your task is to merge effects from two sources (intent-based and code-based) by:
1. Identifying semantically similar effects (even if descriptions differ)
2. Merging similar effects into one, choosing the best description and code
3. Keeping unique effects from both sources
4. Prioritizing code-based effects when there are conflicts

Two effects are considered SIMILAR if they:
- Produce/consume the same items (e.g., "Produces 1 wooden_pickaxe" vs "Creates 1 wooden pickaxe")
- Have equivalent outcomes (e.g., "Consumes 3 planks, 2 sticks" vs "Uses 3 planks and 2 sticks")
- Represent the same state change (e.g., "Mines 1 oak_log" vs "Obtains 1 oak log")

When merging similar effects:
- Prefer code-based description and code (more accurate)
- Preserve the state_representation from the source effects
- IMPORTANT: Preserve is_primary from the source effects (prefer code-based if conflict)

Return ONLY a JSON object in this format:
{
  "merged_effects": [
    {
      "description": "merged description",
      "code": "merged code",
      "state_representation": {...},
      "is_primary": true/false,
      "sources": ["intent", "code"] or ["intent"] or ["code"]
    }
  ]
}"""


# ============================================================================
# Name-based Primary Effect Inference (for wrapper skill repair)
# ============================================================================

def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case: craftWoodenPickaxe → craft_wooden_pickaxe"""
    result = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return result.lower()


def _singularize(name: str) -> str:
    """Basic singularization for Minecraft item names."""
    # Special cases first
    if name.endswith("_ingots"):
        return name[:-1]  # iron_ingots → iron_ingot
    if name.endswith("ches"):  # torches → torch
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss"):  # diamonds → diamond, sticks → stick
        return name[:-1]
    return name


_VERB_PREFIXES = ("craft_", "mine_", "ensure_", "smelt_", "find_", "get_")


def _is_valid_item_name(name: str) -> bool:
    """Check item name validity via domain knowledge or permissive default."""
    dk = _get_domain_knowledge()
    if dk is not None:
        return dk.is_valid_item(name)
    # No domain knowledge — permissive default (accept all names)
    return True


def infer_primary_effect_from_name(skill_name: str) -> Optional[SkillEffect]:
    """
    Infer primary effect from skill function name, validated against known item names.

    Uses domain-injected knowledge when available, falls back to Minecraft item list.

    Examples:
        craftWoodenPickaxe → SkillEffect(item="wooden_pickaxe", is_primary=True)
        craftStonePickaxe  → SkillEffect(item="stone_pickaxe", is_primary=True)
        ensureDiamonds     → SkillEffect(item="diamond", is_primary=True)
        craftWoodenTool    → None (wooden_tool not a valid item)
        getMcDataAndVec3   → None (not a craft/mine/ensure verb)

    Returns:
        SkillEffect with is_primary=True if a valid item can be inferred, else None.
    """
    snake = _camel_to_snake(skill_name)

    # Strip verb prefix
    item_name = None
    for prefix in _VERB_PREFIXES:
        if snake.startswith(prefix):
            item_name = snake[len(prefix):]
            break

    if not item_name:
        return None

    # Try direct match first
    if _is_valid_item_name(item_name):
        pass  # valid
    else:
        # Try singularized
        singular = _singularize(item_name)
        if _is_valid_item_name(singular):
            item_name = singular
        else:
            return None  # not a valid item

    return SkillEffect(
        description=f"Produces {item_name} (inferred from skill name '{skill_name}')",
        code="",
        state_representation={
            "type": "inventory",
            "item": item_name,
            "count": 1,
            "operation": "add",
        },
        is_primary=True,
        confidence_level="inferred",
        importance="core",
        inference_stats={"validation_method": "name_based_inference"},
    )


def synthesize_general_primary_effect(
    products: List[str], param_name: str
) -> Optional[SkillEffect]:
    """
    Build a parameter-bound primary product effect for a parameterized GENERAL
    skill created by sibling refactoring (e.g. craftWoodenTool over the siblings
    craftWoodenAxe / craftWoodenPickaxe).

    A general skill's product is parameter-dependent: the crafted item is selected
    by ``param_name`` (e.g. ``toolType``), so the parametric code contains no
    literal product name. A fixed item effect therefore cannot represent the
    product and would be stripped by ``_validate_effects_against_code`` (the
    "not implemented in code" rule), leaving the skill with no primary effect --
    which lets EffectMatcher's no-primary fallback mis-match intermediate
    by-products.

    Instead we record the product as an OR over the union of the siblings' product
    items, marked ``is_primary=True`` and parameter-bound via ``condition`` so that
    (a) EffectMatcher prefers it over by-products, and (b) effect-validation
    recognises it as implemented-via-parameter (see
    ``MetadataValidationMixin._effect_is_param_implemented``).

    Args:
        products: the siblings' primary product item names,
                  e.g. ``["wooden_axe", "wooden_pickaxe"]``.
        param_name: the product-selecting parameter, e.g. ``"toolType"``.

    Returns:
        ``SkillEffect(is_primary=True)``, or ``None`` if no products are given.
    """
    items = [p for p in dict.fromkeys(products or []) if p]
    if not items:
        return None
    if len(items) == 1:
        state_representation = {
            "type": "inventory", "item": items[0], "count": 1, "operation": "add",
        }
    else:
        state_representation = {
            "logic": "OR",
            "conditions": [
                {"type": "inventory", "item": it, "count": 1, "operation": "add"}
                for it in items
            ],
        }
    condition = {param_name: list(items)} if param_name else None
    return SkillEffect(
        description=(
            f"Produces the requested wooden tool ({' / '.join(items)}), "
            f"selected by parameter '{param_name}'."
        ),
        code="",
        state_representation=state_representation,
        is_primary=True,
        importance="core",
        confidence_level="inferred",
        condition=condition,
        inference_stats={"validation_method": "sibling_product_union"},
    )


# ============================================================================
# EffectExtractor Class
# ============================================================================

class EffectExtractor:
    """
    Effect extractor

    Responsibilities:
    1. Extract effects from task intent (task+context)
    2. Extract effects from code
    3. Merge effects from both sources
    4. Validate that effects are consistent with the code
    """

    def __init__(
        self,
        llm,  # LangChain LLM
        merge_mode: str = "llm",
        custom_logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the effect extractor

        Args:
            llm: LangChain LLM instance
            merge_mode: merge mode, "simple" or "llm"
            custom_logger: optional custom logger
        """
        self.llm = llm
        self.merge_mode = merge_mode
        self.logger = custom_logger or logger

    def extract(
        self,
        code: str,
        description: str,
        task: Optional[str] = None,
        context: Optional[str] = None,
        skill_name: Optional[str] = None,
        execution_traces: Optional[List[SkillExecutionTrace]] = None,
    ) -> Tuple[List[SkillEffect], List[str]]:
        """
        Hybrid effect extraction: extract from both code and task intent, then verify consistency

        Args:
            code: skill code
            description: skill description
            task: task description (optional)
            context: contextual info (optional)
            skill_name: skill name (used to infer the primary effect)
            execution_traces: optional execution history, used for runtime verification

        Returns:
            Tuple[List[SkillEffect], List[str]]: (extracted effects list, warnings list)
        """
        warnings = []

        # ========== Early intercept for FIND tasks ==========
        # FIND tasks produce "nearby_block" effects, not "inventory"
        # Must intercept here, otherwise downstream extract_from_code + merge will overwrite
        if task:
            task_lower = task.lower()
            find_pattern = r'(?:find|locate|discover|search\s+for)\s+(\d+)?\s*(\w+(?:[\s_]\w+)*)'
            match = re.search(find_pattern, task_lower)
            if match:
                count = int(match.group(1)) if match.group(1) else 1
                item = match.group(2).replace(" ", "_")
                self.logger.info(f"[EffectExtractor] FIND task detected: {task}, returning nearby_block effects")
                effects = [SkillEffect(
                    description=f"Locates {item} nearby",
                    code="",  # FIND tasks don't need verification code
                    state_representation={
                        "type": "nearby_block",
                        "item": item,
                        "count": count,
                        "operation": "find"
                    },
                    is_primary=True,
                    confidence_level="high"
                )]
                return effects, warnings  # Return directly, skipping the rest of intent/code extraction and merge
        # ========== End FIND task early intercept ==========

        # Stage 1: extract from intent (task+context)
        intent_effects = []
        if task or context:
            try:
                intent_effects = self.extract_from_intent(
                    task=task, context=context, description=description
                )
                self.logger.info(
                    f"[Intent-based] Extracted {len(intent_effects)} effects from task/context"
                )
            except Exception as e:
                warnings.append(f"Failed to extract effects from intent: {e}")

        # Stage 2: extract from code
        code_effects = []
        try:
            code_effects = self.extract_from_code(code, description, skill_name)
            self.logger.info(
                f"[Code-based] Extracted {len(code_effects)} effects from code"
            )
        except Exception as e:
            warnings.append(f"Failed to extract effects from code: {e}")

        # Stage 3: merge and deduplicate
        merged_effects = self.merge(intent_effects, code_effects)

        # Stage 4: verify that the code implements these effects (supports runtime verification)
        validated_effects, validation_warnings = self.validate_against_code(
            merged_effects, code, execution_traces=execution_traces
        )
        warnings.extend(validation_warnings)

        return validated_effects, warnings

    def extract_from_intent(
        self,
        task: Optional[str] = None,
        context: Optional[str] = None,
        description: Optional[str] = None,
    ) -> List[SkillEffect]:
        """
        Extract effects from task intent (task+context)

        Args:
            task: task description
            context: contextual info
            description: skill description

        Returns:
            List[SkillEffect]: list of effects extracted from intent
        """
        effects = []

        if not task and not context:
            return effects

        try:
            messages = [
                SystemMessage(content=INTENT_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""Task: {task or "N/A"}
Context: {context or "N/A"}
Skill Description: {description or "N/A"}

Analyze the task and context to extract all expected effects. Return only JSON."""
                ),
            ]

            _llm_resp = self.llm.invoke(messages)
            from skillnet.utils.stats_tracker import record_llm_usage
            record_llm_usage(_llm_resp, process_type="metadata_extraction", function_name="metadata.effects.from_task")
            response = _llm_resp.content

            # Parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for e_data in data.get("effects", []):
                    effects.append(
                        SkillEffect(
                            description=e_data.get("description", ""),
                            code=e_data.get("code", ""),
                            state_representation=e_data.get("state_representation", {}),
                            is_primary=e_data.get("is_primary", False),
                        )
                    )
        except Exception as e:
            self.logger.warning(f"Failed to extract effects from intent: {e}")

        return effects

    def extract_from_code(
        self, code: str, description: str, skill_name: Optional[str] = None
    ) -> List[SkillEffect]:
        """
        Extract effects from code

        Args:
            code: skill code
            description: skill description
            skill_name: skill name (used to infer the primary effect)

        Returns:
            List[SkillEffect]: list of effects extracted from code
        """
        effects = []

        try:
            skill_name_hint = ""
            if skill_name:
                skill_name_hint = f"\nSkill name: {skill_name} (use this to infer the primary effect)"

            messages = [
                SystemMessage(content=CODE_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""Skill code:
{code}

Skill description:
{description}{skill_name_hint}

Analyze the code and extract all effects. Return only JSON."""
                ),
            ]

            _llm_resp = self.llm.invoke(messages)
            from skillnet.utils.stats_tracker import record_llm_usage
            record_llm_usage(_llm_resp, process_type="metadata_extraction", function_name="metadata.effects.from_code")
            response = _llm_resp.content

            # Parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for e_data in data.get("effects", []):
                    effects.append(
                        SkillEffect(
                            description=e_data.get("description", ""),
                            code=e_data.get("code", ""),
                            state_representation=e_data.get("state_representation", {}),
                            is_primary=e_data.get("is_primary", False),
                        )
                    )
        except Exception as e:
            self.logger.warning(f"Failed to extract effects from code: {e}")

        return effects

    def merge(
        self,
        intent_effects: List[SkillEffect],
        code_effects: List[SkillEffect],
    ) -> List[SkillEffect]:
        """
        Merge effects extracted from intent and from code

        Args:
            intent_effects: effects extracted from intent
            code_effects: effects extracted from code

        Returns:
            List[SkillEffect]: merged effects
        """
        # If lists are empty or only one source is present, return directly
        if not intent_effects and not code_effects:
            return []
        if not intent_effects:
            return code_effects
        if not code_effects:
            return intent_effects

        # Choose strategy based on merge_mode
        if self.merge_mode == "simple":
            return self._merge_simple(intent_effects, code_effects)
        elif self.merge_mode == "llm":
            total_count = len(intent_effects) + len(code_effects)
            if total_count <= 3:
                return self._merge_simple(intent_effects, code_effects)

            try:
                return self._merge_with_llm(intent_effects, code_effects)
            except Exception as e:
                self.logger.warning(
                    f"[Effect Merge] LLM merge failed: {e}, falling back to simple merge"
                )
                return self._merge_simple(intent_effects, code_effects)
        else:
            self.logger.warning(
                f"[Effect Merge] Unknown merge_mode '{self.merge_mode}', using simple merge"
            )
            return self._merge_simple(intent_effects, code_effects)

    def _merge_simple(
        self,
        intent_effects: List[SkillEffect],
        code_effects: List[SkillEffect],
    ) -> List[SkillEffect]:
        """Simple merge strategy (based on exact description match)"""
        merged = []
        seen_descriptions = set()

        # Prefer effects from code (more accurate)
        for effect in code_effects:
            if effect.description not in seen_descriptions:
                merged.append(effect)
                seen_descriptions.add(effect.description)

        # Add effects unique to intent
        for effect in intent_effects:
            if effect.description not in seen_descriptions:
                merged.append(effect)
                seen_descriptions.add(effect.description)

        return merged

    def _merge_with_llm(
        self,
        intent_effects: List[SkillEffect],
        code_effects: List[SkillEffect],
    ) -> List[SkillEffect]:
        """Use the LLM to intelligently merge effects"""
        # Prepare input data (includes is_primary)
        intent_data = [
            {
                "description": e.description,
                "code": e.code,
                "state_representation": e.state_representation if e.state_representation else {},
                "is_primary": getattr(e, 'is_primary', False),
            }
            for e in intent_effects
        ]
        code_data = [
            {
                "description": e.description,
                "code": e.code,
                "state_representation": e.state_representation if e.state_representation else {},
                "is_primary": getattr(e, 'is_primary', False),
            }
            for e in code_effects
        ]

        human_prompt = f"""Merge these effects:

INTENT-BASED EFFECTS (from task/context):
{json.dumps(intent_data, indent=2, ensure_ascii=False)}

CODE-BASED EFFECTS (from code analysis):
{json.dumps(code_data, indent=2, ensure_ascii=False)}

Analyze and merge semantically similar effects. Return only JSON."""

        messages = [
            SystemMessage(content=MERGE_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ]

        _llm_resp = self.llm.invoke(messages)
        from skillnet.utils.stats_tracker import record_llm_usage
        record_llm_usage(_llm_resp, process_type="metadata_extraction", function_name="metadata.effects.merge")
        response = _llm_resp.content

        # Parse the JSON response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            merged = []
            for e_data in data.get("merged_effects", []):
                state_repr = e_data.get("state_representation", {})

                # If the LLM did not return it, try to look it up from the original effects
                if not state_repr or state_repr == {}:
                    merged_desc = e_data.get("description", "")
                    sources = e_data.get("sources", [])

                    # Prefer to look up in code_effects
                    if "code" in sources:
                        for orig_effect in code_effects:
                            if (
                                orig_effect.description == merged_desc
                                and orig_effect.state_representation
                            ):
                                state_repr = orig_effect.state_representation
                                break

                    # If still not found, look up in intent_effects
                    if (not state_repr or state_repr == {}) and "intent" in sources:
                        for orig_effect in intent_effects:
                            if (
                                orig_effect.description == merged_desc
                                and orig_effect.state_representation
                            ):
                                state_repr = orig_effect.state_representation
                                break

                # Get is_primary (prefer LLM's return value, then look it up from the original effects)
                is_primary = e_data.get("is_primary", None)
                if is_primary is None:
                    merged_desc = e_data.get("description", "")
                    sources = e_data.get("sources", [])
                    # Prefer to look up in code_effects (code-based is more accurate)
                    if "code" in sources:
                        for orig_effect in code_effects:
                            if orig_effect.description == merged_desc:
                                is_primary = getattr(orig_effect, 'is_primary', False)
                                break
                    # If still not found, look up in intent_effects
                    if is_primary is None and "intent" in sources:
                        for orig_effect in intent_effects:
                            if orig_effect.description == merged_desc:
                                is_primary = getattr(orig_effect, 'is_primary', False)
                                break
                    if is_primary is None:
                        is_primary = False

                merged.append(
                    SkillEffect(
                        description=e_data.get("description", ""),
                        code=e_data.get("code", ""),
                        state_representation=state_repr if state_repr else {},
                        is_primary=is_primary,
                    )
                )
            self.logger.info(
                f"[Effect Merge] LLM merged {len(intent_effects) + len(code_effects)} effects into {len(merged)}"
            )
            return merged
        else:
            raise ValueError("Failed to parse LLM response as JSON")

    def validate_against_code(
        self,
        effects: List[SkillEffect],
        code: str,
        execution_traces: Optional[List[SkillExecutionTrace]] = None,
    ) -> Tuple[List[SkillEffect], List[str]]:
        """
        Validate whether effects have a corresponding implementation in the code

        Hybrid validation strategy:
        1. Static analysis: check the code for an explicit implementation
        2. Runtime verification: if static analysis fails, inspect execution history

        Args:
            effects: list of effects to validate
            code: skill code
            execution_traces: optional execution history, used for runtime verification

        Returns:
            Tuple[List[SkillEffect], List[str]]: (validated effects, warnings list)
        """
        validated = []
        warnings = []

        # Initialize the runtime validator (if there is execution history)
        runtime_validator = None
        if execution_traces:
            runtime_validator = get_runtime_validator(min_samples=3)

        for effect in effects:
            # Layer 1: static analysis
            has_static_impl = code_has_effect_implementation(code, effect)

            if has_static_impl:
                # Static analysis confirmed: high confidence
                effect.confidence_value = 1.0
                effect.confidence_level = "high"
                effect.inference_stats = {
                    **(effect.inference_stats or {}),
                    "validation_method": "static_analysis",
                    "validation_status": "confirmed",
                }
                validated.append(effect)
                continue

            # Layer 2: runtime verification (if there is execution history)
            if runtime_validator:
                runtime_result = runtime_validator.validate_effect(
                    effect, execution_traces
                )

                # Decide based on the runtime result
                if runtime_result.status in (
                    ValidationStatus.CONFIRMED,
                    ValidationStatus.LIKELY,
                ):
                    # Runtime confirmed or likely: keep effect, set confidence
                    runtime_validator.update_effect_confidence(effect, runtime_result)
                    effect.inference_stats["validation_method"] = "runtime_evidence"
                    validated.append(effect)
                    self.logger.info(
                        f"[Effect Validation] Runtime verification passed: {effect.description[:50]}... "
                        f"(confidence={runtime_result.confidence_value:.2f})"
                    )
                    continue
                elif runtime_result.status == ValidationStatus.INSUFFICIENT_DATA:
                    # Insufficient data: keep effect, mark as uncertain
                    runtime_validator.update_effect_confidence(effect, runtime_result)
                    effect.inference_stats["validation_method"] = "insufficient_data"
                    validated.append(effect)
                    warnings.append(
                        f"[UNCERTAIN] Effect '{effect.description}' has insufficient runtime data, "
                        f"keeping with low confidence"
                    )
                    self.logger.info(
                        f"[Effect Validation] Insufficient data, temporarily kept: {effect.description[:50]}..."
                    )
                    continue
                else:
                    # Runtime verification indicates unlikely
                    warnings.append(
                        f"[DISCARDED] Effect '{effect.description}' not confirmed by static analysis "
                        f"or runtime evidence ({runtime_result.reason})"
                    )
                    self.logger.warning(
                        f"[Effect Validation] Runtime verification failed, discarded: {effect.description[:50]}..."
                    )
                    continue

            # No execution history and static analysis failed: keep but mark uncertain (cold-start exploration)
            effect.confidence_value = 0.3
            effect.confidence_level = "uncertain"
            effect.inference_stats = {
                **(effect.inference_stats or {}),
                "validation_method": "cold_start",
                "validation_status": "unverified",
            }
            validated.append(effect)
            warnings.append(
                f"[COLD_START] Effect '{effect.description}' kept with uncertain confidence "
                f"(no execution history, static analysis failed)"
            )
            self.logger.info(
                f"[Effect Validation] Cold-start retained: {effect.description[:50]}... (confidence=0.3)"
            )

        return validated, warnings
