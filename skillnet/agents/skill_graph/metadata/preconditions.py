"""
Precondition Extraction Module

Extract preconditions from code and task intent.

v5.0 architectural reorganization — migrated from graph_manager_impl.py.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import HumanMessage, SystemMessage

from skillnet.agents.skill_graph.models import SkillPrecondition
from skillnet.agents.skill_graph.utils import (
    code_has_precondition_check,
    filter_preconditions,
)

logger = logging.getLogger(__name__)


# ============================================================================
# System Prompts
# ============================================================================

_MINECRAFT_TOOL_TIER_FACTS = """
MINECRAFT TOOL REQUIREMENTS — factually correct list (do NOT invent stricter requirements):

Items with NO tool requirement (any tool or bare hands works; do NOT add a tool precondition):
  - All log/wood types: oak_log, birch_log, spruce_log, jungle_log, acacia_log, dark_oak_log, mangrove_log, cherry_log
  - All planks, bamboo, leaves, saplings
  - dirt, grass_block, sand, gravel, clay, snow
  - All flowers, crops, mushrooms, cactus
  - wool, cobweb (shears optional)
  Note: axes SPEED UP mining logs but are NOT required. Do NOT extract "axe" or "pickaxe"
  as a precondition for mining logs, planks, or any wood item.

Items requiring WOODEN pickaxe or better:
  - stone, cobblestone, andesite, diorite, granite, tuff
  - coal_ore, deepslate_coal_ore
  - nether_quartz_ore, sandstone, red_sandstone

Items requiring STONE pickaxe or better:
  - iron_ore, deepslate_iron_ore, raw_iron_block
  - copper_ore, deepslate_copper_ore
  - lapis_lazuli_ore, deepslate_lapis_ore

Items requiring IRON pickaxe or better:
  - diamond_ore, deepslate_diamond_ore
  - gold_ore, deepslate_gold_ore, nether_gold_ore
  - redstone_ore, deepslate_redstone_ore
  - emerald_ore, deepslate_emerald_ore

Items requiring DIAMOND pickaxe or better:
  - obsidian, crying_obsidian, ancient_debris

COMMON MISTAKES TO AVOID:
  Bad: "Bot has a wooden_pickaxe or better in inventory (required to mine oak_log)" — WRONG, oak_log needs no tool
  Bad: "Bot has a tool capable of mining logs" — WRONG, logs need no tool
  Bad: "Required to mine oak_log efficiently" — WRONG, 'efficiently' = optimization hint, not hard requirement
  Good: "At least 3 oak_planks and 2 sticks in inventory" — CORRECT for crafting wooden tools
  Good: "Has a stone_pickaxe or better in inventory" — CORRECT for mining iron_ore

PRODUCTION RULE: Do NOT extract preconditions for items that the skill itself produces or mines.
If the skill's code mines oak_log (e.g., mineBlock(bot, "oak_log", ...)), DO NOT list "oak_log in inventory"
as a precondition — the skill is designed to acquire oak_log, not to require it.
"""


# Single source of truth for the state_representation schema. Embedded in
# both INTENT and CODE extraction prompts so the LLM sees the identical
# contract regardless of extraction path.
_STATE_REPR_SCHEMA_SPEC = """
STATE_REPRESENTATION SCHEMA — STRICT CONTRACT (this is the ONLY legal format):

A state_representation is EITHER:
  (a) a list of leaf conditions (AND is implicit across the list), OR
  (b) a dict with exactly these two keys: {"logic": "AND"|"OR", "conditions": [...]}
      where each condition in the list is itself a state_representation (i.e.
      a leaf dict, a sub-list, or another {"logic":..., "conditions":...} block).

A LEAF CONDITION is a flat dict with exactly these keys:
  - "type": "inventory"                              (required)
  - "item": "<item_name>"                            (required — the Minecraft item id string)
  - "count": <integer >= 1>                          (optional, default 1)
  - "operation": "require" | "require_or_better"     (optional, default "require")

EXAMPLE 1 — "At least 3 oak_planks AND 2 sticks":
  [
    {"type": "inventory", "item": "oak_planks", "count": 3, "operation": "require"},
    {"type": "inventory", "item": "stick",      "count": 2, "operation": "require"}
  ]

EXAMPLE 2 — "Has a wooden_pickaxe OR stone_pickaxe OR iron_pickaxe OR diamond_pickaxe OR netherite_pickaxe"
           (i.e. "wooden_pickaxe or better" — this is THE MOST COMMON TOOL-TIER CASE):
  {
    "logic": "OR",
    "conditions": [
      {"type": "inventory", "item": "wooden_pickaxe",    "count": 1, "operation": "require"},
      {"type": "inventory", "item": "stone_pickaxe",     "count": 1, "operation": "require"},
      {"type": "inventory", "item": "iron_pickaxe",      "count": 1, "operation": "require"},
      {"type": "inventory", "item": "diamond_pickaxe",   "count": 1, "operation": "require"},
      {"type": "inventory", "item": "netherite_pickaxe", "count": 1, "operation": "require"}
    ]
  }

EXAMPLE 3 — "Has 3 planks AND (a wooden_pickaxe OR a stone_pickaxe)":
  {
    "logic": "AND",
    "conditions": [
      {"type": "inventory", "item": "oak_planks", "count": 3, "operation": "require"},
      {"logic": "OR", "conditions": [
        {"type": "inventory", "item": "wooden_pickaxe", "count": 1, "operation": "require"},
        {"type": "inventory", "item": "stone_pickaxe",  "count": 1, "operation": "require"}
      ]}
    ]
  }

FORBIDDEN — these shapes will be rejected by the checker; do NOT produce them:
  ✗ {"any_of": [...]}                    — use {"logic": "OR", "conditions": [...]}
  ✗ {"all_of": [...]}                    — use {"logic": "AND", "conditions": [...]}
  ✗ {"or": [...]} / {"and": [...]}       — use {"logic": "OR"/"AND", "conditions": [...]}
  ✗ {"any": [...]}                       — use {"logic": "OR", "conditions": [...]}
  ✗ {"inventory": {"any_of": [...]}}     — use {"logic": "OR", ...} at top level
  ✗ {"inventory": {"has_any_of": [...]}} — use {"logic": "OR", ...} at top level
  ✗ leaf keys "name" / "slot" / "operator" / "has_item" / "items" / "equipment"
      — the ONLY legal leaf keys are: type, item, count, operation
  ✗ FLAT list with multiple same-tier tools (wooden/stone/iron/diamond pickaxe)
      — this reads as AND (bot must have ALL), not OR. For "or better", USE EXAMPLE 2.

REMEMBER: "or better" / "X or Y or Z" → {"logic": "OR", "conditions": [...]} ONLY.
Never flatten it into a list; never invent new keys.
"""


INTENT_EXTRACTION_SYSTEM_PROMPT = f"""You are a helpful assistant that analyzes Minecraft task descriptions and extracts preconditions for skill planning.

A precondition is a HARD REQUIREMENT that must be met before the skill can be executed successfully. Preconditions are used for skill planning to determine if a skill can be executed given the current game state.

CRITICAL: Only extract HARD REQUIREMENTS that are essential for skill execution, such as:
- Required tools (e.g., "Has a stone_pickaxe or better in inventory" for mining iron_ore)
- Required materials (e.g., "At least 3 oak_planks in inventory" for crafting)
- Required equipment (e.g., "Has a wooden_pickaxe equipped")
- Required blocks in inventory (e.g., "At least 1 crafting_table in inventory")

DO NOT extract:
- Nearby blocks (e.g., "A log block exists within 32 blocks" - the skill can explore to find blocks)
- Empty inventory slots (e.g., "At least 1 empty inventory slot" - usually not a hard requirement)
- Position/biome requirements (unless absolutely critical)
- Technical implementation details
- Internal code dependencies
- Non-checkable assumptions
- OPTIMIZATION HINTS disguised as hard requirements. If you find yourself writing words like "efficiently", "preferably", "for best results", "to avoid", "recommended", "optional", "faster", or "more reliable" — the precondition is NOT a hard requirement, it is an optimization hint. DO NOT extract it.
- Items that the skill itself produces or mines (see Production Rule below).

{_MINECRAFT_TOOL_TIER_FACTS}

Focus on HARD REQUIREMENTS that can be verified by checking:
- bot.inventory.items() for inventory state (tools, materials, blocks)
- bot.entity.equipment for equipment

Extract SPECIFIC item quantities and requirements from the task description and context.

{_STATE_REPR_SCHEMA_SPEC}

Return ONLY a JSON object in this format (no other text):
{{
  "preconditions": [
    {{
      "description": "At least 3 planks and 2 sticks in inventory",
      "code": "...",
      "state_representation": [...]
    }}
  ]
}}"""

CODE_EXTRACTION_SYSTEM_PROMPT = f"""You are a helpful assistant that analyzes Minecraft skill code and extracts preconditions for skill planning.

A precondition is a HARD REQUIREMENT that must be met before the skill can be executed successfully.

CRITICAL: Only extract HARD REQUIREMENTS that are essential for skill execution, such as:
- Required tools (e.g., "Has a stone_pickaxe or better in inventory")
- Required materials (e.g., "At least 3 oak_planks in inventory")
- Required equipment (e.g., "Has a wooden_pickaxe equipped")
- Required blocks in inventory (e.g., "At least 1 crafting_table in inventory")

DO NOT extract:
- Nearby blocks (the skill can explore to find blocks)
- Empty inventory slots (usually not a hard requirement)
- Technical implementation details (e.g., "mcData must contain block entries")
- Internal code dependencies (e.g., "require statements must work")
- Non-checkable assumptions (e.g., "API must be available")
- OPTIMIZATION HINTS. Words like "efficiently", "preferably", "to avoid", "recommended", "optional", "faster", "more reliable" signal an optimization hint, NOT a hard requirement. DO NOT extract them.
- Items that the skill itself produces or mines (see Production Rule below).

{_MINECRAFT_TOOL_TIER_FACTS}

Focus on GAME STATE that can be verified by checking bot.inventory.items() or bot.entity.equipment.

{_STATE_REPR_SCHEMA_SPEC}

Return ONLY a JSON object in this format:
{{
  "preconditions": [
    {{
      "description": "...",
      "code": "...",
      "state_representation": {{}}
    }}
  ]
}}"""

MERGE_SYSTEM_PROMPT = """You are an expert at analyzing and merging Minecraft skill preconditions.

Your task is to merge preconditions from two sources (intent-based and code-based) by:
1. Identifying semantically similar preconditions (even if descriptions differ)
2. Merging similar preconditions into one, choosing the best description and code
3. Keeping unique preconditions from both sources
4. Prioritizing code-based preconditions when there are conflicts

Two preconditions are considered SIMILAR if they:
- Require the same items/materials (e.g., "3 planks" vs "at least 3 wooden planks")
- Check the same conditions (e.g., "crafting_table nearby" vs "crafting table within 6 blocks")
- Have equivalent requirements (e.g., "2 sticks" vs "two sticks")

When merging similar preconditions:
- Prefer code-based description and code (more accurate)
- Preserve the state_representation from the original precondition

Return ONLY a JSON object in this format:
{
  "merged_preconditions": [
    {
      "description": "merged description",
      "code": "merged code",
      "state_representation": {...} or null,
      "original_description": "the original description that had the best state_representation",
      "sources": ["intent", "code"] or ["intent"] or ["code"]
    }
  ]
}"""


# ============================================================================
# PreconditionExtractor Class
# ============================================================================

class PreconditionExtractor:
    """
    Precondition extractor.

    Responsibilities:
    1. Extract preconditions from task intent (task+context)
    2. Extract preconditions from code
    3. Merge preconditions from both sources
    4. Validate consistency between preconditions and code
    """

    def __init__(
        self,
        llm,  # LangChain LLM
        merge_mode: str = "llm",
        custom_logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the precondition extractor.

        Args:
            llm: LangChain LLM instance
            merge_mode: Merge mode; "simple" or "llm"
            custom_logger: Optional custom logger
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
    ) -> Tuple[List[SkillPrecondition], List[str]]:
        """
        Hybrid extraction of preconditions: extract from both code and task intent and validate consistency.

        Args:
            code: Skill code
            description: Skill description
            task: Task description (optional)
            context: Context information (optional)

        Returns:
            Tuple[List[SkillPrecondition], List[str]]: (extracted preconditions, list of warnings)
        """
        warnings = []

        # Stage 1: extract from intent (task+context)
        intent_preconditions = []
        if task or context:
            try:
                intent_preconditions = self.extract_from_intent(
                    task=task, context=context, description=description
                )
                self.logger.info(
                    f"[Intent-based] Extracted {len(intent_preconditions)} preconditions from task/context"
                )
            except Exception as e:
                warnings.append(f"Failed to extract preconditions from intent: {e}")

        # Stage 2: extract from code
        code_preconditions = []
        try:
            code_preconditions = self.extract_from_code(code, description)
            self.logger.info(
                f"[Code-based] Extracted {len(code_preconditions)} preconditions from code"
            )
        except Exception as e:
            warnings.append(f"Failed to extract preconditions from code: {e}")

        # Stage 3: merge and deduplicate
        merged_preconditions = self.merge(intent_preconditions, code_preconditions)

        # Stage 4: filter out unimportant preconditions
        filtered_preconditions = filter_preconditions(merged_preconditions)

        # Stage 5: validate whether the code implements these preconditions
        validated_preconditions, validation_warnings = self.validate_against_code(
            filtered_preconditions, code
        )
        warnings.extend(validation_warnings)

        return validated_preconditions, warnings

    def extract_from_intent(
        self,
        task: Optional[str] = None,
        context: Optional[str] = None,
        description: Optional[str] = None,
    ) -> List[SkillPrecondition]:
        """
        Extract preconditions from task intent (task+context).

        Args:
            task: Task description
            context: Context information
            description: Skill description

        Returns:
            List[SkillPrecondition]: Preconditions extracted from intent
        """
        preconditions = []

        if not task and not context:
            return preconditions

        try:
            messages = [
                SystemMessage(content=INTENT_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""Task: {task or "N/A"}
Context: {context or "N/A"}
Skill Description: {description or "N/A"}

Analyze the task and context to extract all preconditions. Return only JSON."""
                ),
            ]

            _llm_resp = self.llm.invoke(messages)
            from skillnet.utils.stats_tracker import record_llm_usage
            record_llm_usage(_llm_resp, process_type="metadata_extraction", function_name="metadata.preconditions.from_task")
            response = _llm_resp.content

            # Parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for p_data in data.get("preconditions", []):
                    # Coerce None -> "" because ``.get(key, default)`` returns
                    # the stored None when the LLM emits "code": null etc.
                    preconditions.append(
                        SkillPrecondition(
                            description=(p_data.get("description") or ""),
                            code=(p_data.get("code") or ""),
                            state_representation=p_data.get("state_representation") or {},
                        )
                    )
        except Exception as e:
            self.logger.warning(f"Failed to extract preconditions from intent: {e}")

        return preconditions

    def extract_from_code(
        self, code: str, description: str
    ) -> List[SkillPrecondition]:
        """
        Extract preconditions from code.

        Args:
            code: Skill code
            description: Skill description

        Returns:
            List[SkillPrecondition]: Preconditions extracted from code
        """
        preconditions = []

        try:
            messages = [
                SystemMessage(content=CODE_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""Skill code:
{code}

Skill description:
{description}

Analyze the code and extract all preconditions. Return only JSON."""
                ),
            ]

            _llm_resp = self.llm.invoke(messages)
            from skillnet.utils.stats_tracker import record_llm_usage
            record_llm_usage(_llm_resp, process_type="metadata_extraction", function_name="metadata.preconditions.from_code")
            response = _llm_resp.content

            # Parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for p_data in data.get("preconditions", []):
                    # Coerce None -> "" (LLM may emit "code": null etc.)
                    preconditions.append(
                        SkillPrecondition(
                            description=(p_data.get("description") or ""),
                            code=(p_data.get("code") or ""),
                            state_representation=p_data.get("state_representation") or {},
                        )
                    )
        except Exception as e:
            self.logger.warning(f"Failed to extract preconditions from code: {e}")

        return preconditions

    def merge(
        self,
        intent_preconditions: List[SkillPrecondition],
        code_preconditions: List[SkillPrecondition],
    ) -> List[SkillPrecondition]:
        """
        Merge preconditions extracted from intent and code.

        Args:
            intent_preconditions: Preconditions extracted from intent
            code_preconditions: Preconditions extracted from code

        Returns:
            List[SkillPrecondition]: Merged preconditions
        """
        # If both lists are empty or only one source is present, return directly
        if not intent_preconditions and not code_preconditions:
            return []
        if not intent_preconditions:
            return code_preconditions
        if not code_preconditions:
            return intent_preconditions

        # Choose strategy based on merge_mode
        if self.merge_mode == "simple":
            return self._merge_simple(intent_preconditions, code_preconditions)
        elif self.merge_mode == "llm":
            total_count = len(intent_preconditions) + len(code_preconditions)
            if total_count <= 3:
                return self._merge_simple(intent_preconditions, code_preconditions)

            try:
                return self._merge_with_llm(intent_preconditions, code_preconditions)
            except Exception as e:
                self.logger.warning(
                    f"[Precondition Merge] LLM merge failed: {e}, falling back to simple merge"
                )
                return self._merge_simple(intent_preconditions, code_preconditions)
        else:
            self.logger.warning(
                f"[Precondition Merge] Unknown merge_mode '{self.merge_mode}', using simple merge"
            )
            return self._merge_simple(intent_preconditions, code_preconditions)

    def _merge_simple(
        self,
        intent_preconditions: List[SkillPrecondition],
        code_preconditions: List[SkillPrecondition],
    ) -> List[SkillPrecondition]:
        """Simple merge strategy (exact match on description)."""
        merged = []
        seen_descriptions = set()

        # Prefer code-based preconditions (more accurate)
        for precondition in code_preconditions:
            if precondition.description not in seen_descriptions:
                merged.append(precondition)
                seen_descriptions.add(precondition.description)

        # Add preconditions only present in the intent
        for precondition in intent_preconditions:
            if precondition.description not in seen_descriptions:
                merged.append(precondition)
                seen_descriptions.add(precondition.description)

        return merged

    def _merge_with_llm(
        self,
        intent_preconditions: List[SkillPrecondition],
        code_preconditions: List[SkillPrecondition],
    ) -> List[SkillPrecondition]:
        """Use the LLM to intelligently merge preconditions."""
        # Build a description -> state_representation map
        state_repr_map = {}
        for p in intent_preconditions:
            if p.state_representation:
                state_repr_map[p.description] = p.state_representation
        for p in code_preconditions:
            if p.state_representation:
                state_repr_map[p.description] = p.state_representation

        # Prepare input data
        intent_data = [
            {
                "description": p.description,
                "code": p.code,
                "state_representation": p.state_representation,
            }
            for p in intent_preconditions
        ]
        code_data = [
            {
                "description": p.description,
                "code": p.code,
                "state_representation": p.state_representation,
            }
            for p in code_preconditions
        ]

        human_prompt = f"""Merge these preconditions:

INTENT-BASED PRECONDITIONS (from task/context):
{json.dumps(intent_data, indent=2, ensure_ascii=False)}

CODE-BASED PRECONDITIONS (from code analysis):
{json.dumps(code_data, indent=2, ensure_ascii=False)}

Analyze and merge semantically similar preconditions. Return only JSON."""

        messages = [
            SystemMessage(content=MERGE_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ]

        _llm_resp = self.llm.invoke(messages)
        from skillnet.utils.stats_tracker import record_llm_usage
        record_llm_usage(_llm_resp, process_type="metadata_extraction", function_name="metadata.preconditions.merge")
        response = _llm_resp.content

        # Parse the JSON response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            merged = []
            for p_data in data.get("merged_preconditions", []):
                # Try to fetch state_representation from the LLM response
                state_repr = p_data.get("state_representation")

                # If the LLM did not return it, recover from the original map
                if not state_repr:
                    original_desc = p_data.get("original_description", "")
                    if original_desc and original_desc in state_repr_map:
                        state_repr = state_repr_map[original_desc]
                    elif p_data.get("description", "") in state_repr_map:
                        state_repr = state_repr_map[p_data.get("description", "")]
                    else:
                        # Fuzzy match
                        desc_lower = p_data.get("description", "").lower()
                        for orig_desc, orig_repr in state_repr_map.items():
                            if (
                                orig_desc.lower() in desc_lower
                                or desc_lower in orig_desc.lower()
                            ):
                                state_repr = orig_repr
                                break

                # Coerce None -> "" (LLM merge step can emit "code": null).
                merged.append(
                    SkillPrecondition(
                        description=(p_data.get("description") or ""),
                        code=(p_data.get("code") or ""),
                        state_representation=state_repr or {},
                    )
                )
            self.logger.info(
                f"[Precondition Merge] LLM merged {len(intent_preconditions) + len(code_preconditions)} preconditions into {len(merged)}"
            )
            return merged
        else:
            raise ValueError("Failed to parse LLM response as JSON")

    def validate_against_code(
        self, preconditions: List[SkillPrecondition], code: str
    ) -> Tuple[List[SkillPrecondition], List[str]]:
        """
        Validate whether preconditions are checked in the code.

        Args:
            preconditions: Preconditions to validate
            code: Skill code

        Returns:
            Tuple[List[SkillPrecondition], List[str]]: (validated preconditions, warning list)
        """
        validated = []
        warnings = []

        for precondition in preconditions:
            has_check = code_has_precondition_check(code, precondition)

            if has_check:
                validated.append(precondition)
            else:
                # Still add it, but mark a warning
                validated.append(precondition)
                warnings.append(
                    f"Precondition '{precondition.description}' may not be explicitly checked in code"
                )

        return validated, warnings
