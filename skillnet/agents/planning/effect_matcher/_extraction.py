"""
Effect Extraction Mixin (Layer A)

Target effect extraction layer: extract target effects from task descriptions.
Supports both LLM and rule-based extraction with an LLM -> rules -> empty list fallback chain.

extracted from effect_matcher.py.

Methods:
- extract_target_effects: extract target effects from a task description (PUBLIC)
- _extract_target_effects_llm: LLM-based extraction
- _extract_target_effects_rules: rule-based extraction (@staticmethod, called externally by graph_manager_impl.py)

Requires self attributes (from EffectMatcher):
- self.llm: LangChain LLM instance
- self.logger: Logger instance
- self.use_llm_for_extraction: bool
"""

import json
import re
from typing import Any, Dict, List, TYPE_CHECKING

from langchain.schema import HumanMessage, SystemMessage

from skillnet.utils.stats_tracker import record_llm_usage

if TYPE_CHECKING:
    from ._matcher import EffectMatcher


def _extract_effects_with_verbs(
    task: str,
    verbs: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Extract target effects using domain-injected verb patterns.

    Args:
        task: Task description string.
        verbs: Mapping from effect type to verb patterns.
            Keys: "find", "collect", "craft", "ensure", "place", "equip".

    Returns:
        List of effect dicts, same format as _extract_target_effects_rules.
    """
    effects: List[Dict[str, Any]] = []
    task_lower = task.lower()

    # --- FIND verbs (nearby_block, early return) ---
    find_verbs = verbs.get("find", [])
    if find_verbs:
        find_alt = "|".join(find_verbs)
        find_pattern = rf'(?:{find_alt})\s+(\d+)?\s*(\w+(?:[\s_]\w+)*)'
        match = re.search(find_pattern, task_lower)
        if match:
            count = int(match.group(1)) if match.group(1) else 1
            item = match.group(2).replace(" ", "_")
            return [{
                "type": "nearby_block",
                "item": item,
                "count": count,
                "operation": "find",
            }]

    # --- COLLECT verbs (inventory, add) ---
    collect_verbs = verbs.get("collect", [])
    if collect_verbs:
        collect_alt = "|".join(collect_verbs)
        collect_pattern = rf'(?:{collect_alt})\s+(\d+)?\s*(\w+(?:\s+\w+)?)'
        match = re.search(collect_pattern, task_lower)
        if match:
            count = int(match.group(1)) if match.group(1) else 1
            item = match.group(2).replace(" ", "_")
            effects.append({
                "type": "inventory",
                "item": item,
                "count": count,
                "operation": "add",
            })

    # --- CRAFT verbs (inventory, add) ---
    craft_verbs = verbs.get("craft", [])
    if craft_verbs:
        craft_alt = "|".join(craft_verbs)
        craft_pattern = rf'(?:{craft_alt})\s+(\d+)?\s*(\w+(?:\s+\w+)?)'
        match = re.search(craft_pattern, task_lower)
        if match:
            count = int(match.group(1)) if match.group(1) else 1
            item = match.group(2).replace(" ", "_")
            effects.append({
                "type": "inventory",
                "item": item,
                "count": count,
                "operation": "add",
            })

    # --- ENSURE verbs (inventory, ensure) ---
    ensure_verbs = verbs.get("ensure", [])
    if ensure_verbs:
        ensure_alt = "|".join(ensure_verbs)
        ensure_pattern = rf'(?:{ensure_alt})\s+(?:you\s+)?have\s+(\d+)\s+(\w+(?:[\s_]\w+)*)'
        match = re.search(ensure_pattern, task_lower)
        if match:
            count = int(match.group(1))
            item = match.group(2).replace(" ", "_")
            effects.append({
                "type": "inventory",
                "item": item,
                "count": count,
                "operation": "ensure",
            })

    # --- PLACE verbs (block, place) ---
    place_verbs = verbs.get("place", [])
    if place_verbs:
        place_alt = "|".join(place_verbs)
        place_pattern = rf'(?:{place_alt})\s+(?:a\s+)?(\w+(?:\s+\w+)?)'
        match = re.search(place_pattern, task_lower)
        if match:
            item = match.group(1).replace(" ", "_")
            effects.append({
                "type": "block",
                "item": item,
                "count": 1,
                "operation": "place",
            })

    # --- EQUIP verbs (equipment, equip) ---
    equip_verbs = verbs.get("equip", [])
    if equip_verbs:
        equip_alt = "|".join(equip_verbs)
        equip_pattern = rf'(?:{equip_alt})\s+(\w+(?:[\s_]\w+)?)'
        match = re.search(equip_pattern, task_lower)
        if match:
            item = match.group(1).replace(" ", "_")
            effects.append({
                "type": "equipment",
                "item": item,
                "count": 1,
                "operation": "equip",
            })

    # --- Generic number+item fallback ---
    if not effects:
        generic_pattern = r'(\d+)\s+(\w+(?:_\w+)*)'
        matches = re.findall(generic_pattern, task_lower)
        for count_str, item in matches:
            if item not in ['you', 'have', 'the', 'a', 'an', 'is', 'are', 'be', 'to', 'for', 'from', 'with']:
                effects.append({
                    "type": "inventory",
                    "item": item,
                    "count": int(count_str),
                    "operation": "add",
                })
                break

    return effects


class ExtractionMixin:
    """
    Effect Extraction Mixin — target-effect extraction.

    Provides both LLM and rule-based strategies with a fallback chain.
    """

    def extract_target_effects(
        self: "EffectMatcher",
        task: str,
        context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extract target effects from a task description

        Args:
            task: task description
            context: contextual info

        Returns:
            List[Dict]: list of target effects
            For example: [{"type": "inventory", "item": "oak_log", "count": 3, "operation": "add"}]
        """
        # Change 2.4: implement fallback chain (LLM -> rules -> empty list)
        # Strategy 1: LLM extraction
        if self.use_llm_for_extraction and self.llm:
            try:
                llm_effects = self._extract_target_effects_llm(task, context)
                if llm_effects:
                    return llm_effects
                self.logger.warning("[EffectMatcher] LLM extraction returned no results; falling back to rules")
            except Exception as e:
                self.logger.warning(f"[EffectMatcher] LLM extraction failed: {e}; falling back to rules")

        # Strategy 2: Domain-aware rule extraction (uses verb patterns from DomainKnowledge)
        dk = getattr(self, 'domain_knowledge', None)
        if dk:
            verbs = dk.get_effect_extraction_verbs()
            if verbs:
                effects = _extract_effects_with_verbs(task, verbs)
                if effects:
                    self.logger.info(f"[EffectMatcher] Domain-rule extraction: {effects}")
                    return effects

        # Strategy 3: hard-coded rule extraction (fallback)
        effects = self._extract_target_effects_rules(task)
        if effects:
            self.logger.info(f"[EffectMatcher] Rule extraction: {effects}")
            return effects

        self.logger.warning(f"[EffectMatcher] Unable to extract target effects from task: {task}")
        return []

    def _extract_target_effects_llm(
        self: "EffectMatcher",
        task: str,
        context: str
    ) -> List[Dict[str, Any]]:
        """Use the LLM to extract target effects"""
        system_prompt = """You are a task analysis expert. Extract the target effects from a Minecraft task description.

CRITICAL RULES:
1. EVERY effect object MUST have an "item" field - this is MANDATORY
2. If you cannot determine the item, use null or skip that effect
3. NEVER return incomplete effects missing the "item" field
4. Only extract FINAL GOALS (what should be added/created/placed/found), NOT consumed materials
5. Use "add" operation for items that should be obtained/created
6. Use "place" operation for blocks that should be placed
7. Use "find" operation for exploration/navigation/locate tasks (with type "nearby_block")
8. DO NOT extract "remove" operations for consumed materials - those are intermediate steps, not goals

Return ONLY a JSON array. Each effect MUST have these required fields:
{
  "type": "inventory|block|equipment|nearby_block",
  "item": "item_name",  # ← REQUIRED, cannot be omitted
  "count": number,      # ← REQUIRED, use 1 if not specified
  "operation": "add|place|equip|find"  # ← REQUIRED
}

EXAMPLES:
Task: "Mine 3 oak logs"
Output: [{"type": "inventory", "item": "oak_log", "count": 3, "operation": "add"}]

Task: "Craft a wooden pickaxe"
Output: [{"type": "inventory", "item": "wooden_pickaxe", "count": 1, "operation": "add"}]

Task: "Place a chest"
Output: [{"type": "block", "item": "chest", "count": 1, "operation": "place"}]

Task: "Find deepslate layer"
Output: [{"type": "nearby_block", "item": "deepslate", "count": 1, "operation": "find"}]

INVALID (missing item field):
✗ [{"type": "inventory", "count": 3, "operation": "add"}]"""

        human_prompt = f"""Task: {task}
Context: {context}

Extract the target effects. Return only JSON array."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        _llm_resp = self.llm.invoke(messages)
        record_llm_usage(_llm_resp, process_type="planning_effect_match", function_name="effect_matcher.extraction._extract_target_effects_llm", task=task)
        response = _llm_resp.content

        # Debug: log the raw response to diagnose empty-response issues
        self.logger.debug(f"[EffectMatcher] Raw LLM response ({len(response)} chars): {response[:500]}")

        # Parse JSON response
        # First try array format
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                effects = json.loads(json_match.group())

                # Validate each effect
                validated_effects = []
                for effect in effects:
                    if not isinstance(effect, dict):
                        self.logger.warning(f"[EffectMatcher] Skipping non-dict effect: {effect}")
                        continue

                    # Required-field check
                    if 'item' not in effect or not effect['item']:
                        self.logger.warning(f"[EffectMatcher] Skipping effect missing 'item': {effect}")
                        continue

                    # Normalize
                    effect.setdefault('type', 'inventory')
                    effect.setdefault('operation', 'add')
                    effect.setdefault('count', 1)

                    validated_effects.append(effect)

                if validated_effects:
                    return validated_effects
                else:
                    self.logger.warning(f"[EffectMatcher] All effects failed validation, parsed {len(effects)} effects: {effects[:3]}")
                    return []
            except json.JSONDecodeError as e:
                self.logger.warning(f"[EffectMatcher] JSON decode error: {e}")

        # Then try matching an effects field inside an object
        obj_match = re.search(r'\{.*\}', response, re.DOTALL)
        if obj_match:
            try:
                obj = json.loads(obj_match.group())
                if isinstance(obj, dict) and "effects" in obj:
                    effects = obj["effects"]

                    # Same validation logic
                    validated_effects = []
                    for effect in effects:
                        if not isinstance(effect, dict):
                            continue
                        if 'item' not in effect or not effect['item']:
                            continue
                        effect.setdefault('type', 'inventory')
                        effect.setdefault('operation', 'add')
                        effect.setdefault('count', 1)
                        validated_effects.append(effect)
                    return validated_effects if validated_effects else []
            except json.JSONDecodeError:
                pass

        return []

    @staticmethod
    def _extract_target_effects_rules(task: str) -> List[Dict[str, Any]]:
        """Extract target effects using rules"""
        effects = []
        task_lower = task.lower()

        # Match "Find 3 diamond_ore" or "Locate iron ore" (FIND semantics)
        find_pattern = r'(?:find|locate|discover|search\s+for)\s+(\d+)?\s*(\w+(?:[\s_]\w+)*)'
        match = re.search(find_pattern, task_lower)
        if match:
            count = int(match.group(1)) if match.group(1) else 1
            item = match.group(2).replace(" ", "_")
            effects.append({
                "type": "nearby_block",  # FIND tasks use the nearby_block type
                "item": item,
                "count": count,
                "operation": "find"
            })
            # FIND tasks don't need other effects; return directly
            return effects

        # Extended list of gathering verbs
        mine_pattern = (
            r'(?:mine|collect|get|obtain|gather|acquire|harvest|extract|dig)\s+(\d+)?\s*'
            r'(\w+(?:\s+\w+)?)\s*'
            r'(?:log|block|item|ore|stone|wood|plank|stick|pickaxe|axe|sword|'
            r'shovel|hoe|helmet|chestplate|leggings|boots|ingot|gem|food|seed|'
            r'sapling|dye|wool|leather|string|feather|egg|milk|bucket|water|'
            r'bucket|lava|bucket|coal|charcoal|redstone|lapis|diamond|emerald|'
            r'quartz|netherite|gold|iron|copper|raw_iron|raw_gold|raw_copper)?'
        )

        match = re.search(mine_pattern, task_lower)
        if match:
            count = int(match.group(1)) if match.group(1) else 1
            item = match.group(2).replace(" ", "_")
            effects.append({
                "type": "inventory",
                "item": item,
                "count": count,
                "operation": "add"
            })

        # Match "Craft 1 wooden pickaxe"
        craft_pattern = r'craft\s+(\d+)?\s*(\w+(?:\s+\w+)?)'
        match = re.search(craft_pattern, task_lower)
        if match:
            count = int(match.group(1)) if match.group(1) else 1
            item = match.group(2).replace(" ", "_")
            effects.append({
                "type": "inventory",
                "item": item,
                "count": count,
                "operation": "add"
            })

        # Match "Ensure you have 8 raw iron" or "Ensure you have 1 stone pickaxe"
        # add "Ensure you have X" pattern support so the count is extracted correctly for parameter inference
        ensure_pattern = r'ensure\s+(?:you\s+)?have\s+(\d+)\s+(\w+(?:[\s_]\w+)*)'
        match = re.search(ensure_pattern, task_lower)
        if match:
            count = int(match.group(1))
            item = match.group(2).replace(" ", "_")
            effects.append({
                "type": "inventory",
                "item": item,
                "count": count,
                "operation": "ensure"  # Will be converted to "add" by normalize_operation
            })

        # Match "Place a chest"
        place_pattern = r'place\s+(?:a\s+)?(\w+(?:\s+\w+)?)'
        match = re.search(place_pattern, task_lower)
        if match:
            item = match.group(1).replace(" ", "_")
            effects.append({
                "type": "block",
                "item": item,
                "count": 1,
                "operation": "place"
            })

        # Match "Equip iron_helmet" or "Equip diamond chestplate"
        equip_pattern = r'equip\s+(\w+(?:[\s_]\w+)?)'
        match = re.search(equip_pattern, task_lower)
        if match:
            item = match.group(1).replace(" ", "_")
            effects.append({
                "type": "equipment",
                "item": item,
                "count": 1,
                "operation": "equip"
            })

        # Generic number+item pattern as the final fallback
        if not effects:
            generic_pattern = r'(\d+)\s+(\w+(?:_\w+)*)'
            matches = re.findall(generic_pattern, task_lower)
            for count_str, item in matches:
                # Filter common words
                if item not in ['you', 'have', 'the', 'a', 'an', 'is', 'are', 'be', 'to', 'for', 'from', 'with']:
                    effects.append({
                        "type": "inventory",
                        "item": item,
                        "count": int(count_str),
                        "operation": "add"
                    })
                    # Only take the first match
                    break

        return effects
