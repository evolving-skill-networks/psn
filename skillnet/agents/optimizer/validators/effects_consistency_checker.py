"""
Effects-Code consistency checker

Checks whether a skill's declared effects are consistent with its code implementation.
Addresses cases like ensureFuel claiming to produce planks while the code cannot do so.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Set

logger = logging.getLogger(__name__)


@dataclass
class EffectsInconsistency:
    """An inconsistency between declared effects and the code implementation"""
    declared_item: str           # Item the skill claims to produce
    evidence: str                # Evidence


@dataclass
class EffectsConsistencyResult:
    """Consistency check result"""
    skill_name: str
    is_consistent: bool
    issues: List[EffectsInconsistency] = field(default_factory=list)
    checked_items: List[str] = field(default_factory=list)
    quality_score: float = 1.0  # 0.0 (completely inconsistent) ~ 1.0 (fully consistent)


class EffectsConsistencyValidator:
    """
    Checks whether a skill's declared effects are consistent with its code implementation

    Core features:
    1. Detect items declared in effects that the code cannot produce
    2. Compute the consistency quality score (used for Planner filtering)
    3. Determine fix direction (modify code or modify effects)
    """

    def __init__(self, item_aliases: dict = None):
        self.logger = logger
        self._item_aliases = item_aliases or {}

    def check(
        self,
        skill_name: str,
        skill_code: str,
        declared_effects: List[Any],
        target_item: Optional[str] = None,
        skill_description: str = ""
    ) -> EffectsConsistencyResult:
        """
        Check whether a skill implements the capabilities it claims

        Args:
            skill_name: skill name
            skill_code: skill code
            declared_effects: list of declared effects
            target_item: item required by the current task (optional, for filtering)
            skill_description: skill description (used for fix-direction judgment)

        Returns:
            EffectsConsistencyResult: check result
        """
        issues = []
        checked_items = []

        # Extract all declared items
        declared_items = self._extract_declared_items(declared_effects)

        # If target_item is provided, only check that item
        if target_item:
            items_to_check = [target_item] if target_item.lower() in [i.lower() for i in declared_items] else []
        else:
            items_to_check = declared_items

        for item in items_to_check:
            checked_items.append(item)

            if not self._code_can_produce_item(skill_code, item):
                issues.append(EffectsInconsistency(
                    declared_item=item,
                    evidence=f"Effects declare '{item}' but code has no production logic",
                ))

        # Compute quality score
        quality_score = self._calculate_quality_score(
            skill_name, skill_code, declared_effects, target_item
        )

        return EffectsConsistencyResult(
            skill_name=skill_name,
            is_consistent=len(issues) == 0,
            issues=issues,
            checked_items=checked_items,
            quality_score=quality_score
        )

    def quick_check(
        self,
        skill_code: str,
        target_item: str
    ) -> float:
        """
        Lightweight consistency check (used during the Planner phase)

        Returns:
            float: 0.0 (completely inconsistent) ~ 1.0 (fully consistent)
        """
        code_lower = skill_code.lower()
        target_lower = target_item.lower().replace('_', '')

        # Check 1: whether a relevant ensure* function is called
        if f"ensure{target_lower}" in code_lower:
            return 1.0

        # Check 2: whether craftItem is called with the target item
        if "craftitem" in code_lower and target_item.lower() in code_lower:
            return 1.0

        # Check 3: domain-specific alias check
        for keyword, checks in self._item_aliases.items():
            if keyword in target_lower:
                for ensure_fn in checks.get("ensure", []):
                    if ensure_fn in code_lower:
                        return 0.9

        # Check 4: whether the target item name appears in the code
        if target_item.lower() in code_lower:
            return 0.6  # May be a parameter check, not necessarily production

        # Default: indeterminate, return a lower score
        return 0.3

    def _extract_declared_items(self, effects: List[Any]) -> List[str]:
        """Extract all item names from declared effects"""
        items = []

        for effect in effects:
            state_repr = getattr(effect, 'state_representation', None)
            if not state_repr:
                continue

            if isinstance(state_repr, dict):
                # Handle OR logic
                if state_repr.get("logic", "").upper() == "OR":
                    for condition in state_repr.get("conditions", []):
                        item = condition.get("item", "")
                        if item:
                            items.append(item)
                else:
                    # Single item
                    item = state_repr.get("item", "")
                    if item:
                        items.append(item)

        return items

    def _code_can_produce_item(self, code: str, item: str) -> bool:
        """
        Heuristically check whether the code can produce the specified item

        Check strategy:
        1. Whether ensure{ItemBase}() is called
        2. Whether craftItem(bot, "item") is called
        3. Whether mineBlock(bot, "item_ore") is called
        4. Special mappings (e.g. planks -> ensurePlanks, logs -> ensureLogs)
        """
        code_lower = code.lower()
        item_lower = item.lower()
        item_base = item_lower.replace('_', '')

        # === Check 1: ensure* function ===
        # Exact match: ensure{ItemBase}
        if f"ensure{item_base}" in code_lower:
            return True

        # === Check 2: domain-specific alias check ===
        for keyword, checks in self._item_aliases.items():
            if keyword in item_lower:
                for ensure_fn in checks.get("ensure", []):
                    if ensure_fn in code_lower:
                        return True
                for ore_name in checks.get("mine_ore", []):
                    if "mineblock" in code_lower and ore_name in code_lower:
                        return True
                for craft_item in checks.get("craft", []):
                    if "craftitem" in code_lower and craft_item in code_lower:
                        return True

        # === Check 3: craftItem call ===
        if "craftitem" in code_lower:
            # Check whether the target item name is present
            if item_lower in code_lower:
                return True
            # Check variants (e.g. planks within oak_planks)
            item_parts = item_lower.split('_')
            if any(part in code_lower for part in item_parts if len(part) > 3):
                # Stricter check: ensure it appears inside a craftItem call
                craft_pattern = rf"craftitem\s*\([^)]*{item_lower}[^)]*\)"
                if re.search(craft_pattern, code_lower):
                    return True

        # === Check 4: bot.craft call ===
        if "bot.craft" in code_lower and item_lower in code_lower:
            return True

        # === Check 5: mineBlock call ===
        if "mineblock" in code_lower:
            # Check whether a related ore is mined
            ore_name = f"{item_lower}_ore"
            if ore_name in code_lower:
                return True

        return False

    def _calculate_quality_score(
        self,
        skill_name: str,
        skill_code: str,
        declared_effects: List[Any],
        target_item: Optional[str]
    ) -> float:
        """
        Compute the match quality score (0.0 ~ 1.0)

        Scoring dimensions:
        1. OR-condition count penalty: more conditions -> lower score
        2. Name relevance: whether the skill name contains keywords from the target item
        3. Code implementation check: whether there is corresponding production logic
        """
        if not target_item:
            return 1.0

        target_lower = target_item.lower()

        # === Dimension 1: OR condition count ===
        or_penalty = 0.0
        total_or_conditions = 0

        for effect in declared_effects:
            state_repr = getattr(effect, 'state_representation', None)
            if state_repr and isinstance(state_repr, dict):
                if state_repr.get("logic", "").upper() == "OR":
                    conditions = state_repr.get("conditions", [])
                    total_or_conditions = len(conditions)
                    if total_or_conditions > 5:
                        # More than 5 OR conditions; penalize proportionally
                        or_penalty = min(0.5, (total_or_conditions - 5) * 0.05)
                    break

        # === Dimension 2: name relevance ===
        name_score = 0.0
        skill_words = set(re.findall(r'[a-z]+', skill_name.lower()))
        item_words = set(target_lower.replace('_', ' ').split())
        generic_words = {'ensure', 'craft', 'get', 'make', 'create', 'mine', 'obtain'}

        meaningful_skill_words = skill_words - generic_words
        meaningful_item_words = item_words - generic_words

        if meaningful_skill_words & meaningful_item_words:
            name_score = 0.5  # Name relevance adds 0.5

        # === Dimension 3: code implementation check ===
        code_score = 0.0
        if self._code_can_produce_item(skill_code, target_item):
            code_score = 0.3

        # === Compute the final score ===
        base_score = 1.0 - or_penalty + name_score + code_score
        return max(0.0, min(1.0, base_score))

