"""
Effect Matching Mixin (Layer B)

Effect-matching layer: decides whether a skill's expected_effects satisfy the
target effects. Includes exact match, relaxed match, template match, and
category match.

extracted from effect_matcher.py.

Methods:
- effect_matches: check whether a skill's effects match the target (PUBLIC)
- _check_state_representation: exact state_representation match
- _check_state_representation_relaxed: relaxed state_representation match
- _item_matches_with_template: item-name match (with template parameters)
- _get_item_safe_category: get an item's safe-interchange category
- _determine_item_category: determine an item's category
- _get_matched_effect_confidence: get confidence of the matched effect
- _effect_contains_item: check whether state_repr contains the target item

Requires self attributes (from EffectMatcher):
- self.llm: LangChain LLM instance
- self.logger: Logger instance
"""

import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ._utils import (
    is_category_name,
    get_category_items,
    normalize_operation,
    get_safe_interchange_categories,
    get_item_categories,
    # Backward-compat aliases for code that checks membership directly
    SAFE_INTERCHANGE_CATEGORIES,
    ITEM_CATEGORIES,
)

if TYPE_CHECKING:
    from ._matcher import EffectMatcher


class MatchingMixin:
    """
    Effect Matching Mixin - core matching logic.

    Decides whether a skill's expected_effects satisfy the target effect.
    Supports exact match, relaxed match, template-parameter match, and
    LLM-assisted match.
    """

    def effect_matches(
        self: "EffectMatcher",
        target_effect: Dict[str, Any],
        skill_effects: List,
        ignore_count: bool = False
    ) -> bool:
        """
        Check whether a skill's effects match the target effect.

        Prefer matching by state_representation; fall back to description matching.

        Args:
            target_effect: target effect
                format: {"type": "inventory", "item": "oak_log", "count": 3, "operation": "add"}
            skill_effects: list of skill expected_effects
            ignore_count: whether to ignore count checks (for parametric skills)

        Returns:
            bool: whether it matches.
        """
        # Parse target effect
        target_item_raw = target_effect.get("item", "")
        if isinstance(target_item_raw, list):
            target_item = target_item_raw[0].lower() if target_item_raw and isinstance(target_item_raw[0], str) else ''
        elif isinstance(target_item_raw, str):
            target_item = target_item_raw.lower()
        else:
            target_item = ''

        # defensive check - an empty target_item should not match any skill.
        # Usually caused by an upstream state_representation parsing error.
        if not target_item:
            self.logger.warning("[EffectMatcher] target_item is empty, skipping match (possibly a state_representation format issue)")
            return False

        target_count = target_effect.get("count", 1)
        if isinstance(target_count, str):
            try:
                target_count = int(target_count)
            except ValueError:
                target_count = 1

        target_type = target_effect.get("type", "inventory")
        # normalize target_operation — the LLM may extract "ensure" while effects use "add"
        target_operation = normalize_operation(target_effect.get("operation", "add"))

        # is_primary filtering logic.
        # Check whether any effect is marked is_primary=True.
        has_any_primary = any(getattr(e, 'is_primary', False) for e in skill_effects)

        # If primary effects exist, only match against primaries.
        # If not (backward-compat for old skills), match all effects.
        if has_any_primary:
            effects_to_match = [e for e in skill_effects if getattr(e, 'is_primary', False)]
            self.logger.debug(
                f"[EffectMatcher] using is_primary filter: {len(effects_to_match)}/{len(skill_effects)} effects"
            )
        else:
            # Backward-compat: still match all, but warn about data-quality issue.
            # This may cause side-effect effects to be matched incorrectly.
            if skill_effects:  # Only warn when there are effects
                self.logger.warning(
                    f"[EffectMatcher] skill has no primary effect ({len(skill_effects)} effects), using all effects (consider fixing the effects data)"
                )
            effects_to_match = skill_effects

        # First, try exact matching via state_representation
        for effect in effects_to_match:
            state_repr = getattr(effect, 'state_representation', None)
            if state_repr:
                if self._check_state_representation(
                    state_repr, target_item, target_count, target_type, target_operation, ignore_count
                ):
                    return True

        # Fallback: match via description (still using effects_to_match)
        for effect in effects_to_match:
            # Recheck state_representation (second pass, looser match)
            state_repr = getattr(effect, 'state_representation', None)
            if state_repr:
                if self._check_state_representation_relaxed(
                    state_repr, target_item, target_count, target_type, target_operation, ignore_count
                ):
                    return True

            # Description match
            description = effect.description.lower() if hasattr(effect, 'description') else str(effect).lower()

            if target_type == "inventory" and target_operation != "remove":
                # Exclude "equip"-style descriptions
                if re.search(r'^equip', description, re.IGNORECASE):
                    continue

                # Check whether description explicitly contains the target item
                target_item_escaped = re.escape(target_item)
                target_item_space = re.escape(target_item.replace("_", " "))
                target_item_pattern = r'\b(' + target_item_escaped + r'|' + target_item_space + r')\b'

                if not re.search(target_item_pattern, description):
                    continue

                # Check for "remove/equip target_item" pattern
                remove_verbs = ["remove", "removes", "consume", "consumes", "use", "uses", "equip", "equips"]
                remove_pattern = r'\b(' + '|'.join(remove_verbs) + r')\s+(?:[0-9]+\s+)?' + target_item_pattern
                if re.search(remove_pattern, description, re.IGNORECASE):
                    continue

                # Check for explicit "add target_item" pattern
                add_verbs = ["add", "adds", "produce", "produces", "create", "creates",
                             "obtain", "obtains", "get", "gets", "mine", "mines", "collect", "collects"]
                add_pattern = r'\b(' + '|'.join(add_verbs) + r')\s+(?:[0-9]+\s+)?' + target_item_pattern

                if re.search(add_pattern, description, re.IGNORECASE):
                    return True

                # Category match pattern
                category_words = ["log", "plank", "pickaxe", "sword", "axe", "ore", "ingot"]
                for category in category_words:
                    if category in description:
                        category_add_pattern = r'\b(' + '|'.join(add_verbs) + r')\s+(?:[0-9]+\s+)?' + re.escape(category)
                        if re.search(category_add_pattern, description, re.IGNORECASE):
                            if re.search(target_item_pattern, description, re.IGNORECASE):
                                return True

            elif target_type == "block":
                # For "place" operations, require placement verbs in description.
                # A crafting skill's description mentioning "crafting_table" should NOT match
                # a "place crafting_table" target — only skills that actually place blocks should.
                if target_operation == "place":
                    place_verbs = r'\b(place|places|placed|placing|setup|set up)\b'
                    if re.search(place_verbs, description, re.IGNORECASE) and \
                       re.search(r'\b' + re.escape(target_item) + r'\b', description):
                        return True
                else:
                    if re.search(r'\b' + re.escape(target_item) + r'\b', description):
                        return True

            elif target_type == "nearby_block":
                # Effect match for FIND tasks: check whether the description contains find/locate/explore
                if re.search(r'\b(find|locate|explore|search|discover)\b', description, re.IGNORECASE):
                    # Check whether it contains the target item
                    target_item_escaped = re.escape(target_item)
                    target_item_space = re.escape(target_item.replace("_", " "))
                    if re.search(r'\b(' + target_item_escaped + r'|' + target_item_space + r')\b', description, re.IGNORECASE):
                        return True

        return False

    def _check_state_representation(
        self: "EffectMatcher",
        state_repr,
        target_item: str,
        target_count: int,
        target_type: str,
        target_operation: str,
        ignore_count: bool
    ) -> bool:
        """Check whether state_representation matches the target effect."""
        if not state_repr:
            return False

        def count_matches(sr_count: int) -> bool:
            if ignore_count:
                return True
            return sr_count >= target_count

        def safe_get_count(value, default=1):
            if value is None:
                return default
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return default
            return default

        # Handle list case
        if isinstance(state_repr, list):
            for sr in state_repr:
                sr_count = safe_get_count(sr.get("count", 1))
                # normalize sr_operation
                sr_operation = normalize_operation(sr.get("operation", "add"))
                sr_type = sr.get("type", "inventory")
                sr_item = sr.get("item", "")

                if (sr_type == target_type and
                    self._item_matches_with_template(sr_item, target_item) and
                    sr_operation == target_operation and
                    count_matches(sr_count)):
                    return True
            return False

        # Handle dict case
        if isinstance(state_repr, dict):
            if "logic" in state_repr and "conditions" in state_repr:
                logic = state_repr.get("logic", "OR").upper()
                conditions = state_repr.get("conditions", [])

                if logic == "OR":
                    for condition in conditions:
                        if self._check_state_representation(
                            condition, target_item, target_count, target_type, target_operation, ignore_count
                        ):
                            return True
                    return False
                else:  # AND
                    for condition in conditions:
                        if not self._check_state_representation(
                            condition, target_item, target_count, target_type, target_operation, ignore_count
                        ):
                            return False
                    return True
            else:
                sr_count = safe_get_count(state_repr.get("count", 1))
                # normalize sr_operation
                sr_operation = normalize_operation(state_repr.get("operation", "add"))
                sr_type = state_repr.get("type", "inventory")
                sr_item = state_repr.get("item", "")

                return (sr_type == target_type and
                        self._item_matches_with_template(sr_item, target_item) and
                        sr_operation == target_operation and
                        count_matches(sr_count))

        return False

    def _check_state_representation_relaxed(
        self: "EffectMatcher",
        state_repr,
        target_item: str,
        target_count: int,
        target_type: str,
        target_operation: str,
        ignore_count: bool
    ) -> bool:
        """Relaxed state_representation match."""
        if not state_repr:
            return False

        def count_check(sr_count: int) -> bool:
            if ignore_count:
                return True
            return sr_count >= target_count

        def safe_get_count(value, default=1):
            if value is None:
                return default
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return default
            return default

        def item_matches_target(sr_item, target):
            target_lower = target.lower() if isinstance(target, str) else target

            def single_item_matches(item_str):
                item_lower = item_str.lower()
                if item_lower == target_lower:
                    return True
                # Check whether target is a category of item (original logic)
                if is_category_name(target_lower):
                    category_items = get_category_items(target_lower)
                    if item_lower in [ci.lower() for ci in category_items]:
                        return True
                # Bug 5 fix: check whether item is a category of target (reverse match).
                # e.g. effect declares item="log", target requires "oak_log".
                if is_category_name(item_lower):
                    category_items = get_category_items(item_lower)
                    if target_lower in [ci.lower() for ci in category_items]:
                        return True
                return False

            if isinstance(sr_item, list):
                return any(single_item_matches(item) for item in sr_item if isinstance(item, str))
            elif isinstance(sr_item, str):
                return single_item_matches(sr_item)
            return False

        if isinstance(state_repr, dict):
            sr_count = safe_get_count(state_repr.get("count", 1))
            sr_item = state_repr.get("item", "")
            # use normalize_operation to ensure "ensure" -> "add" mapping
            sr_operation = normalize_operation(state_repr.get("operation", "add"))
            if (state_repr.get("type") == target_type and
                item_matches_target(sr_item, target_item) and
                sr_operation == target_operation and
                count_check(sr_count)):
                return True
        elif isinstance(state_repr, list):
            for sr in state_repr:
                sr_count = safe_get_count(sr.get("count", 1))
                sr_item = sr.get("item", "")
                # use normalize_operation to ensure "ensure" -> "add" mapping
                sr_operation = normalize_operation(sr.get("operation", "add"))
                if (sr.get("type") == target_type and
                    item_matches_target(sr_item, target_item) and
                    sr_operation == target_operation and
                    count_check(sr_count)):
                    return True

        return False

    def _item_matches_with_template(self: "EffectMatcher", effect_item, target_item_name: str) -> bool:
        """Check whether an item name in an effect matches the target item.

        Supports template-parameter matching:
        - "{material}_pickaxe" can match "stone_pickaxe"
        - "{type}_log" can match "oak_log"

        Supports list matching (OR logic):
        - ["raw_iron", "iron_ore"] can match "raw_iron" or "iron_ore"
        """
        # Handle list case
        if isinstance(effect_item, list):
            for item in effect_item:
                if self._item_matches_with_template(item, target_item_name):
                    return True
            return False

        if not isinstance(effect_item, str):
            return False

        effect_item_lower = effect_item.lower().strip()
        target_lower = target_item_name.lower().strip()

        # Exact match
        if effect_item_lower == target_lower:
            return True

        # singular/plural match — oak_plank <-> oak_planks
        if (effect_item_lower + 's' == target_lower or
                target_lower + 's' == effect_item_lower):
            return True

        # Category match: check whether target is a category of the effect item
        if is_category_name(target_lower):
            category_items = get_category_items(target_lower)
            if effect_item_lower in [item.lower() for item in category_items]:
                return True

        # Bug 5 fix: reverse category match — check whether effect is a category of target.
        # e.g. effect declares item="log", target requires "oak_log".
        if is_category_name(effect_item_lower):
            category_items = get_category_items(effect_item_lower)
            if target_lower in [item.lower() for item in category_items]:
                return True

        # Within-category interchange: check whether two concrete items belong
        # to the same safe category. E.g. oak_log and birch_log are both "logs"
        # and can be interchanged.
        effect_category = self._get_item_safe_category(effect_item_lower)
        target_category = self._get_item_safe_category(target_lower)
        if (effect_category and target_category and
            effect_category == target_category and
            effect_category in get_safe_interchange_categories()):
            return True

        # Template-parameter match
        if '{' in effect_item_lower and '}' in effect_item_lower:
            # Check whether the entire string is a single template parameter
            pure_template_match = re.match(r'^\{[^}]+\}$', effect_item_lower)
            if pure_template_match:
                return False

            # Convert template parameters into a regex
            pattern = effect_item_lower
            pattern = re.sub(r'\{[^}]+\}', r'(.+)', pattern)
            pattern = '^' + pattern + '$'

            try:
                if re.match(pattern, target_lower):
                    return True
            except re.error:
                pass

        return False

    def _get_item_safe_category(self: "EffectMatcher", item_name: str) -> Optional[str]:
        """Get an item's safe-interchange category.

        Queries RESOURCE_ALIASES to determine the item's safe category. Returns
        only categories on the safe-interchange whitelist.

        Args:
            item_name: item name (lowercase)

        Returns:
            Category name (e.g. "logs"), or None if not in any safe category.
        """
        from skillnet.agents.planning.effect_matcher._utils import _get_resource_aliases

        safe_cats = get_safe_interchange_categories()
        for category, items in _get_resource_aliases().items():
            if category in safe_cats:
                items_lower = [i.lower() for i in items] if items else []
                if item_name in items_lower:
                    return category
        return None

    def _determine_item_category(self: "EffectMatcher", target_item: str) -> Optional[str]:
        """Determine the target item's category."""
        categories = getattr(self, '_item_categories', None) or get_item_categories()
        # First check exact item-list matches
        for category_name, category_info in categories.items():
            if category_info.get("items") and target_item in category_info["items"]:
                return category_name

        # Then check keyword matches
        for category_name, category_info in categories.items():
            if any(keyword in target_item for keyword in category_info.get("keywords", [])):
                return category_name

        return None

    def _get_matched_effect_confidence(
        self: "EffectMatcher",
        target_effect: Dict[str, Any],
        skill_effects: List
    ) -> Tuple[float, str]:
        """
        Get the confidence of the matched effect.

        Args:
            target_effect: target effect
            skill_effects: list of skill expected_effects

        Returns:
            Tuple[float, str]: (confidence value 0.0~1.0, confidence level "high"/"medium"/"low"/"uncertain")
        """
        target_item_raw = target_effect.get("item", "")
        if isinstance(target_item_raw, list):
            target_item = target_item_raw[0].lower() if target_item_raw else ""
        else:
            target_item = target_item_raw.lower() if target_item_raw else ""

        if not target_item:
            return 1.0, "high"  # Without a concrete item, default to high confidence

        # Find the matching effect and return its confidence
        for effect in skill_effects:
            state_repr = getattr(effect, 'state_representation', None)
            if state_repr:
                # Check whether state_repr contains the target item
                if self._effect_contains_item(state_repr, target_item):
                    confidence_value = getattr(effect, 'confidence_value', 1.0)
                    confidence_level = getattr(effect, 'confidence_level', 'high')
                    # Ensure values are valid
                    if confidence_value is None:
                        confidence_value = 1.0
                    if confidence_level is None:
                        confidence_level = 'high'
                    return confidence_value, confidence_level

        # No matching effect found — return defaults
        return 1.0, "high"

    def _effect_contains_item(self: "EffectMatcher", state_repr, target_item: str) -> bool:
        """Check whether state_representation contains the target item."""
        if isinstance(state_repr, list):
            for sr in state_repr:
                sr_item = sr.get("item", "")
                if sr_item and self._item_matches_with_template(sr_item, target_item):
                    return True
        elif isinstance(state_repr, dict):
            if "conditions" in state_repr:
                for cond in state_repr.get("conditions", []):
                    cond_item = cond.get("item", "")
                    if cond_item and self._item_matches_with_template(cond_item, target_item):
                        return True
            else:
                sr_item = state_repr.get("item", "")
                if sr_item and self._item_matches_with_template(sr_item, target_item):
                    return True
        return False

