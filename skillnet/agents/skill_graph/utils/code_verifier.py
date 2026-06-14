"""
Code Verifier - code-verification utilities

Pure functions extracted from graph_manager_impl.py for verifying preconditions and effects.
Supports condition checks and effect-implementation validation in code.

v4.0 modular refactor
"""

import re
import logging
from typing import Any, Dict, List, TYPE_CHECKING

from skillnet.core.dk_registry import get_domain_knowledge, register_on_change

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import SkillPrecondition, SkillEffect

logger = logging.getLogger(__name__)


# ── Item knowledge (lazy caches, invalidated on DK change) ──

_resource_aliases: Dict = None
_item_name_categories: Dict = None


def _on_dk_changed(dk):
    """Invalidate lazy caches when domain knowledge changes."""
    global _resource_aliases, _item_name_categories
    _resource_aliases = None
    _item_name_categories = None


register_on_change(_on_dk_changed)


def _get_resource_aliases() -> Dict:
    """Get resource aliases from domain knowledge or empty fallback."""
    global _resource_aliases
    if _resource_aliases is not None:
        return _resource_aliases
    dk = get_domain_knowledge()
    if dk:
        aliases = dk.get_resource_aliases()
        if aliases:
            _resource_aliases = aliases
            return _resource_aliases
    # Don't cache empty — allows future DI to take effect
    return {}


def _get_item_name_categories() -> Dict:
    """Get item name categories from domain knowledge or empty fallback."""
    global _item_name_categories
    if _item_name_categories is not None:
        return _item_name_categories
    dk = get_domain_knowledge()
    if dk:
        cats = dk.get_item_name_categories()
        if cats:
            _item_name_categories = cats
            return _item_name_categories
    return {}


# ========== Bug 4 fix: Primitive call detection ==========

# Empty — domain knowledge provides primitive semantics at runtime.
_FALLBACK_PRIMITIVE_SEMANTICS = {}


def _get_primitive_semantics() -> Dict[str, Dict[str, bool]]:
    """Return primitive semantics from domain knowledge or empty fallback."""
    dk = get_domain_knowledge()
    if dk is not None:
        semantics = dk.get_primitive_semantics()
        if semantics:
            return semantics
    return _FALLBACK_PRIMITIVE_SEMANTICS


def _check_primitive_call_for_item(code: str, prim_name: str, item: str) -> bool:
    """
    Check whether the code calls a primitive with the target item (direct string match).

    Args:
        code: code string
        prim_name: primitive name (e.g. "mineBlock")
        item: target item name (e.g. "oak_log")

    Returns:
        bool: whether a matching call was found
    """
    # Regex: match await prim_name(bot, "item_name", ...) or await prim_name(bot, 'item_name', ...)
    # Also match the no-await case
    patterns = [
        # await mineBlock(bot, "oak_log", ...)
        rf'(?:await\s+)?{re.escape(prim_name)}\s*\(\s*bot\s*,\s*["\']({re.escape(item)})["\']\s*[,)]',
        # await mineBlock(bot, "oak_log")  (no trailing args)
        rf'(?:await\s+)?{re.escape(prim_name)}\s*\(\s*bot\s*,\s*["\']({re.escape(item)})["\']?\s*\)',
    ]
    for pattern in patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return True
    return False


def _get_item_variants(item: str) -> List[str]:
    """
    Get all variants of an item (including specific items and categories).

    For example:
    - "log" -> ["log", "oak_log", "birch_log", "spruce_log", ...]
    - "oak_log" -> ["oak_log", "log"]

    Args:
        item: item name

    Returns:
        list of the item and its variants
    """
    variants = [item]
    item_lower = item.lower()

    # 1. If it is a category name, add all specific items
    if item_lower in _get_item_name_categories():
        variants.extend(_get_item_name_categories()[item_lower])

    # 2. If it is a specific item, find the matching category
    for category, items in _get_item_name_categories().items():
        if item_lower in [i.lower() for i in items]:
            variants.append(category)
            break

    # 3. Expand via _get_resource_aliases()
    if item_lower in _get_resource_aliases():
        variants.extend(_get_resource_aliases()[item_lower])

    return list(set(variants))


def _code_implements_item_via_primitive(
    code: str,
    item: str,
    operation: str
) -> bool:
    """
    Check whether the code implements an operation on the item via primitive calls.

    Bug 4 fix core function: supports detection of indirectly-implemented effects.

    Args:
        code: code string
        item: target item name
        operation: operation type ("add"/"produce" or "remove"/"consume")

    Returns:
        bool: whether an implementation was found
    """
    # Get all variants of the item
    item_variants = _get_item_variants(item)

    # Determine which primitives are relevant to the operation
    semantics = _get_primitive_semantics()
    if operation in ["add", "produce", "adds", "produces"]:
        relevant_primitives = [
            prim for prim, sem in semantics.items()
            if sem["produces"]
        ]
    elif operation in ["remove", "consume", "removes", "consumes"]:
        relevant_primitives = [
            prim for prim, sem in semantics.items()
            if sem["consumes"]
        ]
    else:
        # For unknown operations, check all primitives
        relevant_primitives = list(semantics.keys())

    # Check whether each relevant primitive is called with the target item
    for prim_name in relevant_primitives:
        for variant in item_variants:
            if _check_primitive_call_for_item(code, prim_name, variant):
                logger.debug(
                    f"[EffectVerify] Found {prim_name}(..., '{variant}', ...) "
                    f"implementing {operation} for '{item}'"
                )
                return True

    return False


def verify_precondition_in_code(code: str, key: str, precond_info: Dict[str, Any]) -> bool:
    """
    Verify whether the precondition has a corresponding check in the code.

    Prevent spurious correlation: if the code lacks a check for a condition,
    do not infer it as a precondition even if execution data correlates it with success.

    Args:
        code: skill code
        key: precondition key (e.g. "inventory:crafting_table")
        precond_info: precondition info

    Returns:
        bool: whether the code contains a corresponding check
    """
    if precond_info["type"] == "inventory":
        item = precond_info["item"]
        # Check whether the code contains a check for this item
        patterns = [
            rf'inventory\.count\(["\']?{re.escape(item)}["\']?\)',  # inventory.count("item")
            rf'inventoryCountByName\(["\']?{re.escape(item)}["\']?\)',  # helper function
            rf'invCountByName\(["\']?{re.escape(item)}["\']?\)',  # another helper
            rf'mcData\.itemsByName\[["\']?{re.escape(item)}["\']?\]',  # mcData lookup
            rf'bot\.inventory\.items\(\).*{re.escape(item)}',  # iterating inventory
            rf'if\s*\([^)]*{re.escape(item)}[^)]*\)',  # checked inside an if statement
            rf'require.*{re.escape(item)}',  # item name in a require condition
        ]

        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True

        # If the item name does not appear in the code at all, it is certainly not a real precondition
        if item not in code:
            return False

        # If the item name appears but there is no explicit check logic, it may not be a precondition either
        # Conservative strategy: confirm only when an explicit check exists
        return False

    elif precond_info["type"] == "equipment":
        slot = precond_info["slot"]
        item = precond_info["item"]
        patterns = [
            rf'inventory\.slots\[.*\].*{re.escape(item)}',
            rf'equipment.*{re.escape(slot)}.*{re.escape(item)}',
            rf'bot\.heldItem.*{re.escape(item)}',
            rf'equip.*{re.escape(item)}',
        ]

        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True

        # Check whether item and slot appear in the code
        if item not in code and slot not in code:
            return False

        return False

    # Unknown type, conservative strategy: do not add
    return False


def code_has_precondition_check(code: str, precondition: "SkillPrecondition") -> bool:
    """
    Check whether the code contains a corresponding precondition check.

    Args:
        code: skill code
        precondition: the precondition to check

    Returns:
        bool: whether the code contains a corresponding check
    """
    # Simple heuristic: look for key elements in the precondition code
    # E.g. if the precondition checks "plankCount >= 3", look for a similar check in the code

    # Extract key item names from the precondition
    # Look for item names (e.g. "plank", "stick", "crafting_table")
    item_patterns = re.findall(r'["\']([a-z_]+)["\']', precondition.code.lower())

    # Check whether the code contains checks for these items
    code_lower = code.lower()
    for item in item_patterns:
        # Look for inventory checks, count checks, etc.
        if (f'inventory' in code_lower and item in code_lower) or \
           (f'count' in code_lower and item in code_lower) or \
           (f'getinventorycount' in code_lower and item in code_lower):
            return True

    # If the precondition description contains a number, check for a corresponding quantity check in code
    quantity_match = re.search(r'(\d+)\s+(\w+)', precondition.description.lower())
    if quantity_match:
        quantity, item = quantity_match.groups()
        if item in code_lower and quantity in code_lower:
            return True

    # If undetermined, return False (conservative strategy)
    return False


def code_has_effect_implementation(code: str, effect: "SkillEffect") -> bool:
    """
    Check whether the code contains a corresponding effect implementation.

    Bug 4 fix: enhanced primitive-call detection, supporting indirectly-implemented effects.

    Matching strategy (in priority order):
    1. Items in state_representation + direct string match
    2. Items in state_representation + primitive-call detection (Bug 4 fix)
    3. Items extracted from description + direct string match
    4. Items extracted from description + primitive-call detection (Bug 4 fix)

    Args:
        code: skill code
        effect: the effect to check

    Returns:
        bool: whether the code contains a corresponding implementation
    """
    code_lower = code.lower()

    # Prefer fetching item name and operation type from state_representation
    state_repr = getattr(effect, 'state_representation', None)

    # Dimension effects are world-state transitions, not item manipulations:
    # the item-pattern strategies below can never match them. Accept when the
    # code references the target dimension or portal machinery; runtime
    # verification (post-state dimension) remains the authoritative check.
    if state_repr and isinstance(state_repr, dict) and state_repr.get('type') == 'dimension':
        target_dim = str(state_repr.get('dimension', '')).lower().replace('minecraft:', '')
        if (target_dim and target_dim in code_lower) \
                or 'dimension' in code_lower or 'portal' in code_lower:
            return True
        return False

    if state_repr and isinstance(state_repr, dict):
        item = state_repr.get('item', '')
        operation = state_repr.get('operation', '')  # Bug 4: fetch operation type

        if item:
            # Handle the case where item may be a list (OR logic)
            items_to_check = item if isinstance(item, list) else [item]
            for single_item in items_to_check:
                if not isinstance(single_item, str):
                    continue
                item_lower = single_item.lower()

                # Strategy 1: use strict string-match patterns
                item_patterns = [
                    f"'{item_lower}'",
                    f'"{item_lower}"',
                    f"name === '{item_lower}'",
                    f'name === "{item_lower}"',
                    f"=== '{item_lower}'",
                    f'=== "{item_lower}"',
                    f"countitem('{item_lower}'",  # possible helper function
                    f'countitem("{item_lower}"',
                ]
                if any(pattern in code_lower for pattern in item_patterns):
                    return True

                # Strategy 2 (Bug 4 fix): check primitive calls
                if _code_implements_item_via_primitive(code, item_lower, operation):
                    return True

    # Fallback: extract item name from description
    desc_lower = effect.description.lower()

    # Look for the item name after "Adds X item_name"
    adds_match = re.search(r'adds\s+(\d+\s+)?(\w+)', desc_lower)
    if adds_match:
        added_item = adds_match.group(2)
        if added_item and added_item not in ['to', 'the', 'a', 'an']:
            # Use _get_resource_aliases() for alias matching
            items_to_check = [added_item]
            if added_item in _get_resource_aliases():
                items_to_check.extend(_get_resource_aliases()[added_item])

            for item_to_check in items_to_check:
                # Strategy 3: direct string match
                item_patterns = [
                    f"'{item_to_check}'",
                    f'"{item_to_check}"',
                ]
                if any(pattern in code_lower for pattern in item_patterns):
                    return True

            # Strategy 4 (Bug 4 fix): check primitive calls
            if _code_implements_item_via_primitive(code, added_item, "add"):
                return True

    # Look for the item name after "Produces"
    produces_match = re.search(r'produces\s+\d+\s+(\w+)', desc_lower)
    if produces_match:
        produced_item = produces_match.group(1)
        items_to_check = [produced_item]
        if produced_item in _get_resource_aliases():
            items_to_check.extend(_get_resource_aliases()[produced_item])

        for item_to_check in items_to_check:
            # Strategy 3: direct string match
            item_patterns = [
                f"'{item_to_check}'",
                f'"{item_to_check}"',
            ]
            if any(pattern in code_lower for pattern in item_patterns):
                return True

        # Strategy 4 (Bug 4 fix): check primitive calls
        if _code_implements_item_via_primitive(code, produced_item, "produce"):
            return True

    # Look for the item name after "Consumes"
    consumes_match = re.search(r'consumes\s+\d+\s+(\w+)', desc_lower)
    if consumes_match:
        consumed_item = consumes_match.group(1)
        items_to_check = [consumed_item]
        if consumed_item in _get_resource_aliases():
            items_to_check.extend(_get_resource_aliases()[consumed_item])

        for item_to_check in items_to_check:
            # Strategy 3: direct string match
            item_patterns = [
                f"'{item_to_check}'",
                f'"{item_to_check}"',
            ]
            if any(pattern in code_lower for pattern in item_patterns):
                return True

        # Strategy 4 (Bug 4 fix): check primitive calls
        if _code_implements_item_via_primitive(code, consumed_item, "consume"):
            return True

    # Look for the item name after "Removes" (Bug 4 fix: new Removes support)
    removes_match = re.search(r'removes\s+(\d+\s+)?(\w+)', desc_lower)
    if removes_match:
        removed_item = removes_match.group(2)
        if removed_item and removed_item not in ['to', 'the', 'a', 'an', 'from']:
            items_to_check = [removed_item]
            if removed_item in _get_resource_aliases():
                items_to_check.extend(_get_resource_aliases()[removed_item])

            for item_to_check in items_to_check:
                # Strategy 3: direct string match
                item_patterns = [
                    f"'{item_to_check}'",
                    f'"{item_to_check}"',
                ]
                if any(pattern in code_lower for pattern in item_patterns):
                    return True

            # Strategy 4 (Bug 4 fix): check primitive calls
            if _code_implements_item_via_primitive(code, removed_item, "remove"):
                return True

    # If undetermined, return False (conservative strategy)
    return False


def classify_effect_importance(
    key: str,
    freq_info: Dict[str, Any],
    skill_name: str,
    skill_code: str
) -> str:
    """
    Classify the importance of an effect.

    Args:
        key: effect key (e.g. "inventory:oak_log:1")
        freq_info: frequency info
        skill_name: skill name
        skill_code: skill code

    Returns:
        "core" | "secondary" | "incidental"
        - core: core effect, the main purpose of the skill
        - secondary: secondary effect, related to but not the main purpose of the skill
        - incidental: incidental effect, present in almost all skills, with no planning value
    """
    effect_type = freq_info.get("type", "")

    # Position changes are almost always incidental (most skills move the bot)
    if effect_type == "position":
        return "incidental"

    # Check inventory changes
    if effect_type == "inventory":
        item = freq_info.get("item", "")

        # Handle the case where item may be a list (OR logic)
        if isinstance(item, list):
            # For lists, check the first valid string entry
            item = item[0] if item and isinstance(item[0], str) else ""
        elif not isinstance(item, str):
            item = str(item) if item else ""

        if not item:
            return "secondary"

        # If the skill name contains this item, it is a core effect
        # E.g.: mineLogs -> oak_log, birch_log etc. are core
        item_base = item.replace("_log", "").replace("_planks", "").replace("_pickaxe", "")
        skill_name_lower = skill_name.lower()

        if item in skill_name_lower or item_base in skill_name_lower:
            return "core"

        # Check whether the code mentions this item as a target
        # E.g. await mineBlock(bot, "oak_log", count)
        if item in skill_code:
            # Further check whether it is inside a key call
            semantics = _get_primitive_semantics()
            key_patterns = [rf'return.*{re.escape(item)}']
            for prim_name in semantics:
                key_patterns.append(rf'{re.escape(prim_name)}\([^)]*["\']?{re.escape(item)}["\']?')
            for pattern in key_patterns:
                if re.search(pattern, skill_code, re.IGNORECASE):
                    return "core"

            return "secondary"

        return "secondary"

    # Equipment changes are usually secondary
    if effect_type == "equipment":
        return "secondary"

    # Block changes (mining/placement) are usually core effects
    if effect_type == "block":
        action = freq_info.get("action", "")
        # If mining/placing, it may be core
        if action in ["mine", "place", "break", "dig"]:
            return "core"
        return "secondary"

    return "secondary"


def validate_naming_effect_consistency(
    skill_name: str,
    effects: List["SkillEffect"]
) -> List[str]:
    """
    Validate consistency between naming and effects; return a list of warnings.

    Args:
        skill_name: skill name
        effects: list of extracted effects

    Returns:
        List[str]: list of warning messages
    """
    warnings = []

    name_lower = skill_name.lower()

    # Rule 1: ensure* should not have a world:place effect
    if name_lower.startswith('ensure'):
        has_world_effect = any(
            e.state_representation and
            isinstance(e.state_representation, dict) and
            e.state_representation.get('type') == 'world'
            for e in effects
        )
        if has_world_effect:
            warnings.append(
                f"Naming-Effect Mismatch: Skill '{skill_name}' starts with 'ensure' but has world effects. "
                f"ensure* should only handle inventory items. Consider renaming to 'setup*' "
                f"if it places blocks in the world."
            )

    # Rule 2: setup* should have a world:place effect
    if name_lower.startswith('setup'):
        has_place_effect = any(
            e.state_representation and
            isinstance(e.state_representation, dict) and
            e.state_representation.get('type') == 'world' and
            e.state_representation.get('operation') in ['place', 'place_usable']
            for e in effects
        )
        if not has_place_effect:
            warnings.append(
                f"Naming-Effect Mismatch: Skill '{skill_name}' starts with 'setup' but has no world:place effect. "
                f"setup* should ensure a functional block is placed and usable in the world."
            )

    # Rule 3: place* should have a world:place effect
    if name_lower.startswith('place') and not name_lower.startswith('placeitem'):
        has_place_effect = any(
            e.state_representation and
            isinstance(e.state_representation, dict) and
            e.state_representation.get('type') == 'world' and
            e.state_representation.get('operation') == 'place'
            for e in effects
        )
        if not has_place_effect:
            warnings.append(
                f"Naming-Effect Mismatch: Skill '{skill_name}' starts with 'place' but has no world:place effect. "
                f"place* should place blocks in the world."
            )

    return warnings
