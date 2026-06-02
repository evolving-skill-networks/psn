"""
Checking Mixin (Layer C)

Precondition-checking layer: responsible for verifying whether preconditions are satisfied.

extracted from the PreconditionChecker class in precondition_checker.py.

Methods:
- check_preconditions: check whether preconditions are satisfied (PUBLIC)
- _check_state_representation: recursively check state_representation (AND/OR logic)
- _check_single_precondition: check a single condition
- _check_biome_resource: check biome resources (v7.5)
- collect_environmental_feedback: gather environmental feedback (PUBLIC)
- _check_precondition_from_description: fallback: description check
- _check_inventory_from_description: fallback: description -> inventory check
- _check_precondition_from_code: fallback: code check
- _check_inventory_from_code: fallback: code -> inventory check
- update_expected_inventory: update expected inventory (PUBLIC)

Requires self attributes (from PreconditionChecker):
- self.skill_graph_manager
- self.effect_matcher
- self.logger
"""

import re
from typing import Any, Callable, Dict, List, Optional

from ._utils import (
    check_tool_requirement,
    condition_matches,
    extract_conditions_from_state_repr,
    extract_generic_type,
    normalize_shorthand_condition,
)
from ._environmental import EnvironmentalFeedback

from . import _utils as _precond_utils


class CheckingMixin:
    """
    Checking Mixin — precondition checking.

    Provides precondition verification, environmental feedback collection, and inventory update functionality.
    """

    def check_preconditions(
        self,
        preconditions: List,
        current_state: Dict[str, Any],
        param_values: Dict[str, Any] = None,
        condition_matcher: Callable = None,
    ) -> List:
        """
        Check whether preconditions are satisfied

        Prefer state_representation for verification; fall back to code/description parsing if absent

        Supports conditional preconditions: if a precondition has a condition field,
        the precondition is only checked when param_values matches the condition.

        Args:
            preconditions: list of preconditions
            current_state: current state (contains inventory, etc.)
            param_values: current parameter values (used for conditional preconditions)
            condition_matcher: optional condition matching function (used to call back to GraphPlanner)

        Returns:
            List: list of unsatisfied preconditions
        """
        missing = []
        inventory = current_state.get('inventory', {})

        # Use the provided matcher or the default one
        _condition_matches = condition_matcher or condition_matches

        for precondition in preconditions:
            # Check conditional precondition: if there's a condition field, check whether it matches
            condition = getattr(precondition, 'condition', None)
            if condition and param_values:
                if not _condition_matches(condition, param_values):
                    # Condition does not match; skip this precondition (not applicable to current parameter values)
                    continue

            # Get the precondition's description
            description = precondition.description if hasattr(precondition, 'description') else str(precondition)

            # Special handling: preconditions starting with "OR:" represent optional paths
            if description.upper().startswith("OR:"):
                self.logger.info(f"\033[36m[Precondition] Skipping optional precondition (starts with OR:): {description[:80]}...\033[0m")
                continue

            # Prefer state_representation for verification
            # pass current_state to support biome_resource type checks
            state_repr = getattr(precondition, 'state_representation', None)
            if state_repr:
                satisfied = self._check_state_representation(state_repr, inventory, current_state)
                if not satisfied:
                    missing.append(precondition)
                continue

            # If no state_representation, check using code/description
            if not hasattr(precondition, 'code') or not precondition.code:
                # Simple description-based check
                if not self._check_precondition_from_description(precondition, description, inventory):
                    missing.append(precondition)
                continue

            # Check using code
            if not self._check_precondition_from_code(precondition, inventory):
                missing.append(precondition)

        return missing

    def _check_state_representation(
        self,
        state_repr,
        inventory: Dict[str, int],
        current_state: Dict[str, Any] = None,
    ) -> bool:
        """
        Recursively check state_representation; supports AND/OR logic

        added current_state parameter to support environment checks (e.g. biome_resource type)

        Args:
            state_repr: state_representation; may be a list (old format) or dict (new format)
            inventory: current inventory
            current_state: full current state (contains biome, position, etc.)

        Returns:
            bool: whether satisfied
        """
        if not state_repr:
            return True

        # If it's a list (old format, default AND logic)
        # Each element may be a simple leaf condition OR a nested logic dict.
        if isinstance(state_repr, list):
            for sr in state_repr:
                if isinstance(sr, dict) and (
                    "logic" in sr or "OR" in sr or "or" in sr
                    or "AND" in sr or "and" in sr or "conditions" in sr
                ):
                    # Nested logic — recurse into _check_state_representation
                    if not self._check_state_representation(sr, inventory, current_state):
                        return False
                else:
                    if not self._check_single_precondition(sr, inventory, current_state):
                        return False
            return True

        # If it's a dict
        if isinstance(state_repr, dict):
            # Check whether a logic field is present (new format)
            if "logic" in state_repr and "conditions" in state_repr:
                logic = state_repr.get("logic", "AND").upper()
                conditions = state_repr.get("conditions", [])

                if logic == "OR":
                    # OR logic: any condition satisfied is sufficient
                    for condition in conditions:
                        if self._check_state_representation(condition, inventory, current_state):
                            return True
                    return False
                else:  # AND logic (default)
                    for condition in conditions:
                        if not self._check_state_representation(condition, inventory, current_state):
                            return False
                    return True

            # support shorthand format {"OR": [...]} and {"AND": [...]}
            # also support lowercase "or"/"and" (LLM-generated preconditions often use lowercase)
            elif "OR" in state_repr or "or" in state_repr:
                or_key = "OR" if "OR" in state_repr else "or"
                conditions = state_repr.get(or_key, [])
                if isinstance(conditions, list):
                    for condition in conditions:
                        # Shorthand conditions may lack a type field; default to inventory
                        normalized = normalize_shorthand_condition(condition)
                        if self._check_single_precondition(normalized, inventory, current_state):
                            return True
                    return False
                return self._check_single_precondition(state_repr, inventory, current_state)

            elif "AND" in state_repr or "and" in state_repr:
                and_key = "AND" if "AND" in state_repr else "and"
                conditions = state_repr.get(and_key, [])
                if isinstance(conditions, list):
                    for condition in conditions:
                        normalized = normalize_shorthand_condition(condition)
                        if not self._check_single_precondition(normalized, inventory, current_state):
                            return False
                    return True
                return self._check_single_precondition(state_repr, inventory, current_state)

            # additional LLM-generated logic keywords that cold-start
            # preconditions commonly use. These require proper recursion because
            # they often contain nested wrappers like {"any_of": [{"inventory": ...}, ...]}.
            elif "any_of" in state_repr:
                conditions = state_repr.get("any_of", [])
                if isinstance(conditions, list):
                    for condition in conditions:
                        if self._check_state_representation(condition, inventory, current_state):
                            return True
                    return False
                return self._check_single_precondition(state_repr, inventory, current_state)

            elif "all_of" in state_repr:
                conditions = state_repr.get("all_of", [])
                if isinstance(conditions, list):
                    for condition in conditions:
                        if not self._check_state_representation(condition, inventory, current_state):
                            return False
                    return True
                return self._check_single_precondition(state_repr, inventory, current_state)

            # {"inventory": inner} wrapper — LLMs sometimes nest
            # inventory-typed conditions under an explicit "inventory" key.
            # Unwrap and recurse into the inner condition.
            elif "inventory" in state_repr and isinstance(state_repr["inventory"], dict):
                return self._check_state_representation(state_repr["inventory"], inventory, current_state)

            # {"slot": N, ...} equipment-slot checks are fragile
            # (bot equipment state is not reliably tracked in the snapshot we
            # receive here). Strip the slot key and fall back to an inventory
            # search — if the item exists in inventory, we treat it as satisfying
            # the slot requirement. This is intentionally permissive: runtime
            # execution will catch any truly-missing items.
            elif "slot" in state_repr:
                stripped = {k: v for k, v in state_repr.items() if k != "slot"}
                return self._check_single_precondition(stripped, inventory, current_state)

            else:
                # Old format: single condition dict
                return self._check_single_precondition(state_repr, inventory, current_state)

        return False

    def _check_single_precondition(
        self,
        state_repr: Dict[str, Any],
        inventory: Dict[str, int],
        current_state: Dict[str, Any] = None,
    ) -> bool:
        """
        Check whether a single precondition's state_representation is satisfied

        added biome_resource type support, used to check whether the current biome has the required resource
        support min_count as an alias for count

        Args:
            state_repr: state representation
            inventory: current inventory
            current_state: full current state (contains biome, position, etc.)

        Returns:
            bool: whether satisfied
        """
        if not isinstance(state_repr, dict):
            return False

        req_type = state_repr.get("type", "inventory")  # default to inventory
        # LLM-generated preconditions routinely use the
        # `"name"` key in place of `"item"` (65% of code-extraction samples
        # on Qwen3-Coder). Both refer to the Minecraft item id — accept either.
        # Without this alias, the `any_of` recursion at line 201 would feed
        # `{"name": "wooden_pickaxe"}` here and silently produce req_item=""
        # → planner reports unsatisfied even with the pickaxe in inventory.
        req_item = state_repr.get("item") or state_repr.get("name", "")
        # support min_count as an alias for count
        req_count = state_repr.get("count") or state_repr.get("min_count", 1)

        # Ensure req_count is an integer
        if isinstance(req_count, str):
            try:
                req_count = int(req_count)
            except ValueError:
                req_count = 1

        req_operation = state_repr.get("operation", "require")
        req_tool_tier = state_repr.get("tool_tier", None)

        # support biome_resource type checks
        if req_type == "biome_resource":
            return self._check_biome_resource(state_repr, current_state)

        if req_type != "inventory":
            # Assume other types are satisfied for now (cannot verify)
            return True

        # Handle dict-valued `item` (LLM-generated name_contains format).
        # Examples from r38:
        # {"item": {"name_contains": "pickaxe"}}  → match any inventory item
        # whose name contains "pickaxe"
        # {"item": {"name": "diamond_pickaxe"}}   → exact match alias
        # Without this, the original code would crash on `inventory.get(dict, 0)`
        # with `TypeError: unhashable type: 'dict'`, which upstream callers
        # silently treated as "precondition unsatisfied".
        if isinstance(req_item, dict):
            name_contains = req_item.get("name_contains")
            name_equals = req_item.get("name") or req_item.get("equals")
            if name_contains:
                nc = str(name_contains).lower()
                for inv_name, inv_count in inventory.items():
                    if nc in inv_name.lower() and inv_count >= req_count:
                        return True
                return False
            if name_equals:
                return inventory.get(str(name_equals), 0) >= req_count
            # Unknown dict shape — conservative: treat as unsatisfied
            return False

        if req_operation == "require":
            # Exact requirement
            item_count = inventory.get(req_item, 0)
            return item_count >= req_count
        elif req_operation == "require_or_better":
            # Required or better tool
            if check_tool_requirement(req_item, req_operation, inventory, req_tool_tier):
                return True
            # If no better tool found, check for an exact match
            item_count = inventory.get(req_item, 0)
            return item_count >= req_count
        else:
            # Unknown operation; conservative: assume unsatisfied
            return False

    def _check_biome_resource(
        self,
        state_repr: Dict[str, Any],
        current_state: Dict[str, Any] = None,
    ) -> bool:
        """
        check whether the current biome has the required resource

        Args:
            state_repr: state representation, e.g.:
                {
                    "type": "biome_resource",
                    "resource": "lava",
                    "operation": "require"  # optional
                }
            current_state: current state; must contain a biome field

        Returns:
            bool: whether the current biome has the resource
        """
        if not current_state:
            # No state info; conservative: assume satisfied
            self.logger.debug("[Precondition] biome_resource check: no current_state, assuming satisfied")
            return True

        current_biome = current_state.get("biome", "")
        if not current_biome or current_biome == "None":
            # No biome info; conservative: assume satisfied
            self.logger.debug("[Precondition] biome_resource check: no biome info, assuming satisfied")
            return True

        required_resource = state_repr.get("resource", "")
        if not required_resource:
            return True

        # Get available resources for the current biome (via DI)
        dk = _precond_utils._domain_knowledge
        biome_resources = dk.get_biome_resources(current_biome) if dk else []

        # Check whether the resource is available in the current biome
        resource_available = required_resource.lower() in [r.lower() for r in biome_resources]

        if resource_available:
            self.logger.info(
                f"\033[32m[Precondition] biome_resource satisfied: {current_biome} has {required_resource}\033[0m"
            )
        else:
            self.logger.warning(
                f"\033[33m[Precondition] biome_resource not satisfied: {current_biome} does not have {required_resource}. "
                f"Available resources: {biome_resources}\033[0m"
            )

        return resource_available

    def collect_environmental_feedback(
        self,
        preconditions: List,
        current_state: Dict[str, Any],
        param_values: Dict[str, Any] = None,
    ) -> Optional[EnvironmentalFeedback]:
        """
        collect environmental feedback

        Inspect biome_resource conditions in preconditions;
        when unsatisfied, generate structured environmental feedback.

        Args:
            preconditions: list of preconditions
            current_state: current state
            param_values: current parameter values

        Returns:
            EnvironmentalFeedback: feedback if there is an environment mismatch; otherwise None
        """
        if not current_state:
            return None

        current_biome = current_state.get("biome", "")
        if not current_biome or current_biome == "None":
            return None

        missing_resources = []

        for precondition in preconditions:
            # Check conditional precondition
            condition = getattr(precondition, 'condition', None)
            if condition and param_values:
                if not condition_matches(condition, param_values):
                    continue

            state_repr = getattr(precondition, 'state_representation', None)
            if not state_repr:
                continue

            # Extract all biome_resource type conditions
            conditions = extract_conditions_from_state_repr(state_repr)
            for cond in conditions:
                if isinstance(cond, dict) and cond.get("type") == "biome_resource":
                    resource = cond.get("resource", "")
                    if resource:
                        dk = _precond_utils._domain_knowledge
                        biome_resources = dk.get_biome_resources(current_biome) if dk else []
                        if resource.lower() not in [r.lower() for r in biome_resources]:
                            missing_resources.append(resource)

        if not missing_resources:
            return None

        # Get available resources (via DI)
        dk = _precond_utils._domain_knowledge
        available_resources = dk.get_biome_resources(current_biome) if dk else []

        # Get bot position
        bot_position = {}
        if "position" in current_state:
            pos = current_state["position"]
            if isinstance(pos, dict):
                bot_position = pos
            elif hasattr(pos, "y"):
                bot_position = {"x": pos.x, "y": pos.y, "z": pos.z}

        return EnvironmentalFeedback(
            feedback_type="environment_mismatch",
            current_biome=current_biome,
            missing_resources=missing_resources,
            available_resources=available_resources,
            bot_position=bot_position,
        )

    def _check_precondition_from_description(
        self,
        precondition,
        description: str,
        inventory: Dict[str, int],
    ) -> bool:
        """
        Check whether a precondition is satisfied from its description

        Args:
            precondition: precondition object
            description: description text
            inventory: current inventory

        Returns:
            bool: whether satisfied
        """
        description_lower = description.lower()

        # Check "has a pickaxe"-style preconditions
        dk = _precond_utils._domain_knowledge
        item_groups = dk.get_item_groups() if dk else {}

        if "pickaxe" in description_lower and ("has" in description_lower or "have" in description_lower):
            pickaxe_types = item_groups.get("pickaxes", [])
            if not pickaxe_types:
                return None  # Can't verify without domain knowledge
            has_pickaxe = any(inventory.get(p, 0) > 0 for p in pickaxe_types)
            if has_pickaxe:
                self.logger.info(f"\033[32m[Precondition] '{description[:80]}...' satisfied: pickaxe is in inventory\033[0m")
                return True
            else:
                self.logger.info(f"\033[33m[Precondition] '{description[:80]}...' not satisfied: no pickaxe in inventory\033[0m")
                return False

        # Check "has an axe"-style preconditions
        if "axe" in description_lower and "pickaxe" not in description_lower and \
           ("has" in description_lower or "have" in description_lower):
            axe_types = item_groups.get("axes", [])
            if not axe_types:
                return None  # Can't verify without domain knowledge
            has_axe = any(inventory.get(a, 0) > 0 for a in axe_types)
            if has_axe:
                self.logger.info(f"\033[32m[Precondition] '{description[:80]}...' satisfied: axe is in inventory\033[0m")
                return True
            return False

        # Check "inventory" or "at least" patterns
        if "inventory" in description_lower or "at least" in description_lower:
            return self._check_inventory_from_description(description_lower, inventory)

        # Cannot determine; conservative: assume unsatisfied
        return False

    def _check_inventory_from_description(
        self,
        description_lower: str,
        inventory: Dict[str, int],
    ) -> bool:
        """
        Extract and check inventory requirements from a description

        Args:
            description_lower: lowercased description text
            inventory: current inventory

        Returns:
            bool: whether satisfied
        """
        # Extract count requirement
        count_match = re.search(r'at least (\d+)', description_lower)
        required_count = int(count_match.group(1)) if count_match else 1

        # Extract item names
        from skillnet.core.dk_registry import get_domain_knowledge
        _dk = get_domain_knowledge()
        _groups = _dk.get_item_groups() if _dk else {}
        item_keywords = []
        if "plank" in description_lower:
            item_keywords.extend(_groups.get("planks", []))
        if "log" in description_lower:
            item_keywords.extend(_groups.get("logs", []))
        if "stick" in description_lower:
            item_keywords.append("stick")
        if "crafting_table" in description_lower:
            item_keywords.append("crafting_table")

        # Check whether the inventory contains enough items
        total_count = 0
        for keyword in item_keywords:
            total_count += inventory.get(keyword, 0)

        return total_count >= required_count

    def _check_precondition_from_code(
        self,
        precondition,
        inventory: Dict[str, int],
    ) -> bool:
        """
        Check whether a precondition is satisfied by inspecting its code

        Args:
            precondition: precondition object
            inventory: current inventory

        Returns:
            bool: whether satisfied
        """
        try:
            code_lower = precondition.code.lower()
            description_lower = (precondition.description.lower()
                               if hasattr(precondition, 'description') else "")

            # Check whether it contains an inventory check
            if "inventory" in code_lower or "items()" in code_lower:
                # Special handling: "preferred" preconditions
                if "preferredlogtypes" in code_lower or "preferred" in description_lower:
                    if "are present in the bot's inventory" in description_lower or "inventory" in description_lower:
                        self.logger.info(f"\033[36m[Precondition] '{precondition.description[:80]}...' is an inventory check, treating as satisfied\033[0m")
                        return True

                # Try to extract items from code and check
                return self._check_inventory_from_code(code_lower, description_lower, inventory)
            else:
                # Check non-inventory preconditions like nearby blocks
                if "findblock" in code_lower or "nearby" in description_lower or "within" in description_lower:
                    return True  # Assume satisfied
                return False

        except Exception as e:
            self.logger.warning(f"\033[33m[Precondition] check failed: {e}\033[0m")
            return False

    def _check_inventory_from_code(
        self,
        code_lower: str,
        description_lower: str,
        inventory: Dict[str, int],
    ) -> bool:
        """
        Extract and check inventory requirements from code

        Args:
            code_lower: lowercased code
            description_lower: lowercased description
            inventory: current inventory

        Returns:
            bool: whether satisfied
        """
        # Item matching patterns
        item_patterns = [
            r'["\'](\w+_plank)["\']',
            r'["\'](\w+_log)["\']',
            r'["\'](\w+_stick)["\']',
            r'["\'](crafting_table)["\']',
            r'["\'](\w+_pickaxe)["\']',
            r'["\'](\w+_axe)["\']',
        ]

        # Count matching patterns
        count_patterns = [
            r'>= (\d+)',
            r'> (\d+)',
            r'count >= (\d+)',
            r'count > (\d+)',
        ]

        required_count = 1
        for pattern in count_patterns:
            match = re.search(pattern, code_lower)
            if match:
                required_count = int(match.group(1))
                break

        # Extract required items
        required_items = []
        for pattern in item_patterns:
            matches = re.findall(pattern, code_lower)
            required_items.extend(matches)

        # If nothing found in code, try extracting from description
        if not required_items:
            from skillnet.core.dk_registry import get_domain_knowledge
            _dk = get_domain_knowledge()
            _groups = _dk.get_item_groups() if _dk else {}
            if "log" in description_lower:
                required_items = list(_groups.get("logs", []))
            elif "plank" in description_lower:
                required_items = list(_groups.get("planks", []))
            elif "stick" in description_lower:
                required_items = ["stick"]

        # Check inventory
        if required_items:
            total_count = 0
            for item in required_items:
                total_count += inventory.get(item, 0)
            return total_count >= required_count

        # Could not extract; check if it's a nearby-block type
        if "findblock" in code_lower:
            return True  # Assume satisfied

        return False

    def update_expected_inventory(
        self,
        expected_inventory: Dict[str, int],
        node,
    ) -> None:
        """
        Update expected inventory (based on the skill's expected_effects)

        Args:
            expected_inventory: expected inventory dict (modified in place)
            node: skill node
        """
        if not node or not hasattr(node, 'expected_effects'):
            return

        from ..effect_matcher import normalize_operation

        for effect in node.expected_effects:
            state_repr = getattr(effect, 'state_representation', None)
            if not state_repr:
                continue

            conditions = extract_conditions_from_state_repr(state_repr)

            for cond in conditions:
                # use normalize_operation to ensure "ensure" → "add" mapping
                cond_operation = normalize_operation(cond.get('operation', 'add'))
                if cond.get('type') == 'inventory' and cond_operation == 'add':
                    item = cond.get('item')
                    count = cond.get('count', 1)

                    if isinstance(count, str):
                        try:
                            count = int(count)
                        except ValueError:
                            count = 1

                    # Handle list-typed items
                    if isinstance(item, list):
                        generic_type = extract_generic_type(item)
                        expected_inventory[generic_type] = expected_inventory.get(generic_type, 0) + count
                    elif isinstance(item, str):
                        expected_inventory[item] = expected_inventory.get(item, 0) + count
