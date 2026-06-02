"""
Skill Finding Mixin (Layer D)

Skill-finding layer: finds skills that satisfy a precondition.

extracted from the PreconditionChecker class in precondition_checker.py.

Methods:
- find_skills_for_precondition: find skills that produce the required precondition (PUBLIC)
- _find_skills_by_state_repr: lookup via state_representation
- _find_skills_by_description: lookup via description
- _find_skills_by_code: lookup via code
- _extract_required_conditions: recursively extract all conditions

Requires self attributes (from PreconditionChecker):
- self.skill_graph_manager
- self.effect_matcher
- self.logger
"""

import re
from typing import Any, Callable, Dict, List, Tuple

from ._utils import (
    extract_generic_type,
    normalize_shorthand_condition,
)


class SkillFindingMixin:
    """
    Skill Finding Mixin - skill lookup.

    Provides functionality for locating skills that can produce a required effect.
    """

    def find_skills_for_precondition(
        self,
        precondition,
        skip_checker: Callable = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find skills that can produce the required precondition.
        Supports OR-logic preconditions.

        Prefer exact matches via state_representation; fall back to description matching.

        Args:
            precondition: precondition object
            skip_checker: optional skip-check function (node) -> bool

        Returns:
            List[Tuple[str, Dict[str, Any]]]: list of (skill name, precondition context) tuples.
        """
        candidate_skills = []

        def build_precond_context(item=None, count=None, is_flexible=False, accepted_items=None):
            """Build the precondition context."""
            return {
                "required_item": item or "unknown",
                "required_count": count or 1,
                "is_flexible": is_flexible,
                "accepted_items": accepted_items or [],
                "precondition_desc": precondition.description if hasattr(precondition, 'description') else str(precondition)
            }

        # Prefer exact matching via state_representation
        state_repr = getattr(precondition, 'state_representation', None)
        if state_repr:
            candidate_skills = self._find_skills_by_state_repr(
                state_repr, precondition, build_precond_context, skip_checker
            )
            if candidate_skills:
                return candidate_skills

        # Fallback: use description and code for matching
        candidate_skills = self._find_skills_by_description(
            precondition, build_precond_context, skip_checker
        )

        if not candidate_skills and hasattr(precondition, 'code') and precondition.code:
            candidate_skills = self._find_skills_by_code(
                precondition, build_precond_context, skip_checker
            )

        return candidate_skills

    def _find_skills_by_state_repr(
        self,
        state_repr,
        precondition,
        build_context: Callable,
        skip_checker: Callable = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find skills that satisfy the precondition via state_representation.

        Args:
            state_repr: state_representation
            precondition: precondition object
            build_context: context builder function
            skip_checker: skip-check function

        Returns:
            List[Tuple[str, Dict[str, Any]]]: list of candidate skills.
        """
        candidate_skills = []
        required_conditions = []

        # Extract all conditions
        self._extract_required_conditions(state_repr, required_conditions)

        # For each condition, find skills that can produce it
        for sr in required_conditions:
            if not isinstance(sr, dict):
                continue

            # Handle item potentially being a list
            raw_item = sr.get("item", "")
            if isinstance(raw_item, list):
                required_item = extract_generic_type(raw_item)
                is_flexible = True
                accepted_items = [item.lower() for item in raw_item if isinstance(item, str)]
            elif isinstance(raw_item, str):
                required_item = raw_item.lower()
                is_flexible = False
                accepted_items = [required_item] if required_item else []
            else:
                continue

            # skip empty item or parameter variable names.
            # Comes from unparseable nested formats (e.g. {"inventory":{"plankName":{...}}}).
            if not required_item:
                continue
            # Detect parameter variable names: no underscore + not a known item name
            if "_" not in required_item:
                # Use domain knowledge if available, else permissive default
                _dk = getattr(self, '_domain_knowledge', None)
                if _dk:
                    _is_valid = _dk.is_valid_item(required_item) or _dk.is_valid_block(required_item)
                else:
                    _is_valid = True  # permissive when no domain knowledge
                if not _is_valid:
                    self.logger.info(f"\033[36m[Precondition] skipping invalid item: '{required_item}'\033[0m")
                    continue

            # support min_count as an alias of count
            required_count = sr.get("count") or sr.get("min_count", 1)
            if isinstance(required_count, str):
                try:
                    required_count = int(required_count)
                except ValueError:
                    required_count = 1

            # default type is inventory
            required_type = sr.get("type", "inventory")
            if required_type != "inventory":
                continue

            # Look in the graph for skills that can produce the required item
            for skill_name, node in self.skill_graph_manager.iter_skills():
                # Skip check
                if skip_checker and skip_checker(node):
                    continue
                if hasattr(node, 'is_deprecated') and node.is_deprecated:
                    continue

                # Check whether the skill's expected_effects match
                target_effect = {
                    "type": required_type,
                    "item": required_item,
                    "count": required_count,
                    "operation": "add"
                }

                if self.effect_matcher and self.effect_matcher.effect_matches(target_effect, node.expected_effects):
                    if skill_name not in [c[0] for c in candidate_skills]:
                        ctx = build_context(required_item, required_count, is_flexible, accepted_items)
                        candidate_skills.append((skill_name, ctx))
                        self.logger.info(f"\033[36m[Precondition] found skill: {skill_name} (state_repr match: {required_item})\033[0m")
                else:
                    # Fallback: description match
                    for effect in node.expected_effects:
                        if hasattr(effect, 'description'):
                            effect_desc = effect.description.lower()
                            if required_item in effect_desc and "add" in effect_desc:
                                if skill_name not in [c[0] for c in candidate_skills]:
                                    ctx = build_context(required_item, required_count, is_flexible, accepted_items)
                                    candidate_skills.append((skill_name, ctx))
                                    self.logger.info(f"\033[36m[Precondition] found skill: {skill_name} (description match)\033[0m")
                                    break

        return candidate_skills

    def _extract_required_conditions(self, state_repr, conditions_list: List):
        """
        Recursively extract all conditions.

        supports {"OR": [...]} and {"AND": [...]} shorthand formats.
        fixed handling of [{"logic": "OR", ...}] nested format.
        """
        if isinstance(state_repr, list):
            # recursively handle nested logic structures inside the list
            for item in state_repr:
                if isinstance(item, dict) and ("logic" in item or "OR" in item or "AND" in item or "or" in item or "and" in item):
                    # If a list item is a dict with logic, recurse
                    self._extract_required_conditions(item, conditions_list)
                else:
                    conditions_list.append(item)
        elif isinstance(state_repr, dict):
            if "logic" in state_repr and "conditions" in state_repr:
                for condition in state_repr.get("conditions", []):
                    self._extract_required_conditions(condition, conditions_list)
            # support {"OR": [...]} shorthand
            # also support lowercase "or", and recurse into nested logic
            elif "OR" in state_repr or "or" in state_repr:
                or_key = "OR" if "OR" in state_repr else "or"
                or_conditions = state_repr.get(or_key, [])
                if isinstance(or_conditions, list):
                    for condition in or_conditions:
                        # Recurse to support nested {"and": [...]} etc.
                        self._extract_required_conditions(condition, conditions_list)
            # support {"AND": [...]} shorthand
            # also support lowercase "and", and recurse into nested logic
            elif "AND" in state_repr or "and" in state_repr:
                and_key = "AND" if "AND" in state_repr else "and"
                and_conditions = state_repr.get(and_key, [])
                if isinstance(and_conditions, list):
                    for condition in and_conditions:
                        self._extract_required_conditions(condition, conditions_list)
            else:
                conditions_list.append(state_repr)

    def _find_skills_by_description(
        self,
        precondition,
        build_context: Callable,
        skip_checker: Callable = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find skills that satisfy the precondition via description.

        Args:
            precondition: precondition object
            build_context: context builder function
            skip_checker: skip-check function

        Returns:
            List[Tuple[str, Dict[str, Any]]]: list of candidate skills.
        """
        candidate_skills = []
        description = precondition.description.lower() if hasattr(precondition, 'description') else str(precondition).lower()

        # Extract keywords
        item_keywords = []
        if "plank" in description:
            item_keywords.append("plank")
        if "log" in description:
            item_keywords.append("log")
        if "stick" in description:
            item_keywords.append("stick")
        if "crafting_table" in description or "crafting" in description:
            item_keywords.append("crafting_table")
        if "pickaxe" in description:
            item_keywords.append("pickaxe")
        if "axe" in description:
            item_keywords.append("axe")
        if "sword" in description:
            item_keywords.append("sword")

        # Look in the graph
        for skill_name, node in self.skill_graph_manager.iter_skills():
            if skip_checker and skip_checker(node):
                continue
            if hasattr(node, 'is_deprecated') and node.is_deprecated:
                continue

            for effect in node.expected_effects:
                effect_desc = effect.description.lower() if hasattr(effect, 'description') else str(effect).lower()

                for keyword in item_keywords:
                    if keyword in effect_desc and "add" in effect_desc:
                        if skill_name not in [c[0] for c in candidate_skills]:
                            ctx = build_context(keyword, None, is_flexible=True, accepted_items=[])
                            candidate_skills.append((skill_name, ctx))
                        break

        return candidate_skills

    def _find_skills_by_code(
        self,
        precondition,
        build_context: Callable,
        skip_checker: Callable = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find skills that satisfy the precondition via code inspection.

        Args:
            precondition: precondition object
            build_context: context builder function
            skip_checker: skip-check function

        Returns:
            List[Tuple[str, Dict[str, Any]]]: list of candidate skills.
        """
        candidate_skills = []
        code_lower = precondition.code.lower()

        item_patterns = [
            r'["\'](\w+_plank)["\']',
            r'["\'](\w+_log)["\']',
            r'["\'](\w+_stick)["\']',
            r'["\'](crafting_table)["\']',
            r'["\'](\w+_pickaxe)["\']',
        ]

        for pattern in item_patterns:
            matches = re.findall(pattern, code_lower)
            for item_name in matches:
                for skill_name, node in self.skill_graph_manager.iter_skills():
                    if skip_checker and skip_checker(node):
                        continue
                    if hasattr(node, 'is_deprecated') and node.is_deprecated:
                        continue

                    for effect in node.expected_effects:
                        effect_desc = effect.description.lower() if hasattr(effect, 'description') else str(effect).lower()
                        if item_name in effect_desc and "add" in effect_desc:
                            if skill_name not in [c[0] for c in candidate_skills]:
                                ctx = build_context(item_name, None, is_flexible=True, accepted_items=[])
                                candidate_skills.append((skill_name, ctx))

        return candidate_skills
