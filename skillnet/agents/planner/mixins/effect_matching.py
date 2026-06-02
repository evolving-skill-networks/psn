"""Effect Matching Mixin - Skill effect extraction, matching, and quality assessment."""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from skillnet.agents.planning import normalize_operation, SUPPORTED_OPERATIONS

if TYPE_CHECKING:
    from ..graph_planner import GraphPlanner


class EffectMatchingMixin:
    """Effect Matching Mixin - Skill effect extraction, matching, and quality assessment.

    Methods:
        _extract_target_effects: Extract target effects from task - uses effect_matcher
        _find_skills_by_effects: Find skills by effects - uses effect_matcher
        _find_uncovered_effects: Find uncovered effects
        _match_effects_to_skills: Match effects to skills
        _build_multi_goal_sequence: Build multi-goal sequence
        _effect_matches: Check if effect matches - delegates to effect_matcher
        _extract_items_from_skill_effects: Extract items from skill effects

    Note:
        Core effect matching logic has been extracted to skillnet.agents.planning.EffectMatcher.
        Methods here are mainly wrappers and coordination calls.
    """

    def _extract_target_effects(self, task: str, context: str) -> List[Dict[str, Any]]:
        """
        Extract target effects from a task description.

        delegates to EffectMatcher.

        Returns:
            List[Dict]: list of target effects, each containing type and value.
            E.g. [{"type": "inventory", "item": "oak_log", "count": 3, "operation": "add"}]
        """
        return self.effect_matcher.extract_target_effects(task, context)

    def _find_skills_by_effects(self, target_effects: List[Dict[str, Any]], task: str = "") -> List[str]:
        """
        Find skills based on target effects.

        delegates to EffectMatcher.

        Args:
            target_effects: list of target effects
            task: current task description (used for semantic-conflict detection)

        Returns:
            List[str]: skill names that can produce the target effects.
        """
        # Delegate to EffectMatcher, passing auxiliary checkers
        candidate_skills = self.effect_matcher.find_skills_by_effects(
            target_effects=target_effects,
            task=task,
            current_state=self._current_state if hasattr(self, '_current_state') else None,
            composite_checker=self._is_composite_skill,
            material_checker=self._player_has_required_materials,
        )

        # Sync quality-score caches (use public property instead of private attribute)
        self._skill_match_quality = self.effect_matcher.skill_match_quality
        self._skill_consistency_penalty = self.effect_matcher.skill_consistency_penalty

        return candidate_skills

    def _find_uncovered_effects(self, target_effects: List[Dict[str, Any]], candidate_skills: List[str]) -> List[Dict[str, Any]]:
        """
        Check which target effects are not covered by candidate skills.

        extended to support place/equip operations.

        Args:
            target_effects: list of target effects
            candidate_skills: list of candidate skill names

        Returns:
            List[Dict]: list of uncovered effects.
        """
        if not target_effects:
            return []

        # If there are no candidate skills, all effects are uncovered
        if not candidate_skills:
            return target_effects

        uncovered = []
        count_param_names = {'count', 'targettotal', 'amount', 'quantity', 'num', 'total', 'target'}

        for effect in target_effects:
            raw_operation = effect.get("operation", "add")
            # normalize operation type (e.g. ensure -> add)
            operation = normalize_operation(raw_operation)
            item = effect.get("item", "unknown")

            # check whether the operation type is supported
            if operation in SUPPORTED_OPERATIONS:
                # Check whether any skill can produce this effect
                covered = False
                for skill_name in candidate_skills:
                    node = self.skill_graph_manager.get_node(skill_name)
                    if node and node.expected_effects:
                        # Check whether there is a count parameter
                        has_count_param = False
                        if hasattr(node, 'parameters') and node.parameters:
                            param_names_lower = {p.lower() for p in node.parameters}
                            has_count_param = bool(param_names_lower & count_param_names)

                        # Create a normalized effect for matching
                        normalized_effect = dict(effect)
                        normalized_effect["operation"] = operation

                        if self._effect_matches(normalized_effect, node.expected_effects, ignore_count=has_count_param):
                            covered = True
                            self.logger.debug(
                                f"[Graph Planner] effect '{item}' (operation={operation}) "
                                f"covered by skill '{skill_name}'"
                            )
                            break

                if not covered:
                    self.logger.info(
                        f"\033[36m[Graph Planner] effect '{item}' (operation={operation}) "
                        f"has no matching skill\033[0m"
                    )
            else:
                # Unsupported operation type (e.g. remove, use, etc.)
                # This triggers fallback to the LLM.
                covered = False
                self.logger.info(
                    f"\033[36m[Graph Planner] effect '{item}' (operation={operation}) "
                    f"is an unsupported operation type; falling back to LLM\033[0m"
                )

            if not covered:
                uncovered.append(effect)

        return uncovered

    def _match_effects_to_skills(
        self,
        target_effects: List[Dict[str, Any]],
        candidate_skills: List[str]
    ) -> Dict[str, List[str]]:
        """
        Map each target effect to the best skill capable of producing it.

        This is the key method for composite-task support. For a task like
        "Mine 8 raw iron and 4 coal" it returns
        {"raw_iron": ["mineIronOre"], "coal": ["mineCoalOre"]}.

        Args:
            target_effects: list of target effects
            candidate_skills: list of candidate skill names

        Returns:
            Dict[str, List[str]]: effect item -> matching skills.
        """
        effect_to_skills: Dict[str, List[str]] = {}

        # [P11 fix] Handle all operation types, not just "add".
        # EffectMatcher._check_state_representation() already supports add/place/remove/use, etc.
        # This lets "Craft planks, then place a chest" match both effects.

        for effect in target_effects:
            item = effect.get("item", "")
            if not item:
                continue

            operation = effect.get("operation", "add")
            # Use "item:operation" as the key so we differentiate operations on the same item.
            # E.g. "chest:add" (obtain) vs "chest:place" (place).
            effect_key = f"{item}:{operation}" if operation != "add" else item

            matching_skills = []
            for skill_name in candidate_skills:
                node = self.skill_graph_manager.get_node(skill_name)
                if node and node.expected_effects:
                    # Check whether there is a count parameter
                    count_param_names = {'count', 'targettotal', 'amount', 'quantity', 'num', 'total', 'target'}
                    has_count_param = False
                    if hasattr(node, 'parameters') and node.parameters:
                        param_names_lower = {p.lower() for p in node.parameters}
                        has_count_param = bool(param_names_lower & count_param_names)

                    if self._effect_matches(effect, node.expected_effects, ignore_count=has_count_param):
                        matching_skills.append(skill_name)

            if matching_skills:
                effect_to_skills[effect_key] = matching_skills

        return effect_to_skills

    def _build_multi_goal_sequence(
        self,
        target_effects: List[Dict[str, Any]],
        candidate_skills: List[str],
        current_state: Dict[str, Any],
        max_depth: int = 5,
        param_values: Dict[str, Any] = None
    ) -> List:
        """
        Build a skill-execution sequence for multiple target effects.

        Unlike _build_skill_sequence, this method picks one skill per target
        effect and composes their execution sequences.

        Args:
            target_effects: list of target effects
            candidate_skills: list of candidate skill names
            current_state: current state
            max_depth: maximum recursion depth
            param_values: parameter values

        Returns:
            List[SkillCallContext]: combined skill-execution sequence.
        """
        # Step 1: map each effect to the best skill
        effect_to_skills = self._match_effects_to_skills(target_effects, candidate_skills)

        if not effect_to_skills:
            self.logger.warning(f"\033[33m[Graph Planner] could not map any target effect to a skill\033[0m")
            return []

        self.logger.info(f"\033[36m[Graph Planner] effect-to-skill mapping: {effect_to_skills}\033[0m")

        # Step 2: pick the best skill for each effect and build its sequence
        all_sequences = []
        used_skills = set()  # Track skills already used to avoid duplicates

        for item, skills in effect_to_skills.items():
            # Pick the best skill (consider success rate, etc.)
            best_skill = self._select_best_skill(skills)

            if best_skill in used_skills:
                # If the skill is already used, skip (one skill may produce multiple effects)
                self.logger.info(f"\033[36m[Graph Planner] skill '{best_skill}' already in sequence, skipping duplicate\033[0m")
                continue

            # Build the single-skill sequence (including its dependencies)
            single_sequence = self._build_skill_sequence(
                [best_skill],
                current_state,
                max_depth=max_depth,
                param_values=param_values,
                context_for_candidates=None
            )

            if single_sequence:
                # Append to the overall sequence and mark used skills
                for ctx in single_sequence:
                    if ctx.skill_name not in used_skills:
                        all_sequences.append(ctx)
                        used_skills.add(ctx.skill_name)
            else:
                self.logger.warning(f"\033[33m[Graph Planner] could not build a skill sequence for effect '{item}'\033[0m")

        if not all_sequences:
            self.logger.warning(f"\033[33m[Graph Planner] multi-goal sequence construction failed\033[0m")
            return []

        self.logger.info(f"\033[36m[Graph Planner] multi-goal sequence: {[ctx.skill_name for ctx in all_sequences]}\033[0m")
        return all_sequences

    def _effect_matches(self, target_effect: Dict[str, Any], skill_effects: List, ignore_count: bool = False) -> bool:
        """
        Check whether a skill's effects match the target effect.

        delegates to EffectMatcher.

        Args:
            target_effect: target effect
            skill_effects: skill's expected_effects list
            ignore_count: whether to ignore count checks

        Returns:
            bool: whether it matches.
        """
        return self.effect_matcher.effect_matches(target_effect, skill_effects, ignore_count)

    def _extract_items_from_skill_effects(self, skill_effects: List) -> dict:
        """
        Extract item names from skill effects, distinguishing deterministic from OR logic.

        Args:
            skill_effects: list of SkillEffect

        Returns:
            dict: {
                "guaranteed": List[str] — items guaranteed to be produced,
                "or_alternatives": List[List[str]] — candidate groups under OR logic
            }
        """
        result = {"guaranteed": [], "or_alternatives": []}
        if not skill_effects:
            return result

        for effect in skill_effects:
            state_repr = getattr(effect, 'state_representation', None)
            if not state_repr:
                continue

            if isinstance(state_repr, dict):
                if state_repr.get("logic", "").upper() == "OR":
                    # OR logic: only one of the alternatives is produced; not all guaranteed
                    conditions = state_repr.get("conditions", [])
                    or_items = []
                    for condition in conditions:
                        item = condition.get("item", "")
                        if item and item not in or_items:
                            or_items.append(item)
                    if or_items:
                        result["or_alternatives"].append(or_items)
                else:
                    item = state_repr.get("item", "")
                    if item and item not in result["guaranteed"]:
                        result["guaranteed"].append(item)

        return result
