"""
EffectMatcher - main effect-matcher class (Facade)

refactored from effect_matcher.py into an internal mixin composition.

The effect matcher is responsible for:
1. Extracting target effects from task descriptions (-> ExtractionMixin)
2. Finding skills that can produce the target effects (this file, orchestration layer)
3. Computing effect-match quality scores (this file)
4. Core effect-matching logic (-> MatchingMixin)

Usage:
    from skillnet.agents.planning import EffectMatcher

    matcher = EffectMatcher(
        skill_graph_manager=skill_graph_manager,
        llm=llm,
        logger=logger
    )

    target_effects = matcher.extract_target_effects(task, context)
    candidate_skills = matcher.find_skills_by_effects(target_effects, task)
"""

import logging
import re
from typing import Any, Dict, List, Optional

from skillnet.agents.skill_graph import should_skip_covered_skill

from ._utils import (
    normalize_operation,
    COUNT_PARAM_NAMES,
    SUPPORTED_OPERATIONS,
)
from ._extraction import ExtractionMixin
from ._matching import MatchingMixin

logger = logging.getLogger(__name__)


class EffectMatcher(ExtractionMixin, MatchingMixin):
    """
    Effect matcher

    Responsibilities:
    1. Extract target effects from task descriptions (ExtractionMixin)
    2. Find skills that can produce the target effects (this class)
    3. Compute effect-match quality scores (this class)
    4. Core effect-matching logic (MatchingMixin)
    """

    def __init__(
        self,
        skill_graph_manager,
        llm=None,
        custom_logger: Optional[logging.Logger] = None,
        use_llm_for_extraction: bool = True,
        domain_knowledge=None,
    ):
        """
        Initialize the effect matcher

        Args:
            skill_graph_manager: SkillGraphManager instance
            llm: LangChain LLM instance (optional)
            custom_logger: optional custom logger
            use_llm_for_extraction: whether to use the LLM to extract target effects
            domain_knowledge: DomainKnowledge instance (optional; provides domain-injected verb patterns)
        """
        self.skill_graph_manager = skill_graph_manager
        self.llm = llm
        self.logger = custom_logger or logger
        self.use_llm_for_extraction = use_llm_for_extraction
        self.domain_knowledge = domain_knowledge

        # Caches
        self._skill_match_quality: Dict[str, float] = {}
        self._skill_consistency_penalty: Dict[str, float] = {}

    # ========================================================================
    # Public read-only properties (replace direct private-state access)
    # ========================================================================

    @property
    def skill_match_quality(self) -> Dict[str, float]:
        """Match quality scores from the latest find_skills_by_effects call (read-only copy)"""
        return dict(self._skill_match_quality)

    @property
    def skill_consistency_penalty(self) -> Dict[str, float]:
        """Consistency penalties from the latest find_skills_by_effects call (read-only copy)"""
        return dict(self._skill_consistency_penalty)

    # ========================================================================
    # Skill lookup (orchestration layer)
    # ========================================================================

    def find_skills_by_effects(
        self,
        target_effects: List[Dict[str, Any]],
        task: str = "",
        current_state: Optional[Dict[str, Any]] = None,
        conflict_checker: Optional[callable] = None,
        composite_checker: Optional[callable] = None,
        material_checker: Optional[callable] = None,
    ) -> List[str]:
        """
        Find skills by target effects

        Args:
            target_effects: list of target effects
            task: current task description (used for semantic conflict detection)
            current_state: current state (used for inventory-aware selection)
            conflict_checker: semantic conflict checker (task, skill_name) -> (bool, str)
            composite_checker: composite-skill checker (skill_name, node) -> bool
            material_checker: material checker (node, inventory, effect) -> bool

        Returns:
            List[str]: list of skill names that can produce the target effects
        """
        candidate_skills = []

        # Clear quality scores and penalty records from the previous call
        self._skill_match_quality = {}
        self._skill_consistency_penalty = {}

        # Check whether the graph is empty
        if self.skill_graph_manager.skill_count == 0:
            self.logger.warning("[EffectMatcher] Skill graph is empty")
            return []

        # find skills that can produce supported operation types (extended with place/equip support)
        supported_effects = []
        for effect in target_effects:
            raw_operation = effect.get("operation", "add")
            # Normalize the operation type (e.g. ensure -> add)
            normalized_op = normalize_operation(raw_operation)
            if normalized_op in SUPPORTED_OPERATIONS:
                # Create a normalized copy of the effect
                normalized_effect = dict(effect)
                normalized_effect["operation"] = normalized_op
                normalized_effect["_original_operation"] = raw_operation  # Preserve original value for debugging
                supported_effects.append(normalized_effect)

        if not supported_effects:
            self.logger.warning("[EffectMatcher] No supported operation-type effects found")
            return []

        # Group log by operation type
        ops_summary = {}
        for e in supported_effects:
            op = e.get("operation")
            if op not in ops_summary:
                ops_summary[op] = []
            ops_summary[op].append(e.get("item", "unknown"))
        for op, items in ops_summary.items():
            self.logger.info(f"[EffectMatcher] Searching for skills with '{op}' operation: {items}")

        skipped_due_to_conflict = []

        for effect in supported_effects:
            for skill_name, node in self.skill_graph_manager.iter_skills():
                # Skip covered skills
                if should_skip_covered_skill(node):
                    # ──────────────────────────────────────────────────────
                    # Layer 2 fix (Phase C validation): when a wrapper would
                    # match the target effect, also check its covered_by
                    # general skill. This ensures the planner can find the
                    # general skill via the wrapper's specific effects when
                    # the general skill itself was created with empty/generic
                    # effects (the production bug). Defense in depth — even
                    # if Layer 1 has gaps in effect propagation, Layer 2
                    # makes the covered_by reachable.
                    # ──────────────────────────────────────────────────────
                    has_count_param_w = False
                    if hasattr(node, 'parameters') and node.parameters:
                        pn_lower = {p.lower() for p in node.parameters}
                        has_count_param_w = bool(pn_lower & COUNT_PARAM_NAMES)
                    eff_type = effect.get("type", "inventory")
                    should_ignore_count_w = has_count_param_w or eff_type == "nearby_block"
                    wrapper_matches = self.effect_matches(
                        effect, node.expected_effects, ignore_count=should_ignore_count_w
                    )
                    if wrapper_matches:
                        covered_by_name = getattr(node, 'covered_by', None)
                        if covered_by_name:
                            cover_node = self.skill_graph_manager.get_node(covered_by_name)
                            if (cover_node is not None
                                    and not getattr(cover_node, 'is_deprecated', False)
                                    and covered_by_name not in candidate_skills):
                                candidate_skills.append(covered_by_name)
                                # Borrow quality score / confidence from the
                                # wrapper match (since wrapper proxies general)
                                target_item = effect.get("item", "")
                                if target_item and hasattr(cover_node, 'code') and cover_node.code:
                                    quality = self.calculate_effect_match_quality(
                                        covered_by_name,
                                        cover_node.code,
                                        node.expected_effects,  # use wrapper's effects
                                        target_item,
                                        effect_confidence=1.0,
                                    )
                                    self._skill_match_quality[covered_by_name] = quality
                                self.logger.info(
                                    f"[EffectMatcher] Layer2: surfaced covered_by "
                                    f"'{covered_by_name}' via wrapper '{skill_name}' "
                                    f"matching effect '{effect.get('item')}'"
                                )
                    continue

                # Skip deprecated skills
                if hasattr(node, 'is_deprecated') and node.is_deprecated:
                    continue

                # Semantic conflict detection
                if task and conflict_checker:
                    has_conflict, conflict_reason = conflict_checker(task, skill_name)
                    if has_conflict:
                        if skill_name not in skipped_due_to_conflict:
                            skipped_due_to_conflict.append(skill_name)
                            self.logger.info(f"[EffectMatcher] Skipping conflicting skill: {skill_name} ({conflict_reason})")
                        continue

                # Check whether the skill has a count parameter
                has_count_param = False
                if hasattr(node, 'parameters') and node.parameters:
                    param_names_lower = {p.lower() for p in node.parameters}
                    has_count_param = bool(param_names_lower & COUNT_PARAM_NAMES)

                # For the nearby_block type (FIND task), always ignore count.
                # FIND tasks are essentially "navigate to target location" — count only specifies how many are expected, not skill availability.
                effect_type = effect.get("type", "inventory")
                should_ignore_count = has_count_param or effect_type == "nearby_block"

                # Check whether the skill's expected_effects match the target effect
                if self.effect_matches(effect, node.expected_effects, ignore_count=should_ignore_count):
                    if skill_name not in candidate_skills:
                        candidate_skills.append(skill_name)

                        # Get the matched effect's confidence (runtime verification result)
                        effect_confidence, confidence_level = self._get_matched_effect_confidence(
                            effect, node.expected_effects
                        )

                        # Compute the effects-code consistency quality score (includes confidence penalty)
                        target_item = effect.get("item", "")
                        if target_item and hasattr(node, 'code') and node.code:
                            quality = self.calculate_effect_match_quality(
                                skill_name, node.code, node.expected_effects, target_item,
                                effect_confidence=effect_confidence
                            )
                            self._skill_match_quality[skill_name] = quality

                            if quality < 0.5:
                                self._skill_consistency_penalty[skill_name] = 1.0 - quality
                                self.logger.warning(
                                    f"[EffectMatcher] {skill_name} may not produce {target_item} "
                                    f"(consistency={quality:.2f}, confidence={confidence_level})"
                                )

                        # Extra logging for low-confidence skills
                        if confidence_level in ("low", "uncertain"):
                            self.logger.info(
                                f"[EffectMatcher] Matched skill: {skill_name} "
                                f"(confidence={confidence_level}, value={effect_confidence:.2f})"
                            )
                        else:
                            self.logger.info(f"[EffectMatcher] Matched skill: {skill_name}")

        # Log statistics of skipped skills
        if skipped_due_to_conflict:
            self.logger.info(
                f"[EffectMatcher] Skipped {len(skipped_due_to_conflict)} conflicting skills"
            )

        # Inventory-Aware Skill Selection
        # Extract target effects with add operations for inventory-aware selection
        target_add_effects = [
            e for e in supported_effects
            if e.get("operation") == "add"
        ]

        if len(candidate_skills) > 1 and current_state and composite_checker and material_checker:
            current_inventory = current_state.get('inventory', {})
            current_inventory = {k.lower(): v for k, v in current_inventory.items()} if current_inventory else {}

            if current_inventory and target_add_effects:
                pure_skills = []
                composite_skills = []
                other_skills = []

                for skill_name in candidate_skills:
                    node = self.skill_graph_manager.get_node(skill_name)
                    is_composite = composite_checker(skill_name, node)
                    has_materials = material_checker(node, current_inventory, target_add_effects[0])

                    if not is_composite and has_materials:
                        pure_skills.append(skill_name)
                        self.logger.info(f"[EffectMatcher] Prioritizing pure-operation skill: {skill_name}")
                    elif is_composite:
                        composite_skills.append(skill_name)
                    else:
                        other_skills.append(skill_name)

                if pure_skills:
                    candidate_skills = pure_skills + other_skills + composite_skills
                    self.logger.info(f"[EffectMatcher] After inventory-aware sorting: {candidate_skills}")

        # Hard filter: if there are high-quality matches, drop low-quality matches
        if self._skill_match_quality and len(candidate_skills) > 1:
            max_quality = max(self._skill_match_quality.values())
            if max_quality > 0.7:
                threshold = max_quality - 0.2
                filtered = [
                    s for s in candidate_skills
                    if self._skill_match_quality.get(s, 1.0) >= threshold
                ]
                if filtered and len(filtered) < len(candidate_skills):
                    removed = set(candidate_skills) - set(filtered)
                    self.logger.info(
                        f"[EffectMatcher] Hard filter removed low-quality matches: {removed} (threshold={threshold:.2f})"
                    )
                    candidate_skills = filtered

        # Absolute minimum quality filter: remove extremely low-quality matches
        # regardless of whether high-quality matches exist. Prevents irrelevant skills
        # (e.g., craftSticks at 10% for iron tasks) from entering backward chaining.
        ABSOLUTE_MIN_QUALITY = 0.25
        if self._skill_match_quality and len(candidate_skills) > 1:
            filtered_by_min = [
                s for s in candidate_skills
                if self._skill_match_quality.get(s, 1.0) >= ABSOLUTE_MIN_QUALITY
            ]
            if filtered_by_min and len(filtered_by_min) < len(candidate_skills):
                removed = set(candidate_skills) - set(filtered_by_min)
                self.logger.info(
                    f"[EffectMatcher] Minimum-quality filter removed: {removed} (min_quality={ABSOLUTE_MIN_QUALITY})"
                )
                candidate_skills = filtered_by_min

        # Name-based fallback: when effect matching finds nothing,
        # try matching by skill name (handles wrapper skills with missing primary effects)
        if not candidate_skills and supported_effects:
            name_matched = self._find_skills_by_name(supported_effects, task, skipped_due_to_conflict)
            if name_matched:
                candidate_skills = name_matched

        if candidate_skills:
            self.logger.info(f"[EffectMatcher] Matched {len(candidate_skills)} skills: {candidate_skills}")

        return candidate_skills

    def _find_skills_by_name(
        self,
        target_effects: List[Dict[str, Any]],
        task: str = "",
        skip_list: List[str] = None,
    ) -> List[str]:
        """
        Name-based fallback when effect matching yields no candidates.

        Converts skill names from camelCase to snake_case and checks if
        the target item appears in the converted name.

        E.g., target_item='wooden_pickaxe' matches 'craftWoodenPickaxe'
        because 'craft_wooden_pickaxe' contains 'wooden_pickaxe'.
        """
        import re
        skip_set = set(skip_list or [])
        matched = []

        # Extract target items from effects
        target_items = []
        for effect in target_effects:
            item = effect.get("item", "")
            if item:
                target_items.append(item.lower())

        if not target_items:
            return []

        for skill_name, node in self.skill_graph_manager.iter_skills():
            if skill_name in skip_set:
                continue
            if should_skip_covered_skill(node):
                continue
            if hasattr(node, 'is_deprecated') and node.is_deprecated:
                continue

            # Convert camelCase to snake_case
            snake_name = re.sub(r'(?<!^)(?=[A-Z])', '_', skill_name).lower()

            for target_item in target_items:
                if target_item in snake_name:
                    matched.append(skill_name)
                    self.logger.info(
                        f"[EffectMatcher] Name-based fallback match: {skill_name} "
                        f"(snake: {snake_name}, target: {target_item})"
                    )
                    # Record quality as moderate (name-only match)
                    self._skill_match_quality[skill_name] = 0.5
                    break

        return matched

    # ========================================================================
    # Quality scoring
    # ========================================================================

    def calculate_effect_match_quality(
        self,
        skill_name: str,
        skill_code: str,
        skill_effects: List,
        target_item: str,
        effect_confidence: Optional[float] = None
    ) -> float:
        """
        Compute the match quality between a skill effect and a target item (0.0 ~ 1.0)

        Scoring dimensions:
        1. OR-condition count penalty: more conditions -> lower score
        2. Name relevance: whether the skill name contains keywords from the target item
        3. Code implementation check: whether the code produces the item
        4. Effect confidence: runtime verification confidence score
        """
        if not target_item:
            return 1.0

        target_lower = target_item.lower()

        # Dimension 1: OR-condition count penalty
        or_penalty = 0.0
        for effect in skill_effects:
            state_repr = getattr(effect, 'state_representation', None)
            if state_repr and isinstance(state_repr, dict):
                if state_repr.get("logic", "").upper() == "OR":
                    conditions = state_repr.get("conditions", [])
                    total_or_conditions = len(conditions)
                    if total_or_conditions > 5:
                        or_penalty = min(0.5, (total_or_conditions - 5) * 0.05)
                    break

        # Dimension 2: name relevance
        name_score = 0.0
        skill_words = set(re.findall(r'[a-z]+', skill_name.lower()))
        item_words = set(target_lower.replace('_', ' ').split())
        generic_words = {'ensure', 'craft', 'get', 'make', 'create', 'mine', 'obtain'}

        meaningful_skill_words = skill_words - generic_words
        meaningful_item_words = item_words - generic_words

        if meaningful_skill_words & meaningful_item_words:
            name_score = 0.5

        # Dimension 3: code implementation check
        code_score = 0.0
        if skill_code:
            try:
                from skillnet.agents.optimizer.validators import EffectsConsistencyValidator
                validator = EffectsConsistencyValidator()
                consistency_score = validator.quick_check(skill_code, target_item)
                if consistency_score >= 0.9:
                    code_score = 0.3
                elif consistency_score >= 0.6:
                    code_score = 0.15
            except Exception as e:
                self.logger.debug(f"[EffectMatcher] Consistency check failed: {e}")

        # Dimension 4: effect confidence penalty (runtime verification)
        confidence_penalty = 0.0
        if effect_confidence is not None and effect_confidence < 1.0:
            # Apply penalty to low confidence:
            # - confidence >= 0.8: no penalty
            # - confidence 0.5~0.8: light penalty 0~0.2
            # - confidence 0.3~0.5: moderate penalty 0.2~0.4
            # - confidence < 0.3: severe penalty 0.4~0.6
            if effect_confidence < 0.3:
                confidence_penalty = 0.6 - effect_confidence
            elif effect_confidence < 0.5:
                confidence_penalty = 0.4 - (effect_confidence - 0.3) * 0.5
            elif effect_confidence < 0.8:
                confidence_penalty = 0.2 - (effect_confidence - 0.5) * 0.67

        # Compute the final score
        base_score = 1.0 - or_penalty + name_score + code_score - confidence_penalty
        final_score = max(0.0, min(1.0, base_score))

        if final_score < 0.7 or confidence_penalty > 0.1:
            self.logger.debug(
                f"[EffectMatcher] {skill_name} quality score for {target_item}: {final_score:.2f} "
                f"(or_penalty={or_penalty:.2f}, name={name_score:.2f}, code={code_score:.2f}, "
                f"confidence_penalty={confidence_penalty:.2f})"
            )

        return final_score

