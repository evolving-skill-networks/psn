"""Skill Sequence Mixin - Planning paths, selecting skills, checking preconditions."""

from __future__ import annotations
import re
import math
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple

from skillnet.agents.planning import normalize_operation
from skillnet.agents.skill_graph import should_skip_covered_skill
from .._types import SkillCallContext

if TYPE_CHECKING:
    from ..graph_planner import GraphPlanner


class SkillSequenceMixin:
    """Skill Sequence Mixin - Planning paths, selecting skills, checking preconditions.

    Methods:
        _build_skill_sequence: Build skill execution sequence - recursive core, manages visited set
        _check_and_warn_semantic_compatibility: Check and warn about semantic compatibility
        _select_best_skill: Select best skill - uses Boltzmann policy
        _extract_param_values_from_task: Extract parameter values from task
        _condition_matches: Check if condition matches
        _check_preconditions: Check preconditions - uses precondition_checker
        _extract_generic_type: Extract generic type from item list
        _is_generic_item_type: Check if item is generic type
        _extract_conditions_from_state_repr: Extract conditions from state representation
        _update_expected_inventory: Update expected inventory
        _find_skills_for_precondition: Find skills that satisfy a precondition

    Warning:
        These methods have complex implicit state dependencies and recursive call chains.
        Not recommended for extraction to standalone modules.
    """

    def _build_skill_sequence(
        self,
        candidate_skills: List[str],
        current_state: Dict[str, Any],
        max_depth: int = 5,
        visited: Optional[set] = None,
        param_values: Dict[str, Any] = None,
        context_for_candidates: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[SkillCallContext]:
        """
        Build the skill execution sequence (taking preconditions into account); return a list of SkillCallContext with context.

        Args:
            candidate_skills: List of candidate skills
            current_state: Current state
            max_depth: Maximum recursion depth
            visited: Set of already-visited skills (prevents cycles)
            param_values: Parameter values (used for conditional precondition checks)
            context_for_candidates: Context mapping for candidate skills {skill_name: precondition_context}

        Returns:
            List[SkillCallContext]: Skill execution sequence (with context)
        """
        if max_depth <= 0:
            return []

        if not candidate_skills:
            return []

        if visited is None:
            visited = set()

        if context_for_candidates is None:
            context_for_candidates = {}

        # Improvement: select the best skill (considers success rate, execution count, etc.)
        selected_skill = self._select_best_skill(candidate_skills)

        # Prevent cyclic dependencies
        if selected_skill in visited:
            # If already visited, try other candidates
            remaining_skills = [s for s in candidate_skills if s != selected_skill]
            if remaining_skills:
                return self._build_skill_sequence(remaining_skills, current_state, max_depth, visited, param_values, context_for_candidates)
            else:
                return []

        node = self.skill_graph_manager.get_node(selected_skill)

        if not node:
            return []

        # Check semantic compatibility and emit warnings
        self._check_and_warn_semantic_compatibility(selected_skill, node)

        # Mark as visited
        visited.add(selected_skill)

        # Check preconditions (pass param_values to support conditional preconditions)
        missing_preconditions = self._check_preconditions(node.preconditions, current_state, param_values)

        if missing_preconditions:
            self.logger.info(f"\033[36m[Graph Planner] Skill '{selected_skill}' has {len(missing_preconditions)} unsatisfied preconditions:\033[0m")
            for i, precond in enumerate(missing_preconditions):
                desc = precond.description if hasattr(precond, 'description') else str(precond)
                self.logger.info(f"\033[36m[Graph Planner]   Precondition {i+1}: {desc[:100]}...\033[0m")

        if not missing_preconditions:
            # All preconditions satisfied; can execute directly
            # Get the context for the current skill from context_for_candidates (if any)
            ctx = context_for_candidates.get(selected_skill)
            return [SkillCallContext(skill_name=selected_skill, precondition_context=ctx)]

        # Preconditions are not satisfied; first execute other skills to satisfy them
        prerequisite_skills = []
        new_context_for_candidates = {}  # Build a new context mapping for recursion

        for precondition in missing_preconditions:
            # Find skills that can produce the required precondition (now returns tuples with context)
            prereq_skills_with_ctx = self._find_skills_for_precondition(precondition)

            # too many candidates indicates the description fallback is too loose;
            # filter using graph children
            MAX_REASONABLE_CANDIDATES = 5
            if len(prereq_skills_with_ctx) > MAX_REASONABLE_CANDIDATES:
                graph_children = set(self.skill_graph_manager.get_children(selected_skill) or [])
                if graph_children:
                    filtered = [(n, ctx) for n, ctx in prereq_skills_with_ctx if n in graph_children]
                    if filtered:
                        self.logger.info(
                            f"\033[36m[Graph Planner] Precondition candidates {len(prereq_skills_with_ctx)} → "
                            f"after graph-children filtering {len(filtered)}: {[f[0] for f in filtered]}\033[0m"
                        )
                        prereq_skills_with_ctx = filtered

            for skill_name, precond_ctx in prereq_skills_with_ctx:
                # Add caller information (the current selected_skill)
                precond_ctx["caller_skill"] = selected_skill
                # Record into the context mapping
                new_context_for_candidates[skill_name] = precond_ctx
                # Collect skill names for subsequent processing
                if skill_name not in prerequisite_skills:
                    prerequisite_skills.append(skill_name)
            desc = precondition.description if hasattr(precondition, 'description') else str(precondition)
            skill_names_only = [s[0] for s in prereq_skills_with_ctx]
            self.logger.info(f"\033[36m[Graph Planner]   Found {len(prereq_skills_with_ctx)} candidate skills for precondition '{desc[:50]}...': {skill_names_only}\033[0m")

        # De-duplicate and exclude already-visited skills
        prerequisite_skills = [s for s in dict.fromkeys(prerequisite_skills) if s not in visited]

        # Avoid cyclic dependencies: if prerequisite_skills contains selected_skill, remove it
        if selected_skill in prerequisite_skills:
            prerequisite_skills.remove(selected_skill)

        if not prerequisite_skills:
            # Check if all remaining missing preconditions are
            # self-descriptive (their only candidate was the current skill
            # itself, which got filtered out by the self-ref removal above).
            #
            # A self-descriptive precondition is one where the only skill
            # that can produce the required state IS the skill we are trying
            # to call. For example, ensureOakLogs has a precondition "Bot
            # must be able to mine oak_log blocks", and the only candidate
            # that satisfies this is ensureOakLogs itself — which is not a
            # dependency, it's the skill's own action.
            #
            # Such preconditions are implicitly satisfied by invoking the
            # current skill, so we treat them as satisfied and proceed with
            # a direct call. Without this handling, the planner would fail
            # on any skill whose only remaining preconditions are self-
            # descriptive, triggering an unnecessary fallback to the LLM
            # planner. Root cause exposed by Issue B's precondition-checker
            # fix (see commit c7a783d).
            # Distinguish "only self-ref candidate" from "no candidates at all":
            # self-descriptive requires that EVERY missing precondition has at
            # least one candidate AND all its candidates are self. Truly
            # unsatisfiable preconditions (zero candidates) fall through to the
            # backtrack path below, preserving legacy behavior.
            all_self_descriptive = bool(missing_preconditions)
            for precond in missing_preconditions:
                candidates_with_ctx = self._find_skills_for_precondition(precond)
                candidate_names = [name for name, _ in candidates_with_ctx]
                if not candidate_names:
                    # No candidates at all — truly unsatisfiable, not self-ref
                    all_self_descriptive = False
                    break
                non_self = [c for c in candidate_names if c != selected_skill]
                if non_self:
                    all_self_descriptive = False
                    break

            if all_self_descriptive:
                self.logger.info(
                    f"\033[32m[Graph Planner] Skill '{selected_skill}' remaining preconditions are all self-descriptive "
                    f"(only itself as a candidate); treated as implicitly satisfied, invoking directly\033[0m"
                )
                ctx = context_for_candidates.get(selected_skill)
                return [SkillCallContext(skill_name=selected_skill, precondition_context=ctx)]

            # Could not find skills to satisfy preconditions; return empty sequence (triggers fallback)
            self.logger.warning(f"\033[33m[Graph Planner] Could not find skills to satisfy preconditions; current state: inventory={current_state.get('inventory', {})}\033[0m")
            visited.remove(selected_skill)  # Backtrack
            return []

        # Recursively build the prerequisite skill sequence
        # Note: prerequisite skills' param_values may differ; passing None for now
        prereq_sequence = self._build_skill_sequence(
            prerequisite_skills,
            current_state,
            max_depth - 1,
            visited.copy(),  # Pass a copy of visited
            param_values=None,  # Parameter values for prerequisite skills must be inferred separately
            context_for_candidates=new_context_for_candidates  # Pass the context mapping
        )

        if not prereq_sequence:
            # Recursion failed; return empty sequence
            visited.remove(selected_skill)  # Backtrack
            return []

        # Compose the sequence: prerequisite skills + target skill
        # Get the context for the current skill
        current_ctx = context_for_candidates.get(selected_skill)
        current_skill_ctx = SkillCallContext(skill_name=selected_skill, precondition_context=current_ctx)

        # Check whether it is already in the sequence (based on skill_name)
        if selected_skill not in [ctx.skill_name for ctx in prereq_sequence]:
            return prereq_sequence + [current_skill_ctx]
        else:
            return prereq_sequence

    def _check_and_warn_semantic_compatibility(self, skill_name: str, node) -> None:
        """
        Check parameter semantics of the skill and emit warnings.

        When the skill has parameters with target_total semantics, emit a warning
        reminding the caller about parameter meaning.

        Args:
            skill_name: Skill name
            node: SkillNode object
        """
        if not node or not node.parameters:
            return

        # Check the semantics of quantity-related parameters
        quantity_param_names = ["count", "amount", "num", "quantity", "number", "total", "target"]

        for param_name, param_info in node.parameters.items():
            if param_name.lower() in quantity_param_names:
                semantic = param_info.get("semantic", "unknown")
                if semantic == "target_total":
                    self.logger.warning(
                        f"\033[33m[Semantic Warning] '{skill_name}' parameter '{param_name}' uses "
                        f"target_total semantics (ensure total count). Callers should pass the desired final total, not a delta.\033[0m"
                    )
                elif semantic == "unknown":
                    # Also emit a hint when semantics are unknown (optional)
                    self.logger.info(
                        f"\033[36m[Semantic] '{skill_name}' parameter '{param_name}' has unknown semantics; "
                        f"recommend inspecting its implementation to determine whether it uses target_total or delta semantics.\033[0m"
                    )

    def _select_best_skill(self, candidate_skills: List[str]) -> str:
        """
        Select the best skill from the candidates (uses a Boltzmann policy based on the value function).

        Uses a Boltzmann (softmax) policy:
        P(s | subgoal) = exp(βV(s)) / Σ exp(βV(s'))

        Where:
        - V(s) is the skill's value function
        - β is the temperature parameter (inverse temperature), default 8.0
        - Larger β: more biased toward skills with high value function (more deterministic)
        - Smaller β: more random, allowing exploration (more stochastic)

        Returns:
            str: Selected skill name
        """
        import random
        import math

        if not candidate_skills:
            return None

        if len(candidate_skills) == 1:
            return candidate_skills[0]

        # Get the value function for each skill
        value_functions = {}
        for skill_name in candidate_skills:
            node = self.skill_graph_manager.get_node(skill_name)
            if not node:
                continue

            # Get the skill's value function
            v_s = node.value_function
            value_functions[skill_name] = v_s

        if not value_functions:
            # If no value function can be obtained, return the first candidate
            self.logger.warning(f"\033[33m[Graph Planner] Unable to obtain the value function for any skill; returning the first candidate skill\033[0m")
            return candidate_skills[0]

        # [Step 1.5+] Apply effects-code consistency penalty
        if hasattr(self, '_skill_consistency_penalty') and self._skill_consistency_penalty:
            for skill_name, penalty in self._skill_consistency_penalty.items():
                if skill_name in value_functions:
                    original_v = value_functions[skill_name]
                    value_functions[skill_name] -= penalty * 0.5  # Penalty coefficient
                    self.logger.info(
                        f"\033[33m[Graph Planner] {skill_name} V(s) consistency penalty: "
                        f"{original_v:.3f} -> {value_functions[skill_name]:.3f} "
                        f"(penalty={penalty:.2f})\033[0m"
                    )

        # Compute the Boltzmann probability distribution
        # Use the log-sum-exp trick to avoid numerical overflow
        # First find the maximum value function
        max_v = max(value_functions.values())

        # Compute exp(β(V(s) - max_v)) and then normalize
        exp_values = {}
        for skill_name, v_s in value_functions.items():
            exp_values[skill_name] = math.exp(self.beta * (v_s - max_v))

        # Compute the normalization constant
        sum_exp = sum(exp_values.values())

        # Compute the probability distribution
        probabilities = {skill_name: exp_val / sum_exp for skill_name, exp_val in exp_values.items()}

        # Log the probability distribution (for debugging)
        self.logger.info(f"\033[36m[Graph Planner] Boltzmann-policy selection (β={self.beta}):\033[0m")
        for skill_name in sorted(probabilities.keys(), key=lambda x: probabilities[x], reverse=True):
            v_s = value_functions[skill_name]
            prob = probabilities[skill_name]
            self.logger.info(f"\033[36m[Graph Planner]   {skill_name}: V(s)={v_s:.3f}, P={prob:.3f}\033[0m")

        # Sample from the probability distribution
        selected_skill = random.choices(
            list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]

        self.logger.info(f"\033[36m[Graph Planner] Selected skill: {selected_skill} (V(s)={value_functions[selected_skill]:.3f}, P={probabilities[selected_skill]:.3f})\033[0m")

        return selected_skill

    def _extract_param_values_from_task(
        self,
        task: str,
        target_effects: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract parameter values from the task description and target effects.

        Used for conditional precondition checking: we need to know the parameter
        values associated with the current task so we only check preconditions
        applicable to those parameter values.

        Args:
            task: Task description, e.g. "mine 3 diamond_ore"
            target_effects: List of target effects

        Returns:
            Dict[str, Any]: Extracted parameter values, e.g. {"targetBlockNames": "diamond_ore", "count": 3}
        """
        import re
        param_values = {}
        task_lower = task.lower() if task else ""

        # 1. Extract item from target_effects
        if target_effects:
            for effect in target_effects:
                if isinstance(effect, dict):
                    item = effect.get("item")
                    if item:
                        # Infer the parameter name
                        if "_ore" in item:
                            param_values["targetBlockNames"] = item
                        elif "log" in item or "plank" in item:
                            param_values["targetBlockNames"] = item
                        else:
                            param_values["targetItem"] = item

                        count = effect.get("count")
                        if count:
                            param_values["count"] = count

        # 2. Extract from the task description
        # Patterns: mine X Y, craft X Y, collect X Y
        patterns = [
            # mine 3 coal_ore / mine coal_ore
            (r'mine\s+(\d+)?\s*(\w+)', ["count", "targetBlockNames"]),
            # craft 1 wooden_pickaxe / craft wooden_pickaxe
            (r'craft\s+(\d+)?\s*(\w+)', ["count", "targetItem"]),
            # smelt 3 iron_ingot
            (r'smelt\s+(\d+)?\s*(\w+)', ["count", "targetItem"]),
            # collect 5 oak_log
            (r'collect\s+(\d+)?\s*(\w+)', ["count", "targetItem"]),
        ]

        for pattern, param_names in patterns:
            match = re.search(pattern, task_lower)
            if match:
                groups = match.groups()
                for i, param_name in enumerate(param_names):
                    if i < len(groups) and groups[i]:
                        value = groups[i]
                        # Try converting to a number
                        if param_name == "count":
                            try:
                                value = int(value)
                            except ValueError:
                                pass
                        if param_name not in param_values:
                            param_values[param_name] = value
                break

        return param_values

    def _condition_matches(
        self,
        condition: Dict[str, Any],
        param_values: Dict[str, Any]
    ) -> bool:
        """
        Check whether a condition matches the parameter values (v5.0 delegates to planning.precondition_checker).

        Used for conditional preconditions/effects: only when parameter values
        match the condition does that precondition/effect apply.

        Args:
            condition: Condition expression, e.g. {"targetBlockNames": "diamond_ore"}
                      or {"targetBlockNames": ["diamond_ore", "gold_ore"]}
            param_values: Current parameter values, e.g. {"targetBlockNames": "diamond_ore", "count": 3}

        Returns:
            bool: Whether the condition matches
        """
        from skillnet.agents.planning.precondition_checker import condition_matches
        return condition_matches(condition, param_values)

    def _check_preconditions(
        self,
        preconditions: List,
        current_state: Dict[str, Any],
        param_values: Dict[str, Any] = None
    ) -> List:
        """
        Check whether preconditions are satisfied (v5.0 delegates to planning.precondition_checker).

        Prefer validation via state_representation; fall back to code/description parsing if absent.

        Supports conditional preconditions: if the precondition has a condition field,
        only check that precondition when param_values match the condition.

        Args:
            preconditions: List of preconditions
            current_state: Current state (includes inventory, etc.)
            param_values: Current parameter values (for conditional preconditions)

        Returns:
            List: List of unsatisfied preconditions
        """
        return self.precondition_checker.check_preconditions(
            preconditions=preconditions,
            current_state=current_state,
            param_values=param_values,
            condition_matcher=self._condition_matches,
        )

    def _extract_generic_type(self, item_list: List[str]) -> str:
        """
        Extract a generic type name from a list of items (v5.0 delegates to planning.precondition_checker).

        For example:
        - ["oak_planks", "birch_planks", "spruce_planks"] → "planks"
        - ["coal", "charcoal", "oak_log", "oak_planks"] → "fuel"
        - ["oak_log", "birch_log", "spruce_log"] → "log"

        Args:
            item_list: List of item names

        Returns:
            Generic type name
        """
        from skillnet.agents.planning.precondition_checker import extract_generic_type
        return extract_generic_type(item_list)

    def _is_generic_item_type(self, item: str) -> bool:
        """
        Check whether an item name is a generic type (v5.0 delegates to planning.precondition_checker).

        For example:
        - "planks" → True (generic planks, not specifying oak/birch/etc.)
        - "oak_planks" → False (specifically oak planks)
        - "log" → True (generic log)
        - "fuel" → True (generic fuel)

        Args:
            item: Item name

        Returns:
            Whether it is a generic type
        """
        from skillnet.agents.planning.precondition_checker import is_generic_item_type
        return is_generic_item_type(item)

    def _extract_conditions_from_state_repr(self, state_repr) -> List[Dict]:
        """
        Extract all conditions from a state_representation.

        Supported formats:
        1. Legacy format (list): [{"type": "inventory", "item": "oak_log", ...}, ...]
        2. New format: {"logic": "OR", "conditions": [...]}
        3. Single condition: {"type": "inventory", "item": "oak_log", ...}

        Args:
            state_repr: state_representation (dict, list, or nested structure)

        Returns:
            List of conditions
        """
        if isinstance(state_repr, list):
            return state_repr
        elif isinstance(state_repr, dict):
            if 'logic' in state_repr and 'conditions' in state_repr:
                # Recursively extract all conditions
                conditions = []
                for cond in state_repr.get('conditions', []):
                    conditions.extend(self._extract_conditions_from_state_repr(cond))
                return conditions
            else:
                # Single condition
                return [state_repr]
        return []

    def _update_expected_inventory(self, expected_inventory: Dict[str, int], node):
        """
        Update the expected inventory (based on the skill's expected_effects).

        Args:
            expected_inventory: Expected inventory dict (modified in place)
            node: Skill node
        """
        if not node or not hasattr(node, 'expected_effects'):
            return

        for effect in node.expected_effects:
            state_repr = getattr(effect, 'state_representation', None)
            if not state_repr:
                continue

            # Parse the state_representation
            conditions = self._extract_conditions_from_state_repr(state_repr)

            for cond in conditions:
                # use normalize_operation to ensure "ensure" → "add" mapping
                cond_operation = normalize_operation(cond.get('operation', 'add'))
                if cond.get('type') == 'inventory' and cond_operation == 'add':
                    item = cond.get('item')
                    count = cond.get('count', 1)
                    # Ensure count is an integer (count may be None, str, etc.)
                    if count is None:
                        count = 1
                    elif isinstance(count, str):
                        try:
                            count = int(count)
                        except ValueError:
                            count = 1
                    elif not isinstance(count, (int, float)):
                        count = 1

                    # Handle list-typed item
                    if isinstance(item, list):
                        generic_type = self._extract_generic_type(item)
                        expected_inventory[generic_type] = expected_inventory.get(generic_type, 0) + count
                    elif isinstance(item, str):
                        expected_inventory[item] = expected_inventory.get(item, 0) + count

    def _find_skills_for_precondition(self, precondition) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find skills that can produce the required precondition (v5.0 delegates to planning.precondition_checker).
        Supports OR-logic preconditions.

        Prefer exact matching via state_representation; fall back to description matching if absent.

        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of (skill name, precondition context) tuples
        """
        return self.precondition_checker.find_skills_for_precondition(
            precondition=precondition,
            skip_checker=should_skip_covered_skill,
        )
