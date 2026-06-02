"""GraphPlanner - Graph-based planner using skill graph for path planning."""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import json
import os
import re
from langchain.schema import HumanMessage, SystemMessage

import skillnet.utils as U
from skillnet.utils.log_utils import LoggerManager
from skillnet.agents.skill_graph import should_skip_covered_skill
from skillnet.agents.constants.task_semantics import (
    detect_task_semantic,
    TaskSemanticType,
    TaskWithSemantic,
)
from skillnet.agents.planning import (
    EffectMatcher,
    PreconditionChecker,
)
from skillnet.agents.planning.inference import (
    ParameterInferenceEngine,
    InferenceContext,
)

from ._types import BasePlanner, SkillCallContext, PlanningResult
from .mixins import (
    RuleLearningMixin,
    EffectMatchingMixin,
    SkillSequenceMixin,
    ParameterResolutionMixin,
    CodeGenerationMixin,
)


class GraphPlanner(
    BasePlanner,
    RuleLearningMixin,
    EffectMatchingMixin,
    SkillSequenceMixin,
    ParameterResolutionMixin,
    CodeGenerationMixin,
):
    """Graph-based Planner: performs path planning using the skill graph.

    Progressive strategy:
    - Stage 1 (0-5 skills): primarily uses LLM; Graph Planner acts as a helper
    - Stage 2 (6-15 skills): hybrid mode; Graph Planner is preferred but fallback is allowed
    - Stage 3 (16+ skills): Graph Planner-led; LLM serves as a supplement
    """

    def __init__(self, skill_graph_manager, action_agent=None, **kwargs):
        """
        Initialize the Graph Planner

        Args:
            skill_graph_manager: SkillGraphManager instance
            action_agent: ActionAgent instance (optional, used for fallback to LLM)
            beta: Boltzmann policy temperature (inverse temperature), defaults to 8.0
                 - Larger beta: more biased toward high value-function skills (more deterministic)
                 - Smaller beta: more random, encourages exploration (more stochastic)
        """
        super().__init__(**kwargs)
        self.skill_graph_manager = skill_graph_manager
        self.action_agent = action_agent
        self.max_planning_depth = kwargs.get("max_planning_depth", 5)
        self.enable_fallback = kwargs.get("enable_fallback", True)
        self.use_llm_for_extraction = kwargs.get("use_llm_for_extraction", True)

        # Boltzmann policy parameter
        self.beta = kwargs.get("beta", 8.0)  # default 8.0

        # Parameter-mapping rule-learning system
        self.use_llm_for_parameters = kwargs.get("use_llm_for_parameters", True)  # enable LLM fallback by default
        self.learn_parameter_rules = kwargs.get("learn_parameter_rules", True)  # whether to learn rules
        self.ckpt_dir = kwargs.get("ckpt_dir", skill_graph_manager.ckpt_dir if hasattr(skill_graph_manager, 'ckpt_dir') else "ckpt")
        self.parameter_rules_file = f"{self.ckpt_dir}/skill_graph/parameter_rules.json"

        # Initialize the logger (initialized early so later code can use it)
        self.logger = LoggerManager.get_logger("GraphPlanner", self.ckpt_dir)

        # If action_agent has an LLM, use it; otherwise try to get one from skill_graph_manager
        if action_agent and hasattr(action_agent, 'llm'):
            self.llm = action_agent.llm
        elif hasattr(skill_graph_manager, 'llm'):
            self.llm = skill_graph_manager.llm
        else:
            self.llm = None
            self.logger.warning(f"\033[33m[Graph Planner] Warning: no LLM found; will use rules to extract target effects\033[0m")

        # LLM parameter-extraction cache (avoids repeated calls)
        self._llm_param_extraction_cache = {}

        # effect matcher (lazy-initialized)
        self.__effect_matcher: Optional[EffectMatcher] = None

        # precondition checker (lazy-initialized)
        self.__precondition_checker: Optional[PreconditionChecker] = None

        # parameter-inference engine (lazy-initialized) - unified parameter-inference interface
        self.__param_inference_engine: Optional[ParameterInferenceEngine] = None

        # Effect-match quality cache (updated by EffectMatcher)
        self._skill_match_quality: Dict[str, float] = {}
        self._skill_consistency_penalty: Dict[str, float] = {}

        # Load learned rules (managed by ParameterInferenceEngine)
        self.parameter_rules = self._load_parameter_rules()

        # Domain knowledge (set via set_domain_knowledge for from_domain path)
        self._domain_knowledge = None

        # Current task
        self._current_task = None

        # Load parameter-passing error history
        self._load_parameter_error_history()

    # ========== effect-matcher property ==========

    @property
    def effect_matcher(self) -> EffectMatcher:
        """Get the effect matcher (lazy-initialized)"""
        if self.__effect_matcher is None:
            self.__effect_matcher = EffectMatcher(
                skill_graph_manager=self.skill_graph_manager,
                llm=self.llm,
                custom_logger=self.logger,
                use_llm_for_extraction=self.use_llm_for_extraction,
                domain_knowledge=getattr(self, '_domain_knowledge', None),
            )
        return self.__effect_matcher

    # ========== precondition-checker property ==========

    @property
    def precondition_checker(self) -> PreconditionChecker:
        """Get the precondition checker (lazy-initialized)"""
        if self.__precondition_checker is None:
            self.__precondition_checker = PreconditionChecker(
                skill_graph_manager=self.skill_graph_manager,
                effect_matcher=self.effect_matcher,
                custom_logger=self.logger,
            )
        return self.__precondition_checker

    # ========== parameter-inference-engine property ==========

    @property
    def param_inference_engine(self) -> ParameterInferenceEngine:
        """Get the parameter-inference engine (lazy-initialized)

        unified parameter-inference interface, supporting:
        - Multi-dimensional semantic inference (quantity_semantic, direction)
        - Strategy-chain fallback mechanism
        - Rule learning and caching
        """
        if self.__param_inference_engine is None:
            self.__param_inference_engine = ParameterInferenceEngine(
                skill_graph_manager=self.skill_graph_manager,
                llm=self.llm,
                ckpt_dir=self.ckpt_dir,
                custom_logger=self.logger,
            )
        return self.__param_inference_engine

    # ========== Parameter-passing error-learning mechanism ==========

    def _load_parameter_error_history(self):
        """Load the parameter-error history (handles corrupted files)"""
        error_file = os.path.join(self.ckpt_dir, "skill_graph", "parameter_errors.json")
        if os.path.exists(error_file):
            try:
                with open(error_file, 'r') as f:
                    self._parameter_error_history = json.load(f)
                self.logger.info(f"[Graph Planner] Loaded {len(self._parameter_error_history)} parameter-error history entries")
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"[Graph Planner] Failed to load parameter-error history (resetting): {e}")
                self._parameter_error_history = []
                # Back up the corrupted file
                try:
                    os.rename(error_file, f"{error_file}.corrupted")
                except:
                    pass
        else:
            self._parameter_error_history = []

    def _has_parameter_error_history(self, skill_name: str, param_name: str) -> bool:
        """Check whether there is an error history for this parameter"""
        if not hasattr(self, '_parameter_error_history'):
            return False

        for error in self._parameter_error_history:
            if error.get('skill_name') == skill_name and error.get('param_name') == param_name:
                return True
        return False

    def _is_composite_skill(self, skill_name: str, skill_node) -> bool:
        """
        Determine whether the skill is a composite skill (contains multiple operation types).

        For example, mineCopperOre contains mine + smelt, so it is a composite skill;
        smeltRawCopper contains only smelt, so it is a pure-operation skill.

        Args:
            skill_name: skill name
            skill_node: skill node

        Returns:
            bool: whether the skill is composite
        """
        code = skill_node.code.lower() if hasattr(skill_node, 'code') and skill_node.code else ""

        # Detect the operation types present in the code
        operations = []
        if any(kw in code for kw in ['mineblock', 'findblock', 'dig', 'collectblock']):
            operations.append('mine')
        if any(kw in code for kw in ['smeltitem', 'furnace.open', 'blast_furnace', 'smelt']):
            operations.append('smelt')
        if any(kw in code for kw in ['craftitem', 'recipe', 'craft(']):
            operations.append('craft')

        # If multiple operation types are present, this is a composite skill
        return len(operations) > 1

    def _player_has_required_materials(
        self,
        skill_node,
        current_inventory: Dict[str, int],
        target_effect: Dict[str, Any]
    ) -> bool:
        """
        Check whether the player already has the input materials required by the skill.

        Inferred by analyzing the skill's preconditions.

        Args:
            skill_node: skill node
            current_inventory: current player inventory
            target_effect: target effect (used to infer required materials)

        Returns:
            bool: whether the player has the required materials
        """
        if not skill_node.preconditions:
            return False

        # Handle the case where target_item may be a list
        target_item_raw = target_effect.get('item', '')
        if isinstance(target_item_raw, list):
            target_item = target_item_raw[0].lower() if target_item_raw and isinstance(target_item_raw[0], str) else ''
        elif isinstance(target_item_raw, str):
            target_item = target_item_raw.lower()
        else:
            target_item = ''
        target_count = target_effect.get('count', 1)
        if not isinstance(target_count, (int, float)) or target_count is None:
            target_count = 1

        # If the task is smelt, check for raw materials
        if 'smelt' in target_item or target_item.endswith('_ingot'):
            # Infer the raw material name
            raw_material = None
            if target_item.endswith('_ingot'):
                # copper_ingot -> raw_copper
                base = target_item.replace('_ingot', '')
                raw_material = f"raw_{base}"
            elif target_item.startswith('raw_'):
                raw_material = target_item

            if raw_material and raw_material in current_inventory:
                if current_inventory[raw_material] >= target_count:
                    return True

        # Check whether any of the skill's preconditions has an inventory requirement that is satisfied
        for precond in skill_node.preconditions:
            if hasattr(precond, 'state_representation'):
                state_repr = precond.state_representation
                # Handle list-form state_representation
                if isinstance(state_repr, list):
                    for sr in state_repr:
                        if self._check_inventory_precondition(sr, current_inventory):
                            return True
                elif isinstance(state_repr, dict):
                    if self._check_inventory_precondition(state_repr, current_inventory):
                        return True

        return False

    def _check_inventory_precondition(self, state_repr: Dict, current_inventory: Dict[str, int]) -> bool:
        """Check whether a single state_representation is satisfied"""
        if not isinstance(state_repr, dict):
            return False

        if state_repr.get('type') == 'inventory':
            item = state_repr.get('item', '')
            if isinstance(item, list):
                # OR logic: any one being satisfied is enough
                for i in item:
                    if i.lower() in current_inventory and current_inventory[i.lower()] >= 1:
                        return True
            elif isinstance(item, str) and item.lower() in current_inventory:
                count = state_repr.get('count', 1)
                if not isinstance(count, (int, float)) or count is None:
                    count = 1
                if current_inventory[item.lower()] >= count:
                    return True

        return False

    def _get_planning_stage(self) -> str:
        """
        Determine the planning stage based on the number of skills in the graph

        Returns:
            str: "early", "middle", or "mature"
        """
        skill_count = self.skill_graph_manager.skill_count

        if skill_count < 5:
            return "early"  # Early: primarily uses LLM
        elif skill_count < 16:
            return "middle"  # Mid: hybrid mode
        else:
            return "mature"  # Mature: Graph Planner-led

    def set_domain_knowledge(self, knowledge):
        """Inject domain-specific knowledge for code generation."""
        self._domain_knowledge = knowledge

    def plan(
        self,
        task: TaskWithSemantic,
        context: str,
        current_state: Dict[str, Any],
        available_skills: List[str],
        skill_metadata: Optional[Dict[str, Any]] = None,
        previous_code: str = "",
        critique: str = "",
    ) -> PlanningResult:
        """
        Perform path planning using the graph

        Strategy:
        1. Analyze the task goal and extract required effects
        2. Look up skills in the graph that can produce those effects
        3. Check preconditions; recursively find skills that satisfy them if not met
        4. Build the skill execution sequence
        5. Generate the composition code
        """
        # Extract the task string and semantic info
        task_str = task.task
        task_semantic = task.semantic_type  # use the passed-in semantic directly, no re-inference

        try:
            # even if the graph is empty, still log and attempt backward chaining
            # (subsequent steps will naturally fall back to the LLM if no suitable skill is found)
            skill_count = self.skill_graph_manager.skill_count
            if skill_count == 0:
                self.logger.info(f"\033[36m[Graph Planner] Skill graph is empty, but still attempting backward chaining (will naturally fall back)\033[0m")

            stage = self._get_planning_stage()
            self.logger.info(f"\033[36m[Graph Planner] Starting backward chaining ({stage} stage, {skill_count} skills)\033[0m")

            # Step 1: Analyze the task and extract target effects
            target_effects = self._extract_target_effects(task_str, context)

            if not target_effects:
                # If target effects cannot be extracted, fall back to LLM
                if self.enable_fallback and self.action_agent:
                    self.logger.warning(f"\033[33m[Graph Planner] Unable to extract target effects; falling back to LLM planner\033[0m")
                    return PlanningResult(
                        success=False,
                        plan_type="graph",
                        error="Cannot extract target effects, fallback to LLM",
                        metadata={"fallback": True, "reason": "no_target_effects"}
                    )
                else:
                    return PlanningResult(
                        success=False,
                        plan_type="graph",
                        error="Cannot extract target effects"
                    )

            # Step 2: Find skills that can produce the target effects
            # Pass in the current task for semantic-conflict detection
            self._current_task = task_str
            self._current_task_semantic = task_semantic  # P1: save semantic info
            # Save the current state for inventory-aware skill selection
            self._current_state = current_state
            candidate_skills = self._find_skills_by_effects(target_effects, task=task_str)

            self.logger.info(f"\033[36m[Graph Planner] Found {len(candidate_skills)} candidate skills: {candidate_skills}\033[0m")

            # Check whether any target effects are not covered
            uncovered_effects = self._find_uncovered_effects(target_effects, candidate_skills)
            if uncovered_effects:
                self.logger.warning(f"\033[33m[Graph Planner] The following target effects are not covered by any skill: {uncovered_effects}\033[0m")
                if self.enable_fallback and self.action_agent:
                    self.logger.warning(f"\033[33m[Graph Planner] Falling back to LLM planner (some effects uncovered)\033[0m")
                    return PlanningResult(
                        success=False,
                        plan_type="graph",
                        error=f"Some target effects not covered by any skill: {[e.get('item') for e in uncovered_effects]}",
                        metadata={"fallback": True, "target_effects": target_effects, "uncovered_effects": uncovered_effects}
                    )

            if not candidate_skills:
                # If no suitable skills are found, fall back to LLM
                self.logger.warning(f"\033[33m[Graph Planner] No skills found that directly produce the target effect\033[0m")
                self.logger.warning(f"\033[33m[Graph Planner] Target effects: {target_effects}\033[0m")
                # Filter out task_specific and deprecated skills; only show formal usable skills
                available_skills = [
                    name for name, node in self.skill_graph_manager.iter_skills(include_task_specific=True)
                    if not getattr(node, 'is_task_specific', False) and not node.is_deprecated
                ]
                self.logger.warning(f"\033[33m[Graph Planner] Available skills ({len(available_skills)}): {available_skills}\033[0m")

                if self.enable_fallback and self.action_agent:
                    self.logger.warning(f"\033[33m[Graph Planner] Falling back to LLM planner\033[0m")
                    return PlanningResult(
                        success=False,
                        plan_type="graph",
                        error="No suitable skills found, fallback to LLM",
                        metadata={"fallback": True, "target_effects": target_effects}
                    )
                else:
                    return PlanningResult(
                        success=False,
                        plan_type="graph",
                        error="No suitable skills found and fallback disabled"
                    )

            # Step 3: Extract parameter values from the task (used for conditioning preconditions)
            param_values = self._extract_param_values_from_task(task_str, target_effects)
            if param_values:
                self.logger.info(f"\033[36m[Graph Planner] Parameter values extracted from task: {param_values}\033[0m")

            # Step 4: Build the skill execution sequence (taking preconditions into account)
            # Improvement: support composite tasks with multiple target effects
            if len(target_effects) > 1:
                # Multi-goal task: use multi-goal sequence construction
                self.logger.info(f"\033[36m[Graph Planner] Detected multi-goal task ({len(target_effects)} effects); using multi-goal sequence construction\033[0m")
                skill_sequence = self._build_multi_goal_sequence(
                    target_effects,
                    candidate_skills,
                    current_state,
                    max_depth=self.max_planning_depth,
                    param_values=param_values
                )
            else:
                # Single-goal task: use the original logic
                skill_sequence = self._build_skill_sequence(
                    candidate_skills,
                    current_state,
                    max_depth=self.max_planning_depth,
                    param_values=param_values,
                    context_for_candidates=None  # top-level call, no external context
                )

            if not skill_sequence:
                if self.enable_fallback and self.action_agent:
                    self.logger.warning(f"\033[33m[Graph Planner] Failed to build skill sequence; falling back to LLM planner\033[0m")
                    return PlanningResult(
                        success=False,
                        plan_type="graph",
                        error="Cannot build skill sequence, fallback to LLM",
                        metadata={"fallback": True}
                    )
                else:
                    return PlanningResult(
                        success=False,
                        plan_type="graph",
                        error="Cannot build skill sequence"
                    )

            # Step 4: Generate the composition code
            code = self._generate_composition_code(
                skill_sequence, skill_metadata, target_effects,
                task=task_str
            )

            # Step 5: Generate the plan description
            plan_description = self._generate_plan_description(skill_sequence, target_effects)

            stage = self._get_planning_stage()
            return PlanningResult(
                success=True,
                plan_type="graph",
                code=code,
                plan=plan_description,
                skill_sequence=skill_sequence,
                metadata={
                    "target_effects": target_effects,
                    "candidate_skills": candidate_skills,
                    "stage": stage,
                    "skill_count": self.skill_graph_manager.skill_count,
                }
            )

        except Exception as e:
            import traceback
            error_msg = f"Graph planning failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(f"\033[31m[Graph Planner] {error_msg}\033[0m")

            if self.enable_fallback and self.action_agent:
                self.logger.warning(f"\033[33m[Graph Planner] Falling back to LLM planner\033[0m")
                return PlanningResult(
                    success=False,
                    plan_type="graph",
                    error=error_msg,
                    metadata={"fallback": True, "exception": str(e)}
                )
            else:
                return PlanningResult(
                    success=False,
                    plan_type="graph",
                    error=error_msg
                )
