"""
PSNCurriculumAgent - Planning-aware, Symbolic, Neurally-guided Curriculum Agent

This agent extends the original CurriculumAgent with:
- Resource planning and threshold monitoring
- Task feasibility checking before proposal
- Long-term goal tracking with milestones
- Enhanced LLM prompting with resource context

Method responsibilities are distributed across 5 mixins:
- TaskValidationMixin: Task format validation, environment checks, special cases
- PromptBuildingMixin: LLM prompt construction and context rendering
- AdaptiveLearningMixin: Adaptive learning path management
- MilestoneManagementMixin: Milestone tracking and failure budgets
- SkillLearningMixin: Skill lookup and cache management
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple, Any

# Import CurriculumAgent base class from the .py file, not the package directory
# (Python prefers curriculum/ over curriculum.py, so we use importlib)
import importlib.util
_curriculum_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "curriculum.py")
_spec = importlib.util.spec_from_file_location("skillnet.agents.curriculum_base", _curriculum_path)
_curriculum_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_curriculum_module)
CurriculumAgent = _curriculum_module.CurriculumAgent

from skillnet.agents.psn_curriculum.resource_tracker import ResourceTracker
from skillnet.agents.psn_curriculum.goal_planner import GoalPlanner
from skillnet.agents.psn_curriculum.decomposition_filter import DecompositionFilter
from skillnet.agents.psn_curriculum.task_decomposer import TaskDecomposer
from skillnet.agents.psn_curriculum.skill_gap_analyzer import SkillGapAnalyzer
from skillnet.agents.psn_curriculum.adaptive_planner import AdaptiveLearningPlanner
from skillnet.agents.constants.task_semantics import TaskWithSemantic, detect_task_semantic
from skillnet.agents.psn_curriculum.mixins import (
    TaskValidationMixin,
    PromptBuildingMixin,
    AdaptiveLearningMixin,
    MilestoneManagementMixin,
    SkillLearningMixin,
)


class PSNCurriculumAgent(
    TaskValidationMixin,
    PromptBuildingMixin,
    AdaptiveLearningMixin,
    MilestoneManagementMixin,
    SkillLearningMixin,
    CurriculumAgent,
):
    """
    Planning-aware, Symbolic, Neurally-guided Curriculum Agent.

    Extends CurriculumAgent with resource planning capabilities:
    - Checks resource thresholds before proposing tasks
    - Verifies task feasibility using symbolic knowledge
    - Tracks long-term goals and milestones
    - Provides enhanced context to LLM for better decisions
    - Skill learning state tracking for progressive task selection
    """

    # Mapping from task types to skill names for learning state tracking
    # These are the core skills that should be learned progressively
    TASK_TO_SKILL_MAPPING = {
        "craft planks": ["craftPlanks", "ensurePlanks"],
        "craft sticks": ["craftSticks", "ensureSticks"],
        "craft crafting_table": ["craftCraftingTable", "ensureCraftingTable", "setupCraftingTable"],
        # Include generic pickaxe skill names (Graph Planner may select these)
        "craft wooden_pickaxe": ["craftWoodenPickaxe", "ensureWoodenPickaxe", "craftPickaxe", "ensurePickaxe"],
        "craft wooden_axe": ["craftWoodenAxe", "ensureWoodenAxe", "craftAxe", "ensureAxe"],
        "craft stone_pickaxe": ["craftStonePickaxe", "ensureStonePickaxe", "craftPickaxe", "ensurePickaxe"],
        "craft stone_axe": ["craftStoneAxe", "ensureStoneAxe", "craftAxe", "ensureAxe"],
        "craft furnace": ["craftFurnace", "ensureFurnace"],
        "craft iron_pickaxe": ["craftIronPickaxe", "ensureIronPickaxe", "craftPickaxe", "ensurePickaxe"],
        "mine logs": ["mineLogs", "mineWoodLogs", "collectLogs", "ensureLogs"],
        "mine cobblestone": ["mineCobblestone", "collectCobblestone", "ensureCobble", "ensureCobblestone"],
    }

    def __init__(
        self,
        model_name="gpt-5-mini",
        temperature=0,
        qa_model_name="gpt-5-mini",
        qa_temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        mode="auto",
        warm_up=None,
        core_inventory_items: str | None = None,
        # PSN-specific parameters
        resource_config_path: Optional[str] = None,
        # Multi-backend LLM support
        openai_api_base=None,
        openai_api_key=None,
        qa_openai_api_base=None,
        qa_openai_api_key=None,
        # Failure threshold tuning
        milestone_task_failure_threshold: int = 5,
    ):
        """
        Initialize PSNCurriculumAgent.

        Args:
            model_name: LLM model for task proposal
            temperature: LLM temperature
            qa_model_name: LLM model for Q&A
            qa_temperature: Q&A LLM temperature
            request_timout: API request timeout
            ckpt_dir: Checkpoint directory
            resume: Whether to resume from checkpoint
            mode: "auto" or "manual"
            warm_up: Warm-up configuration
            core_inventory_items: Regex for core inventory items
            resource_config_path: Path to resource threshold config
            openai_api_base: Custom API base URL for vLLM or compatible endpoints
            openai_api_key: Custom API key
            qa_openai_api_base: Custom API base URL for Q&A LLM
            qa_openai_api_key: Custom API key for Q&A LLM
            milestone_task_failure_threshold: Max consecutive failures before
                filtering a milestone task. Default 5 (raised from 3 to give
                weaker models more optimizer iterations).
        """
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            qa_model_name=qa_model_name,
            qa_temperature=qa_temperature,
            request_timout=request_timout,
            ckpt_dir=ckpt_dir,
            resume=resume,
            mode=mode,
            warm_up=warm_up,
            core_inventory_items=core_inventory_items,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            qa_openai_api_base=qa_openai_api_base,
            qa_openai_api_key=qa_openai_api_key,
        )

        # Knowledge base is wired in by PSNAgent.from_domain() via
        # set_domain_knowledge() → get_symbolic_knowledge_base().
        self.knowledge_base = None

        self.resource_tracker = ResourceTracker(
            config_path=resource_config_path or os.path.join(ckpt_dir, "psn_curriculum", "resource_config.json")
        )

        self.goal_planner = GoalPlanner(
            knowledge_base=self.knowledge_base,
            resource_tracker=self.resource_tracker,
            ckpt_dir=ckpt_dir
        )

        # Domain-owned prompt: set by set_domain_knowledge()
        self.psn_system_prompt = ""

        # domain knowledge for prompt injection
        self._domain_knowledge = None

        # Skill manager reference (will be set by PSN after initialization)
        self.skill_manager = None

        # Cache of learned skills to avoid repeated lookups
        self._learned_skills_cache: Dict[str, bool] = {}

        # Adaptive Learning components (initialized in set_skill_manager)
        self._failed_tasks_history: set = set()
        self.decomposition_filter: Optional[DecompositionFilter] = None
        self.task_decomposer: Optional[TaskDecomposer] = None
        self.skill_gap_analyzer: Optional[SkillGapAnalyzer] = None
        self.adaptive_planner: Optional[AdaptiveLearningPlanner] = None

        # Configuration for adaptive learning
        self.enable_adaptive_learning = True  # Can be disabled via config

        # Milestone failure threshold (configurable to support weaker models)
        self.milestone_task_failure_threshold = milestone_task_failure_threshold

        # Exploration mode cooldown recovery state
        self._exploration_mode_count = 0    # Consecutive exploration mode entries
        self._milestone_retry_budget = 2    # Max resets before permanent give-up

        if self.knowledge_base is not None:
            recipe_count = len(getattr(self.knowledge_base, 'recipes', {}))
            print(f"\033[35m[PSN Curriculum] Initialized with {recipe_count} recipes\033[0m")
        else:
            print(f"\033[35m[PSN Curriculum] Initialized (knowledge base pending from_domain)\033[0m")

    def set_domain_knowledge(self, knowledge):
        """Set domain knowledge for domain-aware prompt loading.

        Called by PSNAgent.from_domain() after construction.
        Overrides the curriculum prompt if the domain provides one.
        Also injects the symbolic knowledge base (recipes, dependencies)
        if the domain provides one — removing the need for a direct import
        of MinecraftKnowledgeBase.
        """
        self._domain_knowledge = knowledge
        if knowledge:
            template = knowledge.get_curriculum_prompt_template()
            if template:
                self.psn_system_prompt = template
            # Inject symbolic knowledge base from domain
            symbolic_kb = knowledge.get_symbolic_knowledge_base()
            if symbolic_kb is not None:
                self.knowledge_base = symbolic_kb
                # Re-wire GoalPlanner if it was already created
                if hasattr(self, 'goal_planner') and self.goal_planner is not None:
                    self.goal_planner.knowledge_base = symbolic_kb
                # Re-wire TaskDecomposer
                if hasattr(self, 'task_decomposer') and self.task_decomposer is not None:
                    self.task_decomposer.knowledge_base = symbolic_kb
        # Wire DomainKnowledge to TaskDecomposer (for tool tier rules)
        if hasattr(self, 'task_decomposer') and self.task_decomposer is not None:
            self.task_decomposer._domain_knowledge = knowledge

    def set_skill_manager(self, skill_manager) -> None:
        """
        Set the skill manager and initialize dependent components.

        This must be called after PSN construction because the skill manager
        is created separately by PSNAgent.
        """
        self.skill_manager = skill_manager

        # Initialize adaptive learning now that we have the skill manager
        if self.enable_adaptive_learning:
            self._init_adaptive_learning()

    def _create_task_with_semantic(self, task_str: str) -> TaskWithSemantic:
        """
        P1: convert a task string into a TaskWithSemantic object.

        This is the core entry point for Curriculum's output semantics.
        All tasks should pass through this method before being returned upward.

        Args:
            task_str: task description string

        Returns:
            TaskWithSemantic: task object carrying semantic information
        """
        semantic_type = detect_task_semantic(task_str)
        return TaskWithSemantic(
            task=task_str,
            semantic_type=semantic_type,
        )

    def has_learned_basic_tools(self) -> bool:
        """
        Check if basic wooden tools have been learned.

        Returns:
            True if both wooden_pickaxe and wooden_axe skills are learned
        """
        has_pickaxe = self.is_skill_learned("craft wooden_pickaxe")
        has_axe = self.is_skill_learned("craft wooden_axe")
        return has_pickaxe and has_axe

    def _validate_milestone_tasks(
        self, tasks: List[str], milestone
    ) -> List[str]:
        """Validate 'Ensure you have N X' tasks against milestone.

        Guards against LLM hallucination or state bugs that substitute wrong
        item names in milestone task lists. In r1_1, "Ensure you have 3 cobblestone"
        (correct, from stone_tools milestone) was replaced by "Ensure you have 3
        cobbled deepslate" (wrong, impossible at Y=70 with wooden tools), blocking
        all downstream progression.

        Checks each "Ensure you have N X" task:
        - Parse item X and verify it's in the milestone's required_items or is a
          valid craft ingredient of a required item
        - If invalid, find the closest matching required item and correct the task
        """
        required_items_dict = milestone.get("required_items", {}) if isinstance(milestone, dict) else getattr(milestone, "required_items", {})
        if not required_items_dict:
            return tasks

        required_items = set(required_items_dict.keys())

        # Build valid items from the BEST recipe variant (preferred path)
        valid_items = set(required_items)
        from skillnet.core.dk_registry import get_domain_knowledge
        dk = get_domain_knowledge()
        item_groups = dk.get_item_groups() if dk else {}
        for ri in list(required_items):
            if ri in item_groups:
                valid_items.update(item_groups[ri])

        # Map: best recipe ingredient → required item it belongs to
        inv = self.goal_planner.resource_tracker.current_inventory
        kb = self.knowledge_base if hasattr(self, 'knowledge_base') else None
        if kb:
            for ri in required_items:
                recipe = kb.get_best_recipe(ri, inv, 1)
                if recipe:
                    valid_items.update(recipe.ingredients.keys())

        # Map: wrong-variant ingredient → correct (best) ingredient
        # e.g., cobbled_deepslate → cobblestone (both are stone_pickaxe
        # recipe ingredients, but cobblestone is preferred)
        wrong_variant_correction = {}
        if kb:
            for ri in required_items:
                best = kb.get_best_recipe(ri, inv, 1)
                all_recipes = kb.get_all_recipes(ri) if hasattr(kb, 'get_all_recipes') else []
                if best and all_recipes:
                    best_ings = set(best.ingredients.keys())
                    for recipe in all_recipes:
                        for ing in recipe.ingredients:
                            if ing not in valid_items and ing not in wrong_variant_correction:
                                # This ingredient is from a non-preferred recipe
                                # Find the equivalent in the best recipe (same role)
                                for best_ing in best_ings:
                                    if best_ing not in ["stick"]:  # Skip universal ingredients
                                        wrong_variant_correction[ing] = best_ing

        validated = []
        for task in tasks:
            task_lower = task.lower()
            if "ensure" in task_lower and "have" in task_lower:
                item, count = self.goal_planner._parse_ensure_task(task)
                if item and item not in valid_items:
                    # Check if it's a wrong-variant ingredient
                    correct_item = wrong_variant_correction.get(item)
                    if correct_item:
                        corrected = f"Ensure you have {count} {correct_item.replace('_', ' ')}"
                        print(
                            f"\033[33m[Milestone Guard] Corrected wrong variant: "
                            f"'{task}' → '{corrected}'\033[0m"
                        )
                        validated.append(corrected)
                        continue
            validated.append(task)
        return validated

    def _try_continue_learning_path(self, inventory=None) -> Optional[Tuple[TaskWithSemantic, str]]:
        """
        Check if there is an ongoing adaptive learning path and return the next task.

        Args:
            inventory: Current inventory for skipping already-satisfied "ensure" tasks.

        Returns:
            (TaskWithSemantic, context) if a learning path task is available, None otherwise.
        """
        if not (self.enable_adaptive_learning and self.adaptive_planner):
            return None
        current_path = self.adaptive_planner.get_current_path()
        if current_path and not self.adaptive_planner.is_path_completed():
            next_task = self.adaptive_planner.get_next_task(inventory=inventory)
            if next_task:
                print(f"\033[32m[PSN Adaptive] Continuing learning path: {next_task}\033[0m")
                learning_context = (
                    f"IMPORTANT: This task is part of an adaptive learning sequence. "
                    f"Complete it to progress toward: {current_path.original_task}"
                )
                return self._create_task_with_semantic(next_task), learning_context
        return None

    def _try_decompose_task(
        self,
        task_str: str,
        inventory: Dict[str, int],
        reason: str,
    ) -> Optional[Tuple[TaskWithSemantic, str]]:
        """
        Check if a task needs decomposition and start a learning path if so.

        Args:
            task_str: The task to check for decomposition
            inventory: Current inventory state for planning the learning path
            reason: Log label for why decomposition is being checked
                    (e.g., "Exploration task", "Milestone task", "LLM task")

        Returns:
            (TaskWithSemantic, context) if decomposition produced a sub-task, None otherwise.
        """
        if not (self.enable_adaptive_learning and self.adaptive_planner):
            return None
        task_obj = self._create_task_with_semantic(task_str)
        skill_names = list(self.skill_manager.skills.keys()) if self.skill_manager else []
        if not self.adaptive_planner.decomposition_filter.should_decompose(task_obj, skill_names):
            return None
        print(f"\033[33m[PSN Adaptive] {reason} needs decomposition: {task_str}\033[0m")

        path = self.adaptive_planner.plan_learning_path(task_str, inventory, skill_names)
        if path.path_type == "learning_first" and path.learning_tasks:
            next_task = self.adaptive_planner.get_next_task(inventory=inventory)
            if next_task:
                print(f"\033[32m[PSN Adaptive] Starting learning path: {next_task}\033[0m")
                learning_context = (
                    f"IMPORTANT: This task is part of an adaptive learning sequence. "
                    f"Complete it to progress toward: {task_str}"
                )
                return self._create_task_with_semantic(next_task), learning_context
        return None

    # ── Public facade for task list state ──────────────────────

    def get_completed_tasks(self) -> List[str]:
        """Return a copy of the completed tasks list."""
        return list(self.completed_tasks)

    def get_failed_tasks(self) -> List[str]:
        """Return a copy of the failed tasks list."""
        return list(self.failed_tasks)

    def add_failed_tasks(self, tasks: List[str]) -> None:
        """Extend the failed tasks list."""
        self.failed_tasks.extend(tasks)

    def remove_matching_failed_tasks(self, predicate) -> int:
        """Remove failed tasks where predicate(task) returns True. Returns count removed."""
        before = len(self.failed_tasks)
        self.failed_tasks = [t for t in self.failed_tasks if not predicate(t)]
        return before - len(self.failed_tasks)

    def reset_task_lists(self) -> None:
        """Clear both completed and failed task lists."""
        self.completed_tasks = []
        self.failed_tasks = []

    def task_count(self) -> int:
        """Return number of completed tasks."""
        return len(self.completed_tasks)

    def count_task_failures(self) -> Dict[str, int]:
        """Public alias for _count_task_failures()."""
        return self._count_task_failures()

    # ─────────────────────────────────────────────────────────────────────

    def propose_next_task(
        self,
        *,
        events,
        chest_observation,
        max_retries=5,
        blacklisted_tasks: Optional[set] = None,
    ) -> Tuple[TaskWithSemantic, str]:
        """
        Propose the next task with resource planning awareness.

        P1 improvement: returns a TaskWithSemantic object carrying semantic info.

        This method:
        1. Updates resource state from events
        2. Handles special cases (first task, underground, full inventory)
        3. Updates milestone progress
        4. Checks for critical resource alerts (only if we have tools)
        5. Uses LLM to make final decision with full context

        Args:
            events: List of (event_type, event_data) from environment
            chest_observation: Observation of nearby chests
            max_retries: Maximum retries for LLM call

        Returns:
            (TaskWithSemantic, context) tuple - P1: task object carrying semantic info
        """
        # 0. Check for manual mode - delegate to parent's manual task input
        if self.mode == "manual":
            task_str, context = self.propose_next_manual_task()
            return self._create_task_with_semantic(task_str), context

        # 1. Update resource state
        self.resource_tracker.update_inventory(events)
        inventory = self.resource_tracker.current_inventory

        # 1.5. Clear skill learning cache at the start of each task proposal
        # This ensures we always check the latest skill graph state
        # Skills may have been learned since the last proposal
        self.clear_skill_cache()

        # 2. Update milestone progress (with optional validation)
        # Moved before special cases so we can check if all milestones are done
        newly_completed = self.goal_planner.update_progress(inventory)
        if newly_completed:
            print(f"\033[32m[PSN] Milestones completed: {newly_completed}\033[0m")
            # Defensive validation: log item breakdown for each completed milestone
            self._log_milestone_validation(newly_completed, inventory)

        # 2.5. Handle special cases FIRST (before milestone-done check)
        # These are HARD requirements that the LLM exploration mode mishandles:
        # - first task (self.progress == 0): seed initial wood logs
        # - emergency (e.g. underground without pickaxe): bot can't progress
        # - armor upgrade: better armor available in inventory
        # - FULL INVENTORY: bot can't pick up new items
        # Previously this check was AFTER the all-milestones-done branch, so
        # once milestones were complete the curriculum entered Phase 4f LLM
        # exploration mode and never generated a "deposit useless items"
        # task even with inventory at 36/36 (observed in
        # ckpt_psnv11_qwen3_fp8_r4r1: 39 'All milestones completed' loops
        # without a chest task). Moving the check up restores the semantic
        # of the existing comment "Handle special cases FIRST".
        special_task = self._check_special_cases(events, chest_observation)
        if special_task:
            context = self._get_task_context_with_planning(special_task, events)
            return self._create_task_with_semantic(special_task), context

        # 2.6. All milestones completed → pure LLM exploration mode
        # Skip rule-based stockpile/milestone tasks and let the LLM decide
        # freely what to do next.
        # Phase 4f: pass feasible_task=None so the LLM isn't nudged toward
        # a hardcoded obsidian/diamond-armor priority ladder
        # (_get_exploration_fallback_task). The LLM reasons over state +
        # completed/failed tasks alone — matches Voyager's open curriculum.
        if self.goal_planner.get_next_milestone() is None:
            print(f"\033[32m[PSN] All milestones completed — open LLM exploration (Phase 4f)\033[0m")
            task_str, context = self._llm_decide_with_resource_context(
                events=events,
                chest_observation=chest_observation,
                alerts=self.resource_tracker.check_resource_levels(),
                milestone_tasks=[],
                feasible_task=None,
                max_retries=max_retries,
                blacklisted_tasks=blacklisted_tasks,
            )

            # ===== Decomposition check for exploration tasks =====
            result = self._try_continue_learning_path(inventory=inventory)
            if result:
                return result
            result = self._try_decompose_task(task_str, inventory, "Exploration task")
            if result:
                return result

            return self._create_task_with_semantic(task_str), context

        # 4. Check for critical resource alerts (only if basic crafting skills are learned)
        # Don't handle resource alerts until we've learned the basic crafting skills
        # This ensures progressive skill learning before stockpiling
        #
        # Strategy: "Learn skills first, stockpile later"
        # - First, learn all basic wooden tool crafting skills
        # - Then, handle resource threshold alerts for stockpiling
        #
        # This prevents the situation where the bot repeatedly mines logs
        # without ever learning the crafting skills
        if self.has_learned_basic_tools():
            critical_alerts = self.resource_tracker.get_critical_alerts()
            if critical_alerts:
                # Handle critical alerts
                stockpile_task = self._handle_critical_alerts(critical_alerts, events)
                if stockpile_task:
                    print(f"\033[33m[PSN Stockpile] Basic skills learned, now stockpiling: {stockpile_task[0]}\033[0m")
                    context = self._get_task_context_with_planning(stockpile_task[0], events)
                    return self._create_task_with_semantic(stockpile_task[0]), context
        else:
            # Log that we're skipping stockpile to focus on skill learning
            critical_alerts = self.resource_tracker.get_critical_alerts()
            if critical_alerts:
                print(f"\033[33m[PSN Skill Learning] Deferring stockpile - basic tools not yet learned\033[0m")

            # 4.5 Prioritize unlearned crafting skills
            # Ensures crafting_table is learned before tools that need 3×3 grid
            next_craft_task = self.get_next_unlearned_crafting_task(inventory)
            if next_craft_task and not self._should_skip_special_task(next_craft_task):
                print(f"\033[35m[PSN Skill Learning] Prioritizing crafting skill: {next_craft_task}\033[0m")
                context = self._get_task_context_with_planning(next_craft_task, events)
                return self._create_task_with_semantic(next_craft_task), context

        # 5. Get next milestone task
        next_milestone = self.goal_planner.get_next_milestone()
        milestone_tasks = self.goal_planner.get_next_milestone_tasks()

        # diagnostic logging — capture raw GoalPlanner
        # output BEFORE any filtering. In r1_1, milestone_tasks arrived at
        # _check_adaptive_learning with "cobbled deepslate" instead of
        # "cobblestone" but get_next_milestone_tasks() provably returns
        # "cobblestone". This log captures the intermediate state to trace
        # the substitution source in future runs.
        if milestone_tasks:
            print(f"\033[36m[PSN Debug] raw milestone_tasks from GoalPlanner: {milestone_tasks[:5]}\033[0m")

        # validate milestone task items against the
        # milestone's required_items and craft ingredients. Guards against
        # LLM hallucination or state bugs that substitute wrong items
        # (e.g., 'cobbled_deepslate' for 'cobblestone' in r1_1).
        if milestone_tasks and next_milestone:
            milestone_tasks = self._validate_milestone_tasks(
                milestone_tasks, next_milestone
            )

        # 5.5. Filter out tasks that have failed too many times (prevent infinite loops)
        consecutive_failure_threshold = self.milestone_task_failure_threshold
        task_failure_counts = self._count_task_failures()

        filtered_milestone_tasks = []
        for task in milestone_tasks:
            # Use shared normalization (consistent with _count_task_failures + PSNAgent._normalize_task_key)
            from skillnet._psn_impl.task_management import TaskManagementMixin
            normalized_task = TaskManagementMixin._normalize_task_key(task)
            failure_count = task_failure_counts.get(normalized_task, 0)
            if failure_count >= consecutive_failure_threshold:
                print(f"\033[33m[PSN] Skipping '{task}' - failed {failure_count} times (threshold: {consecutive_failure_threshold})\033[0m")
            else:
                filtered_milestone_tasks.append(task)

        milestone_tasks = filtered_milestone_tasks

        # 5.6. First check if there's an ongoing adaptive learning path to continue
        if self.enable_adaptive_learning and self.adaptive_planner:
            result = self._try_continue_learning_path(inventory=inventory)
            if result:
                return result

        # 5.7. Adaptive Learning: Check if any milestone task needs decomposition
        # This handles complex tasks like obsidian by breaking them into learnable sub-tasks
        if self.enable_adaptive_learning and self.adaptive_planner and milestone_tasks:
            adaptive_result = self._check_adaptive_learning(milestone_tasks, inventory)
            if adaptive_result:
                task_str, context = adaptive_result
                return self._create_task_with_semantic(task_str), context

        # If all milestone tasks have been filtered out, let LLM decide freely (explore mode)
        original_milestone_tasks = self.goal_planner.get_next_milestone_tasks()
        if not milestone_tasks and original_milestone_tasks:
            self._exploration_mode_count += 1

            # Cooldown recovery: after 3 exploration rounds, reset milestone failures
            # to give the bot a new chance (it may have acquired resources during exploration)
            if self._exploration_mode_count >= 3 and self._milestone_retry_budget > 0:
                self._milestone_retry_budget -= 1
                self._reset_milestone_failures(original_milestone_tasks)
                self._exploration_mode_count = 0
                # Re-enter normal milestone proposal flow
                return self.propose_next_task(
                    events=events,
                    chest_observation=chest_observation,
                    max_retries=max_retries,
                    blacklisted_tasks=blacklisted_tasks,
                )

            print(
                f"\033[33m[PSN] All milestone tasks have failed too many times. "
                f"Entering exploration mode. (count={self._exploration_mode_count}, "
                f"retry_budget={self._milestone_retry_budget})\033[0m"
            )
            # Let LLM decide with exploration context
            task_str, context = self._llm_decide_with_resource_context(
                events=events,
                chest_observation=chest_observation,
                alerts=self.resource_tracker.check_resource_levels(),
                milestone_tasks=[],  # No specific tasks, let LLM explore
                feasible_task=None,
                max_retries=max_retries,
                blacklisted_tasks=blacklisted_tasks,
            )
            return self._create_task_with_semantic(task_str), context

        # Normal milestone path — reset exploration counter
        self._exploration_mode_count = 0

        # 6. Check feasibility of milestone tasks
        # Strategy: Find feasible tasks in order (prereqs → targets → base)
        # Skip base/mining tasks if we already have sufficient materials for crafting
        feasible_task = None
        has_sufficient_materials = self._has_sufficient_materials_for_milestone(inventory)

        # [DEBUG] Log key resource state for diagnosis
        logs_count = sum(v for k, v in inventory.items() if k.endswith("_log"))
        planks_count = sum(v for k, v in inventory.items() if k.endswith("_planks"))
        sticks_count = inventory.get("stick", 0)
        print(f"\033[36m[PSN Debug] Inventory: logs={logs_count}, planks={planks_count}, sticks={sticks_count}\033[0m")
        print(f"\033[36m[PSN Debug] has_sufficient_materials={has_sufficient_materials}, milestone={next_milestone}\033[0m")
        print(f"\033[36m[PSN Debug] milestone_tasks={milestone_tasks[:5]}{'...' if len(milestone_tasks) > 5 else ''}\033[0m")

        for task in milestone_tasks:
            task_lower = task.lower()
            is_mining = "mine" in task_lower or "collect" in task_lower or "gather" in task_lower

            # Skip mining tasks if we have enough materials for crafting
            if is_mining and has_sufficient_materials:
                print(f"\033[33m[PSN] Skipping '{task}' - already have sufficient materials\033[0m")
                continue

            feasibility = self.goal_planner.check_feasibility(TaskWithSemantic.from_legacy_string(task))
            if feasibility.is_feasible:
                feasible_task = task
                break
            elif feasibility.suggested_prerequisites:
                # Return prerequisite task instead, but also check if it failed too many times
                prereq = feasibility.suggested_prerequisites[0]
                if task_failure_counts.get(prereq, 0) < consecutive_failure_threshold:
                    feasible_task = prereq
                    break
                else:
                    print(f"\033[33m[PSN] Skipping prerequisite '{prereq}' - failed {task_failure_counts.get(prereq, 0)} times\033[0m")

        # ===== Milestone task decomposition check =====
        if feasible_task:
            result = self._try_decompose_task(feasible_task, inventory, "Milestone task")
            if result:
                return result
        # ===== Milestone decomposition check end =====

        # ===== First check in-progress learning paths (before the LLM call) =====
        result = self._try_continue_learning_path()
        if result:
            return result
        # ===== End learning-path check =====

        # 7. Use LLM for final decision with enhanced context
        print(f"\033[36m[PSN Debug] feasible_task={feasible_task}\033[0m")
        task_str, context = self._llm_decide_with_resource_context(
            events=events,
            chest_observation=chest_observation,
            alerts=self.resource_tracker.check_resource_levels(),
            milestone_tasks=milestone_tasks,
            feasible_task=feasible_task,
            max_retries=max_retries,
            blacklisted_tasks=blacklisted_tasks,
        )
        print(f"\033[36m[PSN Debug] LLM decided task='{task_str}'\033[0m")

        # ===== LLM task-decomposition check (new tasks only) =====
        # Note: in-progress learning paths were handled before the LLM call; only new tasks are checked here
        # Get current inventory via the active DomainKnowledge — handles
        # both Minecraft tuple-events and dict-events transparently.
        llm_inventory: Dict[str, int] = {}
        if events:
            obs = (
                self._domain_knowledge.extract_observation(events)
                if self._domain_knowledge is not None
                else None
            )
            if obs is not None:
                llm_inventory = dict(obs.inventory)
        result = self._try_decompose_task(task_str, llm_inventory, "LLM task")
        if result:
            return result
        # ===== End LLM task-decomposition check =====

        # 8. Check tool efficiency - if the task is a bulk gathering task,
        # suggest crafting appropriate tool first if possible
        tool_task = self._check_tool_efficiency(task_str, inventory)
        if tool_task:
            print(f"\033[35m[PSN Efficiency] Before '{task_str}', crafting tool first: {tool_task}\033[0m")
            context = self._get_task_context_with_planning(tool_task, events)
            return self._create_task_with_semantic(tool_task), context

        return self._create_task_with_semantic(task_str), context

    def _get_learned_skill_names(self) -> List[str]:
        """Get names of learned, verified skills for LLM context."""
        if not hasattr(self, 'skill_manager') or not self.skill_manager:
            return []
        try:
            all_skills = list(self.skill_manager.get_all_skill_names())
            # Filter for skills with at least one successful execution
            reliable = []
            for name in all_skills:
                node = self.skill_manager.get_node(name)
                if node and getattr(node, 'successful_executions', 0) > 0:
                    reliable.append(name)
            return sorted(reliable)[:20]  # Cap at 20
        except Exception:
            return []

    def _get_exploration_fallback_task(self) -> str:
        """
        Generate a meaningful exploration fallback task based on current inventory.

        Called when all milestones are completed and LLM fails to produce a valid task.
        Returns a concrete, actionable task instead of the generic "Explore the area".
        """
        inventory = self.resource_tracker.current_inventory

        # Prioritized exploration tasks based on what the bot is missing
        # 1. Combat gear
        if not any(k == "diamond_sword" for k in inventory):
            if inventory.get("diamond", 0) >= 2:
                return "Craft 1 diamond sword"
            return "Ensure you have 4 diamonds"

        # 2. Nether preparation (obsidian)
        obsidian_count = inventory.get("obsidian", 0)
        if obsidian_count < 10:
            return f"Ensure you have 10 obsidian"

        # 3. Armor
        if not any(k == "diamond_chestplate" for k in inventory):
            if inventory.get("diamond", 0) >= 8:
                return "Craft 1 diamond chestplate"
            return "Ensure you have 12 diamonds"

        # 4. Food stockpile
        cooked_food_count = sum(
            v for k, v in inventory.items()
            if "cooked" in k
        )
        if cooked_food_count < 20:
            return "Ensure you have 20 cooked food items"

        # 5. Generic resource gathering
        return "Ensure you have 32 iron ingots"

    def _check_tool_efficiency(
        self,
        task: str,
        inventory: Dict[str, int]
    ) -> Optional[str]:
        """
        Check if we should craft a tool before executing a bulk gathering task.

        Delegates to domain knowledge if available, otherwise returns None.

        Args:
            task: The proposed task (e.g., "Ensure you have 17 logs" or "Mine 17 logs")
            inventory: Current inventory state

        Returns:
            A tool crafting task if one should be done first, None otherwise
        """
        dk = getattr(self, '_domain_knowledge', None)
        if dk:
            return dk.check_tool_efficiency(task, inventory)
        return None

    def _get_task_context_with_planning(
        self,
        task: str,
        events: List
    ) -> str:
        """Get task context with planning information"""
        # Get base context
        try:
            base_context = self.get_task_context(task)
        except Exception as e:
            print(f"\033[33m[PSN] Failed to get task context (LLM may be unreachable): {e}\033[0m")
            base_context = ""

        # Add feasibility info
        feasibility = self.goal_planner.check_feasibility(TaskWithSemantic.from_legacy_string(task))
        if not feasibility.is_feasible:
            base_context += f"\nNote: Missing items - {feasibility.missing_items}"
            if feasibility.suggested_prerequisites:
                base_context += f"\nPrerequisites: {feasibility.suggested_prerequisites[:3]}"

        return base_context

    def _count_task_failures(self) -> Dict[str, int]:
        """
        Count how many times each task has failed (using normalized task keys).

        Uses normalized keys to ensure "Mine 3 cobblestone" and "mine 3 Cobblestone"
        are treated as the same task, preventing case differences from bypassing the failure count.

        Returns:
            Dict mapping normalized task name to failure count
        """
        failure_counts = {}
        for task in self.failed_tasks:
            # Use shared normalization (lowercase + whitespace + plural→singular)
            # to be consistent with PSNAgent._normalize_task_key()
            from skillnet._psn_impl.task_management import TaskManagementMixin
            normalized = TaskManagementMixin._normalize_task_key(task)
            failure_counts[normalized] = failure_counts.get(normalized, 0) + 1
        return failure_counts

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        return {
            "tasks_completed": len(self.completed_tasks),
            "tasks_failed": len(self.failed_tasks),
            "current_stage": self.goal_planner.get_current_stage().value,
            "milestones_completed": list(self.goal_planner.completed_milestones),
            "next_milestone": self.goal_planner.get_next_milestone().name if self.goal_planner.get_next_milestone() else None,
            "inventory_summary": self.resource_tracker.get_inventory_summary()
        }
