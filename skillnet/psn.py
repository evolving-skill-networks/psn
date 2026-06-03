import copy
import os
import re
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from skillnet.core.environment import Environment

import skillnet.utils as u

from .agents import ParameterizedActionAgent
from .agents.psn_curriculum.psn_critic import PSNCriticAgent
from .agents import SkillGraphManager
from .agents.optimizer._impl import SkillGraphOptimizer
from .agents.planner import GraphPlanner
from .agents.constants.task_semantics import TaskWithSemantic
from .utils.trajectory_recorder import TrajectoryRecorder
from .utils.log_utils import GlobalOutputLogger
from .tools.progress_recorder import ProgressRecorder
from .core.config import PSNConfig

# PSNAgent implementation mixins (extracted for modularity)
from ._psn_impl import (
    EventProcessingMixin,
    SkillRecordingMixin,
    EffectVerificationMixin,
    StepExecutionMixin,
    StepContext,
    TaskManagementMixin,
)


class PSNAgent(
    EventProcessingMixin,
    SkillRecordingMixin,
    EffectVerificationMixin,
    StepExecutionMixin,
    TaskManagementMixin,
):
    def __init__(
        self,
        config: PSNConfig = None,
        *,
        _env: "Environment" = None,
    ):
        """
        The main class for Evolving Programmatic Skill Networks (PSN).

        Args:
            config: PSNConfig with all agent/environment/checkpoint settings.
                    Defaults to PSNConfig() if None.
            _env: Pre-built Environment instance. Required. The canonical entry
                  points (PSNAgent.from_domain / PSNAgent.from_components) supply
                  this; raw __init__ usage must pass it explicitly.
        """
        if config is None:
            config = PSNConfig()
        self._config = config

        # Unpack frequently-used config sections
        ckpt_dir = config.checkpoint.ckpt_dir
        resume = config.checkpoint.resume
        request_timeout = config.llm.action.request_timeout

        # Handle resume_from_iteration: create new checkpoint directory
        if config.checkpoint.resume_from_iteration is not None:
            ckpt_dir = self._prepare_iteration_resume(
                ckpt_dir, config.checkpoint.resume_from_iteration,
                config.checkpoint.resume_to_dir,
            )
            resume = True

        self.ckpt_dir = ckpt_dir
        self.global_logger = GlobalOutputLogger(ckpt_dir=ckpt_dir)

        # init env (callers must supply via from_domain / from_components)
        if _env is None:
            raise ValueError(
                "PSNAgent.__init__ requires _env to be supplied. "
                "Use the canonical entry points instead:\n"
                "  agent = PSNAgent.from_domain(domain, config)        # recommended\n"
                "  agent = PSNAgent.from_components(env=..., ...)     # for hand-wired components\n"
                "If you're constructing PSNAgent in a new domain, write a "
                "<Domain>Domain(...) factory that builds the env and returns a DomainModule, "
                "then call PSNAgent.from_domain(...) — see skillnet/domains/minecraft/__init__.py."
            )
        self.env = _env
        self.env_wait_ticks = config.environment.wait_ticks
        self.reset_placed_if_failed = config.environment.reset_placed_if_failed
        self.max_iterations = config.max_iterations

        # init agents
        self.action_agent = ParameterizedActionAgent(
            model_name=config.llm.action.model_name,
            temperature=config.llm.action.temperature,
            request_timout=request_timeout,
            ckpt_dir=ckpt_dir,
            resume=resume,
            chat_log=config.action.show_chat_log,
            execution_error=config.action.show_execution_error,
            use_llm_for_normalization=config.action.use_llm_for_normalization,
            include_skill_code=config.action.include_skill_code,
            max_tokens=config.llm.action.max_tokens,
        )
        self.action_agent_task_max_retries = config.action.task_max_retries

        from skillnet.agents.psn_curriculum import PSNCurriculumAgent
        self.curriculum_agent = PSNCurriculumAgent(
            model_name=config.llm.curriculum.model_name,
            temperature=config.llm.curriculum.temperature,
            qa_model_name=config.llm.curriculum_qa.model_name,
            qa_temperature=config.llm.curriculum_qa.temperature,
            request_timout=request_timeout,
            ckpt_dir=ckpt_dir,
            resume=resume,
            mode=config.curriculum.mode,
            warm_up=config.curriculum.warm_up,
            core_inventory_items=config.curriculum.core_inventory_items,
        )
        print(f"\033[35m[PSN] Using PSN Curriculum Agent\033[0m")

        self.critic_agent = PSNCriticAgent(
            model_name=config.llm.critic.model_name,
            temperature=config.llm.critic.temperature,
            request_timout=request_timeout,
            mode=config.critic.mode,
        )

        skill_library_dir = config.checkpoint.skill_library_dir
        self.skill_manager = SkillGraphManager(
            model_name=config.llm.skill_manager.model_name,
            temperature=config.llm.skill_manager.temperature,
            retrieval_top_k=config.skill_manager.retrieval_top_k,
            request_timout=request_timeout,
            ckpt_dir=skill_library_dir if skill_library_dir else ckpt_dir,
            resume=True if resume or skill_library_dir else False,
            merge_mode=config.skill_manager.merge_mode,
            enable_refactor=config.optimization.enable_refactor,
            max_skills=config.skill_manager.max_skills,
        )

        # Connect skill manager to PSN curriculum agent for skill learning state tracking
        if hasattr(self.curriculum_agent, 'set_skill_manager'):
            self.curriculum_agent.set_skill_manager(self.skill_manager)

        # Connect skill manager to action agent for function reference validation
        if hasattr(self.action_agent, 'set_skill_manager'):
            self.action_agent.set_skill_manager(self.skill_manager)

        self.recorder = u.EventRecorder(ckpt_dir=ckpt_dir, resume=resume)
        self.trajectory_recorder = TrajectoryRecorder(
            ckpt_dir=ckpt_dir,
            resume=resume,
            save_trajectories=config.recording.save_trajectories,
        )

        # Initialize progress recorder for visualization and experiment analysis
        self.progress_recorder = ProgressRecorder(
            ckpt_dir=ckpt_dir,
            mode="graph",  # always use graph mode
            resume=resume,
            domain_knowledge=config.domain.knowledge if config.domain else None,
        )

        self.resume = resume
        self.planner_mode = config.planner.mode
        self.enable_optimizer = config.optimization.enable_optimizer
        self.enable_refactor = config.optimization.enable_refactor

        # Initialize optimizer
        self.optimizer = None
        if config.optimization.enable_optimizer:
            self.optimizer = SkillGraphOptimizer(
                skill_graph_manager=self.skill_manager,
                model_name=config.llm.action.model_name,
                temperature=config.llm.action.temperature,
                request_timeout=request_timeout,
                enable_refactor=config.optimization.enable_refactor,
                pure_reasoning=config.optimization.pure_reasoning,
                include_reasoning_examples=config.optimization.include_reasoning_examples,
                use_factual_primitive_doc=config.optimization.use_factual_primitive_doc,
                # B1: Maturity gating parameters
                optimization_threshold=config.optimization.optimization_threshold,
                optimization_epsilon=config.optimization.optimization_epsilon,
                optimization_gamma=config.optimization.optimization_gamma,
            )
            refactor_status = "enabled" if config.optimization.enable_refactor else "disabled"
            print(f"\033[36m[Optimizer] Using TwoPhaseOptimizationEngine (with transaction support, refactor={refactor_status})\033[0m")
        else:
            print(f"\033[33m[Optimizer] Optimizer disabled (--no-optimizer)\033[0m")

        # Initialize escalation tracker
        from skillnet._psn_impl.escalation import EscalationTracker
        self._escalation_tracker = EscalationTracker(
            min_attempts=config.optimization.escalation_min_attempts,
            improvement_threshold=config.optimization.escalation_improvement_threshold,
        )
        self._level2_diagnosis = None  # Set by SEM when Level 2 triggers

        # Initialize Skill Evolution Manager
        from skillnet.agents.evolution import SkillEvolutionManager
        sem_llm = None
        if config.optimization.enable_optimizer:
            try:
                from skillnet.utils.llm_factory import create_chat_llm
                sem_llm = create_chat_llm(
                    model_name=config.llm.action.model_name,
                    component="evolution_manager",
                    temperature=0,
                    request_timeout=60,
                )
            except Exception:
                pass  # No LLM — SEM will use heuristic fallback
        self._evolution_manager = SkillEvolutionManager(llm=sem_llm)

        # Initialize LLM-based HelperExtractor for skill synthesis
        from skillnet.agents.optimizer.synthesis import HelperExtractor
        synthesis_llm = None
        if config.optimization.enable_optimizer:
            try:
                from skillnet.utils.llm_factory import create_chat_llm
                synthesis_llm = create_chat_llm(
                    model_name=config.llm.action.model_name,
                    component="skill_synthesis",
                    temperature=0,
                    request_timeout=60,
                )
            except Exception:
                pass
        self._helper_extractor = HelperExtractor(llm=synthesis_llm)

        # Initialize learning dynamics recorder for experiment analysis
        from skillnet.tools.learning_dynamics_recorder import LearningDynamicsRecorder
        self.dynamics_recorder = LearningDynamicsRecorder(
            ckpt_dir=ckpt_dir,
            graph_manager=self.skill_manager,
            optimization_tracker=(
                self.optimizer.optimization_tracker if self.optimizer else None
            ),
            optimization_threshold=config.optimization.optimization_threshold,
            graph_snapshot_interval=config.checkpoint.graph_snapshot_interval,
        )

        # Initialize planner
        is_adaptive = config.planner.mode != "graph"
        skill_count = self.skill_manager.skill_count
        print(f"\033[36m[Planner] Initializing {'adaptive' if is_adaptive else 'Graph-based'} Planner\033[0m")
        print(f"\033[36m[Planner] Current skill graph has {skill_count} skills\033[0m")

        self.planner = GraphPlanner(
            skill_graph_manager=self.skill_manager,
            action_agent=self.action_agent,
            enable_fallback=True,
            adaptive_mode=is_adaptive,
            min_skills_for_graph_planning=config.planner.min_skills_for_graph_planning,
        )

        # Establish bidirectional connection between optimizer and planner (for parameter-passing issue detection and learning)
        if self.optimizer and hasattr(self, 'planner') and self.planner_mode in ["graph", "adaptive"]:
            self.optimizer.graph_planner = self.planner
            self.planner.optimizer = self.optimizer
            print(f"\033[36m[Init] Established bidirectional Optimizer <-> Graph Planner connection\033[0m")

        # init variables for rollout
        self.action_agent_rollout_num_iter = -1
        self.task = None  # task string (backward compatibility)
        self._task_semantic = None  # P1: TaskWithSemantic object (carries semantic info)
        self.context = ""
        self.messages = None
        self.conversations = []
        self.last_events = None
        self.task_initial_events = None  # initial state at task start; used for state-change computation

        # Skill execution results (used for optimization)
        # Format: {skill_name: {success: bool, effect_verification: {...}}}
        self.last_skill_execution_results = {}
        
        # debug mode variables
        self.debug_mode = config.debug_mode
        self.task_count = 0

        # LLM effect-verification configuration
        self.use_llm_effect_verification = config.optimization.use_llm_effect_verification

        # Global task-failure counter to prevent unbounded retries on the same task
        self.global_task_failure_limit = config.global_task_failure_limit
        self.global_task_attempt_counts = {}  # Dict[normalized_task_key, int]

    @classmethod
    def from_components(
        cls,
        *,
        env: "Environment",
        action_agent: "ParameterizedActionAgent",
        curriculum_agent: "PSNCurriculumAgent",
        critic_agent: "PSNCriticAgent",
        skill_manager: "SkillGraphManager",
        planner: "GraphPlanner",
        config: "PSNConfig",
        optimizer: Optional["SkillGraphOptimizer"] = None,
        event_recorder: Optional["u.EventRecorder"] = None,
        trajectory_recorder: Optional["TrajectoryRecorder"] = None,
        progress_recorder: Optional["ProgressRecorder"] = None,
    ) -> "PSNAgent":
        """
        PyTorch-like component injection constructor.

        Creates a PSNAgent from pre-built components, allowing full control
        over how each component is configured and initialized. The caller is
        responsible for creating and configuring all components; PSNAgent
        handles orchestration and wiring.

        Unlike __init__() which creates all components internally from flat
        kwargs, this method accepts already-constructed component instances.

        Args:
            env: The environment (an Environment subclass; e.g., the
                 MinecraftEnv produced by MinecraftDomain).
            action_agent: Pre-configured action agent.
            curriculum_agent: Pre-configured curriculum agent.
            critic_agent: Pre-configured critic agent.
            skill_manager: Pre-configured skill graph manager.
            planner: Pre-configured graph planner.
            config: PSNConfig with non-component settings (timeouts, flags, etc.).
            optimizer: Optional pre-configured optimizer. If None, optimization is disabled.
            event_recorder: Optional event recorder. Created from config if None.
            trajectory_recorder: Optional trajectory recorder. Created with defaults if None.
            progress_recorder: Optional progress recorder. Created from config if None.

        Returns:
            A fully wired PSNAgent instance.

        Example::

            from skillnet.core import PSNConfig

            config = PSNConfig(
                checkpoint=CheckpointConfig(ckpt_dir="ckpt_exp1"),
                optimization=OptimizationConfig(enable_refactor=False),
            )
            domain = MinecraftDomain(mc_port=25565)
            env = domain.environment
            action_agent = ParameterizedActionAgent(...)
            # ... build other components ...

            agent = PSNAgent.from_components(
                env=env,
                action_agent=action_agent,
                curriculum_agent=curriculum_agent,
                critic_agent=critic_agent,
                skill_manager=skill_manager,
                planner=planner,
                config=config,
            )
            agent.learn()
        """
        instance = cls.__new__(cls)

        # --- Core components ---
        instance.env = env
        instance.action_agent = action_agent
        instance.curriculum_agent = curriculum_agent
        instance.critic_agent = critic_agent
        instance.skill_manager = skill_manager
        instance.planner = planner
        instance.optimizer = optimizer

        # --- Config-derived attributes ---
        instance.ckpt_dir = config.checkpoint.ckpt_dir
        instance.env_wait_ticks = config.environment.wait_ticks
        instance.reset_placed_if_failed = config.environment.reset_placed_if_failed
        instance.max_iterations = config.max_iterations
        instance.debug_mode = config.debug_mode
        instance.use_llm_effect_verification = config.optimization.use_llm_effect_verification
        instance.resume = config.checkpoint.resume
        instance.action_agent_task_max_retries = config.action.task_max_retries
        instance.planner_mode = config.planner.mode
        instance.enable_optimizer = config.optimization.enable_optimizer
        instance.enable_refactor = config.optimization.enable_refactor

        # --- Recorders ---
        instance.global_logger = GlobalOutputLogger(ckpt_dir=config.checkpoint.ckpt_dir)

        if event_recorder is not None:
            instance.recorder = event_recorder
        else:
            instance.recorder = u.EventRecorder(
                ckpt_dir=config.checkpoint.ckpt_dir,
                resume=config.checkpoint.resume,
            )

        if trajectory_recorder is not None:
            instance.trajectory_recorder = trajectory_recorder
        else:
            instance.trajectory_recorder = TrajectoryRecorder(
                ckpt_dir=config.checkpoint.ckpt_dir,
                resume=config.checkpoint.resume,
                save_trajectories=config.recording.save_trajectories,
            )

        if progress_recorder is not None:
            instance.progress_recorder = progress_recorder
        else:
            instance.progress_recorder = ProgressRecorder(
                ckpt_dir=config.checkpoint.ckpt_dir,
                mode="graph",
                resume=config.checkpoint.resume,
                domain_knowledge=config.domain.knowledge if config.domain else None,
            )

        # --- Component wiring ---
        # Connect skill manager to curriculum agent for skill learning state tracking
        if hasattr(curriculum_agent, 'set_skill_manager'):
            curriculum_agent.set_skill_manager(skill_manager)

        # Establish bidirectional optimizer <-> planner connection
        if optimizer is not None and instance.planner_mode in ["graph", "adaptive"]:
            optimizer.graph_planner = planner
            planner.optimizer = optimizer

        # --- Runtime state variables ---
        instance.action_agent_rollout_num_iter = -1
        instance.task = None
        instance._task_semantic = None
        instance.context = ""
        instance.messages = None
        instance.conversations = []
        instance.last_events = None
        instance.task_initial_events = None
        instance.last_skill_execution_results = {}
        instance.task_count = 0
        instance.global_task_failure_limit = config.global_task_failure_limit
        instance.global_task_attempt_counts = {}

        return instance

    @classmethod
    def from_domain(
        cls,
        domain: "DomainModule",
        config: PSNConfig,
    ) -> "PSNAgent":
        """
        Construct a PSNAgent from a DomainModule.

        This is the highest-level factory method: provide a domain-specific
        module (e.g., MinecraftDomain) and a PSNConfig, and get a fully
        wired agent ready to learn.

        Args:
            domain: A DomainModule bundling environment, knowledge,
                    curriculum, and critic for the target domain.
            config: PSNConfig with all settings.

        Returns:
            A fully wired PSNAgent instance.

        Example::

            from skillnet.domains.minecraft import MinecraftDomain
            from skillnet.core import PSNConfig

            domain = MinecraftDomain(mc_port=25565)
            agent = PSNAgent.from_domain(domain, PSNConfig())
            agent.learn()
        """
        config.domain = domain
        env = domain.environment

        # Set central DK registry BEFORE __init__ so that components created
        # during __init__ (e.g., GoalPlanner milestones) can read domain data.
        knowledge = domain.knowledge
        if knowledge:
            from skillnet.core.dk_registry import (
                set_domain_knowledge as _set_registry_dk,
            )
            _set_registry_dk(knowledge)

        instance = cls(config, _env=env)
        instance._domain = domain

        # Wire domain knowledge into sub-agents (instance-level DI)
        if knowledge:
            action_agent = getattr(instance, 'action_agent', None)
            if action_agent:
                action_agent.set_domain_knowledge(knowledge)
            optimizer = getattr(instance, 'optimizer', None)
            if optimizer:
                optimizer.set_domain_knowledge(knowledge)
            planner = getattr(instance, 'planner', None)
            if planner and hasattr(planner, 'set_domain_knowledge'):
                planner.set_domain_knowledge(knowledge)
            skill_mgr = getattr(instance, 'skill_manager', None)
            if skill_mgr and hasattr(skill_mgr, 'set_domain_knowledge'):
                skill_mgr.set_domain_knowledge(knowledge)
            # Modules migrated to dk_registry:
            # effects, decomposition_filter, resource_tracker, goal_planner,
            # _param_constants, prescreener, skill_gap_analyzer,
            # _environmental, code_validation, skill_selection

        # Wire skill language implementation
        lang_impl = domain.skill_language_impl
        if lang_impl:
            instance._skill_language = lang_impl  # for step_plan mixin
            _action_agent = getattr(instance, 'action_agent', None)
            if _action_agent and hasattr(_action_agent, 'set_skill_language'):
                _action_agent.set_skill_language(lang_impl)
            _optimizer = getattr(instance, 'optimizer', None)
            if _optimizer and hasattr(_optimizer, 'set_skill_language'):
                _optimizer.set_skill_language(lang_impl)
            _skill_manager = getattr(instance, 'skill_graph_manager', None)
            if _skill_manager and hasattr(_skill_manager, 'set_skill_language'):
                _skill_manager.set_skill_language(lang_impl)

        # Override core_inventory_items if domain provides a pattern
        if knowledge:
            pattern = knowledge.get_core_inventory_pattern()
            if pattern:
                config.curriculum.core_inventory_items = pattern

        # Wire curriculum strategy
        curriculum = domain.curriculum
        if curriculum:
            old_curriculum = getattr(instance, 'curriculum_agent', None)
            if old_curriculum and hasattr(curriculum, 'set_agent'):
                curriculum.set_agent(old_curriculum)
            instance.curriculum_agent = curriculum

        # Wire critic strategy
        critic = domain.critic
        if critic:
            old_critic = getattr(instance, 'critic_agent', None)
            if old_critic and hasattr(critic, 'set_agent'):
                critic.set_agent(old_critic)
            instance.critic_agent = critic

        # propagate domain knowledge to critic and curriculum agents
        if knowledge:
            critic_obj = getattr(instance, 'critic_agent', None)
            if critic_obj:
                underlying = getattr(critic_obj, 'unwrapped', critic_obj)
                if hasattr(underlying, 'set_domain_knowledge'):
                    underlying.set_domain_knowledge(knowledge)
            curriculum_obj = getattr(instance, 'curriculum_agent', None)
            if curriculum_obj:
                underlying = getattr(curriculum_obj, 'unwrapped', curriculum_obj)
                if hasattr(underlying, 'set_domain_knowledge'):
                    underlying.set_domain_knowledge(knowledge)

        return instance

    @classmethod
    def from_registered_domain(
        cls,
        domain_name: str,
        config: PSNConfig = None,
        **domain_kwargs,
    ) -> "PSNAgent":
        """Create agent from a registered domain name.

        Shortest path to a working agent::

            agent = PSNAgent.from_registered_domain("minecraft", mc_port=25565)

        Args:
            domain_name: Name of a registered domain (e.g. "minecraft").
            config: PSNConfig (defaults to PSNConfig() if omitted).
            **domain_kwargs: Forwarded to the domain factory function.
        """
        from skillnet.domains import get_domain
        domain = get_domain(domain_name, **domain_kwargs)
        return cls.from_domain(domain, config or PSNConfig())

    def reset(self, task: TaskWithSemantic, context="", reset_env=True):
        """
        Reset PSN Agent state to start a new task.

        P1 improvement: the task parameter is now a TaskWithSemantic object carrying semantic info.

        Args:
            task: TaskWithSemantic object (P1: carries semantic info)
            context: task context
            reset_env: whether to reset the environment
        """
        self.action_agent_rollout_num_iter = 0
        # Preflight (contract-gate) reject counter: should NOT consume env.step(), and should NOT
        # be counted as a normal "task attempt" unless it loops too many times.
        self._preflight_rejects_this_task = 0
        # store both the TaskWithSemantic object and string
        self._task_semantic = task
        self.task = task.task  # backward compatibility: keep self.task as a string
        self.context = context
        # Start recording trajectory
        self.trajectory_recorder.start_trajectory(task=self.task, context=context)
        if reset_env:
            self.env.reset(
                options={
                    "mode": "soft",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
        # Build reset code — domain provides ready-to-execute code
        domain = getattr(self, '_domain', None)
        reset_code = ""
        if domain and hasattr(domain, 'knowledge'):
            reset_code = domain.knowledge.get_reset_code()
        # Peek an observation. The server is already paused here, and the reset
        # commands (gamerule/difficulty/time) apply while paused, so run them
        # without unpausing — avoids an unpause/pause cycle and its idle window.
        events = self.env.step(reset_code, is_iteration=False, keep_paused=True)

        # Defensive check: env.step() may return empty results
        if not events or len(events) == 0:
            print("[Warning] env.step() returned empty events in reset(), using fallback")
            events = [("observe", {"status": {}, "inventory": {}})]

        # Save the initial task state for downstream state-change computation
        self.task_initial_events = copy.deepcopy(events)

        # Initialize tracking of task-executed skills (used by on_task_completed lifecycle management)
        self._task_executed_skills = set()

        # Retrieve skills with parameter metadata (always uses ParameterizedActionAgent + SkillGraphManager)
        skills, skill_metadata = self.skill_manager.retrieve_skills(
            query=self.context, return_metadata=True
        )
        print(
            f"\033[33mRender Parameterized Action Agent system message with {len(skills)} skills\033[0m"
        )
        system_message = self.action_agent.render_system_message(
            skills=skills, skill_metadata=skill_metadata,
            task=self.task,
        )
        human_message = self.action_agent.render_human_message(
            events=events, code="", task=self.task, context=context, critique=""
        )
        self.messages = [system_message, human_message]
        print(
            f"\033[32m****Action Agent human message****\n{human_message.content}\033[0m"
        )
        assert len(self.messages) == 2
        self.conversations = []
        return self.messages

    def close(self):
        """Shut down the PSN instance and release resources."""
        # Close the global logger
        if hasattr(self, 'global_logger') and self.global_logger:
            self.global_logger.close()
        self.env.close()

    # ----------------------------------------------------------------
    # Methods below are inherited from mixin classes in _psn_impl/:
    # EventProcessingMixin  - event state extraction, diagnostics
    # SkillRecordingMixin   - skill execution recording, reuse hints
    # EffectVerificationMixin - skill effect checking (rule + LLM)
    # StepExecutionMixin    - step(), _step_plan, _step_execute, etc.
    # TaskManagementMixin   - task normalization, failure limits
    # ----------------------------------------------------------------

    def rollout(self, *, task: TaskWithSemantic, context, reset_env=True):
        """
        Execute a complete rollout for a single task.

        P1 improvement: the task parameter is now a TaskWithSemantic object.

        Args:
            task: TaskWithSemantic object (P1: carries semantic info)
            context: task context
            reset_env: whether to reset the environment
        """
        self.reset(task=task, context=context, reset_env=reset_env)
        while True:
            messages, reward, done, info = self.step()
            if done:
                break
        return messages, reward, done, info

    @staticmethod
    def _prepare_iteration_resume(source_dir: str, target_iteration: int, dest_dir: str = None) -> str:
        """
        Prepare a new checkpoint directory to resume from a given iteration.

        Args:
            source_dir: source checkpoint directory
            target_iteration: target iteration
            dest_dir: optional destination directory name

        Returns:
            path to the new checkpoint directory
        """
        from skillnet.utils.iteration_utils import prepare_iteration_checkpoint

        if dest_dir is None:
            dest_dir = f"{source_dir}_from_{target_iteration}"

        # Create the new (truncated) checkpoint directory
        result = prepare_iteration_checkpoint(source_dir, dest_dir, target_iteration)

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            raise RuntimeError(f"Failed to prepare iteration checkpoint: {error_msg}")

        print(f"\033[35m[PSN] Resume from iteration {target_iteration}, using checkpoint: {dest_dir}\033[0m")
        return dest_dir

    def learn_setup(self, reset_env=True):
        """Initialize learning state. Call before learn_step() for manual control.

        Sets up environment, events, and internal counters. After calling this,
        use learn_step() in a loop for fine-grained iteration control.
        """
        # Save whether this is a resume (since self.resume is set to True below)
        self._is_resume_mode = self.resume
        self._learn_reset_env = reset_env

        if self.resume:
            # keep the inventory
            self.env.reset(
                options={
                    "mode": "soft",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
        else:
            # clear the inventory
            self.env.reset(
                options={
                    "mode": "hard",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
            self.resume = True
        self.last_events = self._ensure_events_list(self.env.step("", is_iteration=False))
        # Initialize task_initial_events in resume mode to ensure correct state-change computation
        self.task_initial_events = copy.deepcopy(self.last_events)

        # Reset task counter (starting from this run)
        self.task_count = 0
        self._consecutive_api_errors = 0

    def learn_step(self):
        """Execute one learning iteration.

        Returns a dict with keys:
            - ``done`` (bool): True when iteration limit reached.
            - ``task`` (str): The task attempted (absent when done/skipped).
            - ``success`` (bool): Whether the task succeeded (absent when done/skipped).
            - ``skipped`` (bool): True if the task was skipped due to failure limit.
            - ``reason`` (str): Short explanation when done or skipped.

        Call ``learn_setup()`` before the first call to ``learn_step()``.
        """
        if self.recorder.iteration > self.max_iterations:
            print("Iteration limit reached")
            return {"done": True, "reason": "iteration_limit"}

        is_resume_mode = getattr(self, '_is_resume_mode', False)
        reset_env = getattr(self, '_learn_reset_env', True)

        # Debug mode: force a specific task for deterministic testing
        # task is now a TaskWithSemantic object
        if self.debug_mode:
            debug_task = self._config.debug_task
            if is_resume_mode and self.task_count == 0:
                task = TaskWithSemantic.from_legacy_string(debug_task)
                context = f"Debug task: {debug_task}"
                print(f"\033[33m[Debug Mode] Resume mode: forcing the first task to: {task.task}\033[0m")
            elif not is_resume_mode and self.task_count == 1:
                task = TaskWithSemantic.from_legacy_string(debug_task)
                context = f"Debug task: {debug_task}"
                print(f"\033[33m[Debug Mode] Non-resume mode: forcing the second task to: {task.task}\033[0m")
            else:
                task, context = self.curriculum_agent.propose_next_task(
                    events=self.last_events,
                    chest_observation=self.action_agent.render_chest_observation(),
                    max_retries=5,
                )
        else:
            task, context = self.curriculum_agent.propose_next_task(
                events=self.last_events,
                chest_observation=self.action_agent.render_chest_observation(),
                max_retries=5,
            )

        # extract task string for backward compatibility
        task_str = task.task

        # Re-proposal loop: if task exceeds global limit, blacklist and re-propose
        blacklisted_tasks = set()
        MAX_REPROPOSE = 5

        for _repropose in range(MAX_REPROPOSE):
            normalized_task = self._normalize_task_key(task_str)
            current_attempts = self.global_task_attempt_counts.get(normalized_task, 0)
            task_limit = self._get_task_failure_limit(task_str)

            if current_attempts < task_limit:
                break  # Task is valid, proceed to execution

            # Task exceeded limit — blacklist and re-propose
            blacklisted_tasks.add(normalized_task)
            print(
                f"\033[33m[Global Limit] Task '{task_str}' blacklisted "
                f"({current_attempts}/{task_limit}), re-proposing...\033[0m"
            )

            task, context = self.curriculum_agent.propose_next_task(
                events=self.last_events,
                chest_observation=self.action_agent.render_chest_observation(),
                max_retries=5,
                blacklisted_tasks=blacklisted_tasks,
            )
            task_str = task.task
        else:
            # All MAX_REPROPOSE proposals were blacklisted — skip this iteration
            print(
                f"\033[33m[Global Limit] All {MAX_REPROPOSE} re-proposals blacklisted. "
                f"Skipping iteration.\033[0m"
            )
            self.progress_recorder.record_iteration_start(
                task=task_str,
                iteration=self.recorder.iteration,
            )
            self._record_iteration_end({
                "task": task_str,
                "success": False,
                "error_message": f"All re-proposals exceeded global limits",
            })
            return {"done": False, "task": task_str, "skipped": True,
                    "reason": "all_proposals_blacklisted"}

        # Increment global attempt counter
        self.global_task_attempt_counts[normalized_task] = current_attempts + 1

        # Increment per-task counter
        self.task_count += 1

        # Record iteration start for progress viewer
        self.progress_recorder.record_iteration_start(
            task=task_str,
            iteration=self.recorder.iteration,
        )

        # Save the pre-task inventory for capability-change detection
        old_inventory = {}
        if self.last_events and len(self.last_events) > 0:
            try:
                _obs = (
                    self._domain.knowledge.extract_observation(self.last_events)
                    if getattr(self, "_domain", None) is not None
                    else None
                )
                old_inventory = dict(_obs.inventory) if _obs is not None else {}
            except (IndexError, KeyError, TypeError):
                pass

        print(
            f"\033[35mStarting task {task_str} for at most {self.action_agent_task_max_retries} times\033[0m"
        )
        try:
            messages, reward, done, info = self.rollout(
                task=task,
                context=context,
                reset_env=reset_env,
            )
        except Exception as e:
            import traceback

            # API/infrastructure error detection
            if self._is_api_error(e):
                self._consecutive_api_errors += 1
                print(
                    f"\033[41m[API Error] {type(e).__name__}: {e} "
                    f"(consecutive: {self._consecutive_api_errors})\033[0m"
                )
                if self._consecutive_api_errors >= 3:
                    print(
                        f"\033[41m[Circuit Breaker] {self._consecutive_api_errors} "
                        f"consecutive API errors. Saving checkpoint and exiting.\033[0m"
                    )
                    # Save checkpoint before exit
                    try:
                        self.curriculum_agent.clean_up_tasks()
                    except Exception:
                        pass
                    raise SystemExit(
                        f"[Circuit Breaker] {self._consecutive_api_errors} consecutive "
                        f"API errors. Last error: {e}"
                    )

                # API error: do NOT count toward task failure limit
                # Decrement the counter that was incremented before rollout
                normalized_task = self._normalize_task_key(task_str)
                if normalized_task in self.global_task_attempt_counts:
                    self.global_task_attempt_counts[normalized_task] -= 1
                    if self.global_task_attempt_counts[normalized_task] <= 0:
                        del self.global_task_attempt_counts[normalized_task]

                info = {
                    "task": task_str,
                    "success": False,
                    "error_type": "api_error",
                }
                # Skip the heavy error diagnosis for API errors, just do env reset
                time.sleep(3)
                reset_options = {
                    "mode": "hard",
                    "wait_ticks": self.env_wait_ticks,
                }
                _obs = (
                    self._domain.knowledge.extract_observation(self.last_events)
                    if self.last_events and getattr(self, "_domain", None) is not None
                    else None
                )
                if _obs is not None and _obs.extra:
                    reset_options["inventory"] = dict(_obs.inventory)
                    reset_options["equipment"] = _obs.extra.get("equipment", [])
                    reset_options["position"] = (
                        _obs.extra.get("raw_observe", {})
                        .get("status", {})
                        .get("position", None)
                    )
                self.last_events = self._ensure_events_list(self.env.reset(options=reset_options))
            else:
                # Non-API error: existing error handling
                time.sleep(3)  # wait for mineflayer to exit

                # Extract exception type for lifecycle management
                error_type = type(e).__name__

                info = {
                    "task": task_str,
                    "success": False,
                    "error_type": error_type,  # store the exception type so we can tell if it is a system-level error
                }

                # Detailed error diagnostics
                error_msg = str(e)
                error_traceback = traceback.format_exc()

                # Diagnose the last_events structure
                events_diagnosis = self._diagnose_events_structure(self.last_events)

                print("\033[41m" + "="*80 + "\033[0m")
                print("\033[41m[ERROR DIAGNOSIS] Rollout terminated due to error\033[0m")
                print("\033[41m" + "="*80 + "\033[0m")
                print(f"\033[31mError Type: {error_type}\033[0m")
                print(f"\033[31mError Message: {error_msg}\033[0m")
                print(f"\033[31mTask: {task_str}\033[0m")
                print(f"\033[31mContext: {context}\033[0m")
                print(f"\033[33m\nEvents Structure Diagnosis:\033[0m")
                print(events_diagnosis)
                print(f"\033[33m\nFull Traceback:\033[0m")
                print(error_traceback)
                print("\033[41m" + "="*80 + "\033[0m\n")

                # reset bot status here
                reset_options = {
                    "mode": "hard",
                    "wait_ticks": self.env_wait_ticks,
                }
                # Safely get inventory, equipment, and position via the
                # active DomainKnowledge so this path stays domain-agnostic.
                _obs = (
                    self._domain.knowledge.extract_observation(self.last_events)
                    if self.last_events and getattr(self, "_domain", None) is not None
                    else None
                )
                if _obs is not None and _obs.extra:
                    reset_options["inventory"] = dict(_obs.inventory)
                    reset_options["equipment"] = _obs.extra.get("equipment", [])
                    reset_options["position"] = (
                        _obs.extra.get("raw_observe", {})
                        .get("status", {})
                        .get("position", None)
                    )
                self.last_events = self._ensure_events_list(self.env.reset(options=reset_options))
                # use red color background to print the error
                print("Your last round rollout terminated due to error:")
                print(f"\033[41m{e}\033[0m")

        if info["success"]:
            # Reset consecutive API error counter on success
            self._consecutive_api_errors = 0

            # Check if the skill should be saved (may be False if it's just calling an existing skill)
            should_save = info.get("should_save_skill", True)
            if should_save:
                # Remember the original name so we can sync _task_executed_skills after add_new_skill renames
                original_program_name = info.get("program_name")

                # Confirm pending code (if any)
                # Handles the case where a stable skill is updated by the LLM
                if original_program_name and self.skill_manager.has_node(original_program_name):
                    node = self.skill_manager.get_node(original_program_name)
                    if node.has_pending_code():
                        result = node.confirm_pending(success=True)
                        if result.adopted:
                            print(f"\033[32m[PSN] Pending code adopted: {original_program_name}\033[0m")
                        elif result.message:
                            print(f"\033[33m[PSN] {result.message}\033[0m")

                # v3.H+ gate: skip redundant add_new_skill when the skill was
                # canonicalized by refactor earlier in this step.
                #
                # Two cases collapse here:
                # (a) sibling / parametric / duplication: skill is now a
                # wrapper (is_covered=True). Writing info["program_code"]
                # (action_agent's original inline) would force Layer 3 to
                # block — we avoid the wasted work and the noisy log.
                # (b) behavioral / extract_common: skill code was modified
                # (no is_covered flag) but tracking set knows. Writing
                # the inline back would destroy the refactor's edit.
                # Layer 3 wrapper protection (graph_manager_impl.py) remains as
                # defense-in-depth for any other code path.
                skip_save_v3h = False
                if original_program_name and self.skill_manager.has_node(original_program_name):
                    existing_node = self.skill_manager.get_node(original_program_name)
                    if getattr(existing_node, 'is_covered', False):
                        skip_save_v3h = True
                        covered_by = getattr(existing_node, 'covered_by', '?')
                        print(
                            f"\033[36m[Step End v3.H] Skipping add_new_skill for "
                            f"'{original_program_name}': skill is now a wrapper of "
                            f"'{covered_by}' (refactor canonicalized this step). "
                            f"Layer 3 protection unchanged.\033[0m"
                        )
                    elif (hasattr(self.skill_manager, 'was_modified_by_refactor_this_step')
                          and self.skill_manager.was_modified_by_refactor_this_step(original_program_name)):
                        skip_save_v3h = True
                        print(
                            f"\033[36m[Step End v3.H] Skipping add_new_skill for "
                            f"'{original_program_name}': skill code was modified by "
                            f"refactor in this step (behavioral / extract_common). "
                            f"Re-writing action_agent inline would undo the refactor.\033[0m"
                        )

                if not skip_save_v3h:
                    self.skill_manager.add_new_skill(info)

                # add_new_skill may rename the skill (semantic rename, type normalization, etc.)
                # Sync _task_executed_skills so on_task_completed uses the correct name
                final_program_name = info.get("program_name")
                if final_program_name and final_program_name != original_program_name:
                    if hasattr(self, '_task_executed_skills') and original_program_name in self._task_executed_skills:
                        self._task_executed_skills.discard(original_program_name)
                        self._task_executed_skills.add(final_program_name)
                        print(f"\033[36m[Skill Rename Sync] Sync _task_executed_skills after add_new_skill rename: '{original_program_name}' → '{final_program_name}'\033[0m")
            else:
                print(f"\033[33m[Skill Manager] Skipping skill save because the task simply reused an existing skill\033[0m")

            # After task success, clear the per-task failure count so it can be retried later
            normalized_task = self._normalize_task_key(task_str)
            if normalized_task in self.global_task_attempt_counts:
                del self.global_task_attempt_counts[normalized_task]
                print(f"\033[32m[Task Counter] Cleared failure count for '{task_str}' after success\033[0m")

            # Also clean up matching entries in failed_tasks to keep data consistent
            # Use normalized comparison; remove all matching failure records
            self.curriculum_agent.remove_matching_failed_tasks(
                lambda t: self._normalize_task_key(t) == normalized_task
            )

        # Detect capability changes (newly acquired tools) and reset related task counters
        new_inventory = {}
        if self.last_events and len(self.last_events) > 0:
            try:
                _obs = (
                    self._domain.knowledge.extract_observation(self.last_events)
                    if getattr(self, "_domain", None) is not None
                    else None
                )
                new_inventory = dict(_obs.inventory) if _obs is not None else {}
            except (IndexError, KeyError, TypeError):
                pass
        self._check_and_reset_on_capability_change(old_inventory, new_inventory)

        # Log Level 2 diagnosis if SEM detected systemic pattern
        if not info["success"] and getattr(self, '_level2_diagnosis', None):
            diag = self._level2_diagnosis
            print(
                f"\033[35m[SEM Level 2] Systemic pattern for '{task_str}': "
                f"{diag.common_cause}\033[0m"
            )
            if diag.proposed_capability:
                print(
                    f"\033[35m[SEM Level 2] Proposed action: "
                    f"{diag.systemic_action.value} — {diag.proposed_capability}\033[0m"
                )
            self._level2_diagnosis = None  # Consumed

        # Shared post-rollout processing
        self._finalize_rollout(self.task, info)

        # Record iteration end for progress viewer
        self._record_iteration_end(info)

        return {"done": False, "task": task_str, "success": info["success"]}

    def _finalize_rollout(self, task, info: dict) -> None:
        """Shared post-rollout processing for both learn() and inference().

        Handles:
        - Curriculum exploration progress update
        - Adaptive planner task result notification
        - Experimental skill lifecycle management (on_task_completed)
        - Clearing per-task executed skills tracking
        - Logging completed/failed task lists

        Args:
            task: The task identifier (TaskWithSemantic from learn(), str from inference())
            info: Rollout result dict with at least 'success' key
        """
        self.curriculum_agent.update_exploration_progress(info)

        # notify adaptive planner of task-completion status
        if hasattr(self.curriculum_agent, 'update_task_result'):
            self.curriculum_agent.update_task_result(task, info["success"])

        # Handle experimental-skills lifecycle after task completion (always uses SkillGraphManager)
        # Use the optimizer engine's full implementation to correctly handle:
        # - On success: validate the skills that participated in success, trigger refactor, clean up unused experimental skills
        # - On failure: clean up all experimental skills created for this task
        completing_skills = list(getattr(self, '_task_executed_skills', set()))

        # Prefer the optimizer's full version (which includes completing_skills and error_type parameters)
        if self.optimizer and hasattr(self.optimizer, 'on_task_completed'):
            self.optimizer.on_task_completed(
                task=task,
                success=info["success"],
                completing_skills=completing_skills,
                error_type=info.get("error_type"),
            )
        elif hasattr(self.skill_manager, 'on_task_completed'):
            # Fall back to the graph_manager version (functionality is incomplete)
            self.skill_manager.on_task_completed(
                task=task,
                success=info["success"],
                error_type=info.get("error_type"),
            )

        # Clear task-scoped parameter corrections from optimizer feedback loop.
        # Bug 8 fix: only clear on success. On failure, keep corrections so the
        # next task instance (when curriculum reissues the same conceptual task)
        # can benefit from Phase 1's prior diagnoses — without this, the same
        # correction was rederived 11 times across 5 independent oak_log tasks
        # in the May 19 diag run because each task ended with a wipe. The
        # per-key cap at metadata_crud.py:283 (keep last 3) prevents unbounded
        # growth across long failure streaks.
        if info.get("success") and hasattr(self.skill_manager, 'clear_task_parameter_corrections'):
            self.skill_manager.clear_task_parameter_corrections()

        # Clear task-level tracking
        self._task_executed_skills = set()

        print(
            f"\033[35mCompleted tasks: {', '.join(self.curriculum_agent.get_completed_tasks())}\033[0m"
        )
        print(
            f"\033[35mFailed tasks: {', '.join(self.curriculum_agent.get_failed_tasks())}\033[0m"
        )

    def learn(self, reset_env=True):
        """Full learning loop. Equivalent to learn_setup() + repeated learn_step()."""
        self.learn_setup(reset_env)
        while True:
            result = self.learn_step()
            if result.get("done"):
                break
        return {
            "completed_tasks": self.curriculum_agent.get_completed_tasks(),
            "failed_tasks": self.curriculum_agent.get_failed_tasks(),
            "skills": self.skill_manager.skills,
        }

    def decompose_task(self, task):
        if not self.last_events:
            self.last_events = self._ensure_events_list(self.env.reset(
                options={
                    "mode": "hard",
                    "wait_ticks": self.env_wait_ticks,
                }
            ))
        return self.curriculum_agent.decompose_task(task, self.last_events)

    def inference(self, task=None, sub_goals=[], reset_mode="hard", reset_env=True):
        if not task and not sub_goals:
            raise ValueError("Either task or sub_goals must be provided")
        if not sub_goals:
            sub_goals = self.decompose_task(task)
        self.env.reset(
            options={
                "mode": reset_mode,
                "wait_ticks": self.env_wait_ticks,
            }
        )
        self.curriculum_agent.reset_task_lists()
        self.last_events = self._ensure_events_list(self.env.step("", is_iteration=False))
        while self.curriculum_agent.progress < len(sub_goals):
            next_task = sub_goals[self.curriculum_agent.progress]
            # get_task_context expects a string, not TaskWithSemantic
            task_str = next_task.task if hasattr(next_task, 'task') else str(next_task)
            context = self.curriculum_agent.get_task_context(task_str)
            print(
                f"\033[35mStarting task {task_str} for at most {self.action_agent_task_max_retries} times\033[0m"
            )
            messages, reward, done, info = self.rollout(
                task=next_task,
                context=context,
                reset_env=reset_env,
            )
            self._finalize_rollout(task_str, info)
