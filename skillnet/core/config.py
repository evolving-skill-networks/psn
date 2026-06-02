"""
PSN Configuration Dataclasses

Structured configuration for PSNAgent, replacing the 50+ kwargs constructor.
All default values are synchronized with PSNAgent.__init__() defaults.

Usage:
    from skillnet.core import PSNConfig

    # Use all defaults
    config = PSNConfig()

    # Customize specific fields
    config = PSNConfig(
        llm=AgentLLMs(
            action=LLMEndpoint(model_name="gpt-5-mini", temperature=0.7),
        ),
        checkpoint=CheckpointConfig(ckpt_dir="ckpt_experiment_1", resume=True),
        optimization=OptimizationConfig(enable_refactor=False),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from skillnet.core.domain import DomainModule


@dataclass
class LLMEndpoint:
    """Configuration for a single LLM endpoint.

    API base/key are resolved automatically by create_chat_llm() from
    environment variables ({COMPONENT}_API_BASE → VLLM_API_BASE).
    """
    model_name: str = "gpt-5-mini"
    temperature: float = 0.0
    request_timeout: int = 1200
    max_tokens: Optional[int] = None  # None = model/server default


@dataclass
class AgentLLMs:
    """LLM configurations for all sub-agents."""
    action: LLMEndpoint = field(default_factory=LLMEndpoint)
    curriculum: LLMEndpoint = field(default_factory=LLMEndpoint)
    curriculum_qa: LLMEndpoint = field(default_factory=LLMEndpoint)
    critic: LLMEndpoint = field(default_factory=LLMEndpoint)
    skill_manager: LLMEndpoint = field(default_factory=LLMEndpoint)
    optimizer: LLMEndpoint = field(default_factory=LLMEndpoint)


@dataclass
class ActionConfig:
    """Action agent behavior configuration."""
    # Per-task retry budget. Optimizer cycles between attempts can take 2+
    # attempts to produce a committable code version (Phase 1 attribution +
    # Phase 2 patch validation each contribute latency). At 4 retries the
    # optimizer's first successful version typically lands too late to be
    # exercised before curriculum gives up; 5 retries gives one extra
    # attempt to run whatever the optimizer just committed.
    task_max_retries: int = 5
    show_chat_log: bool = True
    show_execution_error: bool = True
    use_llm_for_normalization: bool = True
    include_skill_code: bool = False


@dataclass
class CurriculumConfig:
    """Curriculum agent configuration."""
    mode: str = "auto"
    warm_up: Optional[Dict[str, int]] = None
    core_inventory_items: str = (
        r".*_log|.*_planks|stick|crafting_table|furnace"
        r"|cobblestone|dirt|coal|.*_pickaxe|.*_sword|.*_axe"
    )


@dataclass
class CriticConfig:
    """Critic agent configuration."""
    mode: str = "auto"


@dataclass
class SkillManagerConfig:
    """Skill manager configuration."""
    retrieval_top_k: int = 5
    merge_mode: str = "llm"
    max_skills: Optional[int] = None  # None = unlimited; cap graph size for experiments


@dataclass
class PlannerConfig:
    """Planner configuration."""
    mode: str = "adaptive"
    min_skills_for_graph_planning: int = 3


@dataclass
class OptimizationConfig:
    """Optimization-related configuration."""
    enable_optimizer: bool = True
    enable_refactor: bool = True
    use_llm_effect_verification: bool = True
    pure_reasoning: bool = False
    # B1: Maturity gating parameters for P(update s) = (1-ε)·σ(γ(θ-V(s)))+ε
    optimization_threshold: float = 0.6   # θ — pivot point
    optimization_epsilon: float = 0.05    # ε — floor probability
    optimization_gamma: float = 8.0       # γ — sigmoid slope
    # Escalation protocol: Level 0→1 when K failures with no V(s) improvement
    escalation_min_attempts: int = 3      # K — min failures before considering escalation
    escalation_improvement_threshold: float = 0.05  # δ — V(s) delta for "improvement"
    # Gates the reasoning_examples_section in the Phase 1 prompt. Default
    # off — a cross-LLM ablation on Qwen3 + gpt-5-mini showed 0 diagnostic-
    # correctness delta with the guides enabled.
    include_reasoning_examples: bool = False
    # Phase 13.B-9 R4/R5/R6: Tier-2 factual primitive-doc injection.
    # Reduces Phase 1 proximity-validation hallucination ~5-8pp on the
    # ensureCobblestone repro (qwen3 68→60%, gpt5m 63→60% on n=120 each
    # condition). MC R6 (n=5, p=0.008) confirms the bug pattern causes
    # 100% real failure when triggered, so even modest prevention is
    # high-value. TPR 30/30 preserved across all reproducer conditions.
    # Default ON since R6 — researchers can opt-out via False.
    use_factual_primitive_doc: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint and resume configuration."""
    ckpt_dir: str = "ckpt"
    skill_library_dir: Optional[str] = None
    resume: bool = False
    resume_from_iteration: Optional[int] = None
    resume_to_dir: Optional[str] = None
    graph_snapshot_interval: int = 0  # 0=disabled; N=snapshot every N iterations


@dataclass
class RecordingConfig:
    """Trajectory recording configuration."""
    save_trajectories: bool = True


@dataclass
class EnvironmentConfig:
    """Environment connection configuration."""
    wait_ticks: int = 20
    request_timeout: int = 900
    server_port: int = 3000
    view_distance: int = 6
    reset_placed_if_failed: bool = False


@dataclass
class PSNConfig:
    """
    Top-level PSN configuration.

    Groups all PSNAgent configuration into semantic sub-configs.
    Default values match PSNAgent.__init__() exactly.
    """
    llm: AgentLLMs = field(default_factory=AgentLLMs)
    action: ActionConfig = field(default_factory=ActionConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    skill_manager: SkillManagerConfig = field(default_factory=SkillManagerConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    max_iterations: int = 3160
    debug_mode: bool = False
    debug_task: str = "mine 2 Oak Logs"
    global_task_failure_limit: int = 8
    domain: Optional[DomainModule] = None

    # -- Presets --------------------------------------------------------

    @classmethod
    def quick_test(cls, **overrides) -> PSNConfig:
        """Minimal config for quick testing (10 iterations, no optimization)."""
        config = cls(
            max_iterations=10,
            debug_mode=True,
            optimization=OptimizationConfig(
                enable_optimizer=False, enable_refactor=False,
            ),
        )
        return replace(config, **overrides) if overrides else config

    @classmethod
    def no_optimization(cls, **overrides) -> PSNConfig:
        """Standard config with optimizer/refactor disabled."""
        config = cls(
            optimization=OptimizationConfig(
                enable_optimizer=False, enable_refactor=False,
            ),
        )
        return replace(config, **overrides) if overrides else config

    @classmethod
    def production(cls, **overrides) -> PSNConfig:
        """Full optimization, high iterations, trajectory recording."""
        config = cls(
            max_iterations=3160,
            recording=RecordingConfig(save_trajectories=True),
        )
        return replace(config, **overrides) if overrides else config

    # -- Composition helpers --------------------------------------------

    def with_llm(self, model_name: str, **endpoint_kwargs) -> PSNConfig:
        """Return a new config with all LLM endpoints set to the given model."""
        endpoint = LLMEndpoint(model_name=model_name, **endpoint_kwargs)
        return replace(self, llm=AgentLLMs(
            action=endpoint, curriculum=endpoint, curriculum_qa=endpoint,
            critic=endpoint, skill_manager=endpoint, optimizer=endpoint,
        ))

