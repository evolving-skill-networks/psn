# Config Architecture

## Overview

`PSNConfig` replaces the original `PSNAgent.__init__()` constructor that accepted 50+ keyword arguments. Configuration is organized into semantic sub-dataclasses, each grouping related settings. All dataclasses live in `skillnet/core/config.py` and use `dataclasses.dataclass` with sensible defaults synchronized with the legacy constructor.

Three initialization paths exist for `PSNAgent`:

| Path | Signature | Use case |
|------|-----------|----------|
| Direct | `PSNAgent(config=PSNConfig())` | Simple construction |
| Domain | `PSNAgent.from_domain(domain, config)` | Production with domain module |
| Components | `PSNAgent.from_components(pre_built, config)` | Testing with pre-built agents |

## PSNConfig Structure

`PSNConfig` is the top-level dataclass. It composes all sub-configs and adds a handful of agent-wide scalars.

```
PSNConfig
 ├── llm: AgentLLMs
 │    ├── action: LLMEndpoint
 │    ├── curriculum: LLMEndpoint
 │    ├── curriculum_qa: LLMEndpoint
 │    ├── critic: LLMEndpoint
 │    ├── skill_manager: LLMEndpoint
 │    └── optimizer: LLMEndpoint
 ├── action: ActionConfig
 ├── curriculum: CurriculumConfig
 ├── critic: CriticConfig
 ├── skill_manager: SkillManagerConfig
 ├── planner: PlannerConfig
 ├── optimization: OptimizationConfig
 ├── checkpoint: CheckpointConfig
 ├── recording: RecordingConfig
 ├── environment: EnvironmentConfig
 ├── max_iterations: int
 ├── debug_mode: bool
 ├── debug_task: str
 ├── global_task_failure_limit: int
 └── domain: Optional[DomainModule]
```

### Top-level scalar fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iterations` | `int` | `3160` | Maximum learning loop iterations |
| `debug_mode` | `bool` | `False` | Enable debug mode (uses `debug_task`) |
| `debug_task` | `str` | `"mine 2 Oak Logs"` | Task used when `debug_mode=True` |
| `global_task_failure_limit` | `int` | `8` | Max consecutive failures before abandoning a task |
| `domain` | `Optional[DomainModule]` | `None` | Domain module for domain-agnostic operation |

## Sub-dataclass Details

### LLMEndpoint

Configuration for a single LLM endpoint. API base/key are resolved at runtime by `create_chat_llm()` from environment variables (see Environment Variable Resolution below).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"gpt-5-mini"` | Model identifier |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `request_timeout` | `int` | `1200` | API request timeout in seconds |

### AgentLLMs

Holds one `LLMEndpoint` per sub-agent. All default to `LLMEndpoint()`.

| Field | Type | Description |
|-------|------|-------------|
| `action` | `LLMEndpoint` | Skill generation agent |
| `curriculum` | `LLMEndpoint` | Curriculum task generation |
| `curriculum_qa` | `LLMEndpoint` | Curriculum QA verification |
| `critic` | `LLMEndpoint` | Execution critic |
| `skill_manager` | `LLMEndpoint` | Skill retrieval and management |
| `optimizer` | `LLMEndpoint` | Optimization reflection |

### ActionConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_max_retries` | `int` | `4` | Max retries per task |
| `show_chat_log` | `bool` | `True` | Display LLM chat logs |
| `show_execution_error` | `bool` | `True` | Display execution errors |
| `use_llm_for_normalization` | `bool` | `True` | Use LLM for skill name normalization |
| `include_skill_code` | `bool` | `False` | Include skill source code in prompts |

### CurriculumConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `str` | `"auto"` | Curriculum mode |
| `warm_up` | `Optional[Dict[str, int]]` | `None` | Warm-up task mapping |
| `core_inventory_items` | `str` | `r".*_log\|.*_planks\|..."` | Regex for core inventory items |

### CriticConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `str` | `"auto"` | Critic evaluation mode |

### SkillManagerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `retrieval_top_k` | `int` | `5` | Number of skills to retrieve |
| `merge_mode` | `str` | `"llm"` | Skill merge strategy |

### PlannerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `str` | `"adaptive"` | Planning mode |
| `min_skills_for_graph_planning` | `int` | `3` | Minimum skills before enabling graph-based planning |

### OptimizationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_optimizer` | `bool` | `True` | Enable two-phase optimization |
| `enable_refactor` | `bool` | `True` | Enable post-optimization refactoring |
| `use_llm_effect_verification` | `bool` | `True` | Use LLM to verify skill effects |
| `pure_reasoning` | `bool` | `False` | Remove domain data injection (for evaluation) |

### CheckpointConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ckpt_dir` | `str` | `"ckpt"` | Checkpoint directory path |
| `skill_library_dir` | `Optional[str]` | `None` | External skill library to load |
| `resume` | `bool` | `False` | Resume from existing checkpoint |
| `resume_from_iteration` | `Optional[int]` | `None` | Resume from a specific iteration |
| `resume_to_dir` | `Optional[str]` | `None` | Target directory when resuming from iteration |

### RecordingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_trajectories` | `bool` | `True` | Save execution trajectories |

### EnvironmentConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wait_ticks` | `int` | `20` | Ticks to wait between actions |
| `request_timeout` | `int` | `900` | Environment request timeout in seconds |
| `server_port` | `int` | `3000` | Mineflayer server port |
| `view_distance` | `int` | `6` | Bot view distance in chunks |
| `reset_placed_if_failed` | `bool` | `False` | Reset placed blocks on task failure |

## Factory Methods

`PSNConfig` provides three class-method presets and one composition helper. All presets accept `**overrides` applied via `dataclasses.replace()`.

| Method | Key Settings | Purpose |
|--------|-------------|---------|
| `PSNConfig.quick_test()` | `max_iterations=10`, `debug_mode=True`, optimizer and refactor disabled | Fast iteration during development |
| `PSNConfig.no_optimization()` | Optimizer and refactor disabled, all else default | Baseline runs without optimization |
| `PSNConfig.production()` | `max_iterations=3160`, `save_trajectories=True` | Full production training |

The `with_llm(model_name, **endpoint_kwargs)` instance method returns a new config with all six LLM endpoints set to the same model, useful for single-model deployments.

## Config Access Pattern

`PSNAgent` stores the config as `self._config` and unpacks frequently-used values during `__init__`:

```python
def __init__(self, config: PSNConfig = None, *, mc_port=None, _env=None):
    if config is None:
        config = PSNConfig()
    self._config = config

    # Unpack frequently-used config sections
    ckpt_dir = config.checkpoint.ckpt_dir
    resume = config.checkpoint.resume
    request_timeout = config.llm.action.request_timeout
    ...
```

Sub-agents receive individual fields from the config during construction (e.g., `config.llm.action.model_name`, `config.action.show_chat_log`). The config object itself is not passed to sub-agents.

Mixins and methods that need config access later (e.g., `debug_task` in manual mode) read from `self._config` directly.

## Optimizer Sub-configs

Defined in `skillnet/agents/optimizer/config.py`. These are independent from `PSNConfig` and used internally by the optimization engine.

### TwoPhaseOptimizationConfig

Defined in `skillnet/agents/optimizer/engine.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_reflection_depth` | `int` | `3` | Maximum depth for top-down reflection |
| `enable_transactions` | `bool` | `True` | Enable transactional optimization with rollback |
| `auto_rollback_on_failure` | `bool` | `True` | Automatically rollback failed optimizations |
| `enable_post_optimization_refactor` | `bool` | `True` | Run refactor detection after optimization |
| `refactor_min_success_count` | `int` | `3` | Successes required before triggering refactor |
| `refactor_types_enabled` | `List[str]` | `["parametric", "behavioral", "merge_siblings", "duplication", "extract_common"]` | Active refactor strategies |
| `pure_reasoning` | `bool` | `False` | Disable domain data injection for evaluation |

### SkipOptimizationConfig

Controls the probabilistic skip mechanism: `P(optimize s) = (1 - epsilon) * sigmoid(gamma * (threshold - V(s))) + epsilon`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `MAX_VERSIONS_BEFORE_SKIP` | `int` | `20` | Hard skip after this many versions |
| `OPTIMIZATION_EPSILON` | `float` | `0.05` | Minimum optimization probability |
| `OPTIMIZATION_GAMMA` | `float` | `8.0` | Sigmoid slope parameter |
| `OPTIMIZATION_THRESHOLD` | `float` | `0.6` | V(s) threshold (below = likely optimize) |
| `ENABLE_PROBABILITY_SKIP` | `bool` | `True` | Toggle probabilistic skipping |

### BloatPreventionConfig

Prevents unbounded code growth during optimization.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `GROWTH_HARD_LIMIT_RATIO` | `float` | `3.0` | Absolute rejection above 300% growth |
| `GROWTH_SOFT_LIMIT_RATIO` | `float` | `2.0` | Warning above 200% growth |
| `COMPLEX_GROWTH_LIMIT_RATIO` | `float` | `1.75` | Max growth for complex skills (>100 lines) |
| `NORMAL_GROWTH_LIMIT_RATIO` | `float` | `2.0` | Max growth for normal skills (25-100 lines) |
| `MEDIUM_FIX_OVERRIDE_RATIO` | `float` | `3.0` | Override limit for MEDIUM priority fixes |
| `ABSOLUTE_MAX_LINES` | `int` | `800` | Hard cap on any skill's line count |
| `WRAPPER_MAX_LINES` | `int` | `80` | Max lines for wrapper skills |
| `WRAPPER_LINE_THRESHOLD` | `int` | `25` | Below this = wrapper skill |
| `WRAPPER_AWAIT_THRESHOLD` | `int` | `2` | Max awaits in a wrapper |
| `MAX_NEW_HELPERS` | `int` | `2` | Max new helper functions per optimization |
| `COVERED_WRAPPER_MAX_LINES` | `int` | `20` | Stricter limit for covered wrappers |
| `COVERED_WRAPPER_MAX_GROWTH` | `float` | `1.5` | Stricter growth limit for covered wrappers |
| `RETRY_ENABLED` | `bool` | `True` | Enable retry on bloat rejection |
| `RETRY_MAX_DIFF_LINES` | `int` | `10` | Max diff lines allowed on retry |
| `STATS_ENABLED` | `bool` | `True` | Track bloat statistics |
| `STATS_FILE` | `str` | `"bloat_stats.json"` | Statistics output file |

### SemanticDetectionConfig

Detects parameter semantic mismatches (e.g., DELTA vs TARGET_TOTAL).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `CHILD_SUCCESS_PATTERNS` | `List[str]` | 5 regex patterns | Patterns indicating child skill "success" |
| `PARENT_FAIL_PATTERNS` | `List[str]` | 6 regex patterns | Patterns indicating parent skill failure |
| `SEMANTIC_MISMATCH_KEYWORDS` | `List[str]` | 7 keywords | Keywords suggesting semantic mismatch |

### ExecutionValidationConfig

Validates whether child skills were actually executed.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `TIMESTAMP_MATCH_WINDOW` | `int` | `300` | Seconds within which executions are considered co-occurring |
| `MAX_TRACES_TO_CHECK` | `int` | `10` | Number of traces to examine |
| `FEEDBACK_RELEVANCE_THRESHOLD` | `float` | `0.5` | Minimum relevance score for feedback |

## Environment Variable Resolution

LLM endpoint resolution is handled by `create_chat_llm()` in `skillnet/utils/llm_factory.py`. When the `component` parameter is provided, API base and key are resolved from environment variables in a fallback chain:

```
API base:  {COMPONENT}_API_BASE  -->  VLLM_API_BASE  -->  None (OpenAI default)
API key:   {COMPONENT}_API_KEY   -->  VLLM_API_KEY   -->  None
```

For example, with `component="action_agent"`:
1. Check `ACTION_AGENT_API_BASE` -- if set, use it
2. Check `VLLM_API_BASE` -- if set, use it
3. Fall through to `None` -- use the default OpenAI API

When a custom `openai_api_base` is provided, `tiktoken_model_name` is set to `"gpt-4"` to suppress tiktoken warnings for non-OpenAI model names, and `openai_api_key` defaults to `"EMPTY"` (the OpenAI client requires a key even for local endpoints).

The timeout is structured as an `httpx.Timeout` with differentiated values:

| Component | Timeout | Purpose |
|-----------|---------|---------|
| `connect` | 10s | Fast detection of unreachable endpoints |
| `read` | `request_timeout` param | Allow time for long token generation |
| `write` | 30s | Standard write timeout |
| `pool` | 30s | Connection pool timeout |

Temperature adjustment is applied via `get_temperature_for_model()` in `skillnet/utils/model_utils.py`, which forces `temperature=1.0` for models that only support default temperature (e.g., `gpt-5-mini`, `gpt-5-nano`, `gpt-5`).
