# Skill Graph & Evolution

This document describes the skill graph data model, the value function that governs
optimization decisions, the versioning and gradient systems, and the persistence
layer that checkpoints the graph to disk.

---

## 1. Overview

The skill graph is the persistent knowledge base of the SkillNet agent.
It is a **directed acyclic graph (DAG)** where each node represents a learned
skill and each edge represents a caller-to-callee dependency.

Every skill `s` is defined as a quadruple aligned with the PSN formalism:

```
s = (C_s, P_s, E_s, Children(s))
```

| Symbol        | Field                 | Meaning                                        |
|---------------|-----------------------|------------------------------------------------|
| `C_s`         | `code`                | JavaScript async function executed in Mineflayer |
| `P_s`         | `parameters`          | Typed parameter metadata with semantic annotations |
| `E_s`         | `(preconditions, expected_effects)` | Pre- and post-conditions            |
| `Children(s)` | `children`            | Ordered list of child skills called by `s`     |

The graph is managed by `SkillGraphManager`, which wraps a `SkillGraph` instance
and provides facade methods for node access, edge management, persistence, and
semantic operations.

**Source files:**
- `skillnet/agents/skill_graph/models/node.py` -- `SkillNode`
- `skillnet/agents/skill_graph/models/graph.py` -- `SkillGraph`
- `skillnet/agents/skill_graph/_impl/graph_manager_impl.py` -- `SkillGraphManager`
- `skillnet/agents/skill_graph/_impl/mixins/graph_queries.py` -- `GraphQueriesMixin`

---

## 2. SkillNode Model

`SkillNode` is the central data class. It is **not** a `@dataclass`; it uses a
manual `__init__` for flexibility with optional fields and backward compatibility.

### Core fields (paper-aligned)

| Field               | Type                        | Description                                |
|---------------------|-----------------------------|--------------------------------------------|
| `name`              | `str`                       | Unique identifier (also the JS function name) |
| `code`              | `str`                       | Skill code body (`C_s`); a JS async function in the Minecraft domain |
| `description`       | `str`                       | Natural-language summary                   |
| `parameters`        | `Dict[str, Dict[str, Any]]` | Parameter metadata (`P_s`), see below      |
| `preconditions`     | `List[SkillPrecondition]`   | Execution preconditions (`E_s^pre`)        |
| `expected_effects`  | `List[SkillEffect]`         | Expected post-conditions (`E_s^post`)      |
| `children`          | `List[str]`                 | Ordered child skill names (`Children(s)`)  |
| `parents`           | `List[str]`                 | Parent skill names (callers)               |

### Graph structure fields

| Field        | Type   | Description                     |
|--------------|--------|---------------------------------|
| `in_degree`  | `int`  | Number of parent skills         |
| `out_degree` | `int`  | Number of child skills          |

### Statistics and gradients

| Field          | Type               | Description                              |
|----------------|---------------------|------------------------------------------|
| `statistics`   | `SkillStatistics`   | Execution counters and trace history     |
| `gradients`    | `SkillGradients`    | Accumulated feedback for optimization    |
| `versions`     | `List[SkillVersion]`| Full version history                     |

### Value function parameters

| Field                    | Default | Description                        |
|--------------------------|---------|------------------------------------|
| `value_function_lambda`  | `1.0`   | Uncertainty penalty coefficient    |
| `value_function_alpha`   | `1.0`   | Bayesian success prior             |
| `value_function_beta`    | `1.0`   | Bayesian failure prior             |

### Optimization control

| Field             | Type   | Default | Description                              |
|-------------------|--------|---------|------------------------------------------|
| `stop_gradient`   | `bool` | `False` | If `True`, gradient propagation halts here |

### Lifecycle and coverage fields

| Field                          | Type                | Description                                      |
|--------------------------------|---------------------|--------------------------------------------------|
| `is_covered`                   | `bool`              | Whether another skill functionally covers this one |
| `covered_by`                   | `Optional[str]`     | Name of the covering general skill               |
| `coverage_type`                | `Optional[CoverageType]` | Enum: PARAMETRIC, SIBLING, DUPLICATION, etc. |
| `is_general_skill`             | `bool`              | Whether this is a generalized (parameterized) skill |
| `is_experimental`              | `bool`              | Unverified new skill (failed on first execution) |
| `is_task_specific`             | `bool`              | Non-reusable composition (memory-only, not persisted) |
| `is_deprecated`                | `bool`              | Deprecated (legacy flag; the current pipeline removes skills via deletion instead) |
| `consecutive_failures`         | `int`               | Running count of consecutive failures            |
| `deprecation_threshold`        | `int`               | Failures before deprecation candidate (default 3) |

### SkillStatistics

Defined in `skillnet/agents/skill_graph/models/execution.py`.

```python
class SkillStatistics:
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    execution_traces: List[SkillExecutionTrace]  # capped at 100
```

**`success_rate` property:**

```
success_rate = successful_executions / total_executions   (0.0 if no executions)
```

**NOT_EXECUTED handling:** Traces with `execution_status == "not_executed"` are
recorded for audit purposes but are **not** counted in `total_executions`,
`successful_executions`, or `failed_executions`. This prevents skills that were
called but short-circuited (e.g., early return, conditional skip) from distorting
the success rate.

### SkillExecutionTrace

Each execution produces a `SkillExecutionTrace` record:

| Field               | Type                     | Description                              |
|---------------------|--------------------------|------------------------------------------|
| `execution_id`      | `str`                    | Unique ID (auto-generated)               |
| `timestamp`         | `str`                    | ISO 8601 timestamp                       |
| `success`           | `bool`                   | Whether the execution succeeded          |
| `execution_status`  | `str`                    | `success` / `failed` / `not_executed` / `interrupted` |
| `error_message`     | `Optional[str]`          | Error details on failure                 |
| `error_stack`       | `Optional[str]`          | JS stack trace (separate for attribution)|
| `critique`          | `Optional[str]`          | Critic agent feedback                    |
| `actual_effects`    | `List[ActualEffect]`     | Observed state changes                   |
| `call_args`         | `Dict[str, Any]`         | Arguments passed at invocation           |
| `call_depth`        | `int`                    | 1 = top-level, >1 = nested call          |
| `pre_state`         | `Dict[str, Any]`         | Environment state before execution       |
| `post_state`        | `Dict[str, Any]`         | Environment state after execution        |

The `was_executed` property returns `False` when `execution_status == "not_executed"`.

---

## 3. Value Function V(s)

The value function ranks skills for planning and controls optimization gating.
It is implemented as a `@property` on `SkillNode`.

### Formula

```
V(s) = p_hat_s - lambda * u_s
```

Where:

- **`p_hat_s`** is the Bayesian-smoothed success rate (Laplace smoothing):

```
p_hat_s = (n_succ + alpha) / (n_succ + n_fail + alpha + beta)
```

- **`u_s`** is the uncertainty term:

```
u_s = (n_s + 1) ^ (-0.5)
```

- `n_succ` = `statistics.successful_executions`
- `n_fail` = `statistics.failed_executions`
- `n_s` = `statistics.total_executions`

### Configurable parameters

| Parameter | Field on SkillNode           | Default | Role                          |
|-----------|------------------------------|---------|-------------------------------|
| `alpha`   | `value_function_alpha`       | `1.0`   | Success prior (Bayesian)      |
| `beta`    | `value_function_beta`        | `1.0`   | Failure prior (Bayesian)      |
| `lambda`  | `value_function_lambda`      | `1.0`   | Uncertainty penalty weight    |

### Behavior

- **New skill (0 executions):** `p_hat = alpha / (alpha + beta) = 0.5`, `u = 1.0`,
  so `V(s) = 0.5 - 1.0 = -0.5`. New skills are penalized by high uncertainty.
- **Well-tested successful skill (e.g., 10/10):** `p_hat ~ 0.917`, `u ~ 0.302`,
  so `V(s) ~ 0.615`. High value, low optimization probability.
- **Poorly performing skill (e.g., 2/10):** `p_hat ~ 0.25`, `u ~ 0.302`,
  so `V(s) ~ -0.052`. Low value, high optimization probability.

### Precondition and Effect value functions

Separate value functions exist for preconditions and effects, defined in
`skillnet/agents/skill_graph/models/config.py`:

**PreconditionValueFunctionConfig:**

```
V(precond) = p_combined - lambda * uncertainty - gamma * (1 - code_verified)
```

Where `p_combined = w_s * P(precond|success) + w_f * P(missing|failure)`.
Defaults: `alpha=1.0`, `beta=2.0`, `lambda=1.5`, `gamma=0.4`,
`w_s=0.6`, `w_f=0.4`.

**EffectValueFunctionConfig:**

```
V(effect) = p_s - lambda * uncertainty
```

Defaults: `alpha=1.0`, `beta=1.5`, `lambda=1.2`.

---

## 4. Versioning System

### SkillVersion

Every code change creates a `SkillVersion` snapshot, defined in
`skillnet/agents/skill_graph/models/version.py`.

| Field                 | Type                    | Description                               |
|-----------------------|-------------------------|-------------------------------------------|
| `version`             | `str`                   | Semantic version string (e.g., `"1.0.0"`) |
| `created_at`          | `str`                   | ISO 8601 timestamp                        |
| `change_log`          | `str`                   | What changed                              |
| `code`                | `str`                   | Full code snapshot                        |
| `description`         | `str`                   | Description snapshot                      |
| `preconditions`       | `List[SkillPrecondition]` | Precondition snapshot                   |
| `effects`             | `List[SkillEffect]`     | Effect snapshot                           |
| `parameters`          | `Dict`                  | Parameter metadata snapshot               |
| `value_function`      | `float`                 | V(s) at version creation time             |
| `statistics_snapshot`  | `Dict`                 | Execution stats at creation time          |
| `update_source`       | `str`                   | `"optimizer"`, `"refactor"`, `"manual"`, `"rollback"`, etc. |
| `update_reason`       | `str`                   | Human-readable reason                     |
| `optimization_id`     | `Optional[str]`         | Linked optimization flow ID               |
| `used_feedbacks`      | `List[Dict]`            | Feedback items consumed by this version   |
| `affected_subgraph`   | `Dict`                  | Rollback metadata (directly/indirectly affected skills) |

### Contract gate

Each `SkillVersion` includes a **contract validity gate** that checks the
JavaScript code for structural violations:

| Field                  | Type   | Description                                    |
|------------------------|--------|------------------------------------------------|
| `is_contract_valid`    | `bool` | Whether the code passes contract checks        |
| `contract_violations`  | `List[Dict]` | List of detected violations              |

Contract checks (`_check_js_skill_contracts`):

1. **No explicit `undefined` passing** -- detects patterns like `(undefined,`,
   `, undefined)`, `= undefined;` which indicate wrapper parameter mapping failures.
2. Bracket balance (implicit in code parsing).

**Promotion rule:** When `add_version()` is called on a `SkillNode`, only
contract-valid versions are promoted to the node's active `code` and `description`.
Invalid versions are recorded in history but do not update the live skill.

### Sliding window

Each version tracks a sliding window of recent executions (default size: 20):

```python
execution_window: List[Dict]  # [{"execution_id", "success", "timestamp", "version"}]
```

This enables **per-version success rate** tracking and **performance degradation
detection**. The `check_performance_degradation()` method compares the current
version's windowed success rate against the previous version's overall rate.

### Rollback

`SkillNode.rollback_to_version(target_version)` creates a new version entry
that restores code, description, preconditions, effects, and parameters from the
target version. The version number is incremented (not rewound) to maintain a
monotonic history.

### GraphVersion

The graph itself is versioned via `GraphVersion` records:

| Field            | Type              | Description                           |
|------------------|-------------------|---------------------------------------|
| `version`        | `str`             | Graph version string                  |
| `node_versions`  | `Dict[str, str]`  | Snapshot of each node's version       |
| `edges`          | `Dict[str, List[str]]` | Edge snapshot                    |
| `update_source`  | `str`             | `"refactor"`, `"add_skill"`, `"optimization"`, etc. |
| `changes`        | `Dict`            | Added/removed nodes and edges         |

---

## 5. Gradients & Optimization Gating

### SkillGradients

Defined in `skillnet/agents/skill_graph/models/gradients.py`, `SkillGradients`
accumulates optimization signals on a node:

| Channel                    | Description                                  |
|----------------------------|----------------------------------------------|
| `reflections`              | Self-reflection feedback from Phase 1 analysis |
| `feedback`                 | External feedback (e.g., from backpropagation) |
| `optimization_suggestions` | Targeted code change suggestions             |

Each item is a `SkillGradientItem`:

```python
class SkillGradientItem:
    content: str                       # The feedback text
    used_in_optimization: bool = False # Consumed flag
    optimization_id: Optional[str]     # Linked optimization flow
    timestamp: str                     # When created
```

The `get_unused_items()` method returns items not yet consumed by an optimization
pass. After consumption, `mark_as_used()` sets the flag and links the optimization ID.

### stop_gradient

When `node.stop_gradient = True`, gradient propagation from parent failures
halts at this node. This is used for stable, well-tested primitives that should
not be modified in response to upstream failures.

### Optimization gating: P(update s)

The optimizer uses a probabilistic gating mechanism to decide whether to apply
optimization to a skill. This is implemented in
`skillnet/agents/optimizer/validators/skip_checker.py`.

**Formula:**

```
P(update s) = (1 - epsilon) * sigmoid(gamma * (threshold - V(s))) + epsilon
```

Where `sigmoid(x) = 1 / (1 + exp(-x))`.

**Parameters** (from `skillnet/agents/optimizer/config.py`, `SkipOptimizationConfig`):

| Parameter   | Config field              | Default | Description                        |
|-------------|---------------------------|---------|------------------------------------|
| `epsilon`   | `OPTIMIZATION_EPSILON`    | `0.05`  | Minimum optimization probability   |
| `gamma`     | `OPTIMIZATION_GAMMA`      | `8.0`   | Sigmoid slope (sensitivity)        |
| `threshold` | `OPTIMIZATION_THRESHOLD`  | `0.6`   | V(s) crossover point               |

**Behavior:**

- When `V(s) << threshold`: sigmoid output approaches 1, so `P ~ 1.0`.
  Poorly-performing skills are almost always optimized.
- When `V(s) >> threshold`: sigmoid output approaches 0, so `P ~ epsilon = 0.05`.
  Well-performing skills are rarely optimized.
- When `V(s) = threshold`: `P = (1 - epsilon) * 0.5 + epsilon ~ 0.525`.

**Additional skip conditions** (checked before probability gating):

1. **Version count:** Skills with `>= MAX_VERSIONS_BEFORE_SKIP` (default 20)
   versions are always skipped.
2. **CALLER_FIX_MARKER:** If backpropagation analysis determined the fault lies
   in the caller (not this skill), the skill is skipped.

**Pipeline integration:**

- **Phase 1 (analysis/backpropagation):** Never skips. Full gradient propagation
  proceeds with uniform `weight * 0.5` discount.
- **Phase 2 (apply optimization):** `P(update s)` gates whether to actually
  modify the skill's code. Rejected skills produce
  `OptimizationForwardFeedback(analysis_available=True, skipped_reason=...)` so
  parent skills can see the analysis even when the child was not modified.

---

## 6. Preconditions & Effects

### SkillPrecondition

Defined in `skillnet/agents/skill_graph/models/precondition.py`.

A precondition describes a state requirement that must hold before the skill
can execute successfully.

| Field                  | Type               | Description                                 |
|------------------------|--------------------|---------------------------------------------|
| `description`          | `str`              | Human-readable description                  |
| `code`                 | `str`              | Executable check code                       |
| `state_representation` | `Dict[str, Any]`   | Structured state (inventory, position, etc.)|
| `confidence_value`     | `float`            | Value function confidence                   |
| `confidence_level`     | `str`              | `"high"` / `"medium"` / `"low"` / `"uncertain"` / `"unknown"` |
| `inference_stats`      | `Dict`             | Sample counts for Bayesian inference        |
| `condition`            | `Optional[Dict]`   | Parameter-conditional applicability         |
| `source_skill`         | `Optional[str]`    | Origin skill (for refactored preconditions) |

**State representation formats:**

- AND logic: `{"logic": "AND", "conditions": [...]}`
- OR logic: `{"logic": "OR", "conditions": [...]}`
- Nested: `{"logic": "AND", "conditions": [..., {"logic": "OR", "conditions": [...]}, ...]}`
- Legacy (list, implicit AND): `[{"type": "inventory", "item": "oak_log", "count": 1, "operation": "require"}, ...]`

**Conditional preconditions:** The `condition` field enables parameter-dependent
preconditions. For example, `{"targetBlockNames": "diamond_ore"}` means the
precondition only applies when mining diamond ore (which requires an iron pickaxe).

### SkillEffect

Expected effects describe the state changes produced by a successful execution.

| Field                  | Type               | Description                                 |
|------------------------|--------------------|---------------------------------------------|
| `description`          | `str`              | Human-readable description                  |
| `code`                 | `str`              | Executable verification code                |
| `state_representation` | `Dict[str, Any]`   | Structured state change                     |
| `is_primary`           | `bool`             | Whether this is the skill's core output     |
| `importance`           | `str`              | `"core"` / `"secondary"` / `"incidental"`  |
| `confidence_value`     | `float`            | Value function confidence                   |
| `confidence_level`     | `str`              | Same levels as preconditions                |
| `condition`            | `Optional[Dict]`   | Parameter-conditional applicability         |
| `source_skill`         | `Optional[str]`    | Origin skill (for refactored effects)       |

**`is_primary`** distinguishes the skill's core output from by-products.
For example, `craftOakCraftingTable` has `crafting_table` as primary and
`oak_planks` as secondary. The `EffectMatcher` in the planner preferentially
matches `is_primary=True` effects.

**Operations:** `"add"`, `"remove"`, `"place"`, `"equip"`.

### ActualEffect

Recorded after each execution, `ActualEffect` captures the observed state delta:

| Field               | Type                | Description                              |
|---------------------|---------------------|------------------------------------------|
| `execution_id`      | `str`               | Linked execution trace                   |
| `timestamp`         | `str`               | When observed                            |
| `state_changes`     | `List[Dict]`        | Structured changes (preferred format)    |
| `inventory_changes` | `Dict[str, int]`    | Item deltas (legacy)                     |
| `position_changes`  | `Dict[str, float]`  | Position deltas (legacy)                 |
| `pre_state`         | `Dict[str, Any]`    | State before execution                   |
| `post_state`        | `Dict[str, Any]`    | State after execution                    |
| `verified`          | `bool`              | Whether matched against expected effects |

---

## 7. Persistence

### Checkpoint directory structure

```
{ckpt_dir}/skill_graph/
    code/              # {skill_name}.js       -- JavaScript source
    description/       # {skill_name}.txt      -- Natural-language description
    metadata/          # {skill_name}.json     -- Full metadata (graph info, stats, versions, params)
    preconditions/     # {skill_name}.json     -- Precondition list
    effects/           # {skill_name}.json     -- Expected effect list
    graph/             # graph.json            -- Full graph structure (nodes, edges, versions)
```

### Save lifecycle (`_save_to_checkpoint`)

1. **Graph structure:** Serialize `SkillGraph.to_dict()` to `graph/graph.json`.
   This includes all nodes (via `SkillNode.to_dict()`), edges, and graph versions.
   Task-specific and experimental skills are excluded from serialization.

2. **Per-skill files:** For each persistent node:
   - `code/{name}.js` -- current code
   - `description/{name}.txt` -- current description
   - `preconditions/{name}.json` -- precondition list
   - `effects/{name}.json` -- expected effects with `is_primary` flags
   - `metadata/{name}.json` -- comprehensive metadata including:
     - Graph structure (parents, children, in/out degree)
     - Execution statistics
     - Full version history (with code snapshots)
     - Parameter metadata
     - Gradient information
     - Refactor/coverage info
     - Deprecation state
     - Value function and its parameters

3. **Exclusions:** Task-specific skills (saved separately to
   `apply_skills_for_task/`) and experimental skills (memory-only until verified)
   are not written to the main checkpoint directories.

### Load lifecycle (`_load_from_checkpoint`)

1. **Graph deserialization:** Load `graph/graph.json` and reconstruct via
   `SkillGraph.from_dict()`.

2. **Migrations** (run on every load for backward compatibility):
   - Clear coverage marks if refactoring is disabled.
   - Detect and mark missed task-specific skill compositions.
   - Validate and clean erroneous dependency edges.
   - Validate and clean hallucinated effects.
   - Repair missing primary effects on wrapper skills.
   - Fix function-name vs. registration-name mismatches.

3. **VectorDB synchronization:** If the vector database is empty but the graph
   has nodes, rebuild the vectordb from node descriptions. Then synchronize
   to remove stale entries and add missing ones.

---

## 8. Graph Operations

### Edge management

Edges represent caller-to-callee relationships. All operations are exposed via
`GraphQueriesMixin` (which delegates to `SkillGraph`).

| Method                | Description                                           |
|-----------------------|-------------------------------------------------------|
| `add_edge(parent, child)` | Add a dependency edge. Updates `children`, `parents`, `in_degree`, `out_degree` on both nodes. Task-specific parents do not pollute the child's `parents` list. |
| `remove_edge(parent, child)` | Remove a dependency edge. Updates degree counters. |
| `has_edge(parent, child)` | Check if an edge exists.                          |
| `would_create_cycle(from, to)` | DFS check: returns `True` if adding `from -> to` would create a cycle. |

### Node access

| Method                        | Description                                    |
|-------------------------------|------------------------------------------------|
| `get_node(name)`              | Retrieve a `SkillNode` by name                 |
| `has_node(name)`              | Check existence                                |
| `add_skill_node(node)`        | Add a node (returns `False` if already exists)  |
| `remove_skill_node(name)`     | Remove a node and all its edges                |

### Iteration and query

| Method / Property                 | Description                                |
|-----------------------------------|--------------------------------------------|
| `get_all_skill_names(include_task_specific=False)` | List all skill names     |
| `iter_skills(include_task_specific=False)` | Iterate `(name, node)` pairs       |
| `skill_count`                     | Number of nodes in the graph               |
| `get_parents(name)`              | Parent (caller) names                       |
| `get_children(name)`            | Child (callee) names, in call order         |

### Subgraph extraction

```python
get_subgraph(root_name: str, max_depth: int = None) -> SkillGraph
```

Performs a **BFS** from `root_name`, collecting all reachable descendants up to
`max_depth`. Returns a new `SkillGraph` with deep-copied nodes and
reconstructed edges. Used by the optimizer to scope analysis to the affected
dependency subtree.

### Dependency expansion

```python
expand_with_dependencies(skills: List[str], max_depth: int = 5) -> List[str]
```

Given a set of skill names, recursively expands the list to include all
transitive callees up to `max_depth`. Used when loading skill code for
execution -- all dependencies must be available in the Mineflayer environment.

### Cycle detection

`would_create_cycle(from_node, to_node)` uses DFS from `to_node` to check
whether `from_node` is reachable. If so, adding `from_node -> to_node` would
create a cycle, which is forbidden in the DAG. Self-loops (`from == to`) are
also rejected.

### Coverage model

The `CoverageType` enum (in `skillnet/agents/skill_graph/models/coverage.py`)
classifies how a skill is covered by another:

**Wrapper types** (simple delegation, strict bloat limits):
- `PARAMETRIC` -- e.g., `mineOakLogs` becomes a wrapper around `mineLogs(wood_type="oak")`
- `SIBLING` -- sibling unification
- `DUPLICATION` -- deduplicated code
- `FUNCTIONAL_SUPERSET` -- superset skill

**Independent types** (retain their own logic):
- `COMMON_SUBSKILL` -- shared utility (e.g., `setupCraftingTable`)
- `BEHAVIORAL` -- behavioral composition

The `should_skip_covered_skill()` function in `graph.py` determines retrieval
filtering: wrapper-type covered skills are skipped during similarity search,
while independent-type skills remain retrievable.

### Graph persistence filtering

When serializing (`SkillGraph.to_dict()`), the following nodes are excluded:
- `is_task_specific = True` -- non-reusable compositions
- `is_experimental = True` -- unverified skills

Edges involving these nodes are also excluded, ensuring the persisted graph
contains only validated, reusable skills.
