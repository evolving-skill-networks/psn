# Two-Phase Optimization Pipeline

## 1. Overview

The Two-Phase Optimization Pipeline is the system that improves existing skills when they fail during execution. It operates on the skill graph (DAG) by propagating failure signals through parent-child relationships, identifying root causes, and applying targeted code transformations.

The pipeline has two phases:

- **Phase 1 (Top-Down Analysis + Gradient Propagation)**: Starting from the root skill that failed, the system recursively analyzes each skill using an LLM to produce structured "gradients" -- modification directions with magnitude, type, and evidence. Feedback propagates downward to child skills identified as faulty, forming a subgraph DAG. This phase always runs; it is never gated or dampened.

- **Phase 2 (Bottom-Up Code Transformation + Validation)**: Following a reverse topological order (leaves first), each skill's gradients are adjusted by incorporating forward feedback from already-optimized children, then an LLM generates new code. Semantic validators check the result. This phase is gated by `P(update s)` -- skills with high value functions may be probabilistically skipped.

Entry point: `TwoPhaseOptimizationEngine` in `skillnet/agents/optimizer/engine.py`.

---

## 2. Engine Entry Point

### TwoPhaseOptimizationConfig

Defined in `skillnet/agents/optimizer/engine.py`:

```python
@dataclass
class TwoPhaseOptimizationConfig:
    max_reflection_depth: int = 3
    enable_transactions: bool = True
    auto_rollback_on_failure: bool = True
    enable_post_optimization_refactor: bool = True
    refactor_min_success_count: int = 3
    refactor_types_enabled: List[str]  # parametric, behavioral, merge_siblings, duplication, extract_common
    pure_reasoning: bool = False       # Removes domain data injection for evaluation
```

### TwoPhaseOptimizationEngine

The engine is the top-level coordinator. Its constructor accepts:

```python
TwoPhaseOptimizationEngine(
    skill_graph_manager,       # SkillGraphManager instance
    llm,                       # LLM for analysis and code generation
    logger,
    ckpt_dir,
    config: TwoPhaseOptimizationConfig,
    optimizer_callback,        # External callback: (skill_name, task, context, ...) -> {new_code, success}
)
```

During initialization (`_init_components`), the engine creates:

1. **TransactionManager** -- manages nested transactions for rollback safety.
2. **RefactorDetector + refactors** -- post-optimization refactoring (if enabled and importable).
3. **TwoPhaseOptimizationPipeline** -- the core pipeline, created via `create_two_phase_pipeline()`.
4. **SkillInfoGetter** -- adapter that reads skill information from the `SkillGraphManager` for the reflection chain.
5. **SkipChecker** -- `P(update s)` gating for Phase 2.

### Pipeline Creation

`create_two_phase_pipeline()` in `skillnet/agents/optimizer/phases/reflection_chain.py` wires the pipeline:

```python
def create_two_phase_pipeline(
    llm, optimize_fn, max_depth=3, logger=None,
    pure_reasoning=False,
    ckpt_dir=None, cold_start=False, skip_checker=None,
) -> TwoPhaseOptimizationPipeline:
```

It constructs a `PureReflection` (with an `LLMAnalyzer`) and a `ConsiderFunction`, then assembles them into a `TwoPhaseOptimizationPipeline`.

### optimize() Method

The public API is `engine.optimize(skills_to_optimize, current_task, current_error, current_critique, ...)`. It delegates to `_optimize_with_pure_pipeline()`, which:

1. Opens a `SessionTransaction` and `SubgraphTransaction`.
2. Constructs feedback content from error + critique.
3. Calls `pipeline.run()` to execute both phases.
4. Extracts `fix_target` attribution from the chain result.
5. Converts `ChainExecutionResult` into `TwoPhaseOptimizationResult`.
6. Marks successfully optimized skills as `experimental` (pending execution validation).
7. Commits or rolls back the transaction based on outcomes.

---

## 3. Phase 1: Top-Down Symbolic Differentiation

Phase 1 implements the paper's top-down credit assignment: failure signals
are propagated along the executed skill invocation trace to decompose
responsibility across composite skills and their subskills. It is
constructed in two layers:

- **REFLECT Operator (Atomic)** — a single LLM-driven analysis of one
  skill against its feedback, returning a `SkillDelta` (self-gradients) and
  `PropagatedFeedback` for each faulty child. Implements paper Eq. (5):
  $\tilde{\nabla}_s = \mathrm{REFLECT}(f_t, s; \mathcal{T}_t)$.
- **Recursive Application (Chain Rule)** — recursively invokes REFLECT
  along the executed trace, building a DAG of analyses. Implements paper
  Eq. (1)'s recursion:
  $\tilde{\nabla}_{s'} = \mathrm{REFLECT}(\tilde{\nabla}_s, s')$
  for each $s' \in \mathrm{Children}(s)$.

At runtime, the orchestrator `ReflectionChain` (chain rule) repeatedly calls
the atomic `PureReflection.reflect()` (operator) for each visited skill in
the trace; both layers always run together as a single top-down sweep before
Phase 2 begins.

### REFLECT Operator (Atomic)

**Source**: `skillnet/agents/optimizer/phases/pure_reflection.py`

The atomic operator implements:

```
psn_reflection(skill, feedback) -> (delta_skill, {phi_(skill->child) | child in FaultyChildren(skill)})
```

### Core Data Structures

**GradientType** (enum) -- the dimension of modification needed:

| Category | Types |
|----------|-------|
| Logic | `LOGIC`, `CONTROL_FLOW`, `ERROR_HANDLING` |
| Interface | `PARAMETER_SEMANTIC`, `PARAMETER_TYPE`, `RETURN_VALUE` |
| Effects | `PRECONDITION`, `EFFECT`, `SIDE_EFFECT` |
| Environment | `ENVIRONMENT_ADAPTATION`, `RESOURCE_MANAGEMENT`, `PHYSICAL_CONSTRAINT` |

**Gradient** (frozen dataclass) -- an atomic unit of modification direction:

```python
@dataclass(frozen=True)
class Gradient:
    gradient_type: GradientType
    magnitude: float           # [0, 1], urgency of the fix
    direction: str             # Textual description of what to change
    evidence: str              # Supporting evidence from feedback
    suggested_fix: Optional[str] = None
    affected_lines: Optional[List[int]] = None
```

**SkillDelta** -- aggregation of multiple gradients for a single skill:

```python
@dataclass
class SkillDelta:
    skill_name: str
    gradients: List[Gradient]
    total_magnitude: float     # Sum of all gradient magnitudes
    primary_type: GradientType # Type of the largest gradient
```

A delta is considered significant when `total_magnitude > 0.3`.

**PropagatedFeedback** -- feedback passed from parent to child (the chain rule):

```python
@dataclass
class PropagatedFeedback:
    source_skill: str          # Parent skill
    target_skill: str          # Child skill
    issue_description: str
    responsibility: str        # What the child should fix
    expected_behavior: str
    actual_behavior: str
    weight: float = 1.0        # Responsibility proportion
    confidence: float = 0.5
```

### LLM-Driven Root Cause Analysis

The `LLMAnalyzer` builds a structured prompt containing:

- Skill code (up to 8000 chars)
- Feedback content and type
- Execution state (call arguments, inventory changes)
- Chat log from in-game diagnostic messages
- Child skill signatures with parameter types and success rates
- Propagated feedback from parent (if any)
- Domain knowledge sections (environment rules, primitive knowledge, API behaviors, reasoning examples)

The LLM returns a JSON response with three issue categories:

1. **`self_issues`** -- bugs in the skill's own logic (produces self-gradients).
2. **`calling_pattern_issues`** -- this skill calls a child incorrectly (wrong arguments, types, semantics). These produce self-gradients targeting the calling code; the referenced children are added to `caller_fix_children`.
3. **`child_issues`** -- a child skill's own implementation is buggy (produces `PropagatedFeedback` for recursive analysis). Referenced children are added to `faulty_children`.

### Caller/Callee Attribution (Ternary)

The `ReflectionChain` sets `ChainNode.fix_target` based on the LLM's attribution:

| LLM Output | fix_target | Meaning |
|-------------|------------|---------|
| `calling_pattern_issues` only | `caller_fix` | Fix the parent's calling code |
| `child_issues` only | `None` (callee_fix) | Fix the child's implementation |
| Both present | `both_fix` | Fix both parent's calling code and child's implementation |

This ternary classification determines which skills receive Phase 2 optimization and how the `FixTargetType` is recorded in the optimization tracker.

### PureReflection and ReflectionInput

`PureReflection` wraps a `ReflectionAnalyzer` (protocol). The production implementation is `LLMAnalyzer`. The input is:

```python
@dataclass
class ReflectionInput:
    skill_name: str
    skill_code: str
    skill_description: str
    feedback_content: str
    feedback_type: str         # error, critique, backpropagated, etc.
    execution_traces: List[Dict]
    pre_state: Optional[Dict]
    post_state: Optional[Dict]
    children: List[str]
    children_info: Dict[str, Dict]
    propagated_feedback: Optional[PropagatedFeedback]
    chat_log: str = ""
    is_task_specific: bool = False  # Task-specific wrappers focus on child_issues
```

---

### Recursive Application (Chain Rule)

**Source**: `skillnet/agents/optimizer/phases/reflection_chain.py`

`ReflectionChain.build()` recursively constructs a DAG of `ChainNode` objects, one per analyzed skill:

1. Start from the root skill with the original error/critique.
2. Call `PureReflection.reflect()` to produce `ReflectionOutput`.
3. For each child in `faulty_children`, recursively build with the child's `PropagatedFeedback` as input.
4. Stop at `max_depth` (default 3) or when a skill has already been analyzed.

After building, `_topological_sort_reverse()` produces the bottom-up optimization order (leaves first, root last), ensuring children are optimized before their parents in Phase 2.

**Fallback gradient (`weight * 0.5`).** When a parent marks a child as faulty but the child's own LLM analysis produces an insignificant gradient (`total_magnitude <= 0.3`), a fallback gradient is constructed:

```python
fallback_magnitude = propagated_feedback.weight * 0.5
```

This ensures that children identified by their parents still receive optimization attention even if the child's self-analysis does not detect the issue. Phase 1 never dampens gradients by `V(s)` -- full propagation always occurs (see §6 P(update s) Gating for the contrast with Phase 2).

---

## 4. Phase 2: Code Transformation

### Code Edit Operators

**Source**: `skillnet/agents/optimizer/transforms/`

**DiffEngine** (`diff_engine.py`) -- unified engine supporting four diff formats:

| Format | Description |
|--------|-------------|
| `UNIFIED` | Standard unified diff with `@@` hunk headers |
| `SIMPLE` | Simplified `+`/`-` format |
| `INCREMENTAL` | Structured operation list (`replace`, `insert_after`, `delete`) |
| `FULL_CODE` | Complete code replacement |

The engine auto-detects the format via `detect_format()` and applies accordingly. On failure, it falls back to extracting code from the diff text. A `basic_syntax_check()` validates balanced brackets and string termination.

**code_edit_ops.py** -- provides `remove_unused_helper_functions()`, a pure function that identifies and removes helper functions not called by the main function or referenced as callbacks.

### Semantic Validators

**Source**: `skillnet/agents/optimizer/validators/`

| Validator | Purpose |
|-----------|---------|
| `BloatChecker` | Prevents code growth beyond configurable limits (3x hard limit, 800-line absolute max, wrapper-specific limits) |
| `EffectsConsistencyValidator` | Checks that declared effects match code implementation |
| `OverclaimDetector` | Detects parameters claiming broader support than code handles |
| `ExecutionValidator` | Verifies child skills were actually executed (prevents misattribution) |
| `SkipChecker` | `P(update s)` probabilistic gating (see Section 6) |
| `NamingConflictChecker` | Detects function naming conflicts |

### OptimizationForwardFeedback (Bottom-Up Result Propagation)

**Source**: `skillnet/agents/optimizer/feedback/forward_propagation.py`

After a child is processed in Phase 2, its result is packaged as an `OptimizationForwardFeedback` and passed upward in the bottom-up sweep. The parent's `ConsiderFunction` consumes it to adjust the parent's own delta before generating the parent's patch.

```python
@dataclass
class OptimizationForwardFeedback:
    skill_name: str
    optimization_successful: bool
    changes_made: List[str]
    new_code: str              # For persistence
    old_code: str              # For rollback
    interface_changed: bool
    interface_changes: List[InterfaceChange]
    effects_changed: bool
    effect_changes: List[EffectChange]
    parent_suggestions: List[str]
    parent_warnings: List[str]
    analysis_available: bool = False   # True if Phase 1 completed
    skipped_reason: str = ""           # Non-empty when P(update s) rejected
```

`analysis_available=True` even when the skill was gated out at Phase 2 (`skipped_reason` populated) — parents still see that Phase 1 ran, so analysis flows up even when the code transformation does not.

### ConsiderFunction (Delta Adjustment)

**Source**: `skillnet/agents/optimizer/phases/consider_function.py`

The `ConsiderFunction` implements:

```
B := B + Consider(delta_B, [opt_forward_feedback(C) for C in FaultyChildren(B)])
```

It performs five steps:

1. **Collect interface changes** from child forward feedbacks.
2. **Collect effect changes** from child forward feedbacks.
3. **Identify dependency updates** (call signature changes, effect handling, warnings).
4. **Adjust delta** -- reduce magnitude of child-related gradients when children optimized successfully (scale by `1 - success_rate * 0.6`), add new gradients for interface/effect changes.
5. **Evaluate overall impact** (`minimal`, `moderate`, `breaking`, `failed`).

The output `ConsiderResult` contains the `adjusted_delta` that drives the actual code generation.

### ChainExecutor

The `ChainExecutor` iterates over `optimization_order` (bottom-up) and for each node:

1. Skips if gradient is insignificant (`total_magnitude <= 0.3`).
2. Skips task-specific wrappers (Phase 1 backpropagation already handled them).
3. Applies `P(update s)` gating via `SkipChecker.should_skip()`.
4. Calls `ConsiderFunction.consider()` to adjust the delta.
5. Calls `optimize_fn(skill_name, adjusted_delta, context)` to generate new code.
6. Generates `OptimizationForwardFeedback` for parent consumption.

---

## 5. Transaction Management

**Source**: `skillnet/agents/optimizer/tracking/transaction.py`

### Three-Level Hierarchy

```
SessionTransaction
  └── SubgraphTransaction
        └── SkillTransaction
```

- **SessionTransaction**: Top-level session wrapping an entire optimization run. Holds multiple subgraph transactions. Has `session_type` (optimization, refactor, mixed).
- **SubgraphTransaction**: Groups skills involved in a single optimization subgraph (one root skill + its faulty descendants). Tracks `dependency_order` for correct rollback sequencing, `committed_skills`, and `rolled_back_skills`.
- **SkillTransaction**: Tracks a single skill's modification with before/after snapshots.

### SkillSnapshot

Each `SkillTransaction` captures a `SkillSnapshot` at creation time via `SkillSnapshot.from_node(node)`, recording:

- Code, version, description
- Preconditions and expected effects (serialized)
- Parameters, coverage state, dependency lists

### Auto-Rollback on Failure

In `_optimize_with_pure_pipeline()`:

```python
if result.skills_failed and self.config.auto_rollback_on_failure:
    self.transaction_manager.rollback_subgraph(subgraph_txn)
    self.transaction_manager.end_session(session, commit=False)
```

On exception, the transaction is also rolled back:

```python
except Exception:
    self.transaction_manager.rollback_subgraph(subgraph_txn)
    self.transaction_manager.end_session(session, commit=False)
```

Rollback restores skills from their snapshots via `_restore_skill_from_snapshot()`, traversing in reverse dependency order.

### Partial Rollback

`TransactionManager.partial_rollback(subgraph_txn, skills_to_rollback)` allows rolling back specific skills while keeping others committed, setting the subgraph status to `PARTIAL_ROLLBACK`.

### Contract Gate for Version Promotion

After successful optimization, skills are marked as `experimental`:

```python
node.is_experimental = True
node.experimental_task = task
```

Experimental skills must pass actual execution validation before their changes are considered stable. This prevents optimizations that look correct on paper but fail in practice from becoming permanent.

### Persistence

Transaction logs are saved as JSON to `{ckpt_dir}/transactions/{session_id}.json`, recording the complete before/after state of every skill modified.

---

## 6. P(update s) Gating

### Formula

```
P(update s) = (1 - epsilon) * sigma(gamma * (threshold - V(s))) + epsilon
```

Where:
- `epsilon = 0.05` -- minimum optimization probability
- `gamma = 8.0` -- sigmoid slope parameter
- `threshold = 0.6` -- value function threshold
- `sigma` -- sigmoid function
- `V(s)` -- skill value function

### V(s) Value Function

Defined in `skillnet/agents/skill_graph/models/node.py`:

```
V(s) = p_hat_s - lambda * u_s
```

Where:
- `p_hat_s = (n_succ + alpha) / (n_succ + n_fail + alpha + beta)` -- Bayesian-smoothed success rate (alpha=1.0, beta=1.0)
- `u_s = (n_s + 1)^(-1/2)` -- uncertainty measure
- `lambda` -- uncertainty weight (default 0.3)

### SkipOptimizationConfig

Defined in `skillnet/agents/optimizer/config.py`:

```python
@dataclass
class SkipOptimizationConfig:
    MAX_VERSIONS_BEFORE_SKIP: int = 20
    OPTIMIZATION_EPSILON: float = 0.05
    OPTIMIZATION_GAMMA: float = 8.0
    OPTIMIZATION_THRESHOLD: float = 0.6
    ENABLE_PROBABILITY_SKIP: bool = True
```

### SkipChecker Integration

The `SkipChecker` in `skillnet/agents/optimizer/validators/skip_checker.py` runs three checks:

1. **Version count**: Skip if versions >= `MAX_VERSIONS_BEFORE_SKIP` (20).
2. **CALLER_FIX_MARKER**: Skip if feedback indicates the problem is in the caller, not this skill.
3. **Probability skip**: Compute `P(update s)` and compare against a random draw.

### Phase 1 vs Phase 2 Behavior

- **Phase 1**: `SkipChecker` is NOT consulted. Analysis and gradient propagation always run at full strength. No V(s) dampening occurs during backpropagation.
- **Phase 2**: `SkipChecker.should_skip()` is called in `ChainExecutor.execute()`. If rejected, the skill produces an `OptimizationForwardFeedback` with `analysis_available=True` and `skipped_reason` set, so parent skills can still see that analysis was performed.

---

## 7. Domain Knowledge in Optimization

The `LLMAnalyzer` receives domain knowledge through the `_domain_knowledge` attribute (set externally via `PSNAgent.from_domain()`). This is an instance of the `DomainKnowledge` abstract base class.

### Knowledge Sections in the Analysis Prompt

| Section | Source Method | Purpose |
|---------|--------------|---------|
| Environment context | `get_environment_context(pre_state)` | Biome-specific information (e.g., biome type affects available resources) |
| Environment rules | `get_environment_rules(feedback)` | Domain rules relevant to the error (e.g., tool tier requirements) |
| Primitive knowledge | `get_primitive_knowledge(feedback, code)` | API behaviors of control primitives used in the code |
| Detailed error knowledge | `get_detailed_knowledge_for_error(feedback, code, chat_log)` | Indexed knowledge entries matched to the specific error |
| Basic error knowledge | `get_knowledge_for_error(search_text, code)` | Fallback when detailed retrieval returns empty |
| Reasoning examples | `get_reasoning_examples(search_text, max_examples)` | Worked examples showing correct reasoning patterns |

### Knowledge Flow

```
DomainKnowledge (ABC)
  |
  +-- get_environment_rules()     --> env_section in prompt
  +-- get_primitive_knowledge()   --> primitive_section in prompt
  +-- get_detailed_knowledge_for_error() --> api_knowledge_section
  +-- get_reasoning_examples()    --> reasoning_examples_section
  |
  v
LLMAnalyzer._build_analysis_prompt()
  |
  v
LLM generates JSON with self_issues / calling_pattern_issues / child_issues
  |
  v
Gradients + PropagatedFeedback
```

### Code Validators

The `BloatChecker` uses domain-aware configuration:
- `BloatPreventionConfig` sets growth ratios, line limits, and wrapper thresholds.
- Covered wrapper skills (from refactoring) have stricter limits (20 lines max, 1.5x growth).

The `EffectsConsistencyValidator` checks whether a skill's declared effects are actually achievable by its code, catching "overclaim" problems where metadata promises capabilities the implementation cannot deliver.

### Diagnostic Logging

The engine injects a `KnowledgeRetrieval` diagnostic logger into the analyzer when a checkpoint directory is available. This logger tracks which knowledge sections were active, their character counts, and the LLM's token usage, enabling post-hoc analysis of knowledge retrieval effectiveness.
