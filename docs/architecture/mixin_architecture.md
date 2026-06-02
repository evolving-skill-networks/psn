# Mixin Architecture Guide

## Overview

The PSN codebase relies heavily on mixin-based composition to manage
complexity. The original monolithic classes -- PSNAgent at 190KB, SkillGraphManager at
4,783 lines, SkillGraphOptimizer at 2,044 lines -- were split into focused mixins that
each own a single responsibility domain.

There are approximately 44 mixins distributed across 6 facade classes:

| Facade                    | Mixins | Base class        | Location                                      |
|---------------------------|--------|-------------------|-----------------------------------------------|
| PSNAgent                  | 8      | --                | `skillnet/psn.py`                             |
| SkillGraphOptimizer       | 10     | --                | `skillnet/agents/optimizer/_impl/`            |
| ParameterizedActionAgent  | 7      | ActionAgent       | `skillnet/agents/parameterized_action/`       |
| GraphPlanner              | 5      | BasePlanner       | `skillnet/agents/planner/`                    |
| SkillGraphManager         | 10     | --                | `skillnet/agents/skill_graph/_impl/`          |
| PSNCurriculumAgent        | 5      | CurriculumAgent   | `skillnet/agents/psn_curriculum/`             |

Every facade owns its `__init__`; mixins never define `__init__`. Cross-mixin calls
resolve through Python's Method Resolution Order (MRO) via `self.xxx()`.

---

## Pattern Rules

### 1. No `__init__` in mixins

All initialization lives in the facade class. Mixins assume attributes already exist on
`self` when their methods are called. If a mixin needs one-time setup, the facade calls
an explicit `_init_xxx()` method from its own `__init__`.

### 2. Cross-mixin calls via `self.xxx()` (MRO resolution)

Mixins call other mixin methods through `self`, relying on Python's C3 linearization to
resolve the correct implementation. No mixin imports or references another mixin class
directly.

### 3. Shared state via `self` attributes

Mixins communicate through shared instance attributes set by the facade's `__init__`:
- `self._config` -- PSNConfig or component-specific config
- `self._domain_knowledge` -- DomainKnowledge instance (domain-agnostic interface)
- `self.skill_graph_manager` -- SkillGraphManager instance (for optimizer, action agent)
- `self.logger` -- standard Python logger

### 4. No direct mixin-to-mixin imports

Mixins within the same facade never import each other. They are peers in the MRO and
communicate exclusively through `self`. This prevents circular dependencies and keeps
each mixin independently testable.

### 5. Composite mixins are allowed

A mixin may itself compose sub-mixins (e.g., `StepExecutionMixin` composes three
sub-mixins; `ValidationMixin` composes two). The sub-mixins follow the same rules.

---

## PSNAgent

**File**: `skillnet/psn.py`
**Mixin directory**: `skillnet/_psn_impl/`

```python
class PSNAgent(
    EventProcessingMixin,        # Layer 0: event parsing utilities
    SkillRecordingMixin,         # Layer 1: depends on EventProcessing
    EffectVerificationMixin,     # Layer 1: depends on EventProcessing
    StepExecutionMixin,          # Layer 2: depends on SkillRecording + EffectVerification
    TaskManagementMixin,         # Layer 3: standalone
):
```

`StepExecutionMixin` is itself a composite mixin:

```python
class StepExecutionMixin(StepPlanMixin, StepExecutePhaseMixin, StepOptimizeMixin):
```

### Mixin responsibilities

| Mixin                     | File                    | Responsibility                                              |
|---------------------------|-------------------------|-------------------------------------------------------------|
| EventProcessingMixin      | `event_processing.py`   | Extract current state, nearby blocks, and inventory from events |
| SkillRecordingMixin       | `skill_recording.py`    | Auto-record skill executions and find reuse hints           |
| EffectVerificationMixin   | `effect_verification.py`| Verify skill effects via rule-based checks + LLM fallback   |
| StepPlanMixin             | `step_plan.py`          | Phases 1-3: planning, code extraction, recursive error handling |
| StepExecutePhaseMixin     | `step_execute_phase.py` | Phases 4-5: environment execution and diagnostics           |
| StepOptimizeMixin         | `step_optimize.py`      | Phase 6: two-phase skill optimization on failure            |
| StepExecutionMixin        | `step_execution.py`     | Orchestrator `step()`, Phase 7 (message rebuild), preflight |
| TaskManagementMixin       | `task_management.py`    | Task failure tracking, limits, and progress recording       |

### Cross-mixin call graph

```
StepExecutionMixin
  -> SkillRecordingMixin._auto_record_skill_executions()
  -> SkillRecordingMixin._find_reuse_skill_hint()
SkillRecordingMixin
  -> EventProcessingMixin._find_nearby_blocks_from_events()
  -> EffectVerificationMixin._check_skill_effect_achieved()
```

### Required `self` attributes (set by `__init__`)

`skill_manager`, `action_agent`, `planner`, `env`, `curriculum_agent`, `task`,
`context`, `messages`, `conversations`, `last_events`,
`last_skill_execution_results`, `planner_mode`, `enable_optimizer`, `_task_semantic`,
`_task_executed_skills`. Optional: `optimizer`.

---

## SkillGraphOptimizer

**File**: `skillnet/agents/optimizer/_impl/optimizer_impl.py`
**Mixin directory**: `skillnet/agents/optimizer/_impl/mixins/`

```python
class SkillGraphOptimizer(
    FeedbackMixin,
    ValidationMixin,
    CodeEditMixin,
    InterfaceManagementMixin,
    OptimizationMixin,
    ApplyOptimizationMixin,
    EditContextMixin,
    EditAnalysisMixin,
):
```

`ValidationMixin` is itself a composite mixin:

```python
class ValidationMixin(CodeQualityValidationMixin, ResponsibilityValidationMixin):
```

### Mixin responsibilities

| Mixin                        | File                          | Responsibility                                            |
|------------------------------|-------------------------------|-----------------------------------------------------------|
| FeedbackMixin                | `feedback.py`                 | Feedback collection, parsing, and history management      |
| CodeQualityValidationMixin   | `code_quality_validation.py`  | Code growth/bloat checks, duplication detection            |
| ResponsibilityValidationMixin| `responsibility_validation.py`| Skill responsibility boundary and relevance validation    |
| ValidationMixin              | `validation.py`               | Composite: code validation + consistency checking         |
| CodeEditMixin                | `code_edit.py`                | Code formatting, syntax fixing, and validation            |
| InterfaceManagementMixin     | `interface_mgmt.py`           | Interface change analysis and caller update propagation   |
| OptimizationMixin            | `optimization.py`             | Semantic analysis and loop classification delegates       |
| ApplyOptimizationMixin       | `apply_optimization.py`       | Code persistence and validation pipeline                  |
| EditContextMixin             | `edit_context.py`             | Build structured context for LLM code editing             |
| EditAnalysisMixin            | `edit_analysis.py`            | Analyze feedback to produce edit suggestions              |

### Required `self` attributes

`skill_graph_manager`, `logger`, `llm`, and optimizer config fields.

---

## ParameterizedActionAgent

**File**: `skillnet/agents/parameterized_action/_agent.py`
**Mixin directory**: `skillnet/agents/parameterized_action/mixins/`

```python
class ParameterizedActionAgent(
    SkillNamingMixin,
    PromptRenderingMixin,
    CodeValidationMixin,
    CodeParsingMixin,
    SkillSelectionMixin,
    CodeAssemblyMixin,
    CodeFinalizationMixin,
    ActionAgent,           # base class (skillnet/agents/action.py)
):
```

### Mixin responsibilities

| Mixin                | File                  | Responsibility                                          |
|----------------------|-----------------------|---------------------------------------------------------|
| SkillNamingMixin     | `skill_naming.py`     | Skill name normalization, type-aware naming rules       |
| PromptRenderingMixin | `prompt_rendering.py` | Format parameterized skills for LLM prompts             |
| CodeValidationMixin  | `code_validation.py`  | Validate generated code (syntax, references, structure) |
| CodeParsingMixin     | `code_parsing.py`     | Identify main function, parse LLM output                |
| SkillSelectionMixin  | `skill_selection.py`  | Dependency analysis, independent value assessment       |
| CodeAssemblyMixin    | `code_assembly.py`    | Code assembly and function name replacement             |
| CodeFinalizationMixin| `code_finalization.py` | Final validation, Babel transpilation, skill packaging  |

### Base class

`ActionAgent` (`skillnet/agents/action.py`) provides `__init__`, LLM interaction, and
the base `generate_skill()` loop. The mixins override and extend specific steps of this
pipeline.

---

## GraphPlanner

**File**: `skillnet/agents/planner/graph_planner.py`
**Mixin directory**: `skillnet/agents/planner/mixins/`

```python
class GraphPlanner(
    BasePlanner,
    RuleLearningMixin,
    EffectMatchingMixin,
    SkillSequenceMixin,
    ParameterResolutionMixin,
    CodeGenerationMixin,
):
```

### Mixin responsibilities

| Mixin                    | File                     | Responsibility                                         |
|--------------------------|--------------------------|--------------------------------------------------------|
| RuleLearningMixin        | `rule_learning.py`       | Load, save, and learn parameter mapping rules          |
| EffectMatchingMixin      | `effect_matching.py`     | Extract target effects, find skills by effects         |
| SkillSequenceMixin       | `skill_sequence.py`      | Build skill execution sequences via backward chaining  |
| ParameterResolutionMixin | `parameter_resolution.py`| Infer parameter values using multi-strategy chain      |
| CodeGenerationMixin      | `code_generation.py`     | Generate skill composition code (~400 lines)           |

### Base class

`BasePlanner` (`skillnet/agents/planner/_types.py`) is an ABC defining the planner
interface. `GraphPlanner.__init__` initializes the effect matcher, precondition checker,
and parameter inference engine as `@property` attributes.

### Important note

The `@property` decorators on `effect_matcher`, `precondition_checker`, and
`param_inference_engine` in `GraphPlanner` must be preserved -- mixins access these
via `self.effect_matcher` etc.

---

## SkillGraphManager

**File**: `skillnet/agents/skill_graph/_impl/graph_manager_impl.py`
**Mixin directory**: `skillnet/agents/skill_graph/_impl/mixins/`

```python
class SkillGraphManager(
    SkillAdditionMixin,          # defined in graph_manager_impl.py (not extractable)
    SkillExecutionMixin,
    ExecutionLifecycleMixin,
    SemanticsUpdateMixin,
    SkillVersioningMixin,
    RefactorManagementMixin,
    MetadataManagementMixin,
    MetadataCrudMixin,
    MetadataValidationMixin,
    GraphQueriesMixin,
):
```

### Mixin responsibilities

| Mixin                    | File                      | Responsibility                                           |
|--------------------------|---------------------------|----------------------------------------------------------|
| SkillAdditionMixin       | `graph_manager_impl.py`   | New skill creation, naming, initialization (635+ lines)  |
| SkillExecutionMixin      | `skill_execution.py`      | Core execution recording (~270 lines)                    |
| ExecutionLifecycleMixin  | `execution_lifecycle.py`  | Delayed refactor triggers, experimental skill management |
| SemanticsUpdateMixin     | `semantics_update.py`     | Runtime semantics learning from execution data           |
| SkillVersioningMixin     | `skill_versioning.py`     | Version creation, rollback, and history management       |
| RefactorManagementMixin  | `refactor_mgmt.py`        | Refactor candidate screening, execution, relationship analysis |
| MetadataManagementMixin  | `metadata_mgmt.py`        | Metadata orchestration and extraction delegates          |
| MetadataCrudMixin        | `metadata_crud.py`        | Precondition and effect CRUD operations                  |
| MetadataValidationMixin  | `metadata_validation.py`  | Effects validation and load-time repair                  |
| GraphQueriesMixin        | `graph_queries.py`        | Thin wrappers around `self.graph` (16 facade methods)    |

### Note on SkillAdditionMixin

`SkillAdditionMixin` is defined inside `graph_manager_impl.py` rather than in the
`mixins/` directory. The `add_new_skill` method spans 635 lines with 31+ internal method
calls, making it unsafe to extract into a separate file.

---

## PSNCurriculumAgent

**File**: `skillnet/agents/psn_curriculum/agent.py`
**Mixin directory**: `skillnet/agents/psn_curriculum/mixins/`

```python
class PSNCurriculumAgent(
    TaskValidationMixin,
    PromptBuildingMixin,
    AdaptiveLearningMixin,
    MilestoneManagementMixin,
    SkillLearningMixin,
    CurriculumAgent,           # base class (skillnet/agents/curriculum.py)
):
```

### Mixin responsibilities

| Mixin                    | File                  | Responsibility                                          |
|--------------------------|-----------------------|---------------------------------------------------------|
| TaskValidationMixin      | `task_validation.py`  | Task format validation, environment checks, special cases |
| PromptBuildingMixin      | `prompt_building.py`  | LLM prompt construction with resource context           |
| AdaptiveLearningMixin    | `adaptive_learning.py`| Adaptive learning path management and task decomposition |
| MilestoneManagementMixin | `milestone_mgmt.py`   | Milestone tracking, failure budgets, critical alerts    |
| SkillLearningMixin       | `skill_learning.py`   | Skill lookup, task-to-skill mapping, cache management   |

### Base class

`CurriculumAgent` (`skillnet/agents/curriculum.py`) provides `__init__`, LLM-based task
proposal, and basic progress tracking. PSNCurriculumAgent extends this with planning
awareness, resource tracking, and symbolic knowledge.

---

## Cross-cutting Data Flow

### Shared `self` attributes (within a facade)

Mixins within the same facade share state through `self`. The facade's `__init__` is the
single source of truth for attribute initialization. Common shared attributes:

- **PSNAgent**: `self.skill_manager`, `self.action_agent`, `self.planner`, `self.env`,
  `self._config`, `self.messages`, `self.last_events`
- **SkillGraphOptimizer**: `self.skill_graph_manager`, `self.llm`, `self.logger`
- **SkillGraphManager**: `self.graph` (the underlying DAG), `self.vectordb`,
  `self.ckpt_dir`

### Dependency injection between facades

Facades do not inherit from each other. They connect through constructor injection or
setter methods:

- `PSNAgent.__init__` creates `SkillGraphManager`, `ParameterizedActionAgent`,
  `GraphPlanner`, `PSNCurriculumAgent`, and `SkillGraphOptimizer`
- `ParameterizedActionAgent._skill_manager_ref` is set via `set_skill_manager()`,
  called from `psn.py`
- `SkillGraphOptimizer` receives `skill_graph_manager` as a constructor argument
- `GraphPlanner` receives `skill_graph_manager` for graph traversal and effect matching

### The `from_domain()` factory

`PSNAgent.from_domain(DomainModule, PSNConfig)` wires domain knowledge into all
components: action agent, optimizer, curriculum, and critic. This is the standard
production entry point.

---

## Guidelines for Adding New Mixins

### When to create a new mixin

- A facade method group exceeds ~200 lines and has a cohesive responsibility
- Multiple methods share a logical concern distinct from the rest of the class
- You need to test a responsibility in isolation

### Naming conventions

- Class name: `{Responsibility}Mixin` (e.g., `EffectMatchingMixin`)
- File name: `snake_case` of the responsibility (e.g., `effect_matching.py`)
- Place in the `mixins/` subdirectory of the owning package
- Export from `mixins/__init__.py`

### Implementation checklist

1. **No `__init__`** -- document required `self` attributes in the class docstring
2. **No cross-mixin imports** -- call other mixins only via `self.xxx()`
3. **Add to facade MRO** -- insert at the correct layer position
4. **Update `mixins/__init__.py`** -- add the import and `__all__` entry
5. **Preserve MRO order** -- lower layers (fewer dependencies) go later in the base list
6. **Write tests** -- test the mixin methods with a minimal mock host that provides the
   required `self` attributes

### Testing pattern

```python
class MockHost(NewMixin):
    """Minimal host providing required self attributes."""
    def __init__(self):
        self.skill_graph_manager = mock_sgm
        self.logger = logging.getLogger("test")

host = MockHost()
result = host.some_mixin_method(args)
```

### Composite mixins

If a mixin grows beyond ~500 lines, consider splitting it into sub-mixins and creating
a composite mixin that inherits from them (see `StepExecutionMixin` and
`ValidationMixin` as examples). The composite is what appears in the facade's MRO.
