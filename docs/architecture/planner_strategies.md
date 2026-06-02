# Planner Strategies

## Overview

GraphPlanner implements a progressive planning strategy that evolves as the agent's skill library grows. Rather than committing to a single planning approach, it transitions through three phases based on the number of skills in the graph:

- **Phase 1 (0-4 skills, "early")**: LLM-primary planning. The skill graph has too few skills for reliable composition, so GraphPlanner primarily falls back to the LLM action agent for code generation. Backward chaining is still attempted but naturally fails due to insufficient coverage.
- **Phase 2 (5-15 skills, "middle")**: Hybrid planning. GraphPlanner tries backward chaining through the skill graph first, with LLM fallback when the graph cannot cover the required effects.
- **Phase 3 (16+ skills, "mature")**: Graph-primary planning. The skill graph is rich enough to handle most tasks through composition. The LLM serves only as a supplement for novel tasks.

The core planning algorithm is **backward chaining**: given a task, the planner extracts the target effects, finds skills whose expected effects match, checks whether their preconditions are satisfied by the current state, and recursively searches for skills to satisfy any unmet preconditions. When backward chaining cannot produce a complete plan, the planner returns a `PlanningResult` with `fallback=True` metadata so the caller can route the task to LLM-based code generation.

## GraphPlanner Architecture

`GraphPlanner` is defined in `skillnet/agents/planner/graph_planner.py` and uses multiple-inheritance mixin composition. The MRO is:

```
GraphPlanner
  -> BasePlanner               (abstract base with plan() signature)
  -> RuleLearningMixin          (parameter mapping rule persistence)
  -> EffectMatchingMixin        (target effect extraction and skill lookup)
  -> SkillSequenceMixin         (backward chaining and skill selection)
  -> ParameterResolutionMixin   (parameter value inference for skill calls)
  -> CodeGenerationMixin        (JavaScript code assembly from skill sequences)
```

### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `skill_graph_manager` | required | SkillGraphManager instance providing access to the skill DAG |
| `action_agent` | `None` | ActionAgent instance for LLM fallback code generation |
| `max_planning_depth` | `5` | Maximum recursion depth for backward chaining |
| `enable_fallback` | `True` | Whether to allow LLM fallback when graph planning fails |
| `use_llm_for_extraction` | `True` | Whether to use LLM for target effect extraction |
| `beta` | `8.0` | Boltzmann inverse temperature for skill selection |
| `use_llm_for_parameters` | `True` | Whether to use LLM as fallback for parameter inference |
| `learn_parameter_rules` | `True` | Whether to persist learned parameter mapping rules |

### Lazy-Initialized Components

Three core subsystems are created on first access via `@property` descriptors:

- **`effect_matcher`** (`EffectMatcher`): Extracts target effects from tasks and matches them against skill effects.
- **`precondition_checker`** (`PreconditionChecker`): Validates preconditions against the current game state and finds skills to satisfy unmet preconditions.
- **`param_inference_engine`** (`ParameterInferenceEngine`): Infers parameter values through a 6-tier strategy chain.

This lazy initialization avoids circular dependencies and allows domain knowledge to be injected via `set_domain_knowledge()` before the subsystems are first used.

### Plan Method Flow

The `plan()` method orchestrates the full planning pipeline:

1. **Extract target effects** from the task description (LLM or rule-based).
2. **Find candidate skills** whose expected effects match the targets.
3. **Check for uncovered effects** -- if any target effect has no matching skill, fall back to LLM.
4. **Extract parameter values** from the task for precondition conditioning.
5. **Build a skill execution sequence** via backward chaining (single-goal or multi-goal).
6. **Generate JavaScript composition code** that calls the skills in order with inferred parameters.
7. Return a `PlanningResult` containing the code, skill sequence, and metadata.

At each step, failure triggers a fallback `PlanningResult` with `metadata.fallback=True`.

## Effect Matching

The effect matching subsystem lives in `skillnet/agents/planning/effect_matcher/` and is structured as:

```
effect_matcher/
  _utils.py       -- Constants, normalize_operation(), category functions
  _extraction.py  -- ExtractionMixin (target effect extraction)
  _matching.py    -- MatchingMixin (effect comparison logic)
  _matcher.py     -- EffectMatcher facade (skill lookup, quality scoring)
```

### Target Effect Extraction

Effects are extracted from task descriptions using a three-tier degradation strategy:

1. **LLM extraction**: Sends the task to an LLM with a structured prompt requesting a JSON array of effects. Each effect must have `type`, `item`, `count`, and `operation` fields.
2. **Domain-aware rules**: Uses verb patterns injected via `DomainKnowledge.get_effect_extraction_verbs()`. Verb categories include `find`, `collect`, `craft`, `ensure`, `place`, and `equip`.
3. **Hardcoded rules**: Regex patterns matching common task formats like "mine 3 oak_log", "craft wooden_pickaxe", "place a chest", "find diamond_ore".

Effect format:
```json
{"type": "inventory", "item": "oak_log", "count": 3, "operation": "add"}
```

Supported effect types: `inventory`, `block`, `equipment`, `nearby_block`.
Supported operations: `add`, `place`, `equip`, `find` (defined in `SUPPORTED_OPERATIONS`).

### Operation Normalization

The `normalize_operation()` function maps operation aliases to canonical forms. Currently `"ensure"` maps to `"add"` via `OPERATION_ALIASES`. This ensures that LLM-generated operations like "ensure" match skill effects declared with "add".

### Pluralization

Item matching handles singular/plural differences: `oak_plank` matches `oak_planks` and vice versa, via suffix-`s` comparison in `_item_matches_with_template()`.

### Item Category Matching

The matcher supports category-level matching through `is_category_name()` and `get_category_items()`:

- A **category name** (e.g., `"log"`) maps to concrete items (`["oak_log", "birch_log", ...]`) via resource aliases.
- Matching is bidirectional: if the target is a category, any member matches; if the effect declares a category, specific target items that belong to it also match.
- **Template parameters**: Effect items like `"{material}_pickaxe"` are converted to regex patterns and matched against targets like `"stone_pickaxe"`.

### Safe Interchange Categories

Certain item categories are marked as **safely interchangeable** -- items within the category can be substituted without functional impact. The default whitelist includes `logs`, `planks`, and `wood`. Ore categories are explicitly excluded because different ores require different tool tiers.

When two concrete items (e.g., `oak_log` and `birch_log`) both belong to a safe interchange category, the matcher treats them as equivalent.

### Domain Knowledge Injection

Item categories, safe interchange categories, and resource aliases are resolved through a domain-first pattern:

1. Check injected `_domain_knowledge` (set via `set_domain_knowledge()`).
2. Fall back to `_FALLBACK_ITEM_CATEGORIES` / `_FALLBACK_SAFE_INTERCHANGE_CATEGORIES`.
3. Common items and blocks are resolved similarly for the `is_category_name()` guard.

### Quality Scoring

`calculate_effect_match_quality()` produces a 0.0-1.0 score across four dimensions:

1. **OR-condition penalty**: Skills with >5 OR conditions in their effects receive increasing penalties (up to 0.5).
2. **Name relevance**: Whether the skill name shares meaningful words with the target item (+0.5).
3. **Code consistency**: Whether an `EffectsConsistencyValidator` quick check confirms the code can actually produce the target item (+0.15 to +0.3).
4. **Effect confidence**: Runtime-verified confidence from the skill's effect metadata. Low confidence (<0.3) incurs up to 0.6 penalty.

Skills below `ABSOLUTE_MIN_QUALITY = 0.25` are hard-filtered from candidates. When high-quality matches exist (>0.7), low-quality matches more than 0.2 below the maximum are also removed.

### Name-Based Fallback

When effect matching finds zero candidates, a name-based fallback converts skill names from camelCase to snake_case and checks if the target item appears in the converted name (e.g., `craftWoodenPickaxe` -> `craft_wooden_pickaxe` contains `wooden_pickaxe`). These matches receive a moderate quality score of 0.5.

## Precondition Checking

The precondition checker lives in `skillnet/agents/planning/precondition_checker/` and is structured as:

```
precondition_checker/
  _utils.py          -- Pure functions: tool tiers, generic types, condition matching
  _environmental.py  -- EnvironmentalFeedback dataclass, error inference
  _checking.py       -- CheckingMixin (precondition validation)
  _skill_finding.py  -- SkillFindingMixin (finding skills to satisfy preconditions)
  _checker.py        -- PreconditionChecker facade
```

### Precondition Validation

`check_preconditions()` iterates over a skill's precondition list and returns those that are **not** satisfied by the current state. For each precondition:

1. **Conditional preconditions**: If the precondition has a `condition` field, it is only checked when `param_values` matches the condition (e.g., a diamond_ore precondition only applies when `targetBlockNames == "diamond_ore"`).
2. **State representation matching**: Recursively evaluates `state_representation` structures supporting AND/OR logic, shorthand formats (`{"OR": [...]}`, `{"AND": [...]}`), and both uppercase and lowercase keys.
3. **Description/code fallback**: When no `state_representation` exists, falls back to regex-based parsing of the description or code.

Single precondition checking (`_check_single_precondition`) supports:

- `inventory` type: Checks item count against the current inventory. Supports `require` (exact) and `require_or_better` (tool tier comparison) operations.
- `biome_resource` type: Checks if the current biome has the required resource (e.g., lava).
- `min_count` as an alias for `count`.
- Default type is `inventory` when omitted.

### Tool Tier Matching

For `require_or_better` operations, `check_tool_requirement()` compares tool tiers: wooden(1) < stone(2) < iron(3) < gold(4) < diamond(5) < netherite(6). If the player has any tool of the same type at the required tier or higher, the precondition is satisfied.

### Environmental Feedback

`EnvironmentalFeedback` captures biome-resource mismatches (e.g., needing lava in a biome that lacks it). The `collect_environmental_feedback()` method scans preconditions for `biome_resource` type conditions and builds structured feedback. The `infer_environmental_feedback_from_error()` function detects resource problems from runtime error messages using pattern matching (domain-injected or fallback).

### Skill Finding

`find_skills_for_precondition()` searches the skill graph for skills whose effects can satisfy an unmet precondition. It uses three strategies in order:

1. **State representation matching**: Extracts required items from the precondition's `state_representation`, then finds skills whose effects match via `EffectMatcher.effect_matches()`. Falls back to description matching on the effects if no structural match is found.
2. **Description matching**: Scans skill effect descriptions for keyword matches (e.g., "plank", "log", "pickaxe").
3. **Code matching**: Extracts item names from precondition code using regex patterns and matches against skill effects.

Invalid items (parameter variable names without underscores that are not recognized items) are filtered out to prevent false matches.

### Domain Injection

Tool tiers, material prefixes, and generic item patterns all support domain injection via `set_domain_knowledge()`, falling back to Minecraft-specific defaults when no domain is injected.

## Parameter Inference Engine

The parameter inference engine (`skillnet/agents/planning/inference/`) provides a unified interface for resolving parameter values when composing skill calls. It lives in:

```
inference/
  parameter_semantic.py   -- QuantitySemantic, DirectionSemantic, TransformHint, ParameterSemantic
  parameter_engine.py     -- ParameterInferenceEngine, InferenceContext, InferenceResult
  config_validator.py     -- Config context matching, type extraction, fuel helpers
  rules/
    minecraft_rules.py    -- Domain-specific transform rules (lazy shim to domains/)
```

### 6-Tier Strategy Chain

`ParameterInferenceEngine.infer_parameter_value()` tries strategies in priority order, returning the first successful result:

| Priority | Strategy | Confidence | Description |
|----------|----------|------------|-------------|
| 1 | `LEARNED_RULES` | 0.9 | Previously successful inference patterns persisted to disk. Looks up `{skill_name}.{param_name}` in the rules file, extracts value from matching effect fields, and applies stored transforms. |
| 2 | `SEMANTIC` | 0.7-0.85 | Uses `ParameterSemantic` metadata attached to the parameter. Handles quantity semantics (target_total vs delta), direction semantics (input vs output), and transform hints for type conversion. |
| 3 | `EFFECT_EXTRACTION` | 0.6-0.8 | Directly extracts values from `target_effects` based on parameter name patterns. Maps "count"-like params to effect counts, "item/type"-like params to effect items. |
| 4 | `CONTEXT` | 0.5-0.7 | Infers from `precondition_context` (e.g., the `required_item` and `required_count` from the calling skill's unmet precondition). |
| 5 | `LLM` | 0.4-0.6 | Sends a structured prompt to the LLM with the parameter metadata, target effects, and current inventory. Parses the response for the value. |
| 6 | `DEFAULT` | 0.1-0.3 | Uses the parameter's declared default value from its metadata. |

If all strategies fail, the engine returns `InferenceResult` with `strategy_used=NONE` and `value="undefined"`.

### Core Data Classes

**`InferenceContext`** bundles all inputs needed for inference:
- `skill_name`, `param_name`, `param_info` (parameter metadata)
- `target_effects` (what the task aims to achieve)
- `current_inventory`, `current_task`
- `precondition_context` (from backward chaining -- what the caller needs)
- `semantic` (ParameterSemantic if available)
- `upstream_output`, `caller_skill` (call chain support)
- `expected_inventory` (what dependency skills will produce)

**`InferenceResult`** captures the outcome:
- `value` (JS literal format, e.g., `'"oak_log"'`, `'8'`, `'["item1"]'`)
- `strategy_used` (which tier succeeded)
- `confidence` (0.0-1.0)
- `transform_applied`, `validation_passed`

### Parameter Semantics

**`QuantitySemantic`** distinguishes how count parameters should be interpreted:
- `TARGET_TOTAL`: "ensure at least N exist" (e.g., `ensureLogs(count=8)` means "have 8 total")
- `DELTA`: "produce N additional" (e.g., `craftPlanks(count=8)` means "craft 8 more")

**`DirectionSemantic`** classifies type parameters:
- `INPUT`: Specifies source material (e.g., `logType` in `craftPlanks`)
- `OUTPUT`: Specifies product type (e.g., `plankType` in `ensurePlanks`)
- `CONFIG`: Configuration option, not an item type
- `BIDIRECTIONAL`: Context-dependent

**Number-type parameters are never inferred as OUTPUT or INPUT direction** -- those semantics are reserved for item-type parameters.

### TransformHint

`TransformHint` encodes how to convert between input and output item types. It supports four transform strategies:
- `suffix_replace`: Replace suffix (e.g., `_planks` -> `_log`)
- `regex`: Pattern matching (e.g., `deepslate_iron_ore` -> `raw_iron`)
- `lookup`: Direct table lookup
- `chain`: Sequential application of multiple transforms

### Config Validation

`config_validator.py` provides guards against incorrect parameter assignments:
- `item_matches_config_context()`: Validates that an item makes sense for a config parameter (e.g., `coal` matches `fuelPriority`, but `iron_ingot` does not).
- `is_input_material_param()`: Detects parameters expecting raw materials (patterns like `logType`, `inputItem`, `material`).
- `map_output_to_input_material()`: Maps output products to input materials (e.g., `birch_planks` -> `birch_log` for a `logType` parameter).

## Code Generation

The `CodeGenerationMixin` (in `skillnet/agents/planner/mixins/code_generation.py`) assembles executable JavaScript from a skill sequence. The core method is `_generate_composition_code()`.

### Code Assembly Pipeline

1. **Generate function name**: Derived from the task description via `normalize_task_name()` (e.g., "Mine 1 oak_log" -> `mine_1_oak_log`).
2. **Collect require declarations**: Scans each skill's code for `require()` calls (e.g., `minecraft-data`, `vec3`) and deduplicates them. Module-level declarations are tracked separately to avoid temporal dead zone issues.
3. **Resolve parameters**: For each skill in the sequence, resolves parameter values using the ParameterResolutionMixin, which coordinates learned rules, semantic inference, effect extraction, context inference, LLM inference, and defaults.
4. **Generate skill calls**: Produces `await skillName(bot, param1, param2, ...)` statements. Handles destructured parameters and object parameters.
5. **Wrap in async function**: The final code is wrapped in `async function taskName(bot) { ... }`.

### Require Pattern Handling

The code generator extracts require declarations from skill code using:
- **Specific patterns**: Matches known patterns like `const mcData = require('minecraft-data')(bot.version)` and `const Vec3 = require('vec3').Vec3`.
- **Generic pattern**: Matches any `const X = require('Y')...` statement.
- **Module-level detection**: `is_module_level_declaration()` checks whether a declaration is outside any function body (by counting brace balance). Module-level declarations are skipped to avoid redeclaration errors.
- **Global dependency exclusion**: Variables already in `GLOBAL_DEPS_VARS` (e.g., `mcData`, `Vec3`) are not re-declared when they are provided by the runtime's `globalDepsCode`.

### JavaScript Sanitization

`sanitize_python_to_js()` converts Python-isms that LLMs sometimes emit: `False` -> `false`, `True` -> `true`, `None` -> `null`.

## Rule Learning

The `RuleLearningMixin` implements a lightweight rule learning system that persists successful parameter inferences for future reuse.

### Rule Storage

Rules are stored in `{ckpt_dir}/skill_graph/parameter_rules.json` as a dictionary keyed by `"{skill_name}.{param_name}"`. Each rule contains:
- `pattern`: Effect fields to match (e.g., `{"item": "oak_log"}`)
- `extraction`: How to extract the value (e.g., `{"path": "item"}`)
- `param_type`: Expected type (`string`, `number`, `array`, `object`)
- `learned_at`: Timestamp
- `usage_count`: How many times the rule has been used

### Object Parameter Rules

For object-type parameters, rules store per-field extraction rules:
```json
{
  "param_type": "object",
  "field_rules": {
    "fieldName": {"effect_field": "item", "transform": "extract_ore_base"}
  }
}
```

### Learning Process

When LLM inference successfully produces a parameter value, `_learn_parameter_rule()`:
1. Validates the value (rejects invalid mappings like `iron_ingot` to `fuelPriority`).
2. Finds the matching effect that the value was derived from.
3. Constructs a pattern (key fields from the effect) and extraction rule (which field to read).
4. Persists the rule to disk.

### Validation

Before using or learning a rule, `_validate_learned_value()` checks config parameters against their expected context. For example, a `fuelPriority` parameter must receive fuel items (coal, planks, etc.), not arbitrary items.

## Progressive Strategy Detail

### Phase Transitions

The planning phase is determined by `_get_planning_stage()` based on `skill_graph_manager.skill_count`:

| Skill Count | Stage | Label | Behavior |
|-------------|-------|-------|----------|
| 0-4 | Phase 1 | `"early"` | Backward chaining is attempted but almost always fails. Falls back to LLM. The skill graph is too sparse for meaningful composition. |
| 5-15 | Phase 2 | `"middle"` | Graph planning succeeds for tasks with matching skills. LLM fallback handles gaps. The hybrid approach allows the graph to prove its value. |
| 16+ | Phase 3 | `"mature"` | Graph planning handles most tasks. The skill graph has enough coverage for multi-step composition. LLM is used only for genuinely novel tasks. |

Note: The configurable `PlannerConfig.min_skills_for_graph_planning` (default: 3) controls when the planner is first created in `PSNAgent.__init__()`, but the phase thresholds (5 and 16) are hardcoded in `_get_planning_stage()`.

### Backward Chaining Algorithm

The core backward chaining implementation in `_build_skill_sequence()`:

1. **Select best skill** from candidates using Boltzmann policy.
2. **Cycle detection**: If the selected skill is in the `visited` set, try remaining candidates.
3. **Check preconditions** against current state (with parameter conditioning).
4. If preconditions are satisfied, return a single-element sequence.
5. If not, **find prerequisite skills** for each unmet precondition.
6. When candidates exceed `MAX_REASONABLE_CANDIDATES` (5), filter by graph children.
7. **Recurse** on prerequisite skills with `max_depth - 1` and a copy of `visited`.
8. **Compose**: prerequisite sequence + target skill.

### Boltzmann Skill Selection

`_select_best_skill()` uses a softmax probability distribution over skill value functions:

```
P(s | subgoal) = exp(beta * V(s)) / sum(exp(beta * V(s')))
```

Where `V(s)` is the skill's value function (Laplace-smoothed success rate minus uncertainty). A consistency penalty from effect-code quality scoring is applied as `V(s) -= penalty * 0.5`. The log-sum-exp trick prevents numerical overflow. With the default `beta=8.0`, the policy is near-deterministic (strongly favoring high-V skills) while allowing occasional exploration.

### Multi-Goal Planning

For tasks with multiple target effects (e.g., "Mine 8 raw iron and 4 coal"), `_build_multi_goal_sequence()`:

1. Maps each effect to its matching skills via `_match_effects_to_skills()`.
2. For each effect, selects the best skill and builds its backward-chaining sequence.
3. Deduplicates skills across sub-sequences (a skill appearing in multiple chains is included only once).
4. Returns the combined sequence.
