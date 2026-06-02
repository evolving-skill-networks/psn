# Knowledge Retrieval Architecture

This document describes the three-tier knowledge retrieval system used by
PSN's optimizer to diagnose skill failures and inform code
transformations.

---

## 1. Overview

The knowledge retrieval subsystem answers one question: *given a failed skill
execution, what domain facts does the LLM need to produce an accurate root
cause analysis?*

The architecture is organized into three tiers:

| Tier | Component | Responsibility |
|------|-----------|----------------|
| **Data** | `KnowledgeItem` / `KnowledgeEntry` | Pure facts -- no solutions, no fix hints |
| **Index** | `KnowledgeIndex` | Keyword + type scoring over all entries |
| **Retrieval** | `LLMKnowledgeRetriever` | LLM-driven query generation, fallback, formatting |

A strict design principle runs through every layer:

> **Python provides facts; the LLM does reasoning.**

Knowledge items contain factual descriptions, logical implications, and
constraints. They never contain `fix_hint`, `solution_hint`, `fix_strategy`,
or `suggested_fix` fields. The optimizer's LLM is expected to reason from
facts to solutions.

---

## 2. Knowledge Base Types

**Source**: `skillnet/core/knowledge_base.py`

### KnowledgeDomain (Enum)

Separates knowledge by its origin:

| Value | Meaning |
|-------|---------|
| `GAME` | Minecraft game rules and physics (ore distribution, tool tiers, crafting recipes) |
| `API` | Mineflayer API behaviors (pathfinder goals, bot methods, inventory slots) |
| `PRIMITIVE` | Internal primitive function knowledge (preconditions, effects, failure modes) |

### KnowledgeCategory (Enum)

Fine-grained classification within each domain:

- **GAME domain**: `BLOCK_PROPERTY`, `RESOURCE_SPAWN`, `GAME_MECHANIC`, `CRAFTING_RECIPE`
- **API domain**: `PATHFINDER`, `BOT_METHOD`, `INVENTORY`
- **PRIMITIVE domain**: `PRECONDITION`, `EFFECT`, `FAILURE`

### KnowledgeItem (dataclass)

The canonical fact unit. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `domain` | `KnowledgeDomain` | Which knowledge domain |
| `category` | `KnowledgeCategory` | Fine-grained category |
| `name` | `str` | Unique identifier for the knowledge item |
| `fact` | `str` | Core factual description |
| `keywords` | `List[str]` | Keywords used for retrieval matching |
| `logical_implications` | `List[str]` | Conclusions that follow necessarily from the fact |
| `constraints` | `List[str]` | Physical or logical constraints |
| `available_options` | `List[str]` | Enumerated options (without recommendation) |
| `conditions` | `List[str]` | Conditions under which the fact applies |
| `implications` | `List[str]` | **Deprecated** -- use `logical_implications` instead |
| `source` | `str` | Provenance tracking |

Key methods:

- `matches(text: str) -> bool` -- checks if any keyword appears in the text (case-insensitive).
- `to_prompt_text() -> str` -- formats the item for LLM consumption, showing the fact, conditions, logical implications, constraints, and available options.
- `to_dict() / from_dict()` -- serialization round-trip.

**Example of good `logical_implications`** (pure logical deductions):
```
logical_implications=[
    "No diamonds exist above Y=16",      # necessarily follows from spawn range
    "No diamonds exist below Y=-64",
]
```

**Example of bad implications** (strategy leakage):
```
implications=["go deep underground"]     # this is advice, not a deduction
```

### RetrievedKnowledge (dataclass)

A collection returned from knowledge searches:

| Field | Type | Description |
|-------|------|-------------|
| `items` | `List[KnowledgeItem]` | Retrieved knowledge items |
| `relevance_scores` | `dict` | Mapping of item `name` to relevance score |

Methods:

- `to_prompt_section() -> str` -- formats all items grouped by domain under headers like "Minecraft Game Rules", "Mineflayer API Behaviors", "Primitive Function Knowledge".
- `filter_by_domain(domain) -> RetrievedKnowledge` -- returns a filtered subset.
- `filter_by_category(category) -> RetrievedKnowledge` -- returns a filtered subset.

---

## 3. KnowledgeIndex

**Source**: `skillnet/agents/optimizer/knowledge/index.py`

The `KnowledgeIndex` is a domain-agnostic searchable index that scores
`KnowledgeEntry` objects against `KnowledgeQuery` objects.

### KnowledgeEntry (dataclass)

The index-level representation of a knowledge item:

| Field | Type | Description |
|-------|------|-------------|
| `query_type` | `str` | Category label: `"tool_tier"`, `"api_behavior"`, `"game_mechanic"`, `"reasoning_example"`, `"resource"`, `"primitive"` |
| `name` | `str` | Entry name |
| `keywords` | `List[str]` | Keywords for matching |
| `content` | `str` | Formatted knowledge text (typically from `KnowledgeItem.to_prompt_text()`) |
| `priority` | `int` | Tie-breaking priority (higher = preferred), default `0` |

### KnowledgeQuery (dataclass)

An LLM-generated or keyword-based search query:

| Field | Type | Description |
|-------|------|-------------|
| `query_type` | `str` | Desired knowledge type (must match `KnowledgeEntry.query_type` for type bonus) |
| `keywords` | `List[str]` | Search keywords |
| `context` | `str` | Optional context string for secondary matching |

### Search Algorithm

`KnowledgeIndex.search(query, max_results=3) -> List[str]`

1. Score every entry against the query using `_calculate_match_score()`.
2. Filter out entries with score <= 0.
3. Sort by `(-score, -priority)`.
4. Return the `content` strings of the top `max_results` entries.

### Scoring Formula: `_calculate_match_score(entry, query)`

The score is computed as the sum of four components:

```
score = type_bonus + keyword_score + content_score + context_score
```

**Type bonus** (+5.0):
```python
if entry.query_type == query.query_type:
    score += 5.0
```
This is the single largest scoring factor. A matching `query_type` provides a
+5.0 boost, ensuring that type-directed queries strongly prefer entries of the
same type. This makes `query_type` a semantic filter, not just a label.

**Keyword matching** (up to +3.0 per keyword):
```python
for kw in query.keywords:
    if kw in entry.keywords:          # exact match
        score += 3.0
    elif kw is substring of any entry keyword:  # partial (query in entry)
        score += 1.5
    elif any entry keyword is substring of kw:  # partial (entry in query)
        score += 1.0
```
All comparisons are case-insensitive. Each query keyword that matches via any
keyword path is tracked in `matched_via_keywords` to avoid double-counting in
the content matching phase.

**Content text matching** (+1.0 per keyword):
```python
for kw in query.keywords:
    if kw not in matched_via_keywords and len(kw) >= 3 and kw in entry.content:
        score += 1.0
```
Only applies to query keywords that were *not* already matched via the keyword
lists. The minimum length of 3 filters out noise words like "a", "to", "in".
The weight (+1.0) is deliberately lower than keyword matching (+3.0) to
preserve keyword priority.

**Context matching** (+0.5 per keyword):
```python
if query.context:
    for kw in entry.keywords:
        if kw in query.context:
            score += 0.5
```
If the query carries a context string, each entry keyword found in the context
adds a small bonus.

### Builder Registration (Dependency Injection)

The index supports domain-agnostic construction via a builder registration
mechanism:

```python
register_knowledge_builder(builder: Callable[[], List[KnowledgeEntry]])
```

- Stores the builder function in a module-level `_knowledge_builder` variable.
- Invalidates the cached singleton (`_knowledge_index = None`) so the next
  `get_knowledge_index()` call rebuilds with the new builder.

**Fallback**: `_default_build_entries()`

When no builder is registered, the index attempts auto-discovery of the
Minecraft builder:

```python
def _default_build_entries() -> List[KnowledgeEntry]:
    try:
        from skillnet.domains.minecraft.knowledge.index_builder import (
            build_minecraft_knowledge_entries,
        )
        return build_minecraft_knowledge_entries()
    except ImportError:
        logger.warning("[KnowledgeIndex] No knowledge builder available")
        return []
```

This ensures backward compatibility: bare `KnowledgeIndex()` calls (common in
tests) continue to work without explicit registration.

### Singleton Access

```python
get_knowledge_index() -> KnowledgeIndex
```

Returns a module-level singleton. The singleton is lazily created on first
call and invalidated when `register_knowledge_builder()` is called.

---

## 4. LLM Knowledge Retriever

**Source**: `skillnet/agents/optimizer/knowledge/llm_knowledge_retriever.py`

The `LLMKnowledgeRetriever` is the primary entry point for knowledge
retrieval during optimizer analysis. It uses an LLM to generate targeted
search queries, with a keyword-based fallback when the LLM is unavailable.

### Constructor

```python
LLMKnowledgeRetriever(
    llm=None,                         # LLM instance (must have .invoke())
    knowledge_index=None,             # KnowledgeIndex (defaults to global singleton)
    use_llm=True,                     # Whether to attempt LLM query generation
    query_prompt="",                  # Domain-specific prompt template
    fallback_queries_fn=None,         # Domain-specific keyword fallback function
)
```

Domain-specific content is injected via two parameters:
- `query_prompt`: A prompt template with `{feedback_content}`, `{chat_log}`,
  and `{skill_code}` placeholders.
- `fallback_queries_fn`: Signature `(feedback: str, chat_log: str, skill_code: str) -> List[KnowledgeQuery]`.

### Retrieval Workflow

`retrieve(feedback_content, chat_log, skill_code, kr_logger) -> RetrievalResult`

1. **LLM query generation** (primary path):
   - Requires: `use_llm=True`, `llm` is set, `chat_log` is non-empty,
     `query_prompt` is non-empty.
   - Calls `_generate_queries_with_llm()`, which formats the prompt template
     and invokes the LLM up to 3 times with 2-second retry delays.
   - The LLM response is parsed via a chain of three extractors (tried in
     order):
     1. `_extract_json_array()` -- finds `[...]` in the response text.
     2. `_extract_from_markdown_code_block()` -- extracts JSON from
        `` ```json ... ``` `` blocks.
     3. `_extract_json_objects()` -- finds individual `{...}` objects with a
        `query_type` field.
   - Each parsed object becomes a `KnowledgeQuery`.
   - Sets `source = "llm"`.

2. **Keyword fallback** (when LLM path produces no queries):
   - Calls `fallback_queries_fn(feedback_content, chat_log, skill_code)` if
     the function was provided.
   - Sets `source = "fallback"`.

3. **Index search**:
   - For each query, calls `knowledge_index.search(query, max_results=2)`.
   - Collects all result strings.

4. **Deduplication and formatting**:
   - Deduplicates results using `dict.fromkeys()` (preserves order).
   - Formats as `"## Retrieved Knowledge\n\n" + "\n\n".join(results)`.

### RetrievalResult (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `knowledge_text` | `str` | Formatted knowledge text for LLM consumption |
| `queries_used` | `List[KnowledgeQuery]` | The queries that were executed |
| `source` | `str` | `"llm"` or `"fallback"` |

### Diagnostic Counters

Module-level `_kr_stats` dictionary tracks retrieval performance:

| Counter | Description |
|---------|-------------|
| `total_calls` | Total number of `retrieve()` invocations |
| `llm_attempts` | Number of times LLM query generation was attempted |
| `llm_successes` | Number of times LLM successfully generated queries |
| `fallback_used` | Number of times the fallback path was used |
| `empty_results` | Number of times no knowledge was retrieved |

Access via `get_kr_stats() -> dict`.

### Convenience Function

```python
retrieve_knowledge_for_analysis(
    feedback_content, chat_log="", skill_code="",
    llm=None, kr_logger=None,
    query_prompt="", fallback_queries_fn=None,
) -> str
```

Creates a one-shot `LLMKnowledgeRetriever` and returns `result.knowledge_text`
directly. This is the function called by `DomainKnowledge.get_detailed_knowledge_for_error()`.

---

## 5. Minecraft Knowledge Implementation

**Source**: `skillnet/domains/minecraft/knowledge/`

The Minecraft domain populates the knowledge framework with game-specific
facts. Knowledge is organized into three subdirectories, one per
`KnowledgeDomain`.

### game/ -- Game Rules (KnowledgeDomain.GAME)

Contains pure Minecraft game knowledge unrelated to any API:

| Module | Content | Example Facts |
|--------|---------|---------------|
| `blocks.py` | Block hardness, tool tiers, harvest requirements | "Deepslate iron ore requires stone+ pickaxe" |
| `resources.py` | `OreSpawnRange` (min_y, max_y, optimal_y, biome restrictions), `BIOME_RESOURCES` | "Diamond ore spawns Y=-64 to Y=16, optimal at Y=-59" |
| `mechanics.py` | Placement, pathfinding, crafting, combat, inventory, water mechanics | "3x3 crafting recipes require a placed crafting table" |
| `ores.py` | Per-entity ore-to-item drop mappings | "Coal ore drops coal, not coal_ore" |

Aggregated via `get_all_game_knowledge() -> List[KnowledgeItem]`.

### api/ -- Mineflayer API (KnowledgeDomain.API)

Documents Mineflayer JavaScript API behaviors:

| Module | Content |
|--------|---------|
| `pathfinder.py` | GoalNear, GoalBlock, GoalXZ, movement settings |
| `bot_methods.py` | bot.dig(), bot.place(), bot.craft(), bot.equip() behaviors |
| `inventory.py` | Inventory slot semantics, item counting methods |

Aggregated via `get_all_api_knowledge() -> List[KnowledgeItem]`.

### primitives/ -- Primitive Functions (KnowledgeDomain.PRIMITIVE)

Documents the project's own primitive function wrappers:

| Module | Content |
|--------|---------|
| `mining.py` | `mineBlock` preconditions and failure modes |
| `crafting.py` | `craftItem` preconditions, `RecipeDependencyAnalyzer` |
| `placement.py` | `placeItem` position conflicts, standing-on-target issues |
| `smelting.py` | `smeltItem` fuel requirements, furnace placement |
| `combat.py` | `killMob` targeting and combat mechanics |
| `movement.py` | `gotoWithTimeout` pathfinding failures |

Aggregated via `get_all_primitive_knowledge() -> List[KnowledgeItem]`.

### index_builder.py

**Source**: `skillnet/domains/minecraft/knowledge/index_builder.py`

`build_minecraft_knowledge_entries() -> List[KnowledgeEntry]`

Transforms all Minecraft `KnowledgeItem` objects into `KnowledgeEntry` objects
suitable for the `KnowledgeIndex`. It calls seven indexing functions:

| Function | query_type | priority | Source |
|----------|------------|----------|--------|
| `_index_api_behaviors()` | `"api_behavior"` | 5 | `api/` package |
| `_index_game_mechanics()` | `"game_mechanic"` | 3 | `game/mechanics.py` |
| `_index_resource_knowledge()` | `"resource"` | 4 | `game/resources.py` |
| `_index_block_knowledge()` | `"game_mechanic"` | 4 | `game/blocks.py` |
| `_index_reasoning_examples()` | `"reasoning_example"` | 2 | `reasoning_examples.py` |
| `_index_primitive_knowledge()` | `"primitive"` | 4 | `primitives/` package |
| `_index_ore_drop_knowledge()` | `"game_mechanic"` | 5 | `game/ores.py` |

Each function is wrapped in a `try/except ImportError` so missing knowledge
modules degrade gracefully.

### MinecraftKnowledge Registration

The `MinecraftKnowledge.__init__()` method (in `__init__.py`) calls
`_register_knowledge_builder()` which calls
`register_knowledge_builder(build_minecraft_knowledge_entries)` to wire the
index builder for the optimizer's knowledge retrieval system.

### kr_config.py

**Source**: `skillnet/domains/minecraft/kr_config.py`

Contains the two domain-specific injection points for `LLMKnowledgeRetriever`:

- `MINECRAFT_QUERY_PROMPT`: A Chinese-language prompt template that instructs
  the LLM to analyze a failed Minecraft skill execution and return a JSON
  array of up to 3 `KnowledgeQuery` objects. The prompt lists the available
  `query_type` values: `tool_tier`, `api_behavior`, `resource`,
  `game_mechanic`, `primitive`, `mcdata_naming`, `reasoning_example`.

- `minecraft_fallback_queries(feedback, chat_log, skill_code) -> List[KnowledgeQuery]`:
  A keyword-matching function that scans the combined input for domain terms
  (e.g., "pickaxe", "place", "craft", "furnace", "pathfinder", "ore",
  "deepslate", "undefined") and returns up to 5 relevant `KnowledgeQuery`
  objects. Used when the LLM is unavailable or fails to generate queries.

---

## 6. Integration Points

The primary consumer of the knowledge retrieval system is the optimizer's
`PureReflection` engine, specifically its `_build_analysis_prompt()` method in
`skillnet/agents/optimizer/phases/pure_reflection.py`.

### Call Flow

```
PureReflection._build_analysis_prompt(input: ReflectionInput)
│
├── 1. self._domain_knowledge.get_environment_context(pre_state)
│      → biome info, available resources
│
├── 2. self._domain_knowledge.get_environment_rules(feedback)
│      → relevant environment rules as formatted strings
│
├── 3. self._domain_knowledge.get_primitive_knowledge(feedback, code)
│      → primitive function knowledge for the error
│
├── 4. self._domain_knowledge.get_detailed_knowledge_for_error(  ← PRIMARY PATH
│      │   feedback, code, chat_log, kr_logger)
│      │
│      └── MinecraftKnowledge.get_detailed_knowledge_for_error()
│          │
│          └── retrieve_knowledge_for_analysis(
│                  feedback, chat_log, code,
│                  llm=self._kr_llm,
│                  query_prompt=MINECRAFT_QUERY_PROMPT,
│                  fallback_queries_fn=minecraft_fallback_queries)
│              │
│              ├── LLM generates KnowledgeQuery objects
│              ├── KnowledgeIndex.search() scores and retrieves entries
│              └── Returns formatted knowledge_text
│
└── 5. FALLBACK (if step 4 returns empty):
       self._domain_knowledge.get_knowledge_for_error(search_text, code)
       → basic keyword-matched knowledge from api_behaviors.py
```

### Data Flow Summary

1. A skill execution fails. The optimizer creates a `ReflectionInput`
   containing the feedback content, chat log, skill code, pre-execution state,
   child skill information, and any propagated feedback from parent skills.

2. `PureReflection` calls `_build_analysis_prompt()`, which assembles a
   comprehensive LLM prompt. As part of this assembly, it retrieves domain
   knowledge through the `DomainKnowledge` interface.

3. The **primary retrieval path** uses `get_detailed_knowledge_for_error()`,
   which delegates to `retrieve_knowledge_for_analysis()`. This function:
   - Constructs an `LLMKnowledgeRetriever` with the Minecraft-specific prompt
     and fallback function.
   - Invokes the LLM to generate targeted `KnowledgeQuery` objects.
   - Searches the `KnowledgeIndex` for matching entries.
   - Returns formatted knowledge text.

4. If the primary path returns empty, the **fallback path** calls
   `get_knowledge_for_error()`, which uses simpler keyword matching from
   `api_behaviors.py`.

5. The retrieved knowledge text is inserted into the analysis prompt alongside
   environment rules, primitive knowledge, child skill information, execution
   state, and chat logs. The LLM then performs root cause analysis using this
   comprehensive context.

### Diagnostic Logging

When a `kr_logger` is provided (typically during debug runs), every stage of
the retrieval pipeline logs its inputs and outputs:

- `[LLMQuery]` -- number and types of LLM-generated queries
- `[Fallback]` -- number of keyword fallback queries
- `[IndexSearch]` -- query parameters and result counts per query
- `[DomainDetailed]` -- character count and preview of retrieved knowledge
- `[DomainBasic]` -- item count from the basic fallback path
