# Dependency Injection Patterns Handbook

## Overview

PSN is designed as a domain-agnostic lifelong learning framework. While the
primary domain is Minecraft, the architecture ensures that no core module hard-imports
domain-specific data. Instead, domain knowledge flows into the system through three
manually wired dependency injection (DI) patterns plus one composition convention
layered on top. There is no DI framework involved; all wiring is explicit Python
code orchestrated by `PSNAgent.from_domain()`.

The three DI mechanisms, in order of prevalence:

| # | Pattern | Scope | Consumers |
|---|---------|-------|-----------|
| 1 | Instance-level `set_domain_knowledge()` | Per-object | Agents, managers |
| 2 | Central registry `dk_registry` | Process-wide | Utility modules (20 consumers) |
| 3 | Global singleton builder registration | Process-wide | Knowledge index |

Plus one composition convention that the mixins follow when consuming Pattern 1
data:

| Convention | Built on | Use |
|------------|----------|-----|
| Domain-first cascade with fallback | Pattern 1 | Mixins that probe `self` via `getattr`, then fall back to a module-level constant |

All patterns share a common invariant: **when no domain is injected, the module
continues to function using empty/permissive defaults** (empty lists, dicts, sets).
This guarantees backward compatibility and allows tests to run without domain setup.

---

## Pattern 1: Instance-level DI via `set_domain_knowledge()`

### Structure

The most common pattern. An agent or manager class:

1. Initializes `self._domain_knowledge = None` in `__init__`.
2. Exposes a `set_domain_knowledge(knowledge)` setter.
3. At runtime, checks `self._domain_knowledge` before falling back to defaults.

### All consumers (instance-level setters on classes)

Found by searching for `def set_domain_knowledge` on class methods:

| File | Class |
|------|-------|
| `skillnet/agents/action.py` | `ActionAgent` |
| `skillnet/agents/critic.py` | `CriticAgent` |
| `skillnet/agents/planner/graph_planner.py` | `GraphPlanner` |
| `skillnet/agents/optimizer/_impl/optimizer_impl.py` | `SkillGraphOptimizer` |
| `skillnet/agents/psn_curriculum/agent.py` | `PSNCurriculumAgent` |
| `skillnet/agents/psn_curriculum/psn_critic.py` | `PSNCriticAgent` |
| `skillnet/agents/skill_graph/_impl/graph_manager_impl.py` | `GraphManagerImpl` |

### Canonical example: `ActionAgent`

From `skillnet/agents/action.py`:

```python
class ActionAgent:
    def __init__(self, ...):
        ...
        # Domain knowledge (set via set_domain_knowledge for from_domain path)
        self._domain_knowledge = None

    def set_domain_knowledge(self, knowledge):
        """Inject domain-specific knowledge for prompt/primitive overrides."""
        self._domain_knowledge = knowledge

    def render_system_message(self, skills=[]):
        if self._domain_knowledge:
            system_template = self._domain_knowledge.get_system_prompt_template()
            programs = "\n\n".join(
                self._domain_knowledge.load_control_primitive_code() + skills
            )
            ...
        else:
            # Minecraft hardcoded fallback path
            ...
```

### Deep propagation: `SkillGraphOptimizer`

Some classes propagate domain knowledge through an internal object chain. The optimizer
is the most complex example (from `skillnet/agents/optimizer/_impl/optimizer_impl.py`):

```python
def set_domain_knowledge(self, knowledge):
    """Wire domain knowledge to internal LLMAnalyzer and refactors via engine chain."""
    self._domain_knowledge = knowledge
    # Chain: _two_phase_engine -> pure_pipeline -> pure_reflection -> analyzer
    engine = getattr(self, '_two_phase_engine', None)
    if engine:
        pipeline = getattr(engine, 'pure_pipeline', None)
        if pipeline:
            reflection = getattr(pipeline, 'pure_reflection', None)
            if reflection and hasattr(reflection, 'analyzer'):
                reflection.analyzer._domain_knowledge = knowledge
        # Propagate to refactors for function reference validation
        for refactor in getattr(engine, 'refactors', {}).values():
            refactor.domain_knowledge = knowledge
        ...
```

The traversal uses `getattr(..., None)` defensively because internal components may
not exist yet at call time (they are lazily initialized).

### When to use `getattr` vs direct `self._domain_knowledge`

- **Direct access** (`self._domain_knowledge`): Use inside the class that owns the
  attribute and initializes it in `__init__`. Safe because `__init__` guarantees the
  attribute exists.

- **`getattr(self, '_domain_knowledge', None)`**: Use in mixins or any code that may
  run on an instance whose `__init__` did not set the attribute. Mixins do not own
  `__init__`, so they cannot assume any attribute exists. Example from
  `skillnet/agents/parameterized_action/mixins/skill_naming.py`:

  ```python
  dk = getattr(self, '_domain_knowledge', None)
  ```

---

## Pattern 2: Central DK Registry

### Structure

Utility modules that need domain knowledge read it from a **central registry**
(`skillnet/core/dk_registry.py`) instead of maintaining their own module-level
`_domain_knowledge` global. The registry provides a single `get_domain_knowledge()`
accessor, set once by `PSNAgent.from_domain()`.

This replaced an earlier pattern where each module had its own `_domain_knowledge = None`
global plus a `set_domain_knowledge()` setter, and `from_domain()` called 9+ individual
setter functions with unique import aliases.

### The registry module

From `skillnet/core/dk_registry.py`:

```python
_current_dk: Optional["DomainKnowledge"] = None
_on_change_hooks: List[Callable] = []

def get_domain_knowledge() -> Optional["DomainKnowledge"]:
    """Return the current DomainKnowledge instance, or None."""
    return _current_dk

def set_domain_knowledge(dk: Optional["DomainKnowledge"]) -> None:
    """Set the global DomainKnowledge and notify on-change hooks."""
    global _current_dk
    _current_dk = dk
    for hook in _on_change_hooks:
        hook(dk)

def register_on_change(hook: Callable) -> None:
    """Register a callback invoked when DK changes."""
    _on_change_hooks.append(hook)

@contextmanager
def domain_knowledge_context(dk):
    """Context manager for test isolation."""
    old = _current_dk
    set_domain_knowledge(dk)
    try:
        yield
    finally:
        set_domain_knowledge(old)
```

### Consumer pattern

All 20 module-level consumers now follow the same pattern:

```python
from skillnet.core.dk_registry import get_domain_knowledge

def get_tool_tiers():
    """Return tool tier mapping from domain or empty default."""
    dk = get_domain_knowledge()
    if dk:
        config = dk.get_tool_tier_config()
        if config and "tool_tiers" in config:
            return config["tool_tiers"]
    return {}
```

### Cache invalidation hooks

Modules that maintain lazy caches derived from DK use `register_on_change()` to
invalidate their caches when the DK instance changes:

```python
from skillnet.core.dk_registry import register_on_change

_cached_data = None

def _invalidate(dk):
    global _cached_data
    _cached_data = None

register_on_change(_invalidate)
```

### Historical note

In earlier revisions, each module had its own `_domain_knowledge = None` global,
`_FALLBACK_*` constants with Minecraft defaults, and a `set_domain_knowledge()` setter.
`from_domain()` called 9 individual setters with aliased imports. This was replaced
by the central registry to reduce boilerplate and eliminate the risk of missing a new
module in the wiring orchestration.

---

## Pattern 3: Global Singleton Builder Registration

### Structure

Used by knowledge systems that maintain a process-wide singleton. Instead of injecting
data directly, domain code registers a **builder function** that the singleton calls
lazily to populate itself.

### `KnowledgeIndex` — `register_knowledge_builder()`

From `skillnet/agents/optimizer/knowledge/index.py`:

```python
_knowledge_builder: Optional[Callable[[], List[KnowledgeEntry]]] = None

def register_knowledge_builder(builder: Callable[[], List[KnowledgeEntry]]):
    """Register a domain-specific knowledge builder.

    Invalidates the cached singleton so the next get_knowledge_index()
    call rebuilds with the new builder.
    """
    global _knowledge_builder, _knowledge_index
    _knowledge_builder = builder
    _knowledge_index = None          # <-- cache invalidation

_knowledge_index: Optional[KnowledgeIndex] = None

def get_knowledge_index() -> KnowledgeIndex:
    """Get the knowledge index singleton."""
    global _knowledge_index
    if _knowledge_index is None:
        _knowledge_index = KnowledgeIndex()
    return _knowledge_index
```

When `KnowledgeIndex()` is constructed without explicit entries, it calls
`_build_index()`, which delegates to the registered builder (or a backward-compatible
auto-discovery fallback that imports `build_minecraft_knowledge_entries`).

### Domain-side registration: `MinecraftKnowledge.__init__()`

From `skillnet/domains/minecraft/knowledge/__init__.py`:

```python
class MinecraftKnowledge(DomainKnowledge):
    def __init__(self, model_name="gpt-5-mini", kr_llm=None):
        self._model_name = model_name
        self._kr_llm = kr_llm
        self._register_knowledge_builder()

    def _register_knowledge_builder(self):
        from skillnet.agents.optimizer.knowledge.index import register_knowledge_builder
        from skillnet.domains.minecraft.knowledge.index_builder import (
            build_minecraft_knowledge_entries,
        )
        register_knowledge_builder(build_minecraft_knowledge_entries)
```

Registration happens at `MinecraftKnowledge` construction time, which occurs
during `MinecraftDomain` initialization (before `PSNAgent.from_domain()` runs).

---

## Composition Convention: Domain-First Cascade with Fallback (built on Pattern 1)

This is **not a separate DI mechanism**. It is a code convention layered on top of
Pattern 1: the actual injection of `_domain_knowledge` onto the agent instance happens
via `set_domain_knowledge()`, exactly as described in Pattern 1. The convention covers
how mixin methods *read* that injected value.

### Structure

Used inside **mixins** that run as methods on an agent instance. The mixin probes
`self` for domain knowledge via `getattr`, calls a domain method, and falls back to
a module-level constant if the domain is absent or returns nothing.

Unlike Pattern 2 (which uses a central registry for module-level access), this
convention keeps the fallback data as module-level constants but reads domain
knowledge from the owning instance.

### Canonical example: `_get_type_config()` in `SkillNamingMixin`

From `skillnet/agents/parameterized_action/mixins/skill_naming.py`:

```python
class SkillNamingMixin:
    def _get_type_config(self):
        """Return (specific_types: set, type_param_patterns: list).

        Uses domain-provided keywords when available via
        self._domain_knowledge.get_type_keywords(), otherwise returns
        empty defaults (no types recognized, no patterns matched).
        """
        dk = getattr(self, '_domain_knowledge', None)
        if dk:
            cfg = dk.get_type_keywords()
            if cfg:
                types = set(cfg.get('specific_types', []))
                patterns = cfg.get('type_param_patterns', [])
                if types:
                    return types, patterns
        return set(), []
```

The same cascade logic appears in `skillnet/agents/refactor/base.py` where variant
suffixes are resolved. All fallback constants have been emptied — the domain-first
path is now the only source of real data:

```python
_FALLBACK_VARIANT_SUFFIXES = []  # Empty default — domain provides real data
dk = getattr(self, 'domain_knowledge', None)
variant_patterns = (
    dk.get_variant_suffixes() if dk and dk.get_variant_suffixes()
    else _FALLBACK_VARIANT_SUFFIXES
)
```

---

## Wiring Orchestration

All injection is performed in `PSNAgent.from_domain()` at
`skillnet/psn.py:406-544`. The method receives a `DomainModule` containing a
`DomainKnowledge` instance and wires it into every consumer.

### Instance-level DI (agents and managers)

These receive `knowledge` via `set_domain_knowledge()` on the object:

| Target | Access path | Line |
|--------|-------------|------|
| `ActionAgent` | `instance.action_agent` | 450 |
| `SkillGraphOptimizer` | `instance.optimizer` | 453 |
| `GraphPlanner` | `instance.planner` | 456 |
| `GraphManagerImpl` | `instance.skill_manager` | 459 |
| `CriticAgent` / `PSNCriticAgent` | `instance.critic_agent.unwrapped` | 536-537 |
| `PSNCurriculumAgent` | `instance.curriculum_agent.unwrapped` | 540-542 |

### Module-level DI (central registry)

`from_domain()` makes a single call to the central DK registry:

```python
from skillnet.core.dk_registry import set_domain_knowledge as _set_registry_dk
_set_registry_dk(knowledge)
```

All 20 utility modules that previously had individual `set_domain_knowledge()` setters
now read from `get_domain_knowledge()` in the registry. No per-module wiring is needed.

### Global singleton DI (knowledge systems)

These are **not** wired by `from_domain()` directly. Instead, `MinecraftKnowledge.__init__()`
calls `register_knowledge_builder()` at construction time. Since `MinecraftKnowledge` is
created as part of `MinecraftDomain`, this registration happens before `from_domain()` runs.

| Target | Registration function | Registered by |
|--------|-----------------------|---------------|
| `KnowledgeIndex` | `register_knowledge_builder()` | `MinecraftKnowledge.__init__()` |

---

## Thread Safety

The system assumes a **single-agent, single-thread** execution model. None of the
global state is protected by locks:

- Module-level `_domain_knowledge` variables are bare globals.
- `_knowledge_builder` and `_knowledge_index` in `index.py` are unguarded globals.

**Cache invalidation order matters** for Pattern 3. When `register_knowledge_builder()`
is called, it sets `_knowledge_index = None` to force a rebuild on next access. If
a concurrent thread reads `get_knowledge_index()` between the builder assignment and
the cache invalidation, it could see stale data. This is acceptable under the
single-agent assumption.

If multi-agent or multi-threaded execution is ever needed, all module-level globals
would need to be replaced with thread-local storage or protected by locks.

---

## Testing with DI

### Mocking domain knowledge for instance-level DI

Create a stub or `MagicMock` and call the setter directly:

```python
from unittest.mock import MagicMock

def test_action_agent_uses_domain():
    agent = ActionAgent.__new__(ActionAgent)
    agent._domain_knowledge = None

    knowledge = MagicMock()
    knowledge.get_system_prompt_template.return_value = "Test template: {programs}"
    knowledge.load_control_primitive_code.return_value = ["function prim() {}"]

    agent.set_domain_knowledge(knowledge)
    assert agent._domain_knowledge is knowledge
```

### Mocking module-level DI (central registry)

Use the `domain_knowledge_context()` context manager from `dk_registry` for test
isolation. It saves/restores the DK and fires on-change hooks automatically:

```python
from skillnet.core.dk_registry import domain_knowledge_context

def test_effect_matcher_uses_domain():
    mock_dk = MagicMock()
    mock_dk.get_item_categories.return_value = {"custom": {"keywords": ["x"], "items": ["x_item"]}}

    with domain_knowledge_context(mock_dk):
        from skillnet.agents.planning.effect_matcher._utils import get_item_categories
        result = get_item_categories()
        assert "custom" in result
```

### Mocking global singleton DI

For `KnowledgeIndex`, either pass explicit entries or mock the builder:

```python
from skillnet.agents.optimizer.knowledge.index import KnowledgeIndex, KnowledgeEntry

def test_custom_index():
    entries = [KnowledgeEntry(query_type="test", name="t", keywords=["k"], content="c")]
    index = KnowledgeIndex(entries=entries)
    assert len(index.entries) == 1
```

### Important: test isolation for module-level globals

The central DK registry and registered builders persist for the entire pytest session.
Use `domain_knowledge_context()` (which auto-restores) for test isolation:

```python
from skillnet.core.dk_registry import domain_knowledge_context

@pytest.fixture(autouse=True)
def reset_domain_state():
    with domain_knowledge_context(None):
        yield
```

For tests that need a specific DK instance:

```python
@pytest.fixture
def with_mock_dk():
    mock_dk = MagicMock()
    with domain_knowledge_context(mock_dk):
        yield mock_dk
```

### Save/restore pattern for class-scoped tests

Tests that call `from_domain()` with stubs (which calls `set_domain_knowledge()` on the
central registry) should use the `_restore_module_dk` fixture to save and restore the
registry state. This is simpler than `domain_knowledge_context()` for class-level
autouse fixtures:

```python
@pytest.fixture(autouse=True)
def _restore_module_dk(self):
    """Save/restore central DK registry around from_domain() tests."""
    from skillnet.core.dk_registry import get_domain_knowledge, set_domain_knowledge
    saved = get_domain_knowledge()
    yield
    set_domain_knowledge(saved)
```

This pattern is used in `test_from_domain.py`, `test_domain_wiring.py`, and
`test_curriculum_critic_wiring.py`.
