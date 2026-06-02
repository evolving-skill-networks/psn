# AGENTS.md

This file is read by coding agents (Claude Code, Cursor, OpenAI Codex, Aider, etc.) — see [agents.md](https://agents.md/) for the cross-tool convention. Humans should also find it useful as an architectural map of the PSN codebase.

## Quickstart for agents

```bash
make install      # pip install -e . + npm install in skillnet/domains/minecraft/action_space/env/mineflayer
make smoke        # one-line import-graph check
make run          # ./skillnet.sh
```

- **Architecture source of truth**: the package tree below + the 8 deep-dive docs under `docs/architecture/`.
- **Skills are executable code** (JavaScript in the Minecraft domain) executed inside Mineflayer; **Python is the orchestration layer**.
- **One file at a time** when translating or refactoring. Don't use bulk regex substitutions on Chinese-bearing files — the codebase has multilingual comments, and naive substitutions corrupt them.

## Helping a new user set up PSN

If the user is new and asks anything like "set up PSN" / "what do I install" / "how do I run this", execute the checklist below in order. The canonical user-facing instructions live in [README.md](README.md) and [`installation/fabric_mods_install.md`](installation/fabric_mods_install.md); this section is your **execution plan** with explicit gates and the commands you are expected to run for the user.

You drive the setup. Run the scripted parts yourself; only delegate to the user where a step truly requires GUI clicks or secrets they own (Modrinth, Minecraft world creation, API keys).

### Step 1 — STOP and ask which domain (and, if Minecraft, which connection path)

**This is a blocking gate. Do not run `make install`, do not edit any files, do not proceed to Step 2 until you have the user's explicit answers below.** Ask with whatever interactive-question tool your runtime exposes (e.g. `AskUserQuestion` in Claude Code).

**1a. Domain choice.** Ask which domain they want to set up:

> "PSN supports multiple domains. Which do you want to set up?
>
> • **Minecraft** — the original PSN domain. Bot runs in a Minecraft 1.19.4 world via mineflayer. Requires Java + a Minecraft instance.
> • Run `./skillnet.sh --help | grep -A4 'Domain Selection'` to see the live list of registered domains for this PSN install."

Record the answer. If the user picked anything other than Minecraft, **skip step 1b** (no connection path needed — the chosen domain handles its own runtime).

**1b. Minecraft connection path (only if 1a == Minecraft).** Ask:

> "PSN talks to Minecraft in one of two ways. Which do you want to set up?
>
> A. **Modrinth desktop client** — runs Minecraft with a GUI, you watch the bot in the game window. Best for development and demos. Needs a graphical session.
> B. **Headless Fabric server** — no GUI, PSN downloads + manages a local server jar for you. Best for remote machines, CI, or unattended runs. Needs Java 17+.
> C. **Both** — set up A now, also bootstrap B so you can switch at run time."

Record the answer (A / B / C). Steps 4 and 5 branch on it.

### Step 2 — Verify prerequisites (only what their choice needs)

Run these and report any failures back to the user *before* installing anything:

```bash
python --version    # must be 3.10.x; warn on 3.11/3.12; reject ≤3.9
node --version      # must be ≥ 16
```

If the user picked **B or C**, additionally:

```bash
java -version       # must be ≥ 17 (any vendor — OpenJDK / Temurin / Zulu)
```

If `java` is missing or older, **stop and tell the user to install OpenJDK 17+ first**. Do not try to install Java for them — package managers vary too widely. Suggest `apt install openjdk-17-jdk` / `brew install openjdk@17` / Zulu installer as appropriate. Path A users skip the Java check entirely; Modrinth bundles its own JRE inside the instance.

### Step 3 — Install Python/Node deps + apply patches

Run these yourself (no user interaction needed):

```bash
git clone https://github.com/evolving-skill-networks/psn.git && cd psn
conda create -n psn python=3.10 -y && conda activate psn
make install        # pip install -e .  +  npm install (root + mineflayer) + apply patches
make smoke          # expect: "PSN import graph OK"
```

If `make install` fails, read the error, fix the root cause, retry. Do **not** skip `make install` or hand-roll its individual steps; the patch step at the end is required.

### Step 4 — LLM credentials

```bash
cp .env.example .env
```

Then open `.env` and **tell the user which lines to fill in** (do not fill them in yourself — API keys are theirs):

- OpenAI: set `OPENAI_API_KEY=`. PSN is verified against `gpt-5-mini`.
- vLLM / local: uncomment `VLLM_API_BASE`, `VLLM_API_KEY`, `VLLM_MODEL`. Point them at `installation/vllm_setup.md` if they want to host the model themselves.

Once they've saved `.env`, run:

```bash
make verify         # imports / node_modules / patches / .env / java — all should pass
```

If any check fails, fix that one specific check before moving on.

### Step 5 — Minecraft setup, branched on Step 1 answer

#### If the user picked A or C — walk them through Path A

The Modrinth client install needs GUI clicks; you cannot script it. Walk the user through this list one step at a time and **wait for confirmation between blocks 1-3, 4-5, and 6** (use the interactive-question tool again):

1. Install the [Modrinth App](https://modrinth.com/app) (Linux AppImage / macOS dmg / Windows msi).
2. In Modrinth, create a new instance: Minecraft **1.19.4** + Fabric Loader. Modrinth installs both.
3. In that instance, **Mods → Add content** → install:
   - [Fabric API](https://modrinth.com/mod/fabric-api/versions?g=1.19.4&l=fabric)
   - [Multiplayer Server Pause](https://modrinth.com/mod/multiplayer-server-pause/versions?g=1.19.4&l=fabric) — required for PSN's `/pause` command; without it the world ticks during LLM calls and entities drift.
   - [iChunUtil](https://modrinth.com/mod/ichunutil/versions?g=1.19.4&l=fabric) — runtime dep of Multiplayer Server Pause (Modrinth usually auto-installs deps; verify after adding).
4. Launch the instance, create a singleplayer world with cheats allowed.
5. In the pause menu: **Open to LAN** → **Allow Cheats: ON** → **Start LAN World**.
6. Sanity-check the world is listening:
   ```bash
   ss -ltnp 2>/dev/null | awk '$NF ~ /java/' | grep LISTEN
   ```
   At least one Java LISTEN port should appear. If none, the user has not clicked "Open to LAN" — remind them. If multiple, PSN warns and picks the lowest; recommend `--mc-port=PORT` for explicitness.
7. Launch PSN:
   ```bash
   ./skillnet.sh           # auto-detects the LAN port via Minecraft handshake
   ```

#### If the user picked B or C — bootstrap Path B yourself

Path B is fully scripted. Run it for the user:

```bash
./skillnet.sh --mc-mode=headless
```

First run triggers `installation/setup_fabric_server.sh`, which downloads the Fabric 1.19.4 launcher (~165 KB), bootstraps the server jar + libraries (~50 MB), renders `installation/fabric_server/server.properties` from `installation/fabric_server_config.json`, and pre-generates the world. Subsequent runs reuse everything and skip setup. If the user wants non-default seed / gamemode / difficulty / port, **edit `installation/fabric_server_config.json` BEFORE first run** — tell them what you changed and why before changing it.

#### If C (both)

Do the A walkthrough first, then the B bootstrap. At run time the user picks:

```bash
./skillnet.sh                       # → Path A (Modrinth LAN client)
./skillnet.sh --mc-mode=headless    # → Path B (headless Fabric server)
```

### Step 6 — Final sanity check before declaring done

```bash
make verify
```

All five checks must say `OK`. If they do and the user has Minecraft running (Path A) or the headless server running (Path B), they can issue `./skillnet.sh` (or `--mc-mode=headless`) and PSN starts iterating.

### What not to do

- Do not suggest `--no-optimizer` / `--no-refactor` / `--pure-reasoning` / `--include-skill-code` to "fix" a failing setup. These are paper-ablation switches and disabling them will silently degrade PSN.
- Do not generate `.env` content for the user — they own their API keys. Show them what fields to fill and let them paste their own values.
- Do not modify `installation/fabric_server_config.json` defaults on the user's behalf without telling them what changed and why (it controls seed, gamemode, port — material to their experience).
- PSN connects to Minecraft only via README Path A (Modrinth LAN client) or Path B (managed headless Fabric server). There is no Microsoft/Azure auto-login code path; do not suggest an `azure_login=...` parameter or invent a Microsoft-account setup step. Route users to Path A / Path B.

## Project overview

Evolving Programmatic Skill Networks (PSN) is an LLM-powered embodied lifelong learning agent for Minecraft. The system learns reusable, composable skills as executable code organized in a directed skill graph (DAG). Skills improve through iterative feedback loops without fine-tuning the base LLM.

**Key innovation**: Evolving Programmatic Skill Networks — skills are defined as `s = (C_s, P_s, E_s, Children(s))` where:
- `C_s` = executable code
- `P_s` = parameters with types & semantics
- `E_s` = preconditions & expected effects
- `Children(s)` = child skill dependencies

## Build & run commands

### Installation
```bash
./skillnet.sh install        # pip install -e .  +  npm install (root + mineflayer)
```

### Running
```bash
./skillnet.sh                       # default — connect to open-to-LAN Minecraft client (auto-detects port)
./skillnet.sh --mc-mode=headless    # set up + manage a headless Fabric server (no GUI needed)
./skillnet.sh --mc-port=PORT        # explicit port override (skips auto-detect)
./skillnet.sh --manual              # manual task input mode
./skillnet.sh --inference           # single-task mode (don't iterate the curriculum)
./skillnet.sh --ckpt-dir=DIR        # use a specific checkpoint directory
./skillnet.sh --max-iterations=N    # cap iterations (default 3160)
./skillnet.sh --model=openai|vllm   # LLM backend (default reads .env)
./skillnet.sh --help                # full flag list including ablation switches
```

Ablation flags (`--no-optimizer`, `--no-refactor`, `--pure-reasoning`,
`--include-skill-code`) exist for reproducing experiments from the paper
and should NOT be suggested as fixes to a failing run.

### Python environment
Requires Python 3.10 and Node.js >= 16. Recommended conda environment:
```bash
conda create -n psn python=3.10
conda activate psn
```

## Architecture

### Core package structure
```
skillnet/
├── psn.py                        # PSNAgent facade (~1000 LOC), logic in _psn_impl/
├── _psn_impl/                    # 8 PSN agent mixins (+ 4 helper modules)
│   ├── escalation.py             # Optimization escalation protocol (Level 0→1→2)
│   ├── step_execution.py         # Main learning loop step logic
│   ├── task_management.py        # Task lifecycle, failure limits
│   ├── step_plan.py              # Planning phase orchestration
│   ├── step_execute_phase.py     # Execution phase orchestration
│   ├── step_optimize.py          # Optimization phase orchestration
│   ├── step_context.py           # Step context management
│   ├── effect_verification.py    # Post-execution effect checking
│   ├── event_processing.py       # Event handling
│   ├── event_helpers.py          # Event utility functions
│   ├── skill_recording.py        # Skill recording logic
│   └── recording_helpers.py      # Recording utilities
├── core/                         # Domain abstraction layer (ABCs & protocols)
│   ├── config.py                 # PSNConfig + 10 sub-dataclasses
│   ├── domain.py                 # DomainKnowledge ABC (6 abstract + 84 optional)
│   ├── dk_registry.py            # Central domain knowledge registry
│   ├── environment.py            # Environment ABC
│   ├── curriculum.py             # CurriculumStrategy ABC
│   ├── critic.py                 # CriticStrategy ABC
│   ├── skill_language.py         # SkillLanguage Protocol
│   └── knowledge_base.py         # KnowledgeBase for optimizer
├── domains/                      # Domain implementations
│   ├── minecraft/                # Minecraft domain (production)
│   │   ├── knowledge/            # 20+ domain knowledge modules
│   │   ├── prompts/              # 17 domain-owned LLM prompt templates
│   │   ├── environment.py        # MinecraftEnvironment
│   │   ├── curriculum.py         # MinecraftCurriculum
│   │   ├── critic.py             # MinecraftCritic
│   │   └── action_space/        # Minecraft action backend
│   │       ├── env/mineflayer/  # Minecraft bot (JavaScript)
│   │       └── control_primitives/  # JS primitives (mineBlock, craftRecipe, etc.)
├── languages/                    # Skill language implementations
│   ├── javascript.py             # JavaScript (Mineflayer)
│   └── python.py                 # Python support
└── agents/
    ├── parameterized_action/     # Skill generation (package: _agent.py + 7 mixins)
    ├── planner/                  # Multi-strategy planning (graph_planner.py + 5 mixins)
    ├── skill_graph/              # DAG skill storage
    │   ├── models/               # SkillNode, Precondition, Effect
    │   └── _impl/                # GraphManagerImpl
    ├── optimizer/                # Two-phase skill optimization
    │   ├── synthesis/            # Helper extraction for skill synthesis
    │   ├── core/                 # PureReflection engine
    │   ├── phases/               # Phase 1/Phase 2 pipeline
    │   ├── feedback/             # Optimization feedback types & propagation
    │   ├── transforms/           # Code edit operators
    │   ├── validators/           # Semantic validators
    │   ├── tracking/             # Optimization history tracking
    │   ├── knowledge/            # Domain-agnostic knowledge retrieval
    │   ├── analysis/             # Root cause analysis
    │   └── _impl/                # OptimizerImpl
    ├── evolution/                # Skill Evolution Manager (cross-skill pattern analysis)
    ├── psn_curriculum/           # Automatic task generation (agent.py + 5 mixins)
    ├── planning/                 # Effect matching, inference
    └── refactor/                 # Skill deduplication/extraction
```

### Key agent components

1. **PSNAgent** (`psn.py` + 11 mixins in `_psn_impl/`): Main orchestrator facade
2. **PSNConfig** (`core/config.py`): Structured configuration with 10 sub-dataclasses (replaces 50+ kwargs)
3. **DomainModule/DomainKnowledge** (`core/domain.py`): Domain abstraction layer with 6 abstract + 84 optional methods
4. **SkillLanguage** (`core/skill_language.py`): Protocol for JavaScript and Python skill support
5. **ParameterizedActionAgent** (`parameterized_action/_agent.py` + 7 mixins): Generates parametric skills with iterative LLM prompting
6. **SkillGraphManager**: Manages skill DAG with versioning, statistics, and persistence
7. **SkillGraphOptimizer**: Two-phase optimization (Phase 1: root cause analysis, Phase 2: code transformation)
8. **GraphPlanner** (`planner/graph_planner.py` + 5 mixins): Backward chaining through skill graph with fallback to LLM generation
9. **PSNCurriculumAgent** (`psn_curriculum/agent.py` + 5 mixins): Automatic curriculum with goal planning and skill gap analysis
10. **EscalationTracker** (`_psn_impl/escalation.py`): Per-task failure tracking with Level 0→1 escalation when V(s) shows no improvement
11. **SkillEvolutionManager** (`agents/evolution/manager.py`): LLM-driven cross-skill failure pattern analysis; triggers Level 2 escalation
12. **HelperExtractor** (`agents/optimizer/synthesis/extractor.py`): Extracts large inline helpers from optimized code into independent reusable skills

### Data flow
1. Domain wiring: `PSNAgent.from_domain(DomainModule)` injects domain knowledge into all subsystems.
2. Curriculum generates a task → Planner finds a skill composition → Action agent generates/executes code.
3. On failure: Optimizer analyzes root cause → propagates feedback → transforms code → retries.

### Domain abstraction & dependency injection

The system uses a domain abstraction layer to decouple core logic from Minecraft-specific details:

- **4 ABCs + 1 Protocol** in `skillnet/core/`: Environment, DomainKnowledge, CurriculumStrategy, CriticStrategy, SkillLanguage
- **DomainModule** (`core/domain.py`): Bundles all domain components (knowledge, environment, curriculum, critic, skill language)
- **`PSNAgent.from_domain(DomainModule, PSNConfig)`**: Production entry point that wires the domain into all subsystems
- **DomainKnowledge**: 6 abstract + 84 optional methods (90 total); `MinecraftKnowledge` implements all
- **Central DK Registry** (`core/dk_registry.py`): A single `get_domain_knowledge()` accessor replaced 20 module-level singletons. `from_domain()` makes one registry call plus instance-level setters on agents.
- **Domain-owned prompts** (`domains/minecraft/prompts/`): 17 LLM prompt templates loaded via `dk.get_prompt(name)`.
- **Extraction status**: All data constants, LLM prompts, and game mechanics have been extracted from `agents/`. `agents/` and `_psn_impl/` contain zero Minecraft-specific logic. Remaining in `agents/`: LLM prompt context strings (non-logic), `item_name_adapter.py` mappings, `_param_constants.py` transforms (all already DK-wired).
- **3 DI patterns used**:
  1. Instance-level injection (`set_domain_knowledge()` on agent classes)
  2. Central registry (`get_domain_knowledge()` from `dk_registry`)
  3. Global singleton registration (`register_knowledge_builder()` for KnowledgeIndex)
- See `docs/architecture/di_patterns_handbook.md` and `docs/architecture/domain_integration_guide.md` for details.

### Mixin architecture

Large agent classes are split into facade + mixins for maintainability:

- **61 mixins** across 6 facade classes (PSNAgent, ParameterizedActionAgent, GraphPlanner, PSNCurriculumAgent, SkillGraphManager, SkillGraphOptimizer)
- **Pattern**: no mixin `__init__`; cross-mixin calls via `self.xxx()` (resolved by MRO); shared `self` state
- **Facade** owns `__init__` and public API; mixins provide implementation
- See `docs/architecture/mixin_architecture.md` for conventions and examples.

### Checkpoint structure
```
ckpt/
├── skill_graph/
│   ├── code/           # *.js skill files
│   ├── metadata/       # *.json parameter/type info
│   ├── preconditions/  # *.json execution preconditions
│   ├── effects/        # *.json expected outcomes
│   └── graph/          # Skill dependency graph
├── logs/               # Execution logs
└── progress/           # Learning progress tracking
```

## Key concepts

### Skill value function
`V(s) = p̂_s - u_s` where:
- `p̂_s` = Laplace-smoothed success rate
- `u_s` = uncertainty measure

Used to prioritize which skills to optimize.

### Two-phase optimization
1. **Phase 1**: LLM reflection analyzes failures → generates targeted feedback
2. **Phase 2**: Apply code transformations based on the feedback

Phase 1 detects thinking-model context exhaustion (`finish_reason=length`) and auto-retries with thinking disabled.

### Parameter inference engine
Multi-phase cascade mapping skill outputs to next-skill inputs:
- **LLM-first mode** (default): LEARNED_RULES → PureLLM → STATE_MAPPING → SEMANTIC → EFFECT_EXTRACTION → CONTEXT → DEFAULT
- **PureLLMParameterResolver** (`planning/inference/pure_llm_resolver.py`): uses the full 18-field InferenceContext (including environment: nearby_blocks, biome, position, equipment)
- **ParameterSemantic** (`planning/inference/parameter_semantic.py`): StateMapping with 8 StateTypes, TransformHint
- **Optimizer feedback loop**: `reflection_chain.py:_apply_inference_fix()` propagates corrections back to parameter semantic metadata.

### Optimization escalation protocol
Three-level escalation when optimization repeatedly fails:
- **Level 0**: normal optimizer Phase 1 + Phase 2
- **Level 1**: regenerate code with failure history as a negative example (triggered after K consecutive failures with no V(s) improvement)
- **Level 2**: cross-skill pattern analysis via SkillEvolutionManager (LLM-driven, no heuristic)

Config: `OptimizationConfig.escalation_min_attempts` (K=3), `escalation_improvement_threshold` (δ=0.05).
On task success, immediately de-escalate to Level 0.

### Skill synthesis
After a successful optimization, `HelperExtractor`
(`agents/optimizer/synthesis/extractor.py`) uses an LLM to find inline
helper functions in the optimized code that could be reused by other
skills. A candidate is extracted as a new standalone skill via
`add_new_skill()` only when:

- the LLM marks it `is_generic = true` (useful beyond this specific task),
- it has more than 5 lines of actual logic,
- it is **defined inline** in the parent — verified via AST-grade
  pattern check against function declarations, arrow functions, and
  `const|let|var name = function/arrow` bindings. The earlier substring
  check let call sites slip through, which caused the LLM to invent
  helpers from `await mineLogs(...)` call sites and overwrite the real
  `mineLogs` skill (see commit 67bc13b).

Successfully extracted helpers are removed from the parent code and
registered with their LLM-chosen name; if a name collides with an
existing skill the extraction is rejected upstream by `add_new_skill`'s
existing-skill branch.

## Code conventions

- Skills are executable code — JavaScript in the Minecraft domain — executed in the Mineflayer environment
- Python backend that generates skill code in the domain's language (JavaScript for Minecraft)
- Checkpoint directories prefixed with `ckpt_`

## Important files for understanding flow

- `skillnet/psn.py`: main learning-loop orchestration (facade)
- `skillnet/_psn_impl/step_execution.py`: main learning-loop step logic
- `skillnet/core/domain.py`: domain abstraction layer (DomainKnowledge, DomainModule)
- `skillnet/core/config.py`: PSNConfig structured configuration
- `skillnet/domains/minecraft/__init__.py`: Minecraft domain factory
- `skillnet/agents/parameterized_action/_agent.py`: skill generation logic
- `skillnet/agents/optimizer/engine.py`: optimization entry point
- `skillnet/agents/planner/graph_planner.py`: planning and skill composition
- `skillnet/agents/skill_graph/_impl/graph_manager_impl.py`: skill storage/retrieval
- `skillnet/agents/planning/inference/parameter_engine.py`: parameter inference orchestration
- `skillnet/agents/planning/inference/pure_llm_resolver.py`: LLM-based parameter resolution
- `skillnet/_psn_impl/escalation.py`: optimization escalation protocol (Level 0→1→2)
- `skillnet/agents/evolution/manager.py`: cross-skill failure pattern analysis (LLM-driven)
- `skillnet/agents/optimizer/synthesis/extractor.py`: helper extraction for skill synthesis

## Architecture documentation

Detailed architecture docs in `docs/architecture/`:

- `domain_integration_guide.md`: how to implement a new domain (ABCs, wiring, checklist)
- `di_patterns_handbook.md`: dependency injection patterns used across the codebase
- `mixin_architecture.md`: mixin conventions, MRO rules, and extraction patterns
- `config_architecture.md`: PSNConfig structure, sub-dataclasses, and migration guide
- `knowledge_retrieval.md`: knowledge index, LLM-driven retrieval, and domain injection
- `skill_graph_evolution.md`: skill DAG versioning, statistics, and persistence
- `planner_strategies.md`: backward chaining, decomposition, and fallback strategies
- `optimization_pipeline.md`: two-phase pipeline, P(update s) gating, and gradient propagation

