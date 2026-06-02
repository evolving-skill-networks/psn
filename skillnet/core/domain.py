"""
Domain Knowledge ABC and DomainModule Container

DomainKnowledge defines the interface for domain-specific knowledge that
the optimizer, action agent, and planner consume.

DomainModule bundles all domain-specific components into a single object
that can be passed to PSNAgent.from_domain().

Usage:
    from skillnet.core.domain import DomainKnowledge, DomainModule

    class MyKnowledge(DomainKnowledge):
        def get_all_knowledge(self): ...
        ...

    module = DomainModule(
        environment=my_env,
        knowledge=my_knowledge,
        curriculum=my_curriculum,
        critic=my_critic,
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from skillnet.core.environment import Environment
from skillnet.core.curriculum import CurriculumStrategy
from skillnet.core.critic import CriticStrategy

if TYPE_CHECKING:
    from skillnet.core.skill_language import SkillLanguage
    from skillnet.core.observation import Observation


class DomainKnowledge(ABC):
    """
    Abstract interface for domain-specific knowledge.

    Provides knowledge items, control primitives, prompt templates, and
    item categorization rules that the PSN core consumes.
    """

    # ------------------------------------------------------------------
    # Required — subclasses MUST implement these.
    # ------------------------------------------------------------------

    @abstractmethod
    def get_all_knowledge(self) -> List[Any]:
        """Return all knowledge items for this domain."""

    @abstractmethod
    def get_knowledge_for_error(
        self, error_text: str, code: str = ""
    ) -> List[Any]:
        """Return knowledge items relevant to the given error."""

    @abstractmethod
    def get_control_primitives(self) -> List[str]:
        """Return names of control primitive functions."""

    @abstractmethod
    def load_control_primitive_code(self) -> List[str]:
        """Return source code strings for all control primitives."""

    def load_control_primitive_context(
        self,
        primitive_names: Optional[List[str]] = None,
        model_profile: Optional[str] = None,
        for_optimizer: bool = False,
    ) -> List[str]:
        """Return LLM-prompt context descriptions (compact stubs) for primitives.

        These are the short signature/summary versions of primitives that get
        pasted into agent prompts — distinct from the full runtime source
        returned by ``load_control_primitive_code``.

        Default: empty list. Domains with LLM-facing primitive descriptions
        should override.
        """
        return []

    def get_runtime_primitive_names(self) -> List[str]:
        """All function names defined in the runtime action layer, INCLUDING
        helpers (utility functions used by primitives). Used by code validation
        to prevent LLM-generated skill code from shadowing runtime names.

        Default: same as ``get_control_primitives()`` — domains whose runtime
        has helper functions beyond the user-facing primitives should override.
        """
        return self.get_control_primitives()

    @abstractmethod
    def get_skill_language(self) -> str:
        """Return the programming language for skills (e.g., 'javascript', 'python')."""

    @abstractmethod
    def get_skill_language_impl(self) -> "SkillLanguage":
        """Return the SkillLanguage instance for this domain's skill code.

        Used by agents that need to parse/validate/format skill code
        without hard-coding the language. The instance should be cached
        (per-DK) — agents may call this on hot paths.
        """

    @abstractmethod
    def extract_observation(self, events: List[Any]) -> "Observation":
        """Parse the env's event stream into a normalized Observation.

        Each domain implements its own event-shape parsing. The Observation
        must always be returnable, even if the event stream is empty or
        malformed — return an empty Observation rather than raising.
        """

    @abstractmethod
    def render_observation(
        self,
        *,
        events: List[Any],
        chest_observation: str = "",
        completed_tasks: Optional[List[str]] = None,
        failed_tasks: Optional[List[str]] = None,
        progress: int = 0,
        **kwargs,
    ) -> Dict[str, str]:
        """Format the current observation + curriculum context as a dict of
        section-name -> formatted text fragment. Callers concatenate the
        fragments to assemble the final prompt.

        Standardized keys (subset; each domain may add domain-specific keys):
          - "context": optional preamble (often empty)
          - "inventory": "Inventory (used/total): {items}\\n\\n"
          - "chests": chest_observation passed through
          - "completed_tasks": "Completed tasks so far: ...\\n\\n"
          - "failed_tasks": "Failed tasks that are too hard: ...\\n\\n"

        MinecraftKnowledge adds: biome, time, nearby_blocks, other_blocks,
        nearby_entities, health, hunger, position, equipment.
        A custom domain's knowledge might add: health, food, drink, energy, daylight,
        milestones, achievements_count.
        """

    @abstractmethod
    def get_system_prompt_template(self) -> str:
        """Return the system prompt template for the action agent.

        The template MUST contain ``{programs}`` and ``{response_format}``
        placeholders.  These are filled by the action agent's rendering
        pipeline with the concatenated control-primitive source code and
        the standard JSON response format, respectively.
        """

    # ------------------------------------------------------------------
    # Optional — have sensible defaults so domains can implement gradually.
    # ------------------------------------------------------------------

    def get_item_categories(self) -> Dict[str, Dict[str, Any]]:
        """Return item category mapping for effect matching.

        Format::

            {
                "category_name": {
                    "keywords": ["keyword1", ...],   # substring match
                    "items": ["item1", ...],          # exact match
                    "actions": ["action1", ...],      # associated actions
                }
            }

        Empty dict means use built-in defaults.
        """
        return {}

    def get_safe_interchange_categories(self) -> set:
        """Return set of item category names where items are interchangeable.

        E.g., all log types are interchangeable for crafting purposes.
        Used by the effect matcher to allow flexible matching within
        safe categories.

        Empty set means no safe interchange categories.
        """
        return set()

    def get_core_inventory_pattern(self) -> str:
        """Return regex pattern for core inventory items.

        Used by curriculum to identify important inventory items.
        Empty string means use the config default (Minecraft pattern).
        """
        return ""

    def get_task_complexity_tiers(self) -> Dict[str, list]:
        """Return keyword tiers for dynamic task failure limits.

        Format: {"high": ["keyword1", ...], "medium": ["keyword2", ...]}
        High-complexity tasks get 2x the default failure limit,
        medium get 1.5x. Empty dict means all tasks use the default limit.
        """
        return {}

    def get_tool_unlock_mapping(self) -> Dict[str, list]:
        """Return mapping of tools to task keywords they unlock.

        When a tool is newly acquired, tasks matching unlocked keywords
        get their failure counters reset. Empty dict means no resets.
        """
        return {}

    def get_type_keywords(self) -> Dict[str, list]:
        """Return domain-specific type keywords for skill name normalization.

        Format::

            {
                "specific_types": ["oak", "birch", ...],
                "type_param_patterns": [r"\\b(woodType|...)\\b"],
            }

        ``specific_types`` lists item/material variant names that appear in
        skill names (e.g. ``craftOakPlanks``).  When a skill has a type
        parameter AND its name contains one of these, the name is generalized.

        ``type_param_patterns`` lists regex patterns that identify type
        parameters in function signatures or bodies.

        Empty dict means use built-in defaults.
        """
        return {}

    def get_skill_verb_prefixes(self) -> List[str]:
        """Return verb prefixes used in skill naming for target extraction.

        E.g., ['craft', 'mine', 'smelt'] — used to extract 'diamond_pickaxe'
        from 'craftDiamondPickaxe'.
        Default: empty list (no target extraction).
        """
        return []

    def get_known_functions(self) -> Dict[str, Set[str]]:
        """Return domain-specific function sets for code validation.

        Keys: 'bot_methods', 'primitives', 'helpers'
        Used by function reference validation to avoid false positives.
        Empty dict means use built-in defaults.
        """
        return {}

    # ------------------------------------------------------------------
    # Optional — prompt template methods.
    # ------------------------------------------------------------------

    def get_prompt(self, name: str) -> str:
        """Return prompt template by name, or empty string if not available.

        Central dispatcher for domain-owned prompt templates.  Domains
        override this to load from their own prompt files.

        Known names: ``parameterized_action_template``, ``action_template``,
        ``action_response_format``, ``critic``, ``psn_critic``,
        ``curriculum``, ``psn_curriculum``, ``curriculum_task_decomposition``,
        ``curriculum_qa_step1_ask_questions``,
        ``curriculum_qa_step2_answer_questions``, ``skill``,
        ``reparameterize``, ``parameter_extraction``.
        """
        return ""

    def get_critic_prompt_template(self) -> str:
        """Return the system prompt template for the critic agent.

        Used by PSNCriticAgent to evaluate task success/failure.
        Empty string means use the built-in default prompt.
        """
        return ""

    def get_curriculum_prompt_template(self) -> str:
        """Return the system prompt template for the curriculum agent.

        Used by PSNCurriculumAgent to propose tasks.
        Empty string means use the built-in default prompt.
        """
        return ""

    def get_action_response_format(self) -> str:
        """Return the response format template for the action agent.

        Injected into the action system prompt via the ``{response_format}``
        placeholder.  Empty string means use the built-in default format.
        """
        return ""

    def get_reparameterize_prompt_template(self) -> str:
        """Return the prompt for reparameterizing non-parameterized functions.

        Used by CodeAssemblyMixin to add parameters to single-arg functions.
        Empty string means use the built-in default prompt.
        """
        return ""

    # ------------------------------------------------------------------
    # Optional — optimizer knowledge methods.
    # ------------------------------------------------------------------

    def get_domain_name(self) -> str:
        """Return human-readable domain name for prompt text.

        Used in optimizer analysis prompts, e.g.
        ``"You have access to {domain_name} facts..."``.
        """
        return "domain"

    def get_environment_rules(self, error_text: str) -> List[str]:
        """Return environment rules relevant to the given error.

        Each string is a single formatted rule line, e.g.::

            "Block placement requires adjacent solid block. Consequence: ..."

        Used by the optimizer's analysis prompt to provide domain-specific
        environment constraints.  Empty list means no rules matched.
        """
        return []

    def get_primitive_knowledge(self, error_text: str, code: str = "") -> str:
        """Return formatted primitive/API knowledge for the error.

        Returns a formatted string block ready for injection into the
        optimizer analysis prompt, or empty string if nothing relevant.
        """
        return ""

    def get_environment_context(self, state: dict) -> str:
        """Return formatted environment context from execution pre-state.

        The *state* dict comes from the skill's pre-execution snapshot.
        Domains extract whatever context is relevant (biome, room layout,
        agent location, etc.) and return it as a markdown fragment.

        Example return value::

            "\\n\\n**Current Environment:**\\n- Biome: plains\\n- Resources: oak_log, ..."

        Empty string means no environment context available.
        """
        return ""

    def get_reasoning_examples(self, error_text: str, max_examples: int = 2) -> str:
        """Return formatted reasoning examples/guides for the error.

        Returns a formatted string block ready for injection into the
        optimizer analysis prompt, or empty string if no relevant examples.
        """
        return ""

    def get_detailed_knowledge_for_error(
        self, error_text: str, code: str = "", chat_log: str = "",
        kr_logger=None,
    ) -> str:
        """Return detailed knowledge text for error analysis with full context.

        Unlike :meth:`get_knowledge_for_error` which returns structured items,
        this returns a pre-formatted knowledge text string using LLM-driven
        retrieval with domain-specific query generation and keyword fallback.
        Receives *chat_log* for richer error diagnosis.

        Implementations should create an ``LLMKnowledgeRetriever`` with
        domain-specific ``query_prompt`` and ``fallback_queries_fn``.

        Args:
            kr_logger: Optional diagnostic logger for KnowledgeRetrieval log.

        Empty string means fall back to :meth:`get_knowledge_for_error`.
        """
        return ""

    def get_code_validators(self) -> List[Any]:
        """Return domain-specific code validators for the optimizer pipeline.

        Each validator is a callable: ``(code: str, context: dict) -> List[dict]``
        where each dict has ``{method, line, message}`` for errors found.
        Return empty list for no domain-specific validators.

        Context dict may contain: ``primitives`` (set of known primitive names),
        ``skill_name``, ``available_skills``.
        """
        return []

    def get_effect_extraction_verbs(self) -> Dict[str, List[str]]:
        """Return verb patterns for rule-based effect extraction.

        Maps effect types to lists of verb patterns used in regex matching.
        E.g.::

            {
                "find": ["find", "locate", "discover", "search"],
                "collect": ["mine", "collect", "get", "obtain", "gather", ...],
                "craft": ["craft"],
                "ensure": ["ensure"],
                "place": ["place"],
                "equip": ["equip"],
            }

        Empty dict means use no domain-specific verb patterns.
        """
        return {}

    def get_variant_patterns(self) -> List[Tuple[str, str]]:
        """Return variant detection patterns for skill refactoring.

        Each tuple is ``(regex_pattern, suggested_param_name)``.
        E.g.::

            [
                (r'(oak|birch|spruce|...)', 'woodType'),
                (r'(iron|gold|diamond|...)', 'materialType'),
            ]

        Empty list means no variant patterns.
        """
        return []

    def get_variant_suffixes(self) -> List[str]:
        """Return regex patterns that identify skill-name variant words.

        Used by the refactor detector to decide whether a skill name is a
        specialised variant of a more general skill.
        E.g.::

            [
                r'oak|birch|spruce|jungle|acacia|dark_?oak|mangrove|cherry',
                r'iron|gold|diamond|netherite|copper|coal|redstone',
                r'wooden|stone|leather',
            ]

        Empty list means no variant suffix patterns.
        """
        return []

    def get_resource_not_found_patterns(self) -> dict:
        """Return resource-specific error patterns for environmental feedback inference.

        Maps resource names to lists of error message substrings.  Used by
        ``infer_environmental_feedback_from_error`` to detect missing-resource
        errors from execution output.

        Example::

            {
                "diamond": ["no diamond", "diamond not found", "couldn't find diamond"],
            }

        Empty dict means no resource-not-found patterns.
        """
        return {}

    def get_inference_rules(self) -> dict:
        """Return domain-specific inference rules for parameter resolution.

        The returned dict is opaque to the core — domains define the format.
        For Minecraft this contains WOOD_TYPES, RAW_ORE_TYPES, etc.

        Empty dict means no inference rules.
        """
        return {}

    def get_require_patterns(self) -> list:
        """Return require statement patterns for code generation.

        Each entry is a tuple of (regex_pattern, var_name, canonical_declaration).
        Used to detect and normalize require() statements in skill code.
        """
        return []

    def get_global_dependencies(self) -> dict:
        """Return globally-provided dependency declarations.

        Keys are variable names injected by the environment runtime
        (e.g., via globalDepsCode).  Values are the ``require`` / import
        declaration strings used when the dependency is needed inside
        composed code.

        Example (Minecraft)::

            {
                "mcData": "const mcData = require('minecraft-data')(bot.version);",
                "Vec3": "const Vec3 = require('vec3').Vec3;",
            }

        Empty dict means no global dependencies (safe default).
        """
        return {}

    def get_environment_globals(self) -> set:
        """Return variable names provided by the execution runtime.

        These names are available in scope at runtime (injected before
        eval) but do NOT need require/import declaration statements.
        Used by dependency analysis to avoid re-injecting already-provided
        variables.

        Default: empty set.
        """
        return set()

    def get_primitive_semantics(self) -> Dict[str, Dict[str, bool]]:
        """Return produce/consume semantics for control primitives.

        Maps primitive function names to dicts with ``produces`` and
        ``consumes`` boolean flags.  Used by the code verifier to check
        whether a primitive call implements an effect.

        Example (Minecraft)::

            {
                "mineBlock": {"produces": True, "consumes": False},
                "craftItem": {"produces": True, "consumes": True},
            }

        Empty dict means no primitive semantics available.
        """
        return {}

    def get_registry_access_patterns(self) -> list:
        """Return regex patterns for domain-specific data registry lookups.

        Each pattern matches code that accesses domain-specific registries
        (e.g., ``mcData.itemsByName[``, ``bot.registry.blocksByName``).
        Used by code quality validators and skill name normalization.

        Default: empty list (no domain-specific registry patterns).
        """
        return []

    def get_resource_error_patterns(self) -> list:
        """Return domain-specific error patterns for feedback analysis.

        Each entry is ``(regex_pattern, issue_type, description)``.
        Used by edit analysis to match domain-specific runtime errors
        and suggest targeted code edits.

        Default: empty list (no domain-specific error patterns).
        """
        return []

    def get_type_adapter_config(self) -> dict:
        """Return domain-specific type conversion rules and suffix requirements.

        Returns dict with optional keys:

        - ``"conversion_rules"``: Dict of named rules with pattern/adapter/description
        - ``"suffix_requirements"``: Dict mapping param name → expected suffix

        Default: empty dict (only generic JS type conversions apply).
        """
        return {}

    def get_mine_keywords(self) -> list:
        """Return keywords indicating mining/extraction actions.

        Used by ``determine_delta_prefix`` to choose 'mine' vs 'craft' prefix.
        Default: empty list (always returns 'craft').
        """
        return []

    def get_func_prefix_to_direction(self) -> dict:
        """Return ``{func_prefix: direction_name}`` for parameter direction inference.

        Values should be :class:`DirectionSemantic` enum values (``'input'``, ``'output'``).
        Default: empty dict (no prefix-based direction inference).
        """
        return {}

    def get_param_suffix_to_transform(self) -> list:
        """Return ``[(suffix, transform_dict), ...]`` for parameter transform hints.

        Each *transform_dict* has keys matching :class:`TransformHint` fields:
        ``transform_type``, ``source_suffix``/``target_suffix`` or ``pattern``/``replacement``.
        Default: empty list (no suffix-based transforms).
        """
        return []

    def get_plural_to_singular(self) -> dict:
        """Return ``{plural: singular}`` item-name mapping for task-layer normalisation.

        Default: empty dict (no plural normalisation).
        """
        return {}

    def get_suffix_to_group(self) -> dict:
        """Return ``{suffix: group_name}`` for dynamic item pattern detection.

        E.g. ``{'_log': 'logs', '_planks': 'planks', '_pickaxe': 'pickaxes'}``.
        Default: empty dict (no dynamic pattern detection).
        """
        return {}

    def get_task_to_group_mapping(self) -> dict:
        """Return ``{natural_language_word: group_key}`` for task-layer item lookup.

        E.g. ``{'logs': 'logs', 'planks': 'planks', 'pickaxe': 'pickaxes'}``.
        Default: empty dict (no task-to-group mapping).
        """
        return {}

    def get_ore_to_drop_mapping(self) -> dict:
        """Return ``{ore_block_name: drop_item_name}`` for natural ore drops (no Silk Touch).

        E.g. ``{'diamond_ore': 'diamond', 'iron_ore': 'raw_iron'}``.
        Default: empty dict (no ore mapping).
        """
        return {}

    def get_progression_milestones(self) -> list:
        """Return list of milestone dicts defining domain progression.

        Each dict has keys: ``name``, ``description``, ``required_items``
        (``{item_or_group: count}``), ``stage`` (``'early'/'mid'/'late'/'end'``),
        and optional ``required_achievements`` (``list[str]``).
        Default: empty list (no milestones).
        """
        return []

    def get_world_geometry_config(self) -> Dict[str, Any]:
        """World-geometry constants used by PSN-core agents.

        Keys (all optional; PSN-core code tolerates missing keys):
          - "coord_dims": int  -- 2 or 3
          - "has_vertical": bool
          - "surface_y": Optional[int]      -- Y at which "surface" begins
          - "underground_threshold": Optional[int] -- Y below which we're "underground"

        Default: 3D world with no vertical reference, to match domains
        that don't have a meaningful Y axis.
        """
        return {"coord_dims": 3, "has_vertical": False}

    def get_param_type_patterns(self) -> dict:
        """Return ``{param_keyword: [item_suffix_patterns]}`` for parameter compatibility.

        Used to check whether an item name is compatible with a parameter's
        semantic type.  E.g. a parameter containing ``"plank"`` expects items
        matching ``["_planks", "_plank"]``.
        Default: empty dict (all items considered compatible).
        """
        return {}

    def get_milestone_item_mapping(self) -> dict:
        """Return ``{item_name: milestone_name}`` for checkpoint/progress resume.

        Maps inventory items to the milestone they indicate completion of.
        Used by iteration utilities to rebuild progress from inventory snapshots.
        Default: empty dict (no milestone tracking).
        """
        return {}

    def get_resource_name_mappings(self) -> dict:
        """Return ``{resource_key: [item_names]}`` for resource name consistency checks.

        Maps resource keywords (extracted from skill names) to valid item names.
        Used by contract scanning to verify skill code targets correct resources.
        Default: empty dict (no resource validation).
        """
        return {}

    # ------------------------------------------------------------------
    # Optional — curriculum/orchestrator methods.
    # ------------------------------------------------------------------

    def get_complex_task_keywords(self) -> list:
        """Return keywords that indicate a task is complex and should be decomposed.

        Tasks containing any of these keywords will be force-decomposed
        by the DecompositionFilter (unless a reliable matching skill exists).

        Returns:
            List of keyword strings.
        """
        return []

    def get_simple_task_patterns(self) -> list:
        """Return regex patterns that identify simple (non-decomposable) tasks.

        Tasks matching any of these patterns will NOT be decomposed.
        Returns list of regex pattern strings.
        """
        return []

    def get_resource_thresholds(self) -> dict:
        """Return default resource tracking thresholds.

        Returns dict mapping resource names to threshold configs:
        {"logs": {"min_count": 16, "target_count": 26, "priority": 6}, ...}
        """
        return {}

    def get_reset_commands(self) -> List[str]:
        """Return chat commands to run after environment reset.

        E.g., ['/time set day', '/difficulty peaceful'].
        Empty list means no reset commands.
        """
        return []

    def get_reset_code(self) -> str:
        """Return executable code to run after environment reset.

        Returns a ready-to-execute code string in the domain's skill language.
        This code is passed directly to env.step().
        Default: wraps get_reset_commands() in bot.chat() calls,
        or empty string if no commands.
        """
        commands = self.get_reset_commands()
        if not commands:
            return ""
        return "\n".join(f"bot.chat('{cmd}');" for cmd in commands)

    def get_task_skill_mapping(self) -> Dict[str, list]:
        """Return mapping from task descriptions to skill name lists.

        E.g., {"craft planks": ["craftPlanks", "ensurePlanks"]}.
        Empty dict means no predefined mappings.
        """
        return {}

    def get_display_resource_groups(self) -> List[str]:
        """Resource group names for inventory display.

        Returns a list of group or item names to show when displaying
        current inventory status.  E.g. ``["logs", "planks", "diamond"]``.
        Default: empty list (no display groups).
        """
        return []

    def get_tool_type_names(self) -> List[str]:
        """Generic tool type names for task classification.

        Used to detect whether a task is a tool-crafting task.
        E.g. ``["pickaxe", "axe", "sword"]``.
        Default: empty list (no tool classification).
        """
        return []

    def get_initial_task(self) -> str:
        """First task for a new agent.

        Returns a task string to assign at progress == 0,
        e.g. ``"Ensure you have 4 wood logs"``.
        Default: empty string (skip initial task override).
        """
        return ""

    def get_crafting_chain_config(self) -> dict:
        """Return crafting chain configuration for parameter inference.

        Returns dict with optional keys:
        - "config_contexts": Dict mapping config types to item suffix patterns
        - "input_material_params": List of parameter name patterns for input materials
        - "product_to_material_mappings": List of (product_suffix, material_suffix) tuples

        Empty dict means use built-in defaults.
        """
        return {}

    def check_tool_efficiency(self, task: str, inventory: dict) -> Optional[str]:
        """Check if current tools are efficient for the task.

        Returns a tool crafting task string if inefficient, None if ok.
        """
        return None

    def get_revert_failed_action_code(self, blocks: list, positions: list) -> Optional[str]:
        """Return code string to revert side effects of a failed action.

        Args:
            blocks: List of block names that were placed.
            positions: List of position dicts for each placed block.

        Returns:
            A code string to execute for reverting, or None for no-op.
        """
        return None

    # ------------------------------------------------------------------
    # Optional — item knowledge methods.
    # ------------------------------------------------------------------

    def get_item_groups(self) -> Dict[str, List[str]]:
        """Return item substitution groups.

        Items within the same group are functionally equivalent
        (e.g., 'logs' → ['oak_log', 'birch_log', ...]).

        Empty dict means no item groups defined.
        """
        return {}

    def get_item_to_group_mapping(self) -> Dict[str, str]:
        """Return reverse mapping from item names to group names.

        E.g., {'oak_log': 'logs', 'birch_log': 'logs', ...}.

        Empty dict means no mapping available.
        """
        return {}

    def get_resource_aliases(self) -> Dict[str, List[str]]:
        """Return resource name aliases for effect matching.

        Maps generic resource keys to lists of concrete item names
        (e.g., 'planks' → ['oak_planks', 'birch_planks', ...]).

        Empty dict means no aliases defined.
        """
        return {}

    def get_item_name_categories(self) -> Dict[str, List[str]]:
        """Return item name categories for validation and matching.

        Maps category names to lists of item names, including
        singular/plural variants for fuzzy matching.

        Empty dict means no categories defined.
        """
        return {}

    def get_common_items(self) -> Set[str]:
        """Return the set of all known valid items.

        Empty set means no item validation available.
        """
        return set()

    def get_common_blocks(self) -> Set[str]:
        """Return the set of all known valid blocks.

        Empty set means no block validation available.
        """
        return set()

    def is_valid_item(self, item_name: str) -> bool:
        """Check if item_name is a known valid item.

        Default implementation checks get_common_items().
        Returns True if no items are registered (permissive default).
        """
        items = self.get_common_items()
        return not items or item_name in items

    def is_valid_block(self, block_name: str) -> bool:
        """Check if block_name is a known valid block.

        Default implementation checks get_common_blocks().
        Returns True if no blocks are registered (permissive default).
        """
        blocks = self.get_common_blocks()
        return not blocks or block_name in blocks

    def is_armor_item(self, item_name: str) -> bool:
        """Check if item_name is an armor item."""
        return False

    def get_armor_tier(self, item_name: str) -> int:
        """Return armor tier for item_name (-1 if not armor)."""
        return -1

    def get_armor_slot(self, item_name: str) -> Optional[str]:
        """Return equipment slot for armor item (None if not armor)."""
        return None

    def get_tool_tier_config(self) -> dict:
        """Return tool tier configuration.

        Returns dict with optional keys:
        - "tool_tiers": Dict[str, int] mapping material names to tier levels
        - "material_prefixes": List[str] of material prefix names
        - "generic_patterns": List[str] of generic item type patterns

        Empty dict means use built-in defaults.
        """
        return {}

    # ------------------------------------------------------------------
    # Optional — responsibility validation methods.
    # ------------------------------------------------------------------

    def get_symbolic_knowledge_base(self) -> Any:
        """Return a symbolic knowledge base for recipe/dependency queries.

        Used by PSNCurriculumAgent and GoalPlanner for recipe lookup,
        dependency analysis, feasibility checking, and task decomposition.

        The returned object should implement recipe/crafting query methods
        (e.g., ``get_recipe``, ``check_can_craft``, ``get_dependencies``).

        Returns None if no symbolic knowledge base is available.
        """
        return None

    def get_resource_matcher(self) -> Any:
        """Return a resource matcher instance for skill resource consistency checks.

        The returned object should implement:
        - ``extract_resource_from_skill_name(skill_name) -> List[str]``
        - ``find_unexpected_resources(code, expected) -> List[str]``

        Returns None if no resource matcher is available (checks are skipped).
        """
        return None

    def get_responsibility_validation_patterns(self) -> Dict[str, list]:
        """Return patterns for keyword-based responsibility validation.

        Maps skill name pattern conditions to lists of
        ``(regex_pattern, description)`` tuples that flag unrelated logic.

        Format::

            {
                "condition_key": {
                    "match": callable(skill_lower) -> bool,
                    "patterns": [(r'\\bpattern\\b', 'description'), ...],
                },
                ...
            }

        Empty dict means no keyword-based validation patterns.
        """
        return {}

    def get_severe_responsibility_violation_keywords(self) -> set:
        """Keywords that trigger immediate responsibility rejection (no LLM needed).

        When unrelated keywords are detected in optimized code, any keyword
        whose lowercased form contains one of these severe keywords causes
        an immediate rejection without falling through to LLM validation.

        Empty set means no keyword-based immediate rejection.
        """
        return set()

    def get_responsibility_prompt_context(self) -> str:
        """Return domain-specific skill typology text for the LLM responsibility prompt.

        This text is appended to the responsibility validation system prompt
        to provide domain-specific context about skill types and their allowed
        behaviors.

        Empty string means no domain-specific context.
        """
        return ""

    # ------------------------------------------------------------------
    # Optional — environment/biome methods.
    # ------------------------------------------------------------------

    def get_biome_resources(self, biome_name: str) -> List[str]:
        """Return list of resources available in the given biome.

        Used by precondition checker to verify biome-resource feasibility.
        Empty list means no biome resource data available.
        """
        return []

    # ------------------------------------------------------------------
    # Optional — game mechanics methods.
    # ------------------------------------------------------------------

    def get_emergency_task(
        self,
        y: float,
        inventory: Dict[str, int],
        equipment: list,
    ) -> Optional[str]:
        """Return an emergency task based on position and state.

        Called when checking for special-case tasks that override normal
        curriculum selection (e.g., underground without a pickaxe).

        Args:
            y: Vertical position coordinate.
            inventory: Current inventory ``{item_name: count}``.
            equipment: Current equipment list.

        Returns:
            A task string if an emergency is detected, None otherwise.
        """
        return None

    def get_full_inventory_task(
        self,
        inventory: Dict[str, int],
        chest_observation: str,
    ) -> Optional[str]:
        """Return a task to handle a full inventory.

        Implements the domain-specific inventory management waterfall
        (e.g., deposit → place chest → craft chest → free space).

        Args:
            inventory: Current inventory ``{item_name: count}``.
            chest_observation: Raw chest observation string from the environment.

        Returns:
            A task string, or None to use the generic fallback.
        """
        return None

    def has_sufficient_materials_for_progression(
        self,
        inventory: Dict[str, int],
    ) -> bool:
        """Check if inventory has enough materials for early-game progression.

        Returns True if the agent should skip gathering tasks because it
        already has enough raw materials to craft the next required items.

        Default: False (never skip gathering).
        """
        return False

    def get_inventory_config(self) -> dict:
        """Return inventory configuration constants.

        Returns dict with optional keys:

        - ``"stack_size"``: Max items per stack (default: 64).
        - ``"total_slots"``: Total inventory slots (default: 36).
        - ``"full_threshold"``: Slot count triggering full-inventory handling.

        Empty dict means use built-in defaults.
        """
        return {}

    def get_equipment_slot_names(self) -> List[str]:
        """Return ordered list of equipment slot names.

        E.g., ``["head", "torso", "legs", "feet", "hand", "off-hand"]``.
        Empty list means no equipment slot knowledge.
        """
        return []

    def get_skill_name_fallbacks(self) -> Dict[str, str]:
        """Return ``{prefix: normalized_name}`` for skill name normalization.

        When a skill name is reduced to empty after type generalization,
        the first matching prefix determines the fallback name.

        E.g., ``{"craft": "craftItem", "mine": "mineBlock"}``.
        Empty dict means use ``'genericAction'`` for all.
        """
        return {}

    def get_crafting_station_config(self) -> dict:
        """Return crafting station configuration.

        Returns dict with optional keys:

        - ``"item_name"``: The station item (e.g., ``"crafting_table"``).
        - ``"craft_task"``: The task string to craft it.

        Empty dict means no crafting station knowledge.
        """
        return {}

    # ------------------------------------------------------------------
    # Optional — item transform / inference methods.
    # ------------------------------------------------------------------

    def get_group_aliases(self) -> Dict[str, List[str]]:
        """Return group alias mapping.

        Maps a group name to other group names whose items should be
        counted together.  E.g., ``{"fuel": ["logs"]}`` means that
        when counting "fuel", also include items from the "logs" group.

        Empty dict means no aliases.
        """
        return {}

    def get_item_transform_functions(self) -> dict:
        """Return item transform functions for parameter inference.

        Returns a dict with keys:
        - ``"infer_input_from_output"``: ``(str) -> Optional[str]``
        - ``"infer_output_from_input"``: ``(str) -> Optional[str]``
        - ``"apply_transform"``: ``(str, str) -> Optional[str]``

        All values default to ``None`` (no transforms available).
        """
        return {
            "infer_input_from_output": None,
            "infer_output_from_input": None,
            "apply_transform": None,
        }

    # ------------------------------------------------------------------
    # Optional — multi-domain support.
    # ------------------------------------------------------------------

    def get_entry_parameter_name(self) -> str:
        """Return the name of the implicit first parameter for all skill functions.

        This is the environment/runtime object passed to every skill.
        e.g. Minecraft: 'bot' (Mineflayer Bot).

        Default: 'bot' for backward compatibility.
        """
        return "bot"

    # ------------------------------------------------------------------
    # Optional — action prompt knowledge injection.
    # ------------------------------------------------------------------

    def get_action_prompt_knowledge(
        self, task: str, available_skills: list
    ) -> str:
        """Return task-relevant knowledge for the action agent prompt.

        Dynamically injects domain-specific knowledge items based on the
        current task, complementing the static system prompt with contextual
        facts (e.g., item variant patterns, crafting chain rules).

        Args:
            task: The current task description.
            available_skills: List of available skill names.

        Returns:
            Formatted knowledge text to append to the system message,
            or empty string if no relevant knowledge found.
        """
        return ""


@dataclass
class DomainModule:
    """
    Container bundling all domain-specific components.

    Passed to PSNAgent.from_domain() to configure the agent for a
    specific domain (e.g. Minecraft).
    """

    environment: Environment
    knowledge: DomainKnowledge
    curriculum: CurriculumStrategy
    critic: CriticStrategy
    name: str = "unknown"
    skill_language: str = "javascript"
    skill_language_impl: Optional["SkillLanguage"] = None  # Language implementation for skill code ops
