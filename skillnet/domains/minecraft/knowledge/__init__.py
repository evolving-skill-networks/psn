"""
Minecraft Domain Knowledge

Wraps existing Minecraft knowledge modules (optimizer/knowledge/, constants/)
to implement the DomainKnowledge ABC.
"""

import re
from typing import Any, Dict, List, Optional, Set

from skillnet.core.domain import DomainKnowledge
from skillnet.core.observation import Observation


class MinecraftKnowledge(DomainKnowledge):
    """
    DomainKnowledge implementation for Minecraft.

    Delegates to existing knowledge modules in skillnet/agents/optimizer/knowledge/
    and skillnet/agents/constants/.
    """

    def __init__(self, model_name: str = "gpt-5-mini", kr_llm=None, combat: bool = False):
        self._model_name = model_name
        self._kr_llm = kr_llm
        # Combat mode: don't force the world to peaceful/day on each reset, so
        # hostile mobs spawn and the manually-set difficulty/time persist.
        self._combat = combat
        self._register_knowledge_builder()

    def _register_knowledge_builder(self):
        """Register Minecraft knowledge builders for KnowledgeIndex."""
        from skillnet.agents.optimizer.knowledge.index import register_knowledge_builder
        from skillnet.domains.minecraft.knowledge.index_builder import (
            build_minecraft_knowledge_entries,
        )
        register_knowledge_builder(build_minecraft_knowledge_entries)

    def get_all_knowledge(self) -> List[Any]:
        from skillnet.domains.minecraft.knowledge.game import get_all_game_knowledge
        from skillnet.domains.minecraft.knowledge.api import get_all_api_knowledge
        from skillnet.domains.minecraft.knowledge.primitives import get_all_primitive_knowledge
        return get_all_game_knowledge() + get_all_api_knowledge() + get_all_primitive_knowledge()

    def get_knowledge_for_error(
        self, error_text: str, code: str = ""
    ) -> List[Any]:
        from skillnet.domains.minecraft.knowledge.api_behaviors import (
            get_all_knowledge_for_error,
        )
        result = get_all_knowledge_for_error(error_text, code)
        # get_all_knowledge_for_error returns a string; wrap in list for ABC
        return [result] if result else []

    def get_control_primitives(self) -> List[str]:
        return [
            "exploreUntil", "mineBlock", "craftItem", "placeItem", "smeltItem",
            "killMob", "useChest", "mineflayer",
        ]

    def load_control_primitive_code(self) -> List[str]:
        from skillnet.domains.minecraft.action_space.control_primitives import (
            load_control_primitives,
        )
        return load_control_primitives()

    def load_control_primitive_context(
        self,
        primitive_names=None,
        model_profile=None,
        for_optimizer=False,
    ) -> List[str]:
        from skillnet.domains.minecraft.action_space.control_primitives_context import (
            load_control_primitives_context,
        )
        if primitive_names is None:
            primitive_names = self.get_control_primitives()
        return load_control_primitives_context(
            primitive_names,
            model_profile=model_profile,
            for_optimizer=for_optimizer,
        )

    def get_runtime_primitive_names(self) -> List[str]:
        from skillnet.domains.minecraft.action_space.control_primitives import (
            load_control_primitive_names,
        )
        return load_control_primitive_names()

    _skill_language_impl = None  # class-level cache: one Babel-loading instance per process

    def get_skill_language(self) -> str:
        return "javascript"

    def get_skill_language_impl(self):
        if MinecraftKnowledge._skill_language_impl is None:
            from skillnet.languages.javascript import JavaScriptLanguage
            MinecraftKnowledge._skill_language_impl = JavaScriptLanguage()
        return MinecraftKnowledge._skill_language_impl

    def extract_observation(self, events: List[Any]) -> "Observation":
        """Parse Minecraft tuple-event stream into a normalized Observation.

        Minecraft events are tuples of (event_name, payload_dict). The most
        recent ``("observe", payload)`` tuple holds the bot state snapshot;
        its payload contains keys like status.{health, food, biome, position,
        equipment, inventoryUsed, timeOfDay, entities}, voxels, blockRecords,
        inventory.
        """
        if not events:
            return Observation()
        observe = None
        for ev in reversed(events):
            # Accept both tuple and list shapes. Mineflayer JSPyBridge delivers
            # events as tuples, but ANY JSON round-trip (trajectory record/
            # replay, pickling, json.dumps/loads) turns them into lists.
            # Pre-patch the check accepted only tuples, so post-roundtrip
            # events caused observe to be missed → Observation(extra=None) →
            # render_observation fallback (all fields "unknown", "Inventory:
            # Empty"). Observed live in ckpt_diamond_postG0_a iter 6 critic
            # prompt and reproduced via instrumentation in ckpt_diamond_A0diag_a
            # iter 1 (events_len=10, has_observe=True, obs.extra_empty=True).
            # The canonical helper in skillnet/utils/event_utils.py already
            # accepts both shapes via `isinstance(ev, (tuple, list))`.
            if isinstance(ev, (tuple, list)) and len(ev) == 2 and ev[0] == "observe":
                observe = ev[1]
                break
        if observe is None:
            return Observation()
        status = observe.get("status", {})
        position = None
        pos_dict = status.get("position")
        if isinstance(pos_dict, dict):
            position = (
                pos_dict.get("x"),
                pos_dict.get("y"),
                pos_dict.get("z"),
            )
        return Observation(
            inventory=observe.get("inventory", {}),
            milestones_unlocked=[],
            health=status.get("health"),
            food=status.get("food"),
            position=position,
            extra={
                "biome": status.get("biome"),
                "timeOfDay": status.get("timeOfDay"),
                "voxels": observe.get("voxels", []),
                "blockRecords": observe.get("blockRecords", []),
                "entities": status.get("entities", {}),
                "equipment": status.get("equipment", []),
                "inventoryUsed": status.get("inventoryUsed", 0),
                "raw_observe": observe,
            },
        )

    def render_observation(
        self,
        *,
        events: List[Any],
        chest_observation: str = "",
        completed_tasks: Optional[List[str]] = None,
        failed_tasks: Optional[List[str]] = None,
        progress: int = 0,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Format Minecraft observation as Dict[str, str] section fragments.

        Ported verbatim from agents/curriculum.py:155-242, adapted to:
        - Read state via self.extract_observation (returning Observation)
        - Accept curriculum state (completed_tasks, failed_tasks, progress)
          via kwargs rather than reading from CurriculumAgent.self
        - Receive warm_up_optional_inventory threshold via kwargs

        Returns Dict[str, str] where keys are section names and values are
        formatted text fragments. Caller concatenates fragments in order.
        """
        completed_tasks = completed_tasks or []
        failed_tasks = failed_tasks or []

        obs = self.extract_observation(events)
        if not obs.extra:
            # Graceful degradation if no "observe" tuple in events.
            return {
                "context": "",
                "biome": "Biome: unknown\n\n",
                "time": "Time: unknown\n\n",
                "nearby_blocks": "Nearby blocks: None\n\n",
                "other_blocks": "Other blocks that are recently seen: None\n\n",
                "nearby_entities": "Nearby entities: None\n\n",
                "health": "Health: unknown\n\n",
                "hunger": "Hunger: unknown\n\n",
                "position": "Position: unknown\n\n",
                "equipment": "Equipment: []\n\n",
                "inventory": "Inventory: Empty\n\n",
                "chests": chest_observation,
                "completed_tasks": f"Completed tasks so far: {', '.join(completed_tasks) if completed_tasks else 'None'}\n\n",
                "failed_tasks": f"Failed tasks that are too hard: {', '.join(failed_tasks) if failed_tasks else 'None'}\n\n",
            }

        biome = obs.extra.get("biome", "unknown")
        time_of_day = obs.extra.get("timeOfDay", 0)
        voxels = obs.extra.get("voxels", [])
        block_records = obs.extra.get("blockRecords", [])
        entities = obs.extra.get("entities", {})
        health = obs.health if obs.health is not None else 0.0
        hunger = obs.food if obs.food is not None else 0.0
        raw_observe = obs.extra.get("raw_observe", {})
        position = raw_observe.get("status", {}).get("position", {"x": 0, "y": 0, "z": 0})
        equipment = obs.extra.get("equipment", [])
        inventory_used = obs.extra.get("inventoryUsed", 0)
        inventory = dict(obs.inventory)

        # Biome inference from voxel contents (preserves legacy behavior)
        if not any(
            "dirt" in block
            or "log" in block
            or "grass" in block
            or "sand" in block
            or "snow" in block
            for block in voxels
        ):
            biome = "underground"

        other_blocks = ", ".join(
            list(
                set(block_records).difference(set(voxels).union(set(inventory.keys())))
            )
        )
        other_blocks = other_blocks if other_blocks else "None"

        nearby_entities = (
            ", ".join([k for k, v in sorted(entities.items(), key=lambda x: x[1])])
            if entities
            else "None"
        )

        completed_str = ", ".join(completed_tasks) if completed_tasks else "None"

        if failed_tasks:
            failure_counts = {}
            original_names = {}
            for task in failed_tasks:
                normalized = task.lower().strip()
                normalized = re.sub(r"\s+", " ", normalized)
                failure_counts[normalized] = failure_counts.get(normalized, 0) + 1
                if normalized not in original_names:
                    original_names[normalized] = task
            failed_str = ", ".join(
                f"{original_names[key]} (failed {count}x)"
                for key, count in failure_counts.items()
            )
        else:
            failed_str = "None"

        # Filter optional inventory items per warm-up threshold
        warm_up_threshold = kwargs.get("warm_up_optional_inventory", 0)
        if progress < warm_up_threshold:
            pattern = self.get_core_inventory_pattern() or r".*"
            core_re = re.compile(pattern)
            inventory = {
                k: v for k, v in inventory.items()
                if core_re.search(k) is not None
            }

        return {
            "context": "",
            "biome": f"Biome: {biome}\n\n",
            "time": f"Time: {time_of_day}\n\n",
            "nearby_blocks": f"Nearby blocks: {', '.join(voxels) if voxels else 'None'}\n\n",
            "other_blocks": f"Other blocks that are recently seen: {other_blocks}\n\n",
            "nearby_entities": f"Nearby entities: {nearby_entities}\n\n",
            "health": f"Health: {health:.1f}/20\n\n",
            "hunger": f"Hunger: {hunger:.1f}/20\n\n",
            "position": f"Position: x={position['x']:.1f}, y={position['y']:.1f}, z={position['z']:.1f}\n\n",
            "equipment": f"Equipment: {equipment}\n\n",
            "inventory": f"Inventory ({inventory_used}/36): {inventory if inventory else 'Empty'}\n\n",
            "chests": chest_observation,
            "completed_tasks": f"Completed tasks so far: {completed_str}\n\n",
            "failed_tasks": f"Failed tasks that are too hard: {failed_str}\n\n",
        }

    def get_system_prompt_template(self, model_profile: str = None) -> str:
        """Load system prompt template, with model-specific variant if available."""
        if model_profile and model_profile != "default" and model_profile != "gpt-5":
            variant = self.get_prompt(f"parameterized_action_template_{model_profile}")
            if variant:
                return variant
        return self.get_prompt("parameterized_action_template")

    def get_item_categories(self) -> Dict[str, List[str]]:
        from .item_categories import MINECRAFT_ITEM_CATEGORIES
        return dict(MINECRAFT_ITEM_CATEGORIES)

    def get_safe_interchange_categories(self) -> set:
        from .item_categories import MINECRAFT_SAFE_INTERCHANGE_CATEGORIES
        return set(MINECRAFT_SAFE_INTERCHANGE_CATEGORIES)

    def get_core_inventory_pattern(self) -> str:
        return (
            r".*_log|.*_planks|stick|crafting_table|furnace"
            r"|cobblestone|dirt|coal|.*_pickaxe|.*_sword|.*_axe"
        )

    def get_type_keywords(self) -> dict:
        return {
            'specific_types': [
                'birch', 'oak', 'spruce', 'jungle', 'acacia', 'dark_oak', 'mangrove', 'cherry',
                'iron', 'gold', 'diamond', 'netherite', 'stone', 'wooden', 'leather', 'chainmail',
                'white', 'black', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
                'pink', 'brown', 'gray', 'cyan', 'lime', 'magenta', 'light_blue', 'light_gray',
            ],
            'type_param_patterns': [
                r'\b(woodType|plankType|logType|itemType|blockType|materialType|type|material)\b',
                r'\b(wood|plank|log|item|block|material)\w*Type\b',
                r'\b(allowed\w*Types?|allowed\w*Logs?|allowed\w*Planks?)\b',
            ],
        }

    def get_known_functions(self) -> Dict[str, Any]:
        from .function_sets import (
            BOT_METHODS, KNOWN_PRIMITIVES, CONTROL_PRIMITIVE_HELPERS,
            COMMON_HELPER_PATTERNS, KNOWN_INVALID_ITEM_NAMES, AMBIGUOUS_ITEM_NAMES,
            FUNCTION_OPERATION_PATTERNS, CONTROL_PRIMITIVE_API_MAPPINGS,
            ITEM_PRODUCTION_ALIASES,
            MATERIAL_TO_INGREDIENT, PROBLEMATIC_CONCATENATIONS,
            PATTERN_SUPPORTED_VALUES, OVERCLAIM_SEMANTIC_CONTEXTS,
            LOOP_BREAK_API_HINTS, KNOWN_PATHFINDER_GOALS,
        )
        helpers = set(CONTROL_PRIMITIVE_HELPERS)
        return {
            "bot_methods": set(BOT_METHODS),
            "primitives": set(KNOWN_PRIMITIVES),
            "helpers": helpers,
            "common_helpers": set(COMMON_HELPER_PATTERNS),
            # Pathfinder Goal classes are destructured into /step handler scope
            # (mineflayer/index.js:497-523) and reachable from skill code via
            # the JavaScript scope chain. They are NOT undefined at runtime.
            "pathfinder_globals": set(KNOWN_PATHFINDER_GOALS),
            "invalid_item_names": dict(KNOWN_INVALID_ITEM_NAMES),
            "ambiguous_item_names": set(AMBIGUOUS_ITEM_NAMES),
            "operation_patterns": dict(FUNCTION_OPERATION_PATTERNS),
            "api_mappings": list(CONTROL_PRIMITIVE_API_MAPPINGS),
            "item_production_aliases": dict(ITEM_PRODUCTION_ALIASES),
            "material_to_ingredient": dict(MATERIAL_TO_INGREDIENT),
            "concatenation_patterns": list(PROBLEMATIC_CONCATENATIONS),
            "pattern_supported_values": dict(PATTERN_SUPPORTED_VALUES),
            "overclaim_semantic_contexts": list(OVERCLAIM_SEMANTIC_CONTEXTS),
            "loop_break_api_hints": dict(LOOP_BREAK_API_HINTS),
        }

    # ------------------------------------------------------------------
    # Prompt template methods
    # ------------------------------------------------------------------

    def get_prompt(self, name: str) -> str:
        from skillnet.domains.minecraft.prompts import load_minecraft_prompt
        return load_minecraft_prompt(name)

    def get_critic_prompt_template(self) -> str:
        return self.get_prompt("psn_critic")

    def get_curriculum_prompt_template(self) -> str:
        return self.get_prompt("psn_curriculum")

    def get_action_response_format(self) -> str:
        return self.get_prompt("action_response_format")

    def get_reparameterize_prompt_template(self) -> str:
        return self.get_prompt("reparameterize")

    # ------------------------------------------------------------------
    # Optimizer knowledge methods
    # ------------------------------------------------------------------

    def get_domain_name(self) -> str:
        return "Minecraft"

    def get_environment_rules(self, error_text: str) -> List[str]:
        from skillnet.domains.minecraft.knowledge.minecraft_env import (
            create_minecraft_env_knowledge,
        )
        env = create_minecraft_env_knowledge()
        rules = env.find_relevant_rules(error_text)
        return [
            f"{r.description}. Consequence: {r.consequence}"
            for r in rules
        ]

    def get_primitive_knowledge(self, error_text: str, code: str = "") -> str:
        from skillnet.domains.minecraft.knowledge.minecraft_env import (
            create_minecraft_env_knowledge,
        )
        env = create_minecraft_env_knowledge()
        if hasattr(env, 'get_primitive_fix_prompt'):
            return env.get_primitive_fix_prompt(error_text, code) or ""
        return ""

    def get_biome_resources(self, biome_name: str) -> List[str]:
        from .game.resources import get_biome_resources as _get_biome_resources
        return _get_biome_resources(biome_name)

    def get_environment_context(self, state: dict) -> str:
        if not state or not isinstance(state, dict):
            return ""
        biome = state.get("biome")
        if not biome or biome == "unknown":
            return ""
        try:
            from skillnet.domains.minecraft.knowledge.game import get_biome_resources
            resources = get_biome_resources(biome)
            if resources:
                resources_str = ", ".join(resources[:10])
                if len(resources) > 10:
                    resources_str += f" (+{len(resources) - 10} more)"
                return (
                    f"\n\n**Current Environment:**\n"
                    f"- Biome: {biome}\n"
                    f"- Available Resources: {resources_str}"
                )
            return f"\n\n**Current Biome:** {biome}"
        except Exception:
            return f"\n\n**Current Biome:** {biome}"

    def get_reasoning_examples(self, error_text: str, max_examples: int = 2) -> str:
        try:
            from skillnet.domains.minecraft.knowledge.reasoning_examples import (
                get_relevant_examples,
            )
            return get_relevant_examples(error_text, max_examples=max_examples) or ""
        except ImportError:
            return ""

    def get_detailed_knowledge_for_error(
        self, error_text: str, code: str = "", chat_log: str = "",
        kr_logger=None,
    ) -> str:
        from skillnet.agents.optimizer.knowledge.llm_knowledge_retriever import (
            retrieve_knowledge_for_analysis,
        )
        from skillnet.domains.minecraft.kr_config import (
            MINECRAFT_QUERY_PROMPT,
            minecraft_fallback_queries,
        )
        return retrieve_knowledge_for_analysis(
            feedback_content=error_text,
            chat_log=chat_log,
            skill_code=code,
            llm=self._kr_llm,
            kr_logger=kr_logger,
            query_prompt=MINECRAFT_QUERY_PROMPT,
            fallback_queries_fn=minecraft_fallback_queries,
        )

    # Bindings exposed to skill code that hold only data attributes (no
    # methods). When the LLM writes ``obj.something(...)`` against one of
    # these, the call resolves to ``undefined`` at runtime.
    DATA_ONLY_OBJECTS = frozenset({"mcData"})

    def get_code_validators(self):
        from skillnet.agents.optimizer.validators.code_validator import (
            validate_bot_method_calls,
            validate_data_only_object_calls,
        )

        def _bot_method_validator(code: str, context: dict):
            primitives = context.get("primitives")
            return validate_bot_method_calls(code, known_primitives=primitives)

        def _data_only_object_validator(code: str, context: dict):
            return validate_data_only_object_calls(code, self.DATA_ONLY_OBJECTS)

        return [_bot_method_validator, _data_only_object_validator]

    def get_global_dependencies(self) -> dict:
        return {
            "mcData": "const mcData = require('minecraft-data')(bot.version);",
            "Vec3": "const Vec3 = require('vec3').Vec3;",
        }

    def get_environment_globals(self) -> set:
        # v12 fix B2: pull pathfinder Goal classes from KNOWN_PATHFINDER_GOALS
        # so Reference Check (which uses _get_domain_globals() → this method)
        # and Consistency Check (which uses get_known_functions()['pathfinder_globals'])
        # share a single source of truth. Previously this set hand-listed only
        # 10 of the 22 Goal classes destructured at mineflayer/index.js:497-523,
        # which could lead to false-positive Reference Check rejections for the
        # missing 12 (e.g. GoalCompositeAny, GoalInvert, GoalBreakBlock).
        from .function_sets import KNOWN_PATHFINDER_GOALS
        return set(KNOWN_PATHFINDER_GOALS) | {'checkRecipe'}

    def get_primitive_semantics(self) -> Dict[str, Dict[str, bool]]:
        return {
            "mineBlock": {"produces": True, "consumes": False},
            "craftItem": {"produces": True, "consumes": True},
            "smeltItem": {"produces": True, "consumes": True},
            "placeItem": {"produces": False, "consumes": True},
        }

    def get_registry_access_patterns(self) -> list:
        return [
            r'mcData\.itemsByName\[',
            r'mcData\.blocksByName\[',
            r'mcData\.items\[',
            r'mcData\.blocks\[',
            r'bot\.registry\.itemsByName',
            r'bot\.registry\.blocksByName',
        ]

    def get_resource_error_patterns(self) -> list:
        return [
            (r"mcData missing block id for (\w+)|blocksByName\[.*\] is undefined",
             "resource_name",
             "Fix block name: may have used item name (e.g., 'coal') instead of block name (e.g., 'coal_ore')"),
        ]

    def get_type_adapter_config(self) -> dict:
        return {
            "conversion_rules": {
                "planks_to_log": {
                    "pattern": r"(.+)_planks$",
                    "replacement": r"\1_log",
                    "description": "Convert planks type to log type",
                },
                "log_to_planks": {
                    "pattern": r"(.+)_log$",
                    "replacement": r"\1_planks",
                    "description": "Convert log type to planks type",
                },
                "ore_to_raw": {
                    "pattern": r"^(?:deepslate_)?(.+)_ore$",
                    "replacement": r"raw_\1",
                    "description": "Convert ore type to raw material",
                },
                "raw_to_ingot": {
                    "pattern": r"^raw_(.+)$",
                    "replacement": r"\1_ingot",
                    "description": "Convert raw material to ingot",
                },
                "item_name_to_id": {
                    "adapter": "mcData.itemsByName[{value}]?.id",
                    "description": "Convert item name to ID",
                },
                "item_id_to_name": {
                    "adapter": "mcData.items[{value}]?.name",
                    "description": "Convert item ID to name",
                },
                "block_name_to_id": {
                    "adapter": "mcData.blocksByName[{value}]?.id",
                    "description": "Convert block name to ID",
                },
            },
            "suffix_requirements": {
                "logtype": "_log",
                "log_type": "_log",
                "planktype": "_planks",
                "plank_type": "_planks",
                "oretype": "_ore",
                "ore_type": "_ore",
                "woodtype": "_log",
                "wood_type": "_log",
            },
        }

    def get_mine_keywords(self) -> list:
        return [
            "ore", "log", "block", "sand", "gravel",
            "stone", "dirt", "coal", "cobble", "wood",
        ]

    def get_func_prefix_to_direction(self) -> dict:
        return {
            "craft": "input",
            "smelt": "input",
            "mine": "output",
            "ensure": "output",
            "get": "output",
            "collect": "output",
        }

    def get_param_suffix_to_transform(self) -> list:
        return [
            ("logtype", {"transform_type": "suffix_replace", "source_suffix": "_planks", "target_suffix": "_log"}),
            ("log_type", {"transform_type": "suffix_replace", "source_suffix": "_planks", "target_suffix": "_log"}),
            ("oretype", {"transform_type": "regex", "pattern": r"^(?:deepslate_)?(\w+)_ore$", "replacement": "{1}"}),
            ("ore_type", {"transform_type": "regex", "pattern": r"^(?:deepslate_)?(\w+)_ore$", "replacement": "{1}"}),
            ("planktype", {"transform_type": "suffix_replace", "source_suffix": "_log", "target_suffix": "_planks"}),
            ("plank_type", {"transform_type": "suffix_replace", "source_suffix": "_log", "target_suffix": "_planks"}),
        ]

    def get_plural_to_singular(self) -> dict:
        return {
            # Crafting materials
            "sticks": "stick",
            "strings": "string",
            "leathers": "leather",
            "papers": "paper",
            "books": "book",
            "feathers": "feather",
            "flints": "flint",
            # Tools and utilities
            "torches": "torch",
            "crafting_tables": "crafting_table",
            "furnaces": "furnace",
            "chests": "chest",
            "boats": "boat",
            "buckets": "bucket",
            "bowls": "bowl",
            "signs": "sign",
            "ladders": "ladder",
            "fences": "fence",
            "doors": "door",
            "beds": "bed",
            "anvils": "anvil",
            "compasses": "compass",
            "clocks": "clock",
            "maps": "map",
            "shields": "shield",
            "bows": "bow",
            "arrows": "arrow",
            # Building blocks
            "stones": "stone",
            "cobblestones": "cobblestone",
            "bricks": "brick",
            "glasses": "glass",
            # Ores and minerals
            "diamonds": "diamond",
            "emeralds": "emerald",
            "iron_ingots": "iron_ingot",
            "gold_ingots": "gold_ingot",
            "copper_ingots": "copper_ingot",
            "netherite_ingots": "netherite_ingot",
            "coals": "coal",
            "redstones": "redstone",
            "lapis_lazulis": "lapis_lazuli",
            # Raw ores
            "raw_irons": "raw_iron",
            "raw_golds": "raw_gold",
            "raw_coppers": "raw_copper",
            # Food
            "breads": "bread",
            "apples": "apple",
            "carrots": "carrot",
            "potatoes": "potato",
            "melons": "melon",
            "pumpkins": "pumpkin",
        }

    def get_suffix_to_group(self) -> dict:
        return {
            "_log": "logs",
            "_planks": "planks",
            "_pickaxe": "pickaxes",
            "_axe": "axes",
            "_sword": "swords",
            "_shovel": "shovels",
            "_hoe": "hoes",
        }

    def get_task_to_group_mapping(self) -> dict:
        return {
            "wood logs": "logs", "wood log": "logs", "logs": "logs", "log": "logs",
            "planks": "planks", "plank": "planks",
            "cooked food": "cooked_food", "food": "cooked_food",
            "pickaxe": "pickaxes", "pickaxes": "pickaxes",
            "axe": "axes", "axes": "axes",
            "sword": "swords", "swords": "swords",
            "shovel": "shovels", "shovels": "shovels",
            "hoe": "hoes", "hoes": "hoes",
            "fuel": "fuel",
        }

    def get_ore_to_drop_mapping(self) -> dict:
        return {
            "diamond_ore": "diamond",
            "deepslate_diamond_ore": "diamond",
            "coal_ore": "coal",
            "deepslate_coal_ore": "coal",
            "iron_ore": "raw_iron",
            "deepslate_iron_ore": "raw_iron",
            "gold_ore": "raw_gold",
            "deepslate_gold_ore": "raw_gold",
            "copper_ore": "raw_copper",
            "deepslate_copper_ore": "raw_copper",
            "lapis_ore": "lapis_lazuli",
            "deepslate_lapis_ore": "lapis_lazuli",
            "redstone_ore": "redstone",
            "deepslate_redstone_ore": "redstone",
            "emerald_ore": "emerald",
            "deepslate_emerald_ore": "emerald",
            "nether_quartz_ore": "quartz",
            "nether_gold_ore": "gold_nugget",
        }

    def get_display_resource_groups(self):
        return ["logs", "planks", "cobblestone", "iron_ingot", "diamond",
                "fuel", "torch", "cooked_food", "stick"]

    def get_tool_type_names(self):
        return ["pickaxe", "axe", "sword", "shovel", "hoe"]

    def get_initial_task(self):
        return "Ensure you have 4 wood logs"

    def get_progression_milestones(self) -> list:
        return [
            {"name": "get_wood", "description": "Collect initial wood supply",
             "required_items": {"logs": 4}, "stage": "early"},
            {"name": "wooden_tools", "description": "Craft basic wooden tools",
             "required_items": {"wooden_pickaxe": 1, "wooden_axe": 1}, "stage": "early"},
            {"name": "stone_tools", "description": "Upgrade to stone tools",
             "required_items": {"stone_pickaxe": 1, "cobblestone": 16}, "stage": "early"},
            {"name": "iron_ready", "description": "Collected enough iron for tools",
             "required_items": {"raw_iron": 8, "fuel": 8}, "stage": "mid"},
            {"name": "iron_tools", "description": "Iron pickaxe for deep mining",
             "required_items": {"iron_pickaxe": 1}, "stage": "mid"},
            {"name": "diamond_found", "description": "Found diamonds",
             "required_items": {"diamond": 3}, "stage": "late"},
            {"name": "diamond_tools", "description": "Diamond pickaxe for obsidian",
             "required_items": {"diamond_pickaxe": 1}, "stage": "late"},
        ]

    def get_world_geometry_config(self) -> Dict[str, Any]:
        return {
            "coord_dims": 3,
            "has_vertical": True,
            "surface_y": 63,
            "underground_threshold": 58,
        }

    def get_param_type_patterns(self) -> dict:
        return {
            "plank": ["_planks", "_plank"],
            "log": ["_log", "_stem", "_wood"],
            "ore": ["_ore", "raw_"],
            "ingot": ["_ingot"],
            "tool": ["_pickaxe", "_axe", "_shovel", "_hoe", "_sword"],
            "pickaxe": ["_pickaxe"],
            "axe": ["_axe"],
            "shovel": ["_shovel"],
            "sword": ["_sword"],
            "fuel": ["coal", "charcoal", "_planks", "_log", "_wood", "stick", "blaze_rod",
                     "lava_bucket", "dried_kelp_block", "bamboo", "bucket"],
        }

    def get_milestone_item_mapping(self) -> dict:
        return {
            # Wood tier
            "oak_log": "get_wood", "birch_log": "get_wood", "spruce_log": "get_wood",
            "crafting_table": "wooden_tools",
            "wooden_pickaxe": "wooden_tools", "wooden_sword": "wooden_tools", "wooden_axe": "wooden_tools",
            # Stone tier
            "cobblestone": "stone_tools",
            "stone_pickaxe": "stone_tools", "stone_sword": "stone_tools", "stone_axe": "stone_tools",
            "furnace": "stone_tools",
            # Iron tier
            "iron_ore": "iron_ready", "raw_iron": "iron_ready", "iron_ingot": "iron_ready",
            "iron_pickaxe": "iron_tools", "iron_sword": "iron_tools", "iron_axe": "iron_tools",
            "bucket": "iron_tools", "shield": "iron_tools",
            # Diamond tier
            "diamond": "diamond_found",
            "diamond_pickaxe": "diamond_tools", "diamond_sword": "diamond_tools",
            "diamond_axe": "diamond_tools", "enchanting_table": "diamond_tools",
            # Other progress items
            "coal": "stone_tools", "torch": "stone_tools",
            "bread": "iron_ready", "bed": "iron_ready",
        }

    def get_resource_name_mappings(self) -> dict:
        return {
            "cobblestone": ["cobblestone", "stone"],
            "coal": ["coal", "coal_ore", "charcoal"],
            "iron": ["iron", "iron_ore", "raw_iron", "iron_ingot"],
            "ironore": ["iron_ore", "deepslate_iron_ore"],
            "gold": ["gold", "gold_ore", "raw_gold", "gold_ingot"],
            "diamond": ["diamond", "diamond_ore"],
            "lapis": ["lapis", "lapis_ore", "lapis_lazuli"],
            "redstone": ["redstone", "redstone_ore"],
            "copper": ["copper", "copper_ore", "raw_copper", "copper_ingot"],
            "logs": ["oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log"],
            "furnace": ["furnace"],
        }

    # ------------------------------------------------------------------
    # Planner/inference/refactor domain injection (Stage 3)
    # ------------------------------------------------------------------

    def get_effect_extraction_verbs(self) -> Dict[str, List[str]]:
        return {
            "find": ["find", "locate", "discover", "search\\s+for"],
            "collect": [
                "mine", "collect", "get", "obtain", "gather",
                "acquire", "harvest", "extract", "dig",
            ],
            "craft": ["craft"],
            "ensure": ["ensure"],
            "place": ["place"],
            "equip": ["equip"],
        }

    def get_variant_patterns(self) -> list:
        return [
            # Wood types
            (r'(oak|birch|spruce|jungle|acacia|dark_?oak|mangrove|cherry)', 'woodType'),
            # Ore/material types
            (r'(iron|gold|diamond|netherite|copper|coal|redstone|lapis|emerald)', 'materialType'),
            # Tool types
            (r'(pickaxe|axe|shovel|hoe|sword)', 'toolType'),
            # Directions
            (r'(north|south|east|west|up|down)', 'direction'),
            # Colors
            (r'(white|orange|magenta|light_?blue|yellow|lime|pink|gray|light_?gray|cyan|purple|blue|brown|green|red|black)', 'color'),
        ]

    def get_variant_suffixes(self) -> list:
        return [
            r'oak|birch|spruce|jungle|acacia|dark_?oak|mangrove|cherry',
            r'iron|gold|diamond|netherite|copper|coal|redstone',
            r'wooden|stone|leather',
        ]

    def get_resource_not_found_patterns(self) -> dict:
        return {
            "lava": [
                "no lava", "lava not found", "couldn't find lava",
                "no lava found", "cannot find lava", "failed to find lava",
                "exploration timed out without finding lava",
            ],
            "water": [
                "no water", "water not found", "couldn't find water",
                "no water found", "cannot find water", "failed to find water",
            ],
            "diamond": [
                "no diamond", "diamond not found", "couldn't find diamond",
                "no diamonds", "failed to mine diamond",
            ],
        }

    def get_inference_rules(self) -> dict:
        from .minecraft_rules import (
            WOOD_TYPES,
            RAW_ORE_TYPES,
            DIRECT_DROP_ORES,
        )
        return {
            "wood_types": list(WOOD_TYPES),
            "raw_ore_types": list(RAW_ORE_TYPES),
            "direct_drop_ores": list(DIRECT_DROP_ORES),
        }

    def get_require_patterns(self) -> list:
        return [
            (r'const\s+mcData\s*=\s*require\([\'"](?:minecraft-data|minecraft_data)[\'"]\)\s*\([^)]+\)\s*;',
             "mcData",
             "const mcData = require('minecraft-data')(bot.version);"),
            (r'const\s*\{\s*Vec3\s*\}\s*=\s*require\([\'"]vec3[\'"]\)\s*;',
             "Vec3",
             "const Vec3 = require('vec3').Vec3;"),
            (r'const\s+Vec3\s*=\s*require\([\'"]vec3[\'"]\)\s*\.\s*Vec3\s*;',
             "Vec3",
             "const Vec3 = require('vec3').Vec3;"),
        ]

    def get_simple_task_patterns(self) -> list:
        return [
            r"ensure you have \d+ \w+_log",
            r"ensure you have \d+ wood logs?",
            r"ensure you have \d+ cobblestone",
            r"ensure you have \d+ \w+_ore",
            r"ensure you have \d+ coal",
            r"ensure you have \d+ iron_ingot",
            r"ensure you have \d+ diamond",
            r"ensure you have \d+ \w+_planks?",
            r"ensure you have \d+ sticks?",
            r"craft \d+ \w+",
            r"mine \d+ \w+",
            r"smelt \d+ \w+",
            r"cook \d+ \w+",
        ]

    def get_resource_thresholds(self) -> dict:
        return {
            "logs": {"min_count": 16, "target_count": 26, "priority": 6},
            "planks": {"min_count": 0, "target_count": 32, "priority": 4},
            "cobblestone": {"min_count": 0, "target_count": 64, "priority": 3},
            "stick": {"min_count": 0, "target_count": 16, "priority": 4},
            "fuel": {"min_count": 0, "target_count": 32, "priority": 7, "is_consumable": True},
            "torch": {"min_count": 0, "target_count": 16, "priority": 5, "is_consumable": True},
            "cooked_food": {"min_count": 0, "target_count": 16, "priority": 1, "is_consumable": True},
            "iron_ingot": {"min_count": 0, "target_count": 16, "priority": 8},
            "gold_ingot": {"min_count": 0, "target_count": 8, "priority": 4},
            "diamond": {"min_count": 0, "target_count": 8, "priority": 9},
        }

    def get_complex_task_keywords(self) -> list:
        return [
            "obsidian",      # Needs water bucket + lava + diamond pickaxe
            "enchant",       # Enchanting
            "nether",        # Nether-related
            "end",           # End-related
            "portal",        # Portals
            "beacon",        # Beacon
            "brewing",       # Brewing
            "potion",        # Potions
        ]

    # ------------------------------------------------------------------
    # Curriculum/orchestrator methods
    # ------------------------------------------------------------------

    def get_task_complexity_tiers(self) -> dict:
        return {
            'high': [
                'diamond', 'ancient_debris', 'netherite', 'emerald',
                'stronghold', 'end_portal', 'ender_dragon', 'wither',
                'nether', 'blaze', 'ghast', 'fortress',
            ],
            'medium': [
                'iron', 'gold', 'redstone', 'lapis', 'coal',
                'cave', 'mine', 'explore', 'find',
            ],
        }

    def get_tool_unlock_mapping(self) -> dict:
        return {
            'wooden_pickaxe': ['stone', 'cobblestone', 'coal'],
            'stone_pickaxe': ['iron', 'lapis'],
            'iron_pickaxe': ['gold', 'diamond', 'redstone', 'emerald'],
            'diamond_pickaxe': ['obsidian', 'ancient_debris'],
            'wooden_axe': ['log', 'wood'],
            'stone_axe': ['log', 'wood'],
            'iron_axe': ['log', 'wood'],
            'iron_sword': ['zombie', 'skeleton', 'spider', 'creeper'],
            'diamond_sword': ['enderman', 'blaze', 'wither'],
        }

    def get_reset_commands(self) -> List[str]:
        # Default forces peaceful + day for clean skill learning. Combat mode
        # keeps the manual difficulty/time and re-enables the day/night cycle so
        # night falls and hostile mobs spawn.
        if self._combat:
            return ['/gamerule doDaylightCycle true']
        return ['/time set day', '/difficulty peaceful']

    def get_skill_verb_prefixes(self) -> List[str]:
        return ['craft', 'ensure', 'mine', 'smelt', 'cook', 'get']

    def get_task_skill_mapping(self) -> Dict[str, list]:
        return {
            "craft planks": ["craftPlanks", "ensurePlanks"],
            "craft sticks": ["craftSticks", "ensureSticks"],
            "craft crafting_table": ["craftCraftingTable", "ensureCraftingTable", "setupCraftingTable"],
            "craft wooden_pickaxe": ["craftWoodenPickaxe", "ensureWoodenPickaxe", "craftPickaxe", "ensurePickaxe"],
            "craft wooden_axe": ["craftWoodenAxe", "ensureWoodenAxe", "craftAxe", "ensureAxe"],
            "craft stone_pickaxe": ["craftStonePickaxe", "ensureStonePickaxe", "craftPickaxe", "ensurePickaxe"],
            "craft stone_axe": ["craftStoneAxe", "ensureStoneAxe", "craftAxe", "ensureAxe"],
            "craft furnace": ["craftFurnace", "ensureFurnace"],
            "craft iron_pickaxe": ["craftIronPickaxe", "ensureIronPickaxe", "craftPickaxe", "ensurePickaxe"],
            "mine logs": ["mineLogs", "mineWoodLogs", "collectLogs", "ensureLogs"],
            "mine cobblestone": ["mineCobblestone", "collectCobblestone", "ensureCobble", "ensureCobblestone"],
        }

    def check_tool_efficiency(self, task: str, inventory: dict) -> Optional[str]:
        """Minecraft-specific tool efficiency check.

        Before gathering large amounts of resources, suggest crafting
        the appropriate tool first.
        """
        task_lower = task.lower()

        # Parse task: TARGET semantic or DELTA semantic
        match = re.match(r"ensure you have\s+(\d+)\s+(.+)", task_lower)
        if match:
            count = int(match.group(1))
            target = match.group(2).strip()
        else:
            match = re.match(r"(mine|collect|gather|chop)\s+(\d+)\s+(.+)", task_lower)
            if not match:
                return None
            count = int(match.group(2))
            target = match.group(3).strip()

        EFFICIENCY_THRESHOLDS = {
            "logs": 5, "log": 5, "wood": 5,
            "cobblestone": 10, "stone": 10,
        }

        TOOL_FOR_RESOURCE = {
            "logs": "axe", "log": "axe", "wood": "axe",
            "oak_log": "axe", "birch_log": "axe", "spruce_log": "axe",
            "jungle_log": "axe", "acacia_log": "axe", "dark_oak_log": "axe",
            "cobblestone": "pickaxe", "stone": "pickaxe",
            "coal": "pickaxe", "iron_ore": "pickaxe", "raw_iron": "pickaxe",
        }

        threshold = None
        for key, thresh in EFFICIENCY_THRESHOLDS.items():
            if key in target:
                threshold = thresh
                break

        if threshold is None or count < threshold:
            return None

        tool_type = None
        for key, tool in TOOL_FOR_RESOURCE.items():
            if key in target:
                tool_type = tool
                break

        if tool_type is None:
            return None

        tool_tiers = ["wooden", "stone", "iron", "diamond", "netherite"]
        has_tool = any(
            inventory.get(f"{tier}_{tool_type}", 0) > 0
            for tier in tool_tiers
        )

        if has_tool:
            return None

        planks_count = sum(v for k, v in inventory.items() if k.endswith("_planks"))
        logs_count = sum(v for k, v in inventory.items() if k.endswith("_log"))
        sticks_count = inventory.get("stick", 0)
        cobble_count = inventory.get("cobblestone", 0)

        potential_planks = planks_count + (logs_count * 4)

        if cobble_count >= 3:
            potential_sticks_for_stone = sticks_count + ((potential_planks // 2) * 4)
            if potential_sticks_for_stone >= 2:
                return f"Craft 1 stone {tool_type}"

        planks_needed_for_sticks = 0 if sticks_count >= 2 else 2
        total_planks_needed = 3 + planks_needed_for_sticks

        if potential_planks >= total_planks_needed:
            return f"Craft 1 wooden {tool_type}"

        return None

    def get_revert_failed_action_code(self, blocks: list, positions: list) -> Optional[str]:
        import json
        if not blocks:
            return None
        return (
            f"await givePlacedItemBack(bot, {json.dumps(blocks)}, {json.dumps(positions)})"
        )

    # ------------------------------------------------------------------
    # Crafting chain config
    # ------------------------------------------------------------------

    def get_crafting_chain_config(self) -> dict:
        return {
            "config_contexts": {
                "fuel": ["coal", "charcoal", "_planks", "_log", "lava", "blaze", "stick", "wood", "bucket"],
                "tool": ["_pickaxe", "_axe", "_shovel", "_hoe", "_sword"],
                "log": ["_log", "_stem", "_wood"],
                "plank": ["_planks", "_plank"],
                "ore": ["_ore"],
                "block": ["_block", "stone", "dirt", "sand", "gravel", "cobblestone"],
                "material": ["_ingot", "_nugget", "diamond", "emerald", "gold", "iron", "copper"],
            },
            "input_material_params": [
                "logtype", "log_type", "inputlog", "input_log",
                "planktype", "plank_type", "inputplank", "input_plank",
                "oretype", "ore_type", "inputore", "input_ore",
                "inputitem", "input_item", "material", "source",
                "rawinput", "raw_input", "ingredient", "inputmaterial",
            ],
            "product_to_material_mappings": [
                ("_planks", "_log"),
                ("_plank", "_log"),
                ("_ingot", "_ore"),
                ("iron_ingot", "raw_iron"),
                ("gold_ingot", "raw_gold"),
                ("copper_ingot", "raw_copper"),
                ("_pickaxe", ""),
                ("_axe", ""),
                ("_sword", ""),
                ("_shovel", ""),
                ("_hoe", ""),
                ("_helmet", ""),
                ("_chestplate", ""),
                ("_leggings", ""),
                ("_boots", ""),
                ("_block", ""),
            ],
        }

    # ------------------------------------------------------------------
    # Item knowledge methods
    # ------------------------------------------------------------------

    def get_item_groups(self) -> Dict[str, List[str]]:
        from .minecraft_items import ITEM_GROUPS
        return dict(ITEM_GROUPS)

    def get_item_to_group_mapping(self) -> Dict[str, str]:
        from .minecraft_items import ITEM_TO_GROUP
        return dict(ITEM_TO_GROUP)

    def get_resource_aliases(self) -> Dict[str, List[str]]:
        from .minecraft_items import (
            RESOURCE_ALIASES as MC_RESOURCE_ALIASES,
        )
        return dict(MC_RESOURCE_ALIASES)

    def get_item_name_categories(self) -> Dict[str, List[str]]:
        from .minecraft_items import (
            ITEM_CATEGORIES as MC_ITEM_CATEGORIES,
        )
        return dict(MC_ITEM_CATEGORIES)

    def get_common_items(self) -> Set[str]:
        from .minecraft_items import COMMON_ITEMS
        return set(COMMON_ITEMS)

    def get_common_blocks(self) -> Set[str]:
        from .minecraft_items import COMMON_BLOCKS
        return set(COMMON_BLOCKS)

    def is_valid_item(self, item_name: str) -> bool:
        from .mc_items import MC_ITEM_NAMES
        return item_name in MC_ITEM_NAMES

    def is_valid_block(self, block_name: str) -> bool:
        from .minecraft_items import is_valid_block
        return is_valid_block(block_name)

    def is_armor_item(self, item_name: str) -> bool:
        from .minecraft_items import is_armor_item
        return is_armor_item(item_name)

    def get_armor_tier(self, item_name: str) -> int:
        from .minecraft_items import get_armor_tier
        return get_armor_tier(item_name)

    def get_armor_slot(self, item_name: str) -> Optional[str]:
        from .minecraft_items import get_armor_slot
        return get_armor_slot(item_name)

    def get_group_aliases(self) -> Dict[str, List[str]]:
        from .minecraft_items import GROUP_ALIASES
        return dict(GROUP_ALIASES)

    def get_item_transform_functions(self) -> dict:
        from .minecraft_rules import (
            apply_minecraft_transform,
            infer_input_from_output,
            infer_output_from_input,
        )
        return {
            "infer_input_from_output": infer_input_from_output,
            "infer_output_from_input": infer_output_from_input,
            "apply_transform": apply_minecraft_transform,
        }

    # ------------------------------------------------------------------
    # Responsibility validation methods
    # ------------------------------------------------------------------

    def get_symbolic_knowledge_base(self):
        from skillnet.domains.minecraft.knowledge.knowledge_base import (
            MinecraftKnowledgeBase,
        )
        return MinecraftKnowledgeBase()

    def get_resource_matcher(self):
        from skillnet.domains.minecraft.knowledge.resource_matcher import (
            MinecraftResourceMatcher,
        )
        return MinecraftResourceMatcher()

    def get_responsibility_validation_patterns(self) -> dict:
        return {
            "craft_craftingtable": {
                "match": lambda s: 'craft' in s and 'craftingtable' in s,
                "patterns": [
                    (r'\bstone_pickaxe\b', 'stone_pickaxe crafting'),
                    (r'\biron_pickaxe\b', 'iron_pickaxe crafting'),
                    (r'\bdiamond_pickaxe\b', 'diamond_pickaxe crafting'),
                    (r'\bwooden_pickaxe\b', 'wooden_pickaxe crafting'),
                    (r'\bwooden_sword\b', 'wooden_sword crafting'),
                    (r'\bstone_sword\b', 'stone_sword crafting'),
                    (r'\biron_sword\b', 'iron_sword crafting'),
                    (r'\bfurnace\b', 'furnace crafting'),
                    (r'\bchest\b', 'chest crafting'),
                ],
            },
            "mine": {
                "match": lambda s: 'mine' in s,
                "check_fn": lambda s, code: (
                    ['crafting logic']
                    if 'bot.craft(' in code and 'pickaxe' not in s
                    else []
                ),
            },
        }

    def get_severe_responsibility_violation_keywords(self) -> set:
        return {'crafting', 'pickaxe', 'sword'}

    def get_responsibility_prompt_context(self) -> str:
        return """
SKILL TYPES AND THEIR RESPONSIBILITIES:
- "mineXXX" / "collectXXX": Mining/collecting resources from the world
- "craftXXX": Crafting items (may need to place crafting table for 3x3 recipes - this is OK)
- "smeltXXX": Smelting items (may need to ensure fuel - this is OK)
- "equipXXX": Equipping items to player slots
- "placeXXX": Placing blocks in the world

WHAT IS ALLOWED:
1. Calling existing sub-skills to ensure prerequisites
   - e.g., craftPickaxe calling "await mineLogs(bot, 2)" → OK
   - e.g., smeltOre calling "await ensureFuel(bot)" → OK
2. craftXXX placing a crafting table (required for 3x3 recipes in Minecraft) → OK
3. Small helper functions for validation, counting, finding blocks → OK
4. Error handling and retry logic → OK

WHAT IS NOT ALLOWED (VIOLATIONS):
1. Defining NEW standalone functions that implement a DIFFERENT skill's CORE responsibility
   - e.g., Adding a complete "async function mineLogs() { ...full mining logic... }" inside craftPickaxe → VIOLATION
   - e.g., Adding a complete "async function craftWoodenPickaxe() { ...full crafting logic... }" inside craftCraftingTable → VIOLATION
2. Adding logic completely unrelated to achieving the skill's goal
   - e.g., Adding smelting logic to a mining skill → VIOLATION

KEY DISTINCTION:
- "await existingSkill(bot, params)" → OK (calling dependency, not defining new logic)
- "async function newSkill() { ...20+ lines of complete implementation... }" → LIKELY VIOLATION (injecting a new skill)

HELPER FUNCTION GUIDELINES:
- getItemCount(), findBlock(), validateMaterial() → OK (small utilities)
- ensureMaterials() that CALLS existing skills → OK
- ensureMaterials() that IMPLEMENTS mining/crafting from scratch → VIOLATION"""

    # ------------------------------------------------------------------
    # Action prompt knowledge injection
    # ------------------------------------------------------------------

    def get_action_prompt_knowledge(
        self, task: str, available_skills: list
    ) -> str:
        """Inject task-relevant knowledge into the action agent prompt.

        Uses keyword extraction from the task string to query the
        KnowledgeIndex across multiple ``query_type``s and unions the
        top-k results from each, so primitive-level gotchas and
        API-level subtleties reach the first-pass skill-generation prompt
        alongside action_guidance items.

        Without this union, a single query with ``query_type="code_generation"``
        gives a +5.0 score bonus to action_guidance items (via
        ``KnowledgeIndex._calculate_match_score``), which crowds out
        ``primitive`` / ``api_behavior`` items even when they match task
        keywords strongly. That caused r37's diamond failure: the
        ``mineBlock_strict_name_matching`` and ``findBlock_multiple_matching``
        KnowledgeItems about deepslate variants never reached the
        first-pass prompt, so the LLM hardcoded ``"diamond_ore"`` and
        missed ``"deepslate_diamond_ore"``.

        Option 1a relevance filter: items that mention ``deepslate`` are
        only kept when the task contains an ore-related keyword. This
        prevents the LLM from over-generalizing the "ore has a deepslate
        variant" pattern to non-ore tasks like cobblestone (where mining
        deepslate drops ``cobbled_deepslate``, not ``cobblestone``).

        Validated via A/B replay on r37 (9 cases × 10 runs × 2 conditions
        on Qwen3-Coder-Next-FP8): diamond 0/10 → 10/10 HANDLES_DEEPSLATE,
        cobblestone latent bugs 8 → 0, no new regressions on other
        clean/flaky controls. See ``tests/refactor/test_action_knowledge_ab.py``.
        """
        if not task:
            return ""

        from skillnet.agents.optimizer.knowledge.index import (
            KnowledgeQuery,
            get_knowledge_index,
        )

        # Extract keywords from task
        task_lower = task.lower()
        keywords = []

        # Task verb keywords
        verb_map = {
            "ensure": ["ensure", "variant"],
            "craft": ["craft", "recipe"],
            "mine": ["mine", "block"],
            "smelt": ["smelt", "furnace"],
            "place": ["place", "block", "position"],
            "setup": ["setup", "place", "functional"],
            "build": ["build", "construct", "setup"],
            "kill": ["kill", "combat", "health", "damage", "mob"],
            "explore": ["explore", "underground", "surface", "position"],
        }
        for verb, kws in verb_map.items():
            if verb in task_lower:
                keywords.extend(kws)

        # Item-related keywords from task
        item_patterns = [
            "planks", "log", "logs", "ore", "ingot", "pickaxe", "axe",
            "sword", "shovel", "hoe", "stick", "cobblestone", "stone",
            "oak", "birch", "spruce", "iron", "gold", "diamond",
            "crafting_table", "furnace", "wood", "tree",
            # resource names that the original list missed.
            # Without these, ore-targeted tasks like "Ensure you have 4 coal"
            # had no `coal` keyword, so no resource knowledge got retrieved.
            "coal", "copper", "lapis", "redstone", "emerald",
            "raw_iron", "raw_gold", "quartz", "obsidian", "netherite",
            # water/bucket/lava keywords.
            # Without these, bucket_fill_mechanics (which documents the
            # correct API: activateItem, not useOn) is never retrieved
            # for water bucket tasks, causing r1_3's obsidian blockage.
            "water", "bucket", "lava",
        ]
        for pattern in item_patterns:
            if pattern in task_lower:
                keywords.append(pattern)

        if not keywords:
            return ""

        try:
            index = get_knowledge_index()
        except Exception:
            return ""

        # Multi-query union: query each query_type separately and union
        # the top-k results. Preserves per-type representation so primitive
        # and api_behavior items aren't crowded out by the +5.0 type-match
        # bonus that action_guidance gets for a single code_generation query.
        #
        # added `resource` and `game_mechanic` query
        # types to the union so that mining/crafting tasks receive ore
        # Y-range facts and drop/fuel info at INITIAL generation, not only
        # via the optimizer's Phase 1 reflection after a failure.
        # Without this, r0's ensureCoal was generated without any awareness
        # that coal spawns optimally at Y=96 (overworld) or Y=-32 (deepslate),
        # leaving the LLM to guess where to search.
        merged: list = []
        seen: set = set()
        for qt, k in [
            ("code_generation", 3),
            ("primitive", 2),
            ("api_behavior", 2),
            ("resource", 2),        # ore Y-ranges + biome constraints
            ("game_mechanic", 2),   # drop rules, fuel alternatives, etc.
        ]:
            query = KnowledgeQuery(
                query_type=qt,
                keywords=keywords,
                context=task,
            )
            try:
                results = index.search(query, max_results=k)
            except Exception:
                continue
            for text in results:
                if text not in seen:
                    merged.append(text)
                    seen.add(text)

        # Relevance filter: drop deepslate-mentioning items from non-ore tasks.
        # Rationale: deepslate variants only exist for ores (diamond_ore,
        # iron_ore, coal_ore, ...) and mining the deepslate block itself
        # drops cobbled_deepslate, not cobblestone. Keyword matching can
        # surface deepslate knowledge for "cobblestone" tasks (via "stone"
        # partial match), which causes the LLM to invent false variants
        # like mining deepslate for cobblestone. Filter these out when the
        # task clearly doesn't involve an ore that has a deepslate variant.
        ore_relevant_keywords = {
            "diamond", "iron", "gold", "coal", "copper",
            "lapis", "redstone", "emerald", "ore",
        }
        task_has_ore = any(kw in task_lower for kw in ore_relevant_keywords)
        if not task_has_ore:
            merged = [t for t in merged if "deepslate" not in t.lower()]

        if not merged:
            return ""

        section = "\n\n## Task-Relevant Knowledge\n"
        for result_text in merged:
            section += f"\n{result_text}\n"
        return section

    # ------------------------------------------------------------------
    # Game mechanics methods
    # ------------------------------------------------------------------

    def get_emergency_task(
        self, y: float, inventory: dict, equipment: list,
    ) -> Optional[str]:
        """Underground without pickaxe → suggest crafting one."""
        if y >= 50:
            return None

        has_pickaxe = any(key.endswith("_pickaxe") for key in inventory)
        if has_pickaxe:
            return None

        # Check materials for crafting a pickaxe
        has_sticks = inventory.get("stick", 0) >= 2
        has_cobble = inventory.get("cobblestone", 0) >= 3
        has_iron = inventory.get("iron_ingot", 0) >= 3

        # Calculate potential sticks from planks/logs
        planks_count = sum(v for k, v in inventory.items() if k.endswith("_planks"))
        logs_count = sum(v for k, v in inventory.items() if k.endswith("_log"))
        potential_planks = planks_count + (logs_count * 4)
        # 2 planks -> 4 sticks
        potential_sticks = inventory.get("stick", 0) + ((potential_planks // 2) * 4)
        can_make_sticks = potential_sticks >= 2

        # Prefer iron pickaxe > stone pickaxe
        if has_iron and (has_sticks or can_make_sticks):
            return "Craft 1 iron pickaxe"
        if has_cobble and (has_sticks or can_make_sticks):
            return "Craft 1 stone pickaxe"

        return None

    def get_full_inventory_task(
        self, inventory: dict, chest_observation: str,
    ) -> Optional[str]:
        """5-priority waterfall for handling full inventory."""
        # 1. Placed chest nearby → deposit
        has_placed_chest = (
            chest_observation
            and not chest_observation.startswith("Chests: None")
            and "\n(" in chest_observation
        )
        if has_placed_chest:
            return "Deposit useless items into the chest"

        # 2. Chest item in inventory → place it
        if inventory.get("chest", 0) > 0:
            return "Place a chest and deposit useless items into it"

        # 3. Enough planks → craft chest
        if sum(v for k, v in inventory.items() if "planks" in k) >= 8:
            return "Craft 1 chest, place it, and deposit useless items"

        # 4. Enough logs → craft planks then chest
        if sum(v for k, v in inventory.items() if k.endswith("_log")) >= 2:
            return "Craft planks from logs, then craft and place a chest"

        # 5. Last resort
        return "Free up inventory space: drop excess building blocks to make room, then craft and place a chest"

    def has_sufficient_materials_for_progression(
        self, inventory: dict,
    ) -> bool:
        """Check if we have enough materials for early-game crafting."""
        logs_count = sum(v for k, v in inventory.items() if k.endswith("_log"))
        planks_count = sum(v for k, v in inventory.items() if k.endswith("_planks"))
        sticks_count = inventory.get("stick", 0)

        potential_planks = planks_count + (logs_count * 4)
        potential_sticks = sticks_count + ((potential_planks // 2) * 4)

        # 16 planks = 4 logs, enough for crafting_table + wooden_pickaxe + wooden_axe
        has_enough = potential_planks >= 16 and potential_sticks >= 4

        if has_enough:
            print(
                f"\033[32m[PSN] Sufficient materials for early game: "
                f"{logs_count} logs, {planks_count} planks, {sticks_count} sticks\033[0m"
            )
        return has_enough

    def get_inventory_config(self) -> dict:
        return {"stack_size": 64, "total_slots": 36, "full_threshold": 27}

    def get_equipment_slot_names(self) -> list:
        return ["head", "torso", "legs", "feet", "hand", "off-hand"]

    def get_skill_name_fallbacks(self) -> dict:
        return {
            "craft": "craftItem",
            "mine": "mineBlock",
            "place": "placeBlock",
            "get": "getItem",
            "collect": "getItem",
        }

    def get_crafting_station_config(self) -> dict:
        return {
            "item_name": "crafting_table",
            "craft_task": "Craft 1 crafting table",
        }

    def get_tool_tier_config(self) -> dict:
        return {
            "tool_tiers": {
                "wooden": 1, "stone": 2, "iron": 3,
                "gold": 4, "diamond": 5, "netherite": 6,
            },
            "material_prefixes": [
                "oak", "birch", "spruce", "jungle", "acacia", "dark_oak",
                "mangrove", "cherry", "iron", "gold", "diamond", "netherite",
                "copper", "stone", "wooden",
            ],
            "generic_patterns": [
                "planks", "log", "ore", "ingot", "pickaxe", "axe",
                "sword", "shovel", "hoe", "fuel",
            ],
        }
