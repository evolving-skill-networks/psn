"""
Action Guidance Knowledge Items

Facts and API documentation for dynamic injection into the action agent prompt.
These are task-relevant knowledge items that complement the static system prompt.

Design principle: Only FACTS and API documentation, no solution hints.
See knowledge_base.py docstring for the distinction.
"""

from typing import List

from skillnet.core.knowledge_base import (
    KnowledgeDomain,
    KnowledgeCategory,
    KnowledgeItem,
)


ACTION_GUIDANCE_ITEMS: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.CODE_GENERATION,
        name="item_variant_dynamic_patterns",
        fact=(
            "Minecraft items have variant groups that can be discovered via mcData: "
            "Object.keys(mcData.blocksByName).filter(n => n.endsWith('_log')). "
            "Variant groups include: logs (*_log), planks (*_planks), ores (*_ore), "
            "tools (wooden_*/stone_*/iron_*/diamond_* + pickaxe/axe/sword/shovel). "
            "When writing skills that work with variant groups, accept the variant "
            "as a parameter with a sensible default."
        ),
        keywords=[
            "variant", "planks", "logs", "dynamic", "mcData", "pattern",
            "oak", "birch", "spruce", "endsWith", "ensure",
            "item group", "wood type",
        ],
        conditions=["working with item groups", "parameterizing for multiple variants"],
        source="migrated from Section 27 of parameterized_action_template",
    ),

    # Pure API contract documentation: logical_implications contains genuine
    # return-type semantics (recipesFor empty-array ambiguity, checkRecipe.message
    # format, recipe object reuse) that are distinct from the fact field's
    # behavioral description — part of the API contract, not strategy or game
    # knowledge.
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.CODE_GENERATION,
        name="checkRecipe_api_contract",
        fact=(
            "bot.checkRecipe(itemName, count, craftingTable) returns structured info: "
            "{exists: false, reason: 'no_recipe'} if recipe doesn't exist, "
            "{exists: true, available: false, reason: 'insufficient_materials', missing: [...], message: '...'} "
            "if materials insufficient, "
            "{exists: true, available: true, recipe: Recipe, message: '...'} if craftable. "
            "In contrast, bot.recipesFor() returns an empty array for BOTH 'no recipe' AND "
            "'insufficient materials' — it cannot distinguish between these cases."
        ),
        keywords=[
            "craft", "recipe", "checkRecipe", "recipesFor", "materials",
            "insufficient", "no recipe", "missing", "crafting_table", "3x3",
            "cannot craft",
        ],
        logical_implications=[
            "bot.recipesFor() empty array is ambiguous: could be no recipe OR insufficient materials",
            "bot.checkRecipe().message provides human-readable error including missing materials",
            "bot.checkRecipe().recipe can be passed directly to bot.craft()",
        ],
        conditions=["crafting items", "diagnosing recipe failures"],
        source="migrated from Section 22 of parameterized_action_template; study04_retained_api_doc",
    ),

    # Phase 1c: Migrated from Section 27 (Tree Harvesting)
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.CODE_GENERATION,
        name="mineBlock_tree_harvesting_behavior",
        fact=(
            "mineBlock internally prioritizes blocks at or below the bot's Y level for "
            "reachability, mining bottom-most blocks first. "
            "Minecraft game mechanic: log blocks without support below them fall down. "
            "These two behaviors together make tree harvesting efficient: mining bottom "
            "tree logs causes unsupported upper logs to fall and become collectible."
        ),
        keywords=[
            "log", "logs", "oak_log", "birch_log", "spruce_log", "tree",
            "harvest", "wood", "mineBlock", "chop",
        ],
        conditions=["mining logs from trees", "tree harvesting"],
        source="migrated from Section 27 of parameterized_action_template",
    ),

    # Phase 2a: Migrated from Section 24 (Creating setup* Skills)
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.CODE_GENERATION,
        name="setup_skill_parameter_requirement",
        fact=(
            "setup* skills (setupCraftingTable, setupFurnace, etc.) MUST have at least one "
            "optional parameter beyond bot (e.g., maxDistance = 6) to be saved as independent "
            "reusable skills in the skill graph. Without extra parameters, helper functions "
            "are not saved. setup* skills should use placeItem(bot, itemName, position) for "
            "the actual placement step, not low-level bot.placeBlock."
        ),
        keywords=["setup", "crafting_table", "furnace", "functional", "place",
                  "parameter", "maxDistance", "reusable", "helper"],
        conditions=["creating setup* skills", "placing functional blocks"],
        source="migrated from Section 24 of parameterized_action_template",
    ),

    # Phase 2b: Migrated from Section 25 (Placement Best Practices)
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.CODE_GENERATION,
        name="placement_position_selection",
        fact=(
            "When selecting a position for placeItem, the target must be an air block "
            "with at least one solid block adjacent in any of the 6 face directions "
            "(above, below, or one of the 4 horizontal directions). "
            "Bot occupies 2 blocks vertically (foot at bot.entity.position.floored(), "
            "head at foot.offset(0,1,0)). NEVER select a position currently occupied "
            "by foot or head — check: candidate.equals(botFoot) || candidate.equals(botHead). "
            "Use boundingBox === 'block' to verify solid blocks, not just name !== 'air' "
            "(excludes water, lava, flowing fluids)."
        ),
        keywords=["place", "position", "collision", "bot", "standing",
                  "air", "solid", "boundingBox", "placeItem", "setup"],
        conditions=["finding placement position", "setup* skills", "placing functional blocks"],
        source="migrated from Section 25 of parameterized_action_template; "
               "Phase 13.B-4: reverted from 10-cell enumeration cheat (V3_full10) back to "
               "primitives-only fact form. Cheat was load-bearing for Qwen3 placement "
               "reliability but NOT generalizable. Production now relies on raw spatial "
               "neighborhood observation injection (Spatial observer) + LLM-derived "
               "candidate enumeration. See scripts/PHASE_11_13_PLACEMENT_SUMMARY.md.",
    ),

    # Phase 2c: Migrated from Section 27 (Over-Claim Detection)
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.CODE_GENERATION,
        name="tool_material_ingredient_mapping",
        fact=(
            "Minecraft tool crafting uses DIFFERENT ingredients per tier — string concatenation "
            "like material + '_planks' only works for 'wooden'. Correct mapping: "
            "wooden → planks (oak_planks, birch_planks, etc.), stone → cobblestone, "
            "iron → iron_ingot, gold → gold_ingot, diamond → diamond (the gem). "
            "When parameterizing tool crafting across materials, use explicit mapping "
            "(dictionary/object) rather than string concatenation."
        ),
        keywords=["pickaxe", "axe", "sword", "shovel", "hoe", "tool", "material",
                  "wooden", "stone", "iron", "gold", "diamond", "craft",
                  "ingredient", "planks", "cobblestone", "ingot"],
        conditions=["crafting tools", "parameterizing material types"],
        source="migrated from Section 27 of parameterized_action_template",
    ),

    # Phase 4b: Migrated from Section 12b (Bot Status Properties)
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.CODE_GENERATION,
        name="bot_status_properties",
        fact=(
            "Bot status properties (all readable, NOT function calls): "
            "bot.health (Number, 0-20): current health points. "
            "bot.food (Number, 0-20): current hunger level. "
            "bot.oxygenLevel (Number, 0-300): remaining breath ticks when submerged. "
            "bot.entity.isInWater (Boolean): whether the bot is in water. "
            "bot.entity.isInLava (Boolean): whether the bot is in lava. "
            "Note: bot.entity may be undefined if the bot is dead."
        ),
        keywords=[
            "health", "hunger", "food", "water", "oxygen", "damage",
            "combat", "kill", "mob", "underwater", "lava", "swim", "eat",
            "dead", "starving",
        ],
        conditions=["checking bot health", "combat tasks", "underwater tasks"],
        source="migrated from Section 12b of parameterized_action_template",
    ),

    # Phase 4b: Migrated from Sections 19-20 (Environment Awareness + Tool Priority)
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.CODE_GENERATION,
        name="environment_resource_strategy",
        fact=(
            "Check Position.y to determine if underground (y < 50) or on surface. "
            "Underground wood strategy: check inventory for sticks/planks first; "
            "prefer cobblestone + sticks → stone_pickaxe over wooden_pickaxe; "
            "use iron_ingot + sticks if available. If must get wood underground, "
            "explore upward Vec3(0, 1, 0). "
            "Tool priority: iron_pickaxe (3 iron_ingot + 2 sticks) > stone_pickaxe "
            "(3 cobblestone + 2 sticks) > wooden_pickaxe (requires wood). "
            "Always check Nearby blocks to verify resources are accessible."
        ),
        keywords=[
            "underground", "position", "pickaxe", "tool", "cobblestone",
            "stone", "iron", "wood", "log", "craft", "surface", "y level",
            "planks", "stick", "priority",
        ],
        conditions=["underground resource gathering", "tool crafting decisions"],
        source="migrated from Sections 19-20 of parameterized_action_template",
    ),
]
