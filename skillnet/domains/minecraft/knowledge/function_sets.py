"""
Minecraft Function Sets for Code Validation

Minecraft/Mineflayer-specific function sets used by code validators.
Source of truth for bot methods, control primitives, helper functions,
item name rules, and semantic validation patterns.

Epochs 101-102: Extracted from optimizer/validators/code_validator/.
Added LOOP_BREAK_API_HINTS from loop_manager.py.
"""

# Mineflayer bot methods (called via bot.xxx; no validation needed)
BOT_METHODS = {
    "chat", "whisper", "dig", "placeBlock", "equip", "unequip", "toss",
    "activate", "activateBlock", "activateEntity", "activateItem",
    "deactivateItem", "useOn", "attack", "mount", "dismount",
    "moveSlotItem", "setQuickBarSlot", "craft", "openContainer",
    "openChest", "openFurnace", "closeWindow", "clickWindow",
    "creative", "look", "lookAt", "setControlState", "clearControlStates",
    "swingArm", "consume", "fish", "activateEntity",
    # Properties and objects
    "inventory", "entity", "pathfinder", "creative", "quickBarSlot",
    "food", "health", "experience", "scoreboards", "settings",
    "world", "blockAt", "blockAtCursor", "findBlock", "findBlocks",
    "canDigBlock", "canSeeBlock", "canSeeEntity",
    "nearestEntity", "players", "entities", "isRaining",
    "waitForTicks", "waitForChunksToLoad", "awaitMessage",
    # Recipe-related
    "recipesFor", "recipesAll", "checkRecipe",
    # pathfinder-related
    "goto", "setMovements", "setGoal", "stop",
}

# Control primitives known to PSN (originally from the Voyager codebase)
KNOWN_PRIMITIVES = {
    "mineBlock", "craftItem", "smeltItem", "placeItem", "killMob",
    "exploreUntil", "giveItem", "useFurnace", "collectBlock",
    "useChest", "shoot", "givePlacedItemBack", "waitForMobRemoved",
}

# Helper functions in control_primitives/ (injected via globalDepsCode)
CONTROL_PRIMITIVE_HELPERS = {
    # craftHelper.js
    "failedCraftFeedback",
    # adaptive_helpers.js
    "findNearbyOpenArea", "findValidReferenceBlock", "findDiggableBlock",
    "canPlaceAt", "moveNear", "exploreUntilFound", "findBlockExpanded", "safeDig",
    # timeout_utils.js
    "withTimeout", "getBotPosStr", "gotoWithTimeout", "collectWithTimeout",
}

# Pathfinder Goal classes + utility classes destructured from
# `require("mineflayer-pathfinder")` at the /step route handler scope in
# skillnet/domains/minecraft/action_space/env/mineflayer/index.js:497-523. These are accessible to all skill
# code (and to LLM-generated optimizations) via the JavaScript scope chain —
# eval(fullCode) runs INSIDE the /step handler, so bare-name references like
# `new GoalPlaceBlock(...)` resolve via lexical lookup → enclosing handler scope.
#
# v12 forensic evidence: control_primitives/placeItem.js:113 uses
# `new GoalPlaceBlock(...)` directly with no import, and works in production.
# adaptive_helpers.js:15 explicitly documents this pattern.
#
# The Consistency Check / Reference Check passes (which read this set via
# DomainKnowledge.get_known_functions()) previously flagged these as
# "undefined" and triggered false-positive optimization rejections (v12 log
# line 11442) — even when Phase 2 produced correct architecture-aware code.
KNOWN_PATHFINDER_GOALS = {
    # Goal classes (from goals submodule)
    "Goal",
    "GoalBlock", "GoalNear", "GoalXZ", "GoalNearXZ", "GoalY",
    "GoalGetToBlock", "GoalLookAtBlock", "GoalBreakBlock",
    "GoalCompositeAny", "GoalCompositeAll", "GoalInvert",
    "GoalFollow", "GoalPlaceBlock",
    # Top-level pathfinder exports also destructured at handler scope
    "Movements", "Move", "ComputedPath", "PartiallyComputedPath",
    "XZCoordinates", "XYZCoordinates", "SafeBlock", "GoalPlaceBlockOptions",
    "pathfinder",
}

# Common helper function patterns (defined inside the code)
COMMON_HELPER_PATTERNS = {
    "countItemByName",
    "ensureNearPosition",
    "findNearbyBlock",
    "getItemCount",
}

# Unambiguous item name corrections (only one correct answer)
KNOWN_INVALID_ITEM_NAMES = {
    "sticks": "stick",
    "diamonds": "diamond",
    "coals": "coal",
    "torches": "torch",
    "stones": "stone",
    "furnaces": "furnace",
}

# Ambiguous invalid names (need specific type prefix, cannot auto-correct)
AMBIGUOUS_ITEM_NAMES = {
    "wooden_plank", "wood_planks", "wooden_planks",  # → oak_planks? birch_planks?
    "plank", "planks",   # needs prefix e.g. oak_planks
    "log", "logs",       # needs prefix e.g. oak_log
    "wood",              # oak_wood? oak_log?
}


# ============================================================================
# Semantic validation: function name → required operation patterns
# ============================================================================

# Maps function name prefixes to required API call patterns
FUNCTION_OPERATION_PATTERNS = {
    "mine": {
        "required_patterns": [
            r'\bmineBlock\s*\(',
            r'\bbot\.dig\s*\(',
            r'\bcollectBlock\s*\(',
        ],
        "forbidden_only_delegates": [r'^ensure\w*$', r'^get\w*$'],
        "description": "Mining function must contain mining logic (mineBlock, bot.dig)",
    },
    "craft": {
        "required_patterns": [
            r'\bcraftItem\s*\(',
            r'\bbot\.craft\s*\(',
            r'\bbot\.recipesFor\s*\(',
            r'\bbot\.checkRecipe\s*\(',
        ],
        "forbidden_only_delegates": [r'^ensure\w*$', r'^get\w*$'],
        "description": "Crafting function must contain crafting logic (craftItem, bot.craft)",
    },
    "smelt": {
        "required_patterns": [
            r'\bsmeltItem\s*\(',
            r'\bbot\.openFurnace\s*\(',
        ],
        "forbidden_only_delegates": [r'^ensure\w*$'],
        "description": "Smelting function must contain smelting logic (smeltItem, bot.openFurnace)",
    },
    "kill": {
        "required_patterns": [
            r'\bkillMob\s*\(',
            r'\bbot\.attack\s*\(',
        ],
        "forbidden_only_delegates": [],
        "description": "Combat function must contain combat logic (killMob, bot.attack)",
    },
    "place": {
        "required_patterns": [
            r'\bplaceItem\s*\(',
            r'\bbot\.placeBlock\s*\(',
        ],
        "forbidden_only_delegates": [r'^ensure\w*$', r'^setup\w*$'],
        "description": "Placement function must contain placement logic (placeItem, bot.placeBlock)",
    },
}

# Raw API → control primitive diagnostic mappings
# Item keyword → production function aliases
# Used by EffectsConsistencyValidator to detect production capability
# when generic ensure{item_base} pattern doesn't match (e.g., "oak_planks" → "ensureplanks")
ITEM_PRODUCTION_ALIASES = {
    "planks": {"ensure": ["ensureplanks"]},
    "log": {"ensure": ["ensurelogs"]},
    "coal": {"ensure": ["ensurecoal"], "mine_ore": ["coal_ore"]},
    "stick": {"ensure": ["ensuresticks"], "craft": ["stick"]},
}

# Overclaim detector: Minecraft material→ingredient mappings
MATERIAL_TO_INGREDIENT = {
    "wooden": ["oak_planks", "birch_planks", "spruce_planks", "jungle_planks",
               "acacia_planks", "dark_oak_planks", "mangrove_planks", "cherry_planks"],
    "stone": ["cobblestone"],
    "iron": ["iron_ingot"],
    "gold": ["gold_ingot"],
    "diamond": ["diamond"],
    "netherite": ["netherite_ingot"],
}

# Overclaim detector: problematic string concatenation patterns
# Each entry: (regex_pattern, pattern_name, warning_message)
PROBLEMATIC_CONCATENATIONS = [
    (r'material\s*\+\s*["\']_planks["\']', "material_planks",
     "Only works for 'wooden'. Use explicit mapping for other materials."),
    (r'`\$\{material\}_planks`', "material_planks_template",
     "Only works for 'wooden'. Use explicit mapping for other materials."),
    (r'material\s*\+\s*["\']_ingot["\']', "material_ingot",
     "Only works for iron/gold. Wooden uses planks, stone uses cobblestone."),
    (r'`\$\{material\}_ingot`', "material_ingot_template",
     "Only works for iron/gold. Wooden uses planks, stone uses cobblestone."),
]

# Overclaim detector: pattern name → actually supported material values
PATTERN_SUPPORTED_VALUES = {
    "material_planks": ["wooden"],
    "material_planks_template": ["wooden"],
    "material_ingot": ["iron", "gold"],
    "material_ingot_template": ["iron", "gold"],
}

# Overclaim detector: semantic context keywords for parameter classification
OVERCLAIM_SEMANTIC_CONTEXTS = ["material", "tool", "block", "ore", "item", "fuel", "log"]

CONTROL_PRIMITIVE_API_MAPPINGS = [
    {
        "pattern": r'\bbot\.craft\s*\(',
        "api": "bot.craft()",
        "primitive": "craftItem(bot, name, count)",
        "benefit": "Automatically handles crafting-table lookup, 2x2/3x3 recipe distinction, and inventory sync",
    },
    {
        "pattern": r'\bbot\.recipesFor\s*\(',
        "api": "bot.recipesFor()",
        "primitive": "craftItem(bot, name, count)",
        "benefit": "Automatically handles recipe lookup and crafting logic",
    },
    {
        "pattern": r'\bbot\.recipesAll\s*\(',
        "api": "bot.recipesAll()",
        "primitive": "craftItem(bot, name, count)",
        "benefit": "Automatically handles recipe lookup and crafting logic",
    },
    {
        "pattern": r'\bbot\.checkRecipe\s*\(',
        "api": "bot.checkRecipe()",
        "primitive": None,
        "benefit": "Distinguishes 'recipe not found' from 'insufficient materials' and provides detailed error info",
        "is_recommended": True,
    },
    {
        "pattern": r'\bbot\.dig\s*\(',
        "api": "bot.dig()",
        "primitive": "mineBlock(bot, name, count)",
        "benefit": "Automatically handles pathfinding, tool selection, and inventory sync",
    },
    {
        "pattern": r'\bbot\.placeBlock\s*\(',
        "api": "bot.placeBlock()",
        "primitive": "placeItem(bot, name, position)",
        "benefit": "Automatically handles pathfinding, position validation, and reference-block calculation",
    },
    {
        "pattern": r'\bbot\.openFurnace\s*\(',
        "api": "bot.openFurnace()",
        "primitive": "smeltItem(bot, itemName, fuelName, count)",
        "benefit": "Automatically handles furnace interaction and item placement",
    },
]

# Loop-breaking API hints per strategy issue type
LOOP_BREAK_API_HINTS = {
    "placement_failure": [
        "Use placeItem(bot, name, position) instead of inline placement logic (it has built-in movement)",
        "Or call bot.pathfinder.goto(new GoalNear(x, y, z, 2)) to move first, then retry",
        "findNearbyOpenArea(bot, position, radius) can find an open area",
    ],
    "pathfinding_stuck": [
        "Check whether the target is reachable: bot.pathfinder.getPathTo(goal)",
        "Consider using bot.dig(block) to clear obstacles",
        "Or move to a closer intermediate point and retry",
    ],
    "mining_obstructed": [
        "Use bot.findBlock() to find a nearer block of the same kind",
        "Use bot.canSeeBlock() to check for obstacles",
        "Consider clearing nearby obstacles first",
    ],
}
