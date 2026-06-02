"""
API Knowledge - Bot Methods
Mineflayer Bot core methods knowledge

Migrated from api_behaviors.py
All code examples removed, only behavior descriptions retained
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# Bot Core Methods Knowledge
# ============================================================

BOT_METHODS_KNOWLEDGE: List[KnowledgeItem] = [
    # Block Placement
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.placeBlock",
        fact=(
            "bot.placeBlock(referenceBlock, faceVector) places equipped block on the face of referenceBlock. "
            "Requires: item equipped in hand, valid reference block, target position is air, within 4.5 blocks. "
            "Waits for 'blockUpdate' event (5000ms timeout) to confirm placement."
        ),
        keywords=["place", "block", "placeblock", "equip", "reference", "timeout"],
        conditions=["placing blocks"],
    ),

    # Block Mining
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.dig",
        fact=(
            "bot.dig(block, forceLook) breaks the specified block. "
            "Bot must be within reach (4.5 blocks). Time depends on block hardness and equipped tool. "
            "Returns when block is broken; may take significant time without proper tool."
        ),
        keywords=["dig", "mine", "break", "block", "tool", "reach"],
        conditions=["breaking blocks", "mining"],
    ),

    # Item Equipping
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.equip",
        fact=(
            "bot.equip(item, destination) moves item to specified slot (hand, head, torso, legs, feet, off-hand). "
            "Item must exist in inventory. Must be called immediately before actions that need the item."
        ),
        keywords=["equip", "item", "hand", "slot", "inventory", "hold"],
        conditions=["preparing to use an item"],
    ),

    # Crafting
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.craft",
        fact=(
            "bot.craft(recipe, count, craftingTable) crafts items using a recipe. "
            "CRITICAL: 'recipe' must be a Recipe OBJECT obtained from bot.recipesFor() or bot.checkRecipe(), "
            "NOT an item name string or numeric item ID. "
            "Passing wrong type (e.g., mcData.itemsByName[name].id) causes TypeError at craft.js: "
            "'Cannot read properties of undefined (reading 0)'. "
            "This error looks like a mcData naming issue but is actually a bot.craft() argument type error. "
            "The craftItem(bot, name, count) control primitive handles Recipe object resolution internally, "
            "accepting a simple item name string instead of requiring a Recipe object. "
            "If recipe requires 3x3 grid, craftingTable block must be provided and within range. "
            "WARNING: bot.craft may fail silently (no error thrown) if materials are insufficient or "
            "the recipe/count combination is invalid — always verify inventory after crafting."
        ),
        keywords=["craft", "recipe", "crafting", "table", "materials", "bot.craft",
                  "craft.js", "bot.recipesFor", "bot.checkRecipe", "craftItem"],
        conditions=[
            "crafting items",
            "TypeError at craft.js",
            "Cannot read properties of undefined (reading '0') in craft context",
        ],
    ),

    # Container Interaction
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.openContainer",
        fact=(
            "bot.openContainer(containerBlock) opens a container (chest, furnace, etc.) and returns Window object. "
            "Bot must be within interaction range (about 4 blocks). Must close window when done."
        ),
        keywords=["open", "container", "chest", "furnace", "window", "interact"],
        conditions=["accessing containers"],
    ),

    # Block Finding
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.findBlock",
        fact=(
            "bot.findBlock({matching, maxDistance, count}) searches for blocks matching criteria. "
            "Returns null if no matching block found within range. "
            "Only finds blocks in loaded chunks; range limited by server render distance."
        ),
        keywords=["find", "block", "search", "matching", "distance", "null"],
        conditions=["searching for specific blocks"],
    ),

    # findBlock null return crash pattern
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="findBlock_null_return",
        fact=(
            "bot.findBlock({matching, maxDistance}) returns null when no matching block "
            "is found within maxDistance. ALWAYS null-check the return value before "
            "accessing .position, .name, or any other property. "
            "Common crash pattern: bot.findBlock({...}).position.offset(...) crashes with "
            "'Cannot read properties of undefined (reading offset)' when no block found. "
            "Defensive access requires null-checking the return value before accessing properties."
        ),
        keywords=["findBlock", "null", "undefined", "offset", "position",
                  "Cannot read properties", "reading", "crafting_table",
                  "matching", "maxDistance"],
        conditions=["bot.findBlock() called on block that may not exist nearby"],
    ),

    # General null/undefined property access patterns
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="null_property_access_general",
        fact=(
            "Many Mineflayer API calls return null or undefined when their target "
            "is not found. Accessing properties (.x, .y, .z, .position, "
            ".plus(), .offset(), .distanceTo(), .floored()) on null/undefined causes "
            "'Cannot read properties of null/undefined' TypeError crash. "
            "Common nullable sources: "
            "bot.findBlock() → null when no block found; "
            "bot.blockAt(pos) → null for unloaded chunks or invalid positions; "
            "bot.entity.position → undefined if bot entity not initialized; "
            "bot.nearestEntity() → null if no entity matches filter; "
            "exploreUntil() callback result → undefined if no match found; "
            "array.find() → undefined if no element matches. "
            "bot.findBlocks() internal matching callback receives Block objects "
            "with position=null during palette scanning — must check position before use."
        ),
        keywords=["null", "undefined", "Cannot read properties", "reading",
                  "position", "x", "y", "z", "plus", "offset", "Vec3",
                  "findBlock", "blockAt", "entity", "nearestEntity",
                  "TypeError", "null check", "defensive", "floored", "distanceTo",
                  "matching", "palette"],
        conditions=[
            "Cannot read properties of null",
            "Cannot read properties of undefined",
            "TypeError accessing position-related property",
        ],
    ),

    # Block At Position
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.blockAt",
        fact=(
            "bot.blockAt(position) returns the Block object at the given Vec3 position. "
            "Returns block info even for air blocks. Returns null for unloaded chunks. "
            "Instant operation, no pathfinding involved."
        ),
        keywords=["blockat", "position", "get", "block", "check", "instant"],
        conditions=["checking block at position"],
    ),

    # Item Activation
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.activateItem",
        fact=(
            "bot.activateItem() uses the currently held item (right-click behavior). "
            "Used for: eating, throwing, placing water/lava, using buckets, shooting bow. "
            "Behavior depends on what item is equipped and what bot is looking at. "
            "NOTE: bot.useOn(target) is for ENTITIES (mobs/players) only, not blocks. "
            "For block interactions (bucket filling, placing), use bot.activateItem()."
        ),
        keywords=["activate", "item", "use", "rightclick", "bucket", "eat",
                  "useOn", "water"],
        conditions=["using held item", "filling bucket", "placing water/lava"],
    ),

    # Block Activation
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.activateBlock",
        fact=(
            "bot.activateBlock(block) right-clicks on a block (interact). "
            "Used for: opening doors, pressing buttons, using levers, interacting with block-based UIs."
        ),
        keywords=["activate", "block", "interact", "door", "button", "lever"],
        conditions=["interacting with interactable blocks"],
    ),

    # Looking
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.lookAt",
        fact=(
            "bot.lookAt(point, force) makes bot look at the specified Vec3 point. "
            "Important for accurate placement, activation, and attacks. "
            "Some actions require looking at the target before executing."
        ),
        keywords=["look", "lookat", "face", "direction", "aim"],
        conditions=["aiming at target"],
    ),
]


# ============================================================
# Special API Behavior Knowledge
# ============================================================

SPECIAL_API_KNOWLEDGE: List[KnowledgeItem] = [
    # mineBlock primitive behavior
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="mineBlock_primitive",
        fact=(
            "mineBlock(bot, blockName, count) is a high-level mining function that: "
            "1) Auto-equips best tool for the block type, "
            "2) Uses bot.collectBlock.collect() to pick up dropped items, "
            "3) Waits for inventory sync after collection. "
            "Different from bot.dig() which only breaks blocks without tool equip or collection."
        ),
        keywords=["mineblock", "primitive", "tool", "collect", "equip", "mining"],
        conditions=["using mineBlock primitive"],
    ),

    # NoChests error
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="collectBlock_NoChests",
        fact=(
            "mineBlock/collectBlock throws 'NoChests' error when inventory is full and no chest locations configured. "
            "The block IS mined, but dropped items cannot be collected. "
            "Error chain: mineBlock() -> collectBlock() -> emptyInventoryIfFull() -> throws NoChests."
        ),
        keywords=["nochests", "inventory", "full", "collectblock", "mineblock", "error"],
        conditions=["mining with full inventory"],
    ),

    # Invalid API warning
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="firstEmptySlot_invalid",
        fact=(
            "bot.inventory.firstEmptySlot() does NOT exist in Mineflayer API. "
            "Calling it throws 'bot.inventory.firstEmptySlot is not a function'. "
            "Use bot.inventory.emptySlotCount() instead, or iterate bot.inventory.slots to find null entries."
        ),
        keywords=["firstemptyslot", "invalid", "not a function", "error", "slots"],
        conditions=["checking for empty slots"],
    ),

    # Weak-LLM Minecraft knowledge corrigendum. Patches incomplete pretraining
    # knowledge for tier-mismatch failure modes (iron+obsidian, wooden+iron_ore,
    # ...) where smaller LLMs hallucinate context-anchored rules from partial
    # data ("lava destroys obsidian", "raw_iron doesn't exist in vanilla", etc.).
    # Cross-LLM ablation: Qwen3 gains ~33% on synthetic tier-mismatch with the
    # rule; gpt-5-mini is unaffected. Re-validate on the target LLM family
    # before deleting.
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.BLOCK_PROPERTY,
        name="tool_tier_rules",
        fact=(
            "Tool tiers: Tier 0 (wooden/golden pickaxe), Tier 1 (stone pickaxe), "
            "Tier 2 (iron pickaxe), Tier 3 (diamond pickaxe), Tier 4 (netherite pickaxe). "
            "Block requirements: coal_ore/nether_quartz_ore Tier 0+, iron_ore/copper_ore/lapis_ore Tier 1+, "
            "diamond_ore/gold_ore/emerald_ore/redstone_ore Tier 2+, obsidian/ancient_debris Tier 3+. "
            "Using a lower tier tool BREAKS the block but drops NOTHING."
        ),
        keywords=["pickaxe", "tier", "mine", "ore", "diamond", "iron", "stone", "harvest", "drop", "tool"],
        conditions=["mining ores", "harvesting blocks with tools"],
        logical_implications=[
            "stone pickaxe cannot harvest diamond ore - block breaks but no drops",
            "iron pickaxe cannot harvest obsidian - block breaks but no drops",
            "block break animation completes even with wrong tool tier",
            "pattern 'mined but got 0 items' indicates tool tier mismatch",
        ],
        constraints=[
            "tool tier < required tier = block destroyed, no drops",
            "mining animation and time are independent of drop success",
        ],
        source="study02_study04_retained_weak_llm_minecraft_corrigendum",
    ),

    # Water source vs flowing (migrated from WATER_SOURCE_VS_FLOWING APIBehavior)
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="water_source_vs_flowing",
        fact=(
            "Water blocks have two types: source (level=0, static, can fill bucket) and "
            "flowing (level>0, moving, CANNOT fill bucket). "
            "bot.findBlock matching 'water' returns BOTH types indiscriminately."
        ),
        keywords=["water", "source", "flowing", "bucket", "level", "fill"],
        conditions=["filling buckets", "working with water blocks"],
        constraints=[
            "only source blocks (level=0) can fill buckets",
            "bot.findBlock matches both source and flowing water",
            "activateItem on flowing water does nothing",
        ],
    ),

    # prismarine-block field/method API contract (Phase 13.B-7).
    # Pure Tier-2 framework API contract — analogous to vec3_api_contract.
    # Origin: ckpt_psnv11_qwen3_fp8_r4r1 ensureWaterBucket round-3 optimizer
    # feedback hallucinated `block.meta === 0`; LLM copied it; subsequent 7
    # rounds couldn't fix because the API itself is non-existent. See
    # scripts/PHASE_11_12_FINDINGS.md (Phase 13.B-7) for the production trace.
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="prismarine_block_property_api",
        fact=(
            "The Block object returned by bot.blockAt(pos) is a prismarine-block 1.16+ "
            "instance. To read blockstate properties (e.g. water level, redstone power, "
            "door open/closed) use ONE of: "
            "block.metadata (number, default 0), "
            "block._properties (object, e.g. {level: 0} for water source), "
            "block.getProperties() (method, returns _properties), "
            "block.stateId (full block state ID). "
            "There is NO `block.meta` field and NO `block.state` field; both are "
            "common LLM hallucinations. `block.meta === 0` is `undefined === 0` = false, "
            "so any findBlocks() callback using it returns an empty array regardless of "
            "actual block state."
        ),
        keywords=[
            "prismarine", "block", "property", "metadata", "_properties",
            "getProperties", "stateId", "meta", "state", "level", "findBlock",
            "findBlocks", "matching", "callback", "hallucination",
        ],
        conditions=[
            "reading block state properties",
            "filtering blocks by level/state in findBlocks callbacks",
            "checking water source vs flowing",
        ],
        constraints=[
            "block.meta does not exist (always undefined)",
            "block.state does not exist (always undefined)",
            "use block._properties.level / getProperties().level / metadata for level checks",
        ],
        source="study_phase13b7_prismarine_block_api_hallucination",
    ),

    # Bucket fill mechanics (migrated from BUCKET_FILL_MECHANICS APIBehavior)
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bucket_fill_mechanics",
        fact=(
            "Bucket filling requires: have empty bucket in inventory, find water SOURCE block (not flowing), "
            "move within interaction range (<4 blocks), equip bucket to hand, look at water source, "
            "call activateItem(), wait for inventory sync (2-10 ticks), verify water_bucket in inventory."
        ),
        keywords=["bucket", "fill", "water", "activateItem", "equip", "source"],
        conditions=["filling buckets with water"],
        constraints=[
            "must be within interaction range (<4 blocks)",
            "must look at water source block center",
            "inventory sync takes 2-10 ticks after activation",
        ],
    ),

    # findBlock/findBlocks multiple matching
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="findBlock_multiple_matching",
        fact=(
            "bot.findBlock/findBlocks matching parameter accepts a single block ID, an ARRAY of block IDs, "
            "or a function(block) returning boolean. "
            "When searching for an ore that has both regular and deepslate variants, "
            "matching must include BOTH block IDs or use a function matching both names. "
            "Block ID pairs: diamond_ore/deepslate_diamond_ore, iron_ore/deepslate_iron_ore, "
            "gold_ore/deepslate_gold_ore, coal_ore/deepslate_coal_ore, copper_ore/deepslate_copper_ore, "
            "lapis_ore/deepslate_lapis_ore, redstone_ore/deepslate_redstone_ore, "
            "emerald_ore/deepslate_emerald_ore."
        ),
        keywords=["findBlock", "findBlocks", "matching", "multiple", "array", "block id",
                  "deepslate", "ore", "variant", "diamond_ore", "iron_ore", "both"],
        conditions=["searching for ores", "findBlock returns empty for ore that should exist",
                    "ore not found at expected Y level"],
        constraints=[
            "Each ore variant has a unique numeric block ID in mcData",
            "mcData.blocksByName['diamond_ore'].id !== mcData.blocksByName['deepslate_diamond_ore'].id",
            "All deepslate ore variants follow naming pattern: 'deepslate_' + ore_name",
        ],
    ),
    # mcData item/block naming convention
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="mcdata_item_naming",
        fact=(
            "minecraft-data (mcData) uses Minecraft's internal registry names, which are always SINGULAR. "
            "mcData.itemsByName[name] and mcData.blocksByName[name] return undefined if the name doesn't exist in the registry. "
            "Accessing .id on undefined throws 'Cannot read properties of undefined (reading id)'. "
            "This error may appear far from the wrong name if caught by a try/catch and re-thrown as a different error "
            "(e.g., a crafting failure may actually be caused by a wrong mcData name upstream)."
        ),
        keywords=["undefined", "reading 'id'", "itemsByName", "blocksByName", "mcData",
                  "Cannot read properties", "item name", "registry"],
        conditions=[
            "mcData.itemsByName or mcData.blocksByName returns undefined",
            "Cannot read properties of undefined (reading 'id')",
            "crafting/inventory operations fail unexpectedly",
        ],
        constraints=[
            "minecraft-data uses Minecraft internal registry names (always singular, underscore-separated)",
            "no automatic correction — undefined is silently returned for unrecognized names",
            "item and block registries are separate: same name may have different IDs in each",
        ],
    ),

    # Bot Status Properties — readable properties for health/safety awareness
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.health",
        fact=(
            "bot.health (Number, 0-20): the bot's current health points. "
            "Readable property, always up-to-date. 20 = full health, 0 = dead."
        ),
        keywords=["health", "dead", "death", "died", "damage", "hurt", "missing entity"],
        conditions=[
            "bot dies during execution",
            "bot health-related failure",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.food",
        fact=(
            "bot.food (Number, 0-20): the bot's current hunger level. "
            "Readable property. 20 = fully satiated, 0 = starving. "
            "At 0, the bot takes starvation damage."
        ),
        keywords=["food", "hunger", "starving", "saturation"],
        conditions=[
            "bot starves during execution",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.oxygenLevel",
        fact=(
            "bot.oxygenLevel (Number, 0-300): remaining breath ticks when submerged in water. "
            "Readable property. Decreases while underwater, resets when surfacing. "
            "At 0, the bot takes drowning damage."
        ),
        keywords=["oxygen", "drowning", "drowned", "submerged", "underwater", "breath"],
        conditions=[
            "bot drowns during execution",
            "bot dies underwater",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.entity.isInWater",
        fact=(
            "bot.entity.isInWater (Boolean): whether the bot is currently in water. "
            "bot.entity.isInLava (Boolean): whether the bot is currently in lava. "
            "Both are readable properties on bot.entity."
        ),
        keywords=["isInWater", "isInLava", "swimming", "drowning", "burning", "lava damage"],
        conditions=[
            "bot enters water or lava during exploration",
            "bot takes environmental damage",
        ],
    ),

    # Movement Control
    KnowledgeItem(
        domain=KnowledgeDomain.API,
        category=KnowledgeCategory.BOT_METHOD,
        name="bot.setControlState",
        fact=(
            "bot.setControlState(control, state) sets a movement control to true or false. "
            "Available controls: 'forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak'. "
            "state is a boolean. The bot continues the action until it is set to false. "
            "Example: bot.setControlState('jump', true) makes the bot jump continuously. "
            "bot.setControlState('sprint', true) makes the bot sprint while moving forward. "
            "bot.setControlState('sneak', true) makes the bot sneak (shift-walk). "
            "Call bot.clearControlStates() to reset all controls to false."
        ),
        keywords=["setControlState", "jump", "sprint", "sneak", "forward", "back", "movement", "control"],
        conditions=[
            "bot needs to jump",
            "bot needs to sprint",
            "bot needs to sneak",
            "manual movement control",
        ],
    ),
]


# Combine all
BOT_KNOWLEDGE: List[KnowledgeItem] = BOT_METHODS_KNOWLEDGE + SPECIAL_API_KNOWLEDGE


def get_all_bot_methods_knowledge() -> List[KnowledgeItem]:
    """Get all bot methods knowledge items."""
    return BOT_KNOWLEDGE.copy()
