"""
Game Knowledge - Mechanics
Game mechanics knowledge: placement, pathfinding, crafting, combat, inventory, water/lava physics

Migrated from minecraft_env.py and api_behaviors.py
All solution_hint fields removed
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# Placement Mechanics Knowledge
# ============================================================

PLACEMENT_KNOWLEDGE: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="block_placement_requirement",
        fact=(
            "Block placement requires an adjacent solid block as reference. "
            "Cannot place blocks floating in air without support. "
            "The bot must be within 4.5 blocks (interaction range) of target position."
        ),
        keywords=["place", "block", "adjacent", "solid", "reference", "support"],
        conditions=["placing any block"],
        implications=[
            "floating placement fails",
            "need existing block as anchor",
            "must be close enough to interact",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="interaction_distance",
        fact=(
            "Player interaction distance is 4.5 blocks in survival mode. "
            "This applies to block placement, breaking, opening containers, and entity interaction."
        ),
        keywords=["distance", "range", "interaction", "reach", "blocks"],
        conditions=["any block/entity interaction"],
        implications=[
            "too far causes failure",
            "need to move closer first",
            "applies to all interactions",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="position_occupation",
        fact=(
            "A block cannot be placed where an entity (including the player) is standing. "
            "The target position must be air or replaceable (water, tall grass). "
            "Entity hitboxes prevent block placement in occupied spaces."
        ),
        keywords=["place", "occupied", "entity", "standing", "position", "hitbox",
                  "blockupdate", "timeout", "did not fire", "blockUpdate"],
        conditions=["attempting placement at occupied position"],
        implications=[
            "placement times out or fails",
            "bot itself can block placement",
            "must ensure position is clear",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="fence_and_wall_collision",
        fact=(
            "Fences and walls have 1.5 block collision height, preventing most mobs from jumping over. "
            "Players can jump onto 1-block high fences if adjacent to higher ground."
        ),
        keywords=["fence", "wall", "collision", "jump", "height", "barrier"],
        conditions=["navigating near fences/walls"],
        implications=[
            "blocks walking paths",
            "may need to break or circumvent",
            "affects pathfinding",
        ],
    ),

    # pure-fact placement rule knowledge (so the LLM reasons rather than copies a strategy)
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="block_placement_reference_directions",
        fact=(
            "Block placement requires an adjacent solid block as reference. "
            "ANY of the 6 adjacent positions can serve as reference: above, below, north, south, east, west. "
            "Placement is NOT limited to having a solid block below - walls and ceiling also work. "
            "In narrow spaces like 1x1 shafts or tunnels, horizontal wall blocks are valid references."
        ),
        keywords=["place", "block", "adjacent", "reference", "wall", "shaft", "tunnel", "narrow", "direction"],
        conditions=["placing blocks", "no valid position", "placement failed", "confined space"],
        implications=[
            "walls can serve as placement reference",
            "ceiling can serve as placement reference",
            "checking only 'below' is insufficient",
            "all 6 adjacent directions should be checked",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="block_placement_y_levels",
        fact=(
            "Valid placement positions exist at any Y level within interaction range (~4.5 blocks from bot's eye). "
            "Positions above bot's head (y+1, y+2) are often valid when surrounded by walls. "
            "Only searching at foot level (y=0) may miss valid positions in vertical shafts."
        ),
        keywords=["place", "position", "y level", "height", "above", "head", "search", "shaft"],
        conditions=["finding placement position", "no valid position found", "vertical shaft"],
        implications=[
            "search should include multiple Y levels",
            "y+2 above head often valid in shafts",
            "y=0 foot level may be blocked by bot",
        ],
    ),

    # bot capability knowledge (pure facts, so the LLM reasons about environment-shaping strategies)
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="bot_mining_capability",
        fact=(
            "Bot can mine/dig any block using bot.dig(block). "
            "Mining a solid block removes it, leaving air in its place. "
            "Mining requires appropriate tool for the block type (pickaxe for stone, axe for wood, etc.)."
        ),
        keywords=["mine", "dig", "block", "remove", "air", "terrain", "modify"],
        conditions=["cannot find position", "no valid position", "stuck", "confined"],
        implications=[
            "solid blocks can be converted to air",
            "terrain is not fixed, it can be modified",
            "mining creates new placement opportunities",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="bot_movement_and_jump",
        fact=(
            "Bot can move to any reachable position using pathfinder. "
            "Bot can jump using setControlState('jump', true). "
            "While jumping/airborne, the foot-level position is temporarily unoccupied by the bot."
        ),
        keywords=["move", "jump", "position", "airborne", "foot", "pathfind"],
        conditions=["cannot place", "position occupied", "bot blocking"],
        implications=[
            "bot position is not fixed",
            "foot position becomes air during jump",
            "can place at foot position while airborne",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="placement_requirements_summary",
        fact=(
            "Block placement requires three conditions: "
            "(1) Target position is air or replaceable (water, tall_grass). "
            "(2) At least one adjacent solid block exists as reference. "
            "(3) Bot is within 4.5 blocks interaction range. "
            "If any condition fails, placement fails."
        ),
        keywords=["place", "requirement", "condition", "air", "adjacent", "reference", "range"],
        conditions=["placement failed", "cannot place", "no valid position"],
        implications=[
            "all three conditions must be met",
            "if any condition fails, need to change environment or position",
            "conditions can be satisfied by modifying terrain",
        ],
    ),
]


# ============================================================
# Pathfinding Mechanics Knowledge
# ============================================================

PATHFINDING_KNOWLEDGE: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="pathfinding_obstacles",
        fact=(
            "Pathfinding considers solid blocks as obstacles. "
            "Transparent/non-solid blocks (water, lava, fences, doors) have special handling. "
            "Falling damage is considered when calculating paths."
        ),
        keywords=["path", "pathfind", "obstacle", "blocked", "navigate"],
        conditions=["movement fails", "path not found"],
        implications=[
            "may need to clear obstacles",
            "liquids block normal pathing",
            "high drops avoided by default",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="water_lava_traversal",
        fact=(
            "Water and lava block standard walking paths. "
            "Swimming in water is possible but slows movement. "
            "Lava causes damage; bots typically avoid it completely."
        ),
        keywords=["water", "lava", "swim", "liquid", "path"],
        conditions=["path blocked by liquid"],
        implications=[
            "may need bridge or boat",
            "swimming is slow",
            "lava is deadly",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="pathfinder_timeout",
        fact=(
            "Pathfinding has a timeout (typically 60 seconds). "
            "Very long paths may timeout before completion. "
            "Unreachable goals cause immediate failure."
        ),
        keywords=["timeout", "path", "unreachable", "long", "distance"],
        conditions=["path timeout", "unreachable"],
        implications=[
            "distant goals may fail",
            "blocked paths timeout",
            "may need intermediate waypoints",
        ],
    ),
]


# ============================================================
# Crafting Mechanics Knowledge
# ============================================================

CRAFTING_KNOWLEDGE: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="crafting_table_requirement",
        fact=(
            "Complex recipes requiring 3x3 grid need a crafting table. "
            "Player inventory only has 2x2 crafting grid. "
            "Crafting table must be placed and bot within interaction range."
        ),
        keywords=["craft", "table", "3x3", "recipe", "grid"],
        conditions=["crafting complex items"],
        implications=[
            "simple recipes work in inventory",
            "most tools need crafting table",
            "must place table first",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="furnace_requirement",
        fact=(
            "Smelting requires a furnace (or blast furnace/smoker). "
            "Furnace needs fuel: coal (8 items), charcoal (8), planks (1.5), logs (1.5). "
            "Common smelting recipes: raw_iron -> iron_ingot, raw_gold -> gold_ingot, "
            "raw_copper -> copper_ingot, oak_log -> charcoal, sand -> glass. "
            "IMPORTANT: In Minecraft 1.17+, smelt raw_iron (NOT iron_ore). "
            "iron_ore is the block you mine to get raw_iron."
        ),
        keywords=["smelt", "furnace", "fuel", "coal", "charcoal", "raw_iron",
                  "iron_ore", "iron_ingot", "oak_log"],
        conditions=["smelting items", "cooking food"],
        implications=[
            "need fuel supply",
            "must place furnace",
            "takes time per item",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="crafting_table_item_vs_block",
        fact=(
            "A crafting_table in inventory is an Item object. "
            "craftItem and bot.craft require a placed crafting_table Block in the world "
            "(found by bot.findBlock). These are different object types. "
            "An inventory Item must be placed in the world to become a Block. "
            "bot.placeBlock(referenceBlock, faceVector) is a low-level API whose first argument "
            "must be a Block object, not an inventory Item — passing an Item causes TypeError."
        ),
        keywords=["crafting_table", "item", "block", "inventory", "placed", "placeblock",
                  "typeof", "typeerror", "reference"],
        conditions=["crafting table needed but only in inventory"],
        logical_implications=[
            "having crafting_table in inventory does not satisfy craftItem's requirement",
            "inventory Item must be placed as world Block first",
            "bot.placeBlock expects Block object as first argument, not Item",
        ],
    ),

    # Cross-domain workflow: crafting 3x3 recipes requires placement
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="crafting_table_placement_workflow",
        fact=(
            "3x3 recipes require a placed crafting_table block within 32 blocks of the bot. "
            "craftItem internally searches for a placed crafting_table via bot.findBlock; "
            "if none is found, it throws 'requires a crafting table'. "
            "A crafting_table item in inventory is NOT the same as a placed crafting_table block — "
            "it must be placed in the world first (placeItem places inventory items as world blocks). "
            "The crafting_table itself is a 2x2 recipe (4 planks) and does not require an existing table to craft. "
            "Common 3x3 recipes: all tools (pickaxe, axe, sword, shovel, hoe), furnace, chest, bucket, armor."
        ),
        keywords=["crafting_table", "3x3", "place", "placeitem", "workflow",
                  "requires a crafting table", "no available recipes",
                  "pickaxe", "axe", "sword", "furnace", "chest", "tools"],
        logical_implications=[
            "crafting_table in inventory ≠ placed crafting_table block — must be placed to be usable",
            "a crafting_table item must be placed as a world block before craftItem can use it",
            "after placing, craftItem auto-finds and navigates to the placed table",
            "crafting_table itself is a 2x2 recipe (4 planks) — can be crafted without a table",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="multi_step_material_budget",
        fact=(
            "When a crafting workflow has multiple steps sharing the same base material, "
            "the TOTAL material budget must be computed BEFORE any crafting begins. "
            "Each intermediate craft consumes materials unavailable for later steps. "
            "Example: wooden_pickaxe from logs requires: "
            "3 planks (pickaxe) + 2 planks (sticks) + 4 planks (crafting_table if needed) "
            "= 9 planks = 3 logs. "
            "If code crafts the crafting_table first (consuming 4 planks) without reserving "
            "for sticks and the recipe, the final craftItem call fails with "
            "'insufficient materials'. "
            "Failing to account for the total budget before starting intermediate crafts "
            "can cause later steps to fail with 'insufficient materials'."
        ),
        keywords=["insufficient_materials", "budget", "planks", "total",
                  "crafting_table", "sticks", "wooden_pickaxe", "wooden_axe",
                  "not enough", "ran out"],
        conditions=["multi-step crafting with shared base materials",
                    "insufficient materials after intermediate craft"],
        logical_implications=[
            "intermediate crafts consume shared materials, reducing availability for later steps",
            "total budget must account for ALL steps before ANY crafting begins",
            "crafting order matters: early over-consumption causes later failures",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="material_requirements",
        fact=(
            "Crafting fails if insufficient materials in inventory. "
            "Materials must be correct type; similar items don't substitute. "
            "Check exact material names in recipes."
        ),
        keywords=["material", "ingredient", "enough", "missing", "craft"],
        conditions=["crafting with insufficient materials"],
        implications=[
            "need exact materials",
            "gather before crafting",
            "check inventory first",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="delta_vs_total_crafting",
        fact=(
            "When 'have X need Y more': calculate delta = Y - X, craft delta amount. "
            "When 'need Y total': check current amount first. "
            "Common error: crafting Y when already have X."
        ),
        keywords=["delta", "total", "already", "have", "need", "amount"],
        conditions=["calculating craft amounts"],
        implications=[
            "don't over-craft",
            "check inventory first",
            "subtract existing from target",
        ],
    ),
]


# ============================================================
# Water and Fluid Mechanics Knowledge
# ============================================================

WATER_MECHANICS_KNOWLEDGE: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="water_source_vs_flowing",
        fact=(
            "Water has two types: source blocks (level=0) and flowing water (level>0). "
            "Only source blocks can fill buckets; flowing water cannot. "
            "Source blocks are full, static water; flowing water moves from source."
        ),
        keywords=["water", "source", "flowing", "bucket", "level", "fill"],
        conditions=["filling bucket", "collecting water"],
        implications=[
            "bucket fill fails on flowing water",
            "must find source block",
            "check water level property",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="infinite_water_source",
        fact=(
            "2x2 or larger water pool creates infinite water source. "
            "Each source block regenerates when adjacent to 2+ sources. "
            "Can infinitely collect water from such pools."
        ),
        keywords=["infinite", "water", "source", "pool", "2x2", "regenerate"],
        conditions=["need unlimited water"],
        implications=[
            "create 2x2 pool for unlimited water",
            "sources regenerate each other",
            "oceans and rivers are infinite sources",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="water_on_lava",
        fact=(
            "Water poured on lava SOURCE block creates obsidian. "
            "Water on flowing lava creates cobblestone. "
            "Lava on water creates stone or cobblestone."
        ),
        keywords=["water", "lava", "obsidian", "cobblestone", "pour", "create"],
        conditions=["creating obsidian", "mixing water and lava"],
        implications=[
            "need lava source for obsidian",
            "flowing lava gives cobblestone",
            "method to get obsidian",
        ],
    ),

    # Lava bucket collection (similar to water)
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="lava_source_vs_flowing",
        fact=(
            "Lava has source blocks (level=0) and flowing lava (level>0), same as water. "
            "Only lava SOURCE blocks can fill buckets; flowing lava cannot be collected. "
            "Unlike water, lava does NOT have infinite source mechanics (removed in 1.3)."
        ),
        keywords=["lava", "source", "flowing", "bucket", "level", "fill", "collect"],
        conditions=["filling lava bucket", "collecting lava"],
        implications=[
            "bucket fill fails on flowing lava",
            "must find lava source block",
            "lava sources are finite (except via dripstone)",
        ],
    ),

    # Renewable lava via dripstone (1.17+)
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="dripstone_lava_farming",
        fact=(
            "Pointed dripstone under lava source slowly fills cauldron with lava (1.17+). "
            "Place lava source above dripstone, cauldron below. "
            "Only renewable lava source in the game. Takes ~15 minutes per bucket."
        ),
        keywords=["dripstone", "lava", "renewable", "cauldron", "farm", "infinite"],
        conditions=["need renewable lava"],
        implications=[
            "slow but infinite lava",
            "need dripstone from caves",
            "setup: lava -> dripstone -> cauldron",
        ],
    ),

    # --- Environment Physics ---
    # Facts about water movement, drowning, item behavior, mining, pathfinder.
    # Facts describe WHAT IS, never WHAT TO DO.

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="water_movement_speed",
        fact=(
            "Walking on land: 4.317 blocks/s. Sprint on land: 5.612 blocks/s. "
            "Swimming on still water surface: 2.20 blocks/s. "
            "Partially submerged: 1.97 blocks/s. "
            "Swimming downstream underwater: 1.81 blocks/s. "
            "Swimming upstream underwater: 0.39 blocks/s. "
            "Sprint-swimming (fully submerged + sprint key): 3.918 blocks/s."
        ),
        keywords=["water", "swim", "swimming", "speed", "movement", "underwater", "sprint", "slow"],
        logical_implications=[
            "Water reduces horizontal movement to ~50% of land speed",
            "Sprint-swimming is ~9% slower than walking on land",
            "Upstream movement is reduced to ~9% of walking speed",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="swimming_activation",
        fact=(
            "Swimming mode activates when the player is fully submerged in water AND presses sprint key. "
            "In swim mode the player's hitbox becomes horizontal and 1 block tall (vs upright ~1.8 blocks). "
            "Without swim mode, the player wades through water upright at 1.97-2.20 blocks/s."
        ),
        keywords=["swim", "swimming", "mode", "submerge", "sprint", "hitbox", "horizontal"],
        logical_implications=[
            "Swimming mode requires both full submersion and sprint input",
            "Swim hitbox fits through 1-block gaps that standing hitbox cannot",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="water_block_placement",
        fact=(
            "bot.placeBlock(referenceBlock, faceVector) requires referenceBlock to be a solid block. "
            "Water source blocks are NOT solid — cannot serve as reference blocks for placement. "
            "Solid blocks (stone, sand, etc.) CAN be used as reference even when submerged in water — "
            "the new block replaces the adjacent water block. "
            "Non-cube blocks (slabs, stairs, fences, signs, ladders, trapdoors) can be 'waterlogged' — "
            "both the block and a water source occupy the same space."
        ),
        keywords=["water", "place", "placement", "reference", "solid", "block", "waterlog"],
        logical_implications=[
            "Placement near water requires finding a solid block as reference",
            "Placing a block adjacent to a submerged solid block replaces the water",
            "Water source blocks cannot be used as placeBlock reference",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="underwater_mining_speed",
        fact=(
            "Mining with the player's head submerged in water takes 5x as long as on land. "
            "Mining while floating (not standing on a solid block) takes an additional 5x penalty "
            "(25x total if both apply). "
            "Aqua Affinity enchantment (helmet, level 1) removes the 5x underwater penalty. "
            "The floating penalty is separate and NOT removed by Aqua Affinity."
        ),
        keywords=["mine", "mining", "underwater", "slow", "aqua", "affinity", "floating", "speed"],
        logical_implications=[
            "Underwater mining without enchantments is 5x-25x slower",
            "Standing on a solid block halves the total penalty to 5x",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="water_item_buoyancy",
        fact=(
            "Item entities in still water float slowly upward to the surface. "
            "Item entities in flowing water are pushed by the current direction. "
            "Items despawn after 6,000 game ticks (5 minutes) in a loaded chunk. "
            "The despawn timer pauses when the chunk is unloaded."
        ),
        keywords=["item", "water", "float", "collect", "drop", "drift", "despawn", "buoyancy", "surface"],
        logical_implications=[
            "Dropped items underwater will not be at the break position — they float up",
            "collectBlock may fail underwater because items drift away from expected position",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="oxygen_drowning",
        fact=(
            "Player has 300 game ticks (15 seconds) of air supply when fully submerged. "
            "Air decreases by 1 tick per game tick while the player's head is submerged. "
            "When air reaches -20 (1 second after depletion), player takes 2 HP (1 heart) drowning damage. "
            "Drowning damage repeats every 20 ticks (1 second) after that. "
            "Air recovers at 1 bubble per 4 game ticks (0.2 seconds) when the player exits water."
        ),
        keywords=["oxygen", "air", "drown", "drowning", "breath", "submerge", "damage", "health"],
        logical_implications=[
            "Player can work underwater for at most 15 seconds before taking damage",
            "Extended underwater operations require periodic surfacing or Respiration enchantment",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="depth_strider_enchantment",
        fact=(
            "Depth Strider enchantment (boots, max level 3): each level removes 1/3 of water's "
            "horizontal slowdown. Level 3 removes all water slowdown — swim speed equals walking on land. "
            "Sprint-swimming with Depth Strider III: 5.305 blocks/s. "
            "Also reduces flowing water push proportionally per level. "
            "Only affects horizontal movement; vertical speed is unchanged."
        ),
        keywords=["depth", "strider", "enchant", "boots", "water", "speed", "swim"],
        logical_implications=[
            "Without Depth Strider, water reduces horizontal speed by ~50%",
            "Depth Strider III fully negates water speed penalty",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="respiration_enchantment",
        fact=(
            "Respiration enchantment (helmet): adds +15 seconds of air per level (max level 3 = 60s total). "
            "Also grants level/(level+1) chance to skip drowning damage per tick "
            "(Level 1: 50%, Level 2: 67%, Level 3: 75%)."
        ),
        keywords=["respiration", "enchant", "helmet", "air", "breath", "underwater", "drown"],
        logical_implications=[
            "Respiration III extends total air to 60 seconds and reduces drowning damage by 75%",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="pathfinder_water_behavior",
        fact=(
            "mineflayer-pathfinder has a 'liquidCost' parameter: additional cost penalty for paths through "
            "liquid blocks. mineflayer-pathfinder does NOT implement sprint-swimming — it uses standard "
            "movement in water. The pathfinder avoids breaking blocks that touch liquid blocks (prevents "
            "flooding). Bots can get stuck in water when attempting to place blocks under themselves while "
            "swimming in shallow water. GoalBlock targeting underwater positions may produce paths the bot "
            "cannot follow, resulting in 'Path was stopped' or timeout errors."
        ),
        keywords=["pathfinder", "water", "liquid", "cost", "swim", "stuck", "goal", "path", "timeout"],
        conditions=["pathfinding near water", "navigating underwater"],
        logical_implications=[
            "Pathfinding through water is more expensive and less reliable than on land",
            "GoalBlock for underwater targets may produce unfollowable paths",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="kelp_mechanics",
        fact=(
            "Kelp is a multi-block plant: kelp_plant (stem blocks) + kelp (top growth block). "
            "Breaking any segment drops the kelp item, which floats upward in water due to buoyancy."
        ),
        keywords=["kelp", "kelp_plant", "ocean", "underwater", "plant", "break", "float"],
        logical_implications=[
            "Kelp items float upward after breaking — cannot be collected at break position",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="bubble_columns",
        fact=(
            "Soul sand placed under a column of water source blocks creates an upward bubble column "
            "(~11 blocks/s upward). Magma blocks under water source blocks create a downward bubble column "
            "(~4.9 blocks/s downward). Bubble columns only propagate through water source blocks — they stop "
            "at flowing water. Players in a bubble column replenish air supply as if they left the water."
        ),
        keywords=["bubble", "column", "soul", "sand", "magma", "upward", "downward", "water"],
        logical_implications=[
            "Bubble columns require water source blocks (not flowing) to propagate",
            "Upward bubble columns can restore air supply without surfacing",
        ],
    ),
]


# ============================================================
# Combat Mechanics Knowledge
# ============================================================

COMBAT_KNOWLEDGE: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="attack_range",
        fact=(
            "Melee attack range is 3 blocks. Entity must be within range to attack. "
            "Bow/crossbow has unlimited range but requires line of sight."
        ),
        keywords=["attack", "range", "melee", "distance", "combat"],
        conditions=["attacking entity"],
        implications=[
            "must be close for melee",
            "move toward target first",
            "ranged weapons for distance",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="attack_cooldown",
        fact=(
            "Swords have 0.625 second cooldown for full damage. "
            "Axes have 1 second cooldown. "
            "Attacking before cooldown resets deals reduced damage."
        ),
        keywords=["cooldown", "attack", "damage", "sword", "axe", "speed"],
        conditions=["combat timing"],
        implications=[
            "rapid clicking reduces damage",
            "wait for full cooldown",
            "swords faster than axes",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="entity_despawn",
        fact=(
            "Hostile mobs despawn when player is >128 blocks away. "
            "Passive mobs (animals) don't despawn. "
            "Named mobs never despawn."
        ),
        keywords=["despawn", "mob", "entity", "distance", "disappear"],
        conditions=["entity not found", "mob disappeared"],
        implications=[
            "stay close to target",
            "mob may have despawned",
            "named mobs persist",
        ],
    ),
]


# ============================================================
# Inventory Mechanics Knowledge
# ============================================================

INVENTORY_KNOWLEDGE: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="inventory_capacity",
        fact=(
            "Player inventory has 36 slots (27 main + 9 hotbar). "
            "Most items stack to 64; some (ender pearls, snowballs) stack to 16; "
            "tools, armor, and potions don't stack."
        ),
        keywords=["inventory", "full", "slots", "stack", "capacity", "36"],
        conditions=["inventory management", "collecting items"],
        implications=[
            "36 slots total",
            "unstackable items fill quickly",
            "may need to drop or store items",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="item_collection",
        fact=(
            "Items dropped on ground are auto-collected when player walks within 1 block. "
            "Dropped items despawn after 5 minutes (300 seconds). "
            "Collection fails if inventory is full."
        ),
        keywords=["collect", "pickup", "drop", "despawn", "ground", "item"],
        conditions=["collecting dropped items"],
        implications=[
            "walk over items to collect",
            "hurry before despawn",
            "make room in inventory",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="inventory_full_mining",
        fact=(
            "When inventory is full, mined items drop on ground instead of being collected. "
            "mineBlock/collectBlock plugin may fail with 'NoChests' error when inventory full. "
            "Use bot.dig() as fallback when collection fails."
        ),
        keywords=["inventory", "full", "mining", "nochests", "collect", "drop"],
        conditions=["mining with full inventory"],
        implications=[
            "items drop on ground",
            "mineBlock may fail",
            "bot.dig() as alternative",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="chest_storage",
        fact=(
            "Single chest has 27 slots, double chest has 54 slots. "
            "Chests must be placed with air above to open. "
            "Items in chests persist across sessions."
        ),
        keywords=["chest", "storage", "store", "slots", "container"],
        conditions=["storing items", "using chests"],
        implications=[
            "good for long-term storage",
            "need space above chest",
            "items are safe in chests",
        ],
    ),
]


# ============================================================
# Special Resource Acquisition Knowledge
# ============================================================

SPECIAL_ACQUISITION_KNOWLEDGE: List[KnowledgeItem] = [
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="blaze_rod_acquisition",
        fact=(
            "Blaze rods only drop from Blaze mobs in Nether Fortresses. "
            "Cannot be crafted or found elsewhere. "
            "Required for brewing stands and blaze powder."
        ),
        keywords=["blaze", "rod", "nether", "fortress", "brewing"],
        conditions=["need blaze rods"],
        implications=[
            "must go to Nether",
            "find Nether Fortress",
            "kill Blaze mobs",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="ender_pearl_acquisition",
        fact=(
            "Ender pearls drop from Endermen or can be traded from Piglins. "
            "Endermen spawn at night or in The End dimension. "
            "Required for End portal and teleportation."
        ),
        keywords=["ender", "pearl", "enderman", "piglin", "trade"],
        conditions=["need ender pearls"],
        implications=[
            "hunt Endermen at night",
            "or trade with Piglins",
            "needed for The End",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="nether_portal_creation",
        fact=(
            "Nether portal requires at least 10 obsidian in a 4x5 frame. "
            "Corners are optional (14 obsidian for full frame). "
            "Light with flint and steel to activate."
        ),
        keywords=["nether", "portal", "obsidian", "flint", "steel"],
        conditions=["building nether portal"],
        implications=[
            "need 10+ obsidian",
            "need flint and steel",
            "diamond pickaxe to mine obsidian",
        ],
    ),
]


# ============================================================
# Dimension-Specific Mechanics Knowledge
# ============================================================

DIMENSION_KNOWLEDGE: List[KnowledgeItem] = [
    # Bed explosion in Nether/End
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="bed_explosion_nether_end",
        fact=(
            "Beds EXPLODE when used in the Nether or The End (more powerful than TNT). "
            "This is because beds set spawn points, which is not allowed in these dimensions. "
            "The explosion causes fire and destroys nearby blocks."
        ),
        keywords=["bed", "explode", "explosion", "nether", "end", "sleep", "spawn"],
        conditions=["trying to sleep in nether", "trying to sleep in end"],
        implications=[
            "DO NOT use beds in Nether/End",
            "can be used as weapon against mobs",
            "respawn anchor for Nether instead",
        ],
    ),

    # Respawn anchor in Overworld/End
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="respawn_anchor_explosion",
        fact=(
            "Respawn anchor EXPLODES when used in Overworld or The End. "
            "It only works in the Nether (opposite of beds). "
            "Crafted with 6 crying obsidian and 3 glowstone. Charge with glowstone blocks."
        ),
        keywords=["respawn", "anchor", "explode", "nether", "overworld", "end", "glowstone"],
        conditions=["using respawn anchor outside nether"],
        implications=[
            "only use in Nether",
            "explodes in Overworld/End",
            "needs glowstone to charge",
        ],
    ),

    # Crying obsidian portal limitation
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="crying_obsidian_portal",
        fact=(
            "Crying obsidian CANNOT be used to build Nether portals. "
            "It looks similar to obsidian but has purple particles and light. "
            "Only regular obsidian works for portal frames."
        ),
        keywords=["crying", "obsidian", "portal", "nether", "frame", "cannot"],
        conditions=["building nether portal", "using crying obsidian"],
        implications=[
            "crying obsidian is NOT for portals",
            "use regular obsidian only",
            "crying obsidian is for respawn anchors",
        ],
    ),
]


# ============================================================
# Farming Mechanics Knowledge
# ============================================================

FARMING_KNOWLEDGE: List[KnowledgeItem] = [
    # Farmland hydration
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="farmland_hydration",
        fact=(
            "Farmland must be within 4 blocks of water (horizontally) to stay hydrated. "
            "Dry farmland eventually reverts to dirt. "
            "Water source can be same Y level or 1 block above farmland."
        ),
        keywords=["farmland", "water", "hydrate", "irrigate", "crop", "farm", "4 blocks"],
        conditions=["farming crops", "farmland turning to dirt"],
        implications=[
            "place water within 4 blocks",
            "one water block hydrates 9x9 area",
            "dry farmland becomes dirt",
        ],
    ),

    # Farmland creation
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="farmland_creation",
        fact=(
            "Farmland is created by using hoe on dirt or grass blocks. "
            "Jumping on farmland turns it back to dirt (trampling). "
            "Mobs walking on farmland can also trample it."
        ),
        keywords=["farmland", "hoe", "create", "dirt", "trample", "jump"],
        conditions=["creating farmland", "farmland trampled"],
        implications=[
            "use hoe on dirt",
            "don't jump on farmland",
            "fence off from mobs",
        ],
    ),

    # Crop light requirements
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="crop_light_requirement",
        fact=(
            "Crops require light level 9+ to grow. "
            "Crops uproot if light level drops below 8. "
            "Torches (level 14) or glowstone (level 15) provide sufficient light."
        ),
        keywords=["crop", "light", "grow", "torch", "wheat", "carrot", "potato"],
        conditions=["crops not growing", "underground farming"],
        implications=[
            "add torches for underground farms",
            "light level 9 minimum",
            "plants uproot in darkness",
        ],
    ),
]


# ============================================================
# Mob Spawning Mechanics Knowledge
# ============================================================

SPAWNING_KNOWLEDGE: List[KnowledgeItem] = [
    # Light level for hostile mob spawning (1.18+ changes)
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="hostile_mob_spawning_light",
        fact=(
            "Hostile mobs spawn ONLY at light level 0 (1.18+). "
            "Before 1.18, they spawned at light level 7 or below. "
            "Single torch prevents spawning in a large area."
        ),
        keywords=["spawn", "mob", "light", "level", "hostile", "torch", "dark"],
        conditions=["preventing mob spawns", "mob farm design"],
        implications=[
            "light level 0 = complete darkness",
            "one torch covers large area",
            "caves need minimal lighting now",
        ],
    ),

    # Mob spawning conditions
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="mob_spawning_conditions",
        fact=(
            "Mobs spawn on solid blocks with 2+ air blocks above. "
            "Spawning requires player to be 24-128 blocks away. "
            "Mobs don't spawn on transparent blocks (glass, slabs, leaves)."
        ),
        keywords=["spawn", "mob", "block", "distance", "solid", "transparent"],
        conditions=["mob spawning mechanics"],
        implications=[
            "24+ blocks from player",
            "solid opaque floor needed",
            "2 blocks height required",
        ],
    ),

    # Passive mob spawning
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="passive_mob_spawning",
        fact=(
            "Passive mobs (animals) spawn on grass blocks with light level 9+. "
            "They spawn during world generation and rarely afterwards. "
            "Most biomes have specific animal types (cows, pigs, sheep, chickens)."
        ),
        keywords=["passive", "animal", "spawn", "grass", "cow", "pig", "sheep", "chicken"],
        conditions=["finding animals", "animal farm"],
        implications=[
            "need grass blocks",
            "breed existing animals",
            "explore to find animals",
        ],
    ),
]


# ============================================================
# Physics / Falling Blocks Knowledge
# ============================================================

PHYSICS_KNOWLEDGE: List[KnowledgeItem] = [
    # Falling blocks
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="falling_blocks",
        fact=(
            "Sand, gravel, anvils, concrete powder, and dragon eggs fall when unsupported. "
            "They become falling entities and drop as items if landing on non-solid blocks. "
            "Falling blocks can suffocate entities they land on."
        ),
        keywords=["sand", "gravel", "fall", "gravity", "anvil", "concrete", "suffocate"],
        conditions=["mining under sand/gravel", "building with gravity blocks"],
        implications=[
            "check for falling blocks when mining upward",
            "can cause suffocation damage",
            "useful for traps and farms",
            "torch trick: place torch, break block above, sand lands on torch and drops",
        ],
    ),

    # Sand/gravel mining hazard
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="sand_gravel_mining_hazard",
        fact=(
            "Mining upward into sand or gravel is dangerous - blocks fall and can suffocate. "
            "Always check what's above before mining upward in caves or underground. "
            "Gravel is common in caves; sand near beaches and deserts."
        ),
        keywords=["sand", "gravel", "mine", "upward", "danger", "suffocate", "cave"],
        conditions=["mining underground", "digging upward"],
        implications=[
            "look up before mining",
            "have escape route",
            "use water to break fall",
        ],
    ),

    # Torch breaking trick
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="torch_breaking_trick",
        fact=(
            "Place torch on ground, break block above - falling sand/gravel lands on torch and breaks into items. "
            "This is the fastest way to clear vertical columns of sand or gravel. "
            "Works because torches are non-solid blocks that break falling blocks into items."
        ),
        keywords=["torch", "sand", "gravel", "break", "trick", "fast", "column"],
        conditions=["clearing sand/gravel columns", "efficient mining"],
        implications=[
            "much faster than mining each block",
            "requires only 1 torch",
            "collect all items at bottom",
        ],
    ),

    # Anvil fall damage
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="anvil_fall_damage",
        fact=(
            "Falling anvils deal damage based on fall distance (2 damage per block, max 40). "
            "Anvils are damaged when they fall (can break after multiple falls). "
            "Can be used as a weapon in PvP or mob traps."
        ),
        keywords=["anvil", "fall", "damage", "weapon", "trap", "break"],
        conditions=["anvil falls on entity", "using anvil as trap"],
        implications=[
            "dangerous trap mechanism",
            "anvil wears out from falling",
            "max 20 hearts damage from high fall",
        ],
    ),
]


# ============================================================
# Door Mechanics Knowledge
# ============================================================

DOOR_KNOWLEDGE: List[KnowledgeItem] = [
    # Iron door vs wooden door
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="iron_door_mechanics",
        fact=(
            "Iron doors REQUIRE redstone signal to open (button, lever, pressure plate). "
            "Right-clicking iron doors does nothing. "
            "Zombies cannot break iron doors (only wooden on hard difficulty)."
        ),
        keywords=["iron", "door", "redstone", "button", "lever", "pressure", "zombie"],
        conditions=["using iron doors", "zombie-proofing"],
        implications=[
            "need redstone component to open",
            "safe from zombies on any difficulty",
            "use button or pressure plate for convenience",
        ],
    ),

    # Wooden door mechanics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="wooden_door_mechanics",
        fact=(
            "Wooden doors open by right-clicking or with redstone. "
            "Zombies can break wooden doors on Hard difficulty. "
            "Villagers can open and close wooden doors."
        ),
        keywords=["wooden", "door", "open", "zombie", "villager", "hard"],
        conditions=["base defense", "villager interaction"],
        implications=[
            "easy to use without redstone",
            "not zombie-proof on hard",
            "villagers use them for AI",
        ],
    ),

    # Trapdoor mechanics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="trapdoor_mechanics",
        fact=(
            "Wooden trapdoors open by right-click; iron trapdoors need redstone. "
            "Open trapdoors act like ladders for climbing. "
            "Mobs see open trapdoors as solid blocks (useful for traps)."
        ),
        keywords=["trapdoor", "iron", "wooden", "climb", "ladder", "mob", "trap"],
        conditions=["vertical movement", "mob trap design"],
        implications=[
            "iron trapdoors need redstone",
            "can climb open trapdoors like ladders",
            "mobs walk into open trapdoors and fall",
        ],
    ),
]


# ============================================================
# Growth Mechanics Knowledge
# ============================================================

GROWTH_KNOWLEDGE: List[KnowledgeItem] = [
    # Bonemeal mechanics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="bonemeal_mechanics",
        fact=(
            "Bonemeal instantly grows crops, saplings, and other plants. "
            "Made from bones (skeleton drops) - 1 bone = 3 bonemeal. "
            "Multiple applications may be needed for full growth (especially saplings)."
        ),
        keywords=["bonemeal", "bone", "grow", "instant", "crop", "sapling", "skeleton"],
        conditions=["speeding up growth", "farming"],
        implications=[
            "skeleton farm provides unlimited bonemeal",
            "saplings may need 2-5 applications",
            "crops usually need 1-3 applications",
        ],
    ),

    # Sapling growth requirements
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="sapling_growth_requirements",
        fact=(
            "Saplings need light level 9+ and space above to grow into trees. "
            "Oak/birch need 5-7 blocks vertical space; large trees need more. "
            "Saplings can grow on dirt, grass, coarse dirt, podzol, or farmland."
        ),
        keywords=["sapling", "tree", "grow", "space", "light", "oak", "birch", "spruce"],
        conditions=["planting trees", "tree farm"],
        implications=[
            "clear blocks above sapling",
            "ensure light level 9+",
            "bonemeal speeds up growth",
        ],
    ),

    # Large tree requirements
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="large_tree_requirements",
        fact=(
            "Large trees (dark oak, 2x2 spruce/jungle) require 4 saplings in 2x2 pattern. "
            "Dark oak ONLY grows as large tree (4 saplings required). "
            "Large jungle trees can be grown from single sapling or 2x2 pattern."
        ),
        keywords=["large", "tree", "dark oak", "spruce", "jungle", "2x2", "sapling"],
        conditions=["growing large trees", "dark oak farming"],
        implications=[
            "dark oak needs 4 saplings",
            "arrange in 2x2 square",
            "apply bonemeal to any of the 4",
        ],
    ),

    # Crop tick mechanics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="crop_tick_mechanics",
        fact=(
            "Crops grow through random ticks (average 20 minutes from plant to harvest). "
            "Growth speed affected by: hydration, light, surrounding crops. "
            "Alternate crop rows grow faster than full fields of same crop."
        ),
        keywords=["crop", "tick", "grow", "time", "speed", "hydrate", "alternate"],
        conditions=["optimizing farm", "crop growth time"],
        implications=[
            "hydrated farmland grows crops faster",
            "alternate crop rows for speed",
            "bonemeal bypasses tick requirements",
        ],
    ),
]


# ============================================================
# Hunger / Food System Knowledge
# ============================================================

HUNGER_KNOWLEDGE: List[KnowledgeItem] = [
    # Hunger basics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="hunger_basics",
        fact=(
            "Hunger bar (10 drumsticks) depletes from actions like sprinting, jumping, mining. "
            "Health regenerates when hunger is 18+ (9 drumsticks). "
            "At 0 hunger: can't sprint, take starvation damage (1 heart every 4 seconds)."
        ),
        keywords=["hunger", "food", "drumstick", "starve", "regenerate", "sprint"],
        conditions=["managing hunger", "health regeneration"],
        implications=[
            "keep food available",
            "eat when below 9 drumsticks",
            "starvation kills on Hard difficulty",
        ],
    ),

    # Saturation mechanics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="saturation_mechanics",
        fact=(
            "Saturation is hidden hunger value that depletes before visible hunger bar. "
            "High saturation foods (steak, golden carrots) keep you full longer. "
            "Low saturation foods (cookies, melon) require frequent eating."
        ),
        keywords=["saturation", "food", "steak", "golden", "carrot", "full", "hungry"],
        conditions=["choosing food type", "efficient eating"],
        implications=[
            "steak/porkchop are most efficient",
            "golden carrots best saturation",
            "bread is good early-game balance",
        ],
    ),

    # Food healing
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="food_healing_values",
        fact=(
            "Cooked meat restores most hunger: steak/porkchop (8), chicken (6). "
            "Bread restores 5 hunger, baked potato restores 5. "
            "Raw meat restores less and raw chicken can cause food poisoning."
        ),
        keywords=["food", "heal", "steak", "porkchop", "bread", "chicken", "poison"],
        conditions=["choosing food", "cooking meat"],
        implications=[
            "always cook meat for more hunger",
            "avoid raw chicken (30% poison chance)",
            "golden apples provide regeneration effect",
        ],
    ),

    # Golden apple effects
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="golden_apple_effects",
        fact=(
            "Golden apple gives Absorption (2 extra hearts) and Regeneration II for 5 seconds. "
            "Crafted with 8 gold ingots + apple. "
            "Enchanted golden apple (uncraftable, dungeon loot) gives much stronger effects."
        ),
        keywords=["golden", "apple", "absorption", "regeneration", "heal", "gold"],
        conditions=["combat preparation", "emergency healing"],
        implications=[
            "use in dangerous situations",
            "expensive to craft (8 gold ingots)",
            "enchanted version is rare loot only",
        ],
    ),

    # Suspicious stew
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="suspicious_stew_effects",
        fact=(
            "Suspicious stew gives different effects based on flower used in crafting. "
            "Effects: regeneration (oxeye daisy), saturation (dandelion), night vision (poppy). "
            "Can be found in shipwrecks or crafted with mushroom, bowl, and flower."
        ),
        keywords=["suspicious", "stew", "flower", "effect", "mushroom", "saturation"],
        conditions=["special food effects", "early game buffs"],
        implications=[
            "dandelion stew gives great saturation",
            "useful for early game with limited food",
            "effects vary by flower type",
        ],
    ),
]


# ============================================================
# Enchanting System Knowledge
# ============================================================

ENCHANTING_KNOWLEDGE: List[KnowledgeItem] = [
    # Enchanting table basics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="enchanting_table_basics",
        fact=(
            "Enchanting table requires lapis lazuli and experience levels to enchant items. "
            "Without bookshelves, maximum enchantment level is 8. "
            "Enchanting table is crafted with 4 obsidian, 2 diamonds, and 1 book."
        ),
        keywords=["enchant", "table", "lapis", "experience", "level", "xp"],
        conditions=["enchanting items"],
        implications=[
            "need lapis lazuli (1-3 per enchant)",
            "need experience levels",
            "bookshelves increase max level",
        ],
    ),

    # Bookshelf requirement for max enchants
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="bookshelf_enchanting_requirement",
        fact=(
            "15 bookshelves around enchanting table unlock level 30 enchants. "
            "Bookshelves must be 1 block away with air between. "
            "Each bookshelf adds 2 levels (max 30 at 15 bookshelves)."
        ),
        keywords=["bookshelf", "enchant", "level", "30", "15", "setup"],
        conditions=["maximizing enchantments"],
        implications=[
            "need 15 bookshelves for max level",
            "arrange in square with 1 block gap",
            "air or torches between table and shelves",
        ],
    ),

    # Common enchantments for tools
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="tool_enchantments",
        fact=(
            "Efficiency (I-V) increases mining speed. Fortune (I-III) increases ore drops. "
            "Silk Touch lets you collect blocks as-is (glass, grass, ice). "
            "Unbreaking (I-III) increases durability. Fortune and Silk Touch are mutually exclusive."
        ),
        keywords=["efficiency", "fortune", "silk touch", "unbreaking", "tool", "pickaxe"],
        conditions=["enchanting tools"],
        implications=[
            "fortune III triples diamond drops on average",
            "silk touch for glass and spawners (no mob spawner)",
            "unbreaking III roughly triples tool life",
        ],
    ),

    # Common enchantments for weapons
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="weapon_enchantments",
        fact=(
            "Sharpness (I-V) adds damage to all mobs. Smite (I-V) extra damage to undead. "
            "Bane of Arthropods (I-V) extra damage to spiders. Fire Aspect sets mobs on fire. "
            "Looting (I-III) increases mob drops. Knockback pushes mobs back."
        ),
        keywords=["sharpness", "smite", "looting", "fire aspect", "sword", "weapon"],
        conditions=["enchanting weapons"],
        implications=[
            "sharpness is most versatile",
            "smite best for wither and zombies",
            "looting increases rare drops",
        ],
    ),

    # Common enchantments for armor
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="armor_enchantments",
        fact=(
            "Protection (I-IV) reduces all damage. Fire/Blast/Projectile Protection for specific damage. "
            "Feather Falling (I-IV) reduces fall damage (boots only). "
            "Thorns (I-III) reflects damage to attackers. Mending repairs with XP."
        ),
        keywords=["protection", "feather falling", "thorns", "mending", "armor"],
        conditions=["enchanting armor"],
        implications=[
            "protection IV is most versatile",
            "feather falling essential for exploration",
            "thorns damages your armor faster",
        ],
    ),

    # Mending enchantment
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="mending_enchantment",
        fact=(
            "Mending repairs items using collected experience orbs. "
            "Cannot be obtained from enchanting table - only treasure (fishing, trading, loot). "
            "Mending and Infinity are mutually exclusive on bows."
        ),
        keywords=["mending", "repair", "experience", "xp", "treasure", "villager"],
        conditions=["obtaining mending", "repairing items"],
        implications=[
            "trade with librarian villagers",
            "find in dungeon chests",
            "makes items nearly permanent",
        ],
    ),

    # Anvil combining and repair
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="anvil_mechanics",
        fact=(
            "Anvil combines enchanted items and repairs with materials. "
            "Repair cost increases each time (prior work penalty). "
            "Too expensive error at 40+ levels (item becomes unrepairable via anvil)."
        ),
        keywords=["anvil", "combine", "repair", "too expensive", "level", "cost"],
        conditions=["combining enchantments", "repairing items"],
        implications=[
            "plan enchant order to minimize cost",
            "mending bypasses anvil limits",
            "anvil damages with use",
        ],
    ),
]


# ============================================================
# Shield Mechanics Knowledge
# ============================================================

SHIELD_KNOWLEDGE: List[KnowledgeItem] = [
    # Shield basics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="shield_basics",
        fact=(
            "Shield blocks all frontal melee and projectile attacks when raised (right-click). "
            "Crafted with 6 planks and 1 iron ingot. "
            "Blocking reduces movement speed significantly."
        ),
        keywords=["shield", "block", "attack", "defend", "arrow", "melee"],
        conditions=["defending against attacks"],
        implications=[
            "essential for skeleton fights",
            "blocks creeper explosion damage",
            "movement slowed while blocking",
        ],
    ),

    # Shield vs specific attacks
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="shield_attack_interactions",
        fact=(
            "Shield blocks arrows, tridents, and melee attacks completely. "
            "Axe attacks disable shield for 5 seconds (shield stun). "
            "Warden's sonic boom attack bypasses shields entirely."
        ),
        keywords=["shield", "axe", "stun", "disable", "warden", "sonic", "bypass"],
        conditions=["combat with shield"],
        implications=[
            "vulnerable after axe hit",
            "useless against Warden",
            "blocks creeper and TNT damage",
        ],
    ),

    # Shield durability
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="shield_durability",
        fact=(
            "Shield has 336 durability, reduced when blocking attacks. "
            "Explosion damage reduces durability significantly. "
            "Can be repaired with planks on anvil or with Mending enchantment."
        ),
        keywords=["shield", "durability", "repair", "plank", "mending", "break"],
        conditions=["maintaining shield"],
        implications=[
            "blocking explosions wears shield fast",
            "keep spare shield or repair materials",
            "mending makes shield last indefinitely",
        ],
    ),
]


# ============================================================
# Tool Durability Knowledge
# ============================================================

DURABILITY_KNOWLEDGE: List[KnowledgeItem] = [
    # Tool durability values
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="tool_durability_values",
        fact=(
            "Tool durability: wood (59), stone (131), iron (250), diamond (1561), netherite (2031). "
            "Each block mined or mob hit reduces durability by 1. "
            "Tools break when durability reaches 0."
        ),
        keywords=["durability", "tool", "diamond", "iron", "stone", "wood", "netherite"],
        conditions=["tool selection", "resource planning"],
        implications=[
            "diamond lasts 6x longer than iron",
            "netherite is most durable",
            "bring backup tools for long trips",
        ],
    ),

    # Armor durability
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="armor_durability_values",
        fact=(
            "Armor durability varies by piece: helmet (lowest), chestplate (highest). "
            "Diamond armor has ~4x durability of iron. "
            "Durability reduced when taking damage."
        ),
        keywords=["armor", "durability", "helmet", "chestplate", "diamond", "iron"],
        conditions=["armor management"],
        implications=[
            "chestplate protects most, lasts longest",
            "helmet wears out faster",
            "always repair before breaking",
        ],
    ),

    # Unbreaking enchantment effect
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="unbreaking_effect",
        fact=(
            "Unbreaking I/II/III gives 50%/67%/75% chance to not consume durability. "
            "Effectively doubles/triples/quadruples tool lifespan. "
            "Essential enchantment for expensive tools (diamond, netherite)."
        ),
        keywords=["unbreaking", "durability", "enchant", "lifespan", "tool"],
        conditions=["extending tool life"],
        implications=[
            "unbreaking III is ~4x durability",
            "combine with mending for permanent tools",
            "prioritize on diamond/netherite gear",
        ],
    ),

    # Repair methods
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="tool_repair_methods",
        fact=(
            "Anvil repair: combine two items or item + material (diamonds for diamond tools). "
            "Grindstone: combines two items, removes enchantments, returns some XP. "
            "Mending: automatically repairs when picking up XP orbs."
        ),
        keywords=["repair", "anvil", "grindstone", "mending", "xp", "material"],
        conditions=["repairing tools"],
        implications=[
            "anvil preserves enchantments",
            "grindstone for unenchanted items",
            "mending is best long-term solution",
        ],
    ),
]


# ============================================================
# Day/Night Cycle Knowledge
# ============================================================

TIME_KNOWLEDGE: List[KnowledgeItem] = [
    # Day/night cycle basics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="day_night_cycle",
        fact=(
            "Full day/night cycle is 20 minutes real time. "
            "Daytime lasts 10 minutes, night lasts 10 minutes. "
            "Hostile mobs spawn at night (light level 0) and burn in sunlight."
        ),
        keywords=["day", "night", "cycle", "time", "sun", "moon", "20 minutes"],
        conditions=["time management", "mob spawning"],
        implications=[
            "10 min of safe daytime",
            "sleep to skip night",
            "undead burn at dawn",
        ],
    ),

    # Sleeping mechanics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="sleeping_mechanics",
        fact=(
            "Sleeping in bed skips night and thunderstorms. "
            "Can only sleep during night or thunderstorms. "
            "Sets spawn point at bed location. All players must sleep in multiplayer."
        ),
        keywords=["sleep", "bed", "night", "skip", "spawn", "thunderstorm"],
        conditions=["skipping night", "setting spawn"],
        implications=[
            "must wait for night to sleep",
            "bed sets respawn point",
            "phantoms spawn if not sleeping for 3+ days",
        ],
    ),

    # Phantom spawning from insomnia
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="phantom_spawning",
        fact=(
            "Phantoms spawn if player hasn't slept for 3+ in-game days. "
            "They attack at night and swoop down from above. "
            "Sleeping resets insomnia timer. Phantoms drop phantom membrane."
        ),
        keywords=["phantom", "insomnia", "sleep", "3 days", "membrane", "swoop"],
        conditions=["avoiding phantoms", "getting phantom membrane"],
        implications=[
            "sleep at least every 3 days",
            "phantoms are annoying at night",
            "membrane repairs elytra and makes slow falling potions",
        ],
    ),

    # Weather effects
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="weather_effects",
        fact=(
            "Rain lowers light level, allows fishing in any water body. "
            "Endermen take damage in rain and teleport away. "
            "Thunderstorms allow sleeping during day and spawn charged creepers (lightning)."
        ),
        keywords=["rain", "weather", "thunder", "lightning", "storm", "enderman"],
        conditions=["weather effects", "fishing"],
        implications=[
            "rain makes endermen flee",
            "thunderstorms dangerous (lightning)",
            "charged creepers from lightning",
        ],
    ),
]


# ============================================================
# Brewing System Knowledge
# ============================================================

BREWING_KNOWLEDGE: List[KnowledgeItem] = [
    # Brewing stand basics
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="brewing_stand_basics",
        fact=(
            "Brewing stand requires blaze powder as fuel (20 brews per powder). "
            "Base potion: water bottle + nether wart = awkward potion. "
            "Most potions start from awkward potion, not water bottle."
        ),
        keywords=["brewing", "stand", "blaze", "powder", "nether wart", "awkward"],
        conditions=["brewing potions"],
        implications=[
            "need blaze rods for blaze powder",
            "need nether wart from nether fortress",
            "always start with awkward potion",
        ],
    ),

    # Essential potions
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="essential_potions",
        fact=(
            "Healing: awkward + glistering melon. Strength: awkward + blaze powder. "
            "Fire Resistance: awkward + magma cream. Night Vision: awkward + golden carrot. "
            "Slow Falling: awkward + phantom membrane."
        ),
        keywords=["potion", "healing", "strength", "fire resistance", "night vision"],
        conditions=["brewing useful potions"],
        implications=[
            "fire resistance essential for nether",
            "healing potions for combat",
            "slow falling for end exploration",
        ],
    ),

    # Potion modifiers
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="potion_modifiers",
        fact=(
            "Redstone extends duration. Glowstone increases potency (level II). "
            "Gunpowder makes splash potion (throwable). Dragon's breath makes lingering potion. "
            "Fermented spider eye corrupts potions (healing → harming)."
        ),
        keywords=["redstone", "glowstone", "gunpowder", "splash", "lingering", "modifier"],
        conditions=["modifying potions"],
        implications=[
            "redstone vs glowstone mutually exclusive",
            "splash potions for combat",
            "fermented spider eye for harmful potions",
        ],
    ),

    # Nether ingredients location
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="brewing_nether_ingredients",
        fact=(
            "Nether wart: found in nether fortress soul sand gardens. "
            "Blaze rods: dropped by blazes in nether fortress. "
            "Magma cream: dropped by magma cubes or crafted (slime ball + blaze powder)."
        ),
        keywords=["nether wart", "blaze rod", "magma cream", "fortress", "ingredient"],
        conditions=["gathering brewing ingredients"],
        implications=[
            "nether fortress is essential",
            "farm nether wart on soul sand",
            "blaze spawner provides unlimited blaze rods",
        ],
    ),
]


# ============================================================
# Aggregated Knowledge
# ============================================================

MECHANICS_KNOWLEDGE: List[KnowledgeItem] = (
    PLACEMENT_KNOWLEDGE +
    PATHFINDING_KNOWLEDGE +
    CRAFTING_KNOWLEDGE +
    WATER_MECHANICS_KNOWLEDGE +
    COMBAT_KNOWLEDGE +
    INVENTORY_KNOWLEDGE +
    SPECIAL_ACQUISITION_KNOWLEDGE +
    DIMENSION_KNOWLEDGE +
    FARMING_KNOWLEDGE +
    SPAWNING_KNOWLEDGE +
    PHYSICS_KNOWLEDGE +
    DOOR_KNOWLEDGE +
    GROWTH_KNOWLEDGE +
    HUNGER_KNOWLEDGE +
    ENCHANTING_KNOWLEDGE +
    SHIELD_KNOWLEDGE +
    DURABILITY_KNOWLEDGE +
    TIME_KNOWLEDGE +
    BREWING_KNOWLEDGE
)


def get_all_mechanics_knowledge() -> List[KnowledgeItem]:
    """Get all game mechanics knowledge items."""
    return MECHANICS_KNOWLEDGE.copy()


def get_relevant_mechanics_knowledge(error_message: str, skill_code: str = "") -> str:
    """
    Look up relevant game-mechanics knowledge based on an error message.

    lets the LLM derive solutions from pure facts.

    Args:
        error_message: error message
        skill_code: skill code (optional)

    Returns:
        Formatted relevant knowledge text
    """
    combined_text = f"{error_message} {skill_code}".lower()
    relevant_items = []

    for item in MECHANICS_KNOWLEDGE:
        # Check condition match
        condition_match = any(
            cond.lower() in combined_text
            for cond in item.conditions
        )
        # Check keyword match
        keyword_match = any(
            kw.lower() in combined_text
            for kw in item.keywords
        )

        if condition_match or keyword_match:
            relevant_items.append(item)

    if not relevant_items:
        return ""

    lines = ["## Relevant Game Mechanics\n"]
    for item in relevant_items:
        lines.append(f"### {item.name}")
        lines.append(f"**Fact**: {item.fact}")
        if item.implications:
            lines.append("**Implications**:")
            for impl in item.implications:
                lines.append(f"  - {impl}")
        lines.append("")

    return "\n".join(lines)
