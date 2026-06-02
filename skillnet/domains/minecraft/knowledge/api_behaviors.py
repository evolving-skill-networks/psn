"""
API Knowledge Facade

Facade functions for knowledge retrieval; the underlying data source is the KnowledgeItem objects in the api/ directory.

Active interfaces:
  - get_relevant_api_knowledge(): keyword-match retrieval of relevant KnowledgeItems
  - get_all_knowledge_for_error(): aggregate retrieval of all relevant knowledge (LLMAnalyzer fallback path)
  - get_ore_knowledge(), get_relevant_resource_knowledge(): domain-specific knowledge retrieval

The actual knowledge data is stored in:
  - api/pathfinder.py: pathfinding API knowledge
  - api/bot_methods.py: bot-method knowledge
  - api/inventory.py: inventory API knowledge
"""

from typing import Dict, List

# import ore data from the new module
from .game.resources import ORE_SPAWN_RANGES, OreSpawnRange, RESOURCE_KNOWLEDGE
from skillnet.core.knowledge_base import KnowledgeItem

# Backward-compatible alias (deprecated)
OreSpawnInfo = OreSpawnRange


def _ore_spawn_to_prompt_text(ore: OreSpawnRange, name: str) -> str:
    """Convert an OreSpawnRange to prompt text"""
    text = f"**{name}**: Y range [{ore.min_y} to {ore.max_y}], best at Y={ore.optimal_y}"
    if ore.biome_restrictions:
        text += f", only in: {', '.join(ore.biome_restrictions)}"
    return text


# ============================================================
# Helper Functions
# ============================================================

def get_relevant_api_knowledge(error_message: str, skill_code: str = "") -> str:
    """
    Return relevant API behavior knowledge given the error message and skill code.

    data source switched from the APIBehavior dict to KnowledgeItem objects in the api/ directory.
    Retains all hand-tuned AND conditions and pattern-matching logic; only the data source changed.
    Output goes through KnowledgeItem.to_prompt_text() with no JS code leaking.

    Args:
        error_message: error message
        skill_code: skill code (optional)

    Returns:
        Formatted API-knowledge text
    """
    from .api import get_all_api_knowledge

    # Build name→KnowledgeItem lookup (replaces the legacy APIBehavior dict.get())
    all_items = {item.name: item for item in get_all_api_knowledge()}
    relevant_items = []
    combined_text = (error_message + " " + skill_code).lower()

    # ===== Retain all existing matching logic =====

    # Detect blockUpdate/placement related issues
    if "blockupdate" in combined_text or "placeblock" in combined_text or "place" in combined_text:
        relevant_items.extend([
            all_items.get("GoalNear"),
            all_items.get("GoalPlaceBlock"),
            all_items.get("bot.placeBlock"),
        ])

    # Detect pathfinding-related issues
    if "path" in combined_text or "goto" in combined_text or "blocked" in combined_text:
        relevant_items.extend([
            all_items.get("GoalNear"),
            all_items.get("GoalBlock"),
            all_items.get("pathfinder_goto"),
        ])

    # Detect mining-related issues
    if "mine" in combined_text or "dig" in combined_text or "ore" in combined_text:
        relevant_items.extend([
            all_items.get("bot.dig"),
            all_items.get("bot.findBlock"),
        ])

    # Detect mineBlock primitive usage - especially the bot.dig vs mineBlock choice
    mineblock_patterns = ["mineblock", "typeof bot.dig", "bot.dig(block)", "log", "wood", "tree"]
    if any(p in combined_text for p in mineblock_patterns):
        relevant_items.append(all_items.get("mineBlock_primitive"))

    # Detect crafting-related issues
    if "craft" in combined_text:
        relevant_items.extend([
            all_items.get("bot.craft"),
            all_items.get("bot.openContainer"),
        ])

    # Detect equip-related issues
    if "equip" in combined_text or "hand" in combined_text:
        relevant_items.append(all_items.get("bot.equip"))

    # Detect inventory-related issues
    if "inventory" in combined_text or "slot" in combined_text or "item" in combined_text:
        relevant_items.extend([
            all_items.get("bot.inventory.items"),
            all_items.get("bot.inventory.count"),
            all_items.get("bot.inventory.findInventoryItem"),
            all_items.get("bot.inventory.emptySlotCount"),
        ])
        # Specifically detect the firstEmptySlot mistake - a common misuse
        if "firstemptyslot" in combined_text or "is not a function" in combined_text:
            relevant_items.append(all_items.get("firstEmptySlot_invalid"))

    # ===== Retain AND-conditions =====

    # Detect tool-tier issues - mining but no drops obtained
    mining_keywords = ["pickaxe", "mining", "diamond", "ore", "obsidian", "dig"]
    no_drop_patterns = ["now have 0", "currently have 0", "no drops", "didn't drop", "obtained 0", "0/"]
    if any(kw in combined_text for kw in mining_keywords):
        if any(p in combined_text for p in no_drop_patterns):
            relevant_items.append(all_items.get("tool_tier_rules"))

    # Detect bucket-fill issues - water source vs flowing water
    bucket_patterns = ["bucket", "water_bucket", "fill bucket", "water bucket"]
    water_fail_patterns = ["failed to fill", "failed to get a water bucket", "could not fill",
                           "no water bucket", "bucket fill failed", "bucket still empty"]
    if any(p in combined_text for p in bucket_patterns):
        relevant_items.append(all_items.get("water_source_vs_flowing"))
        relevant_items.append(all_items.get("bucket_fill_mechanics"))
    # Specifically detect bucket-fill failures
    if any(p in combined_text for p in water_fail_patterns):
        relevant_items.append(all_items.get("water_source_vs_flowing"))
        relevant_items.append(all_items.get("bucket_fill_mechanics"))

    # Detect NoChests/inventory-full issues - mineBlock failing because the inventory is full
    if "nochests" in combined_text or "no defined chest" in combined_text:
        relevant_items.append(all_items.get("collectBlock_NoChests"))

    # Detect inventory-management issues - inventory full or repeated mineBlock failures
    inventory_full_patterns = ["36/36", "inventory full", "no empty slot", "emptyslotcount"]
    mineblock_fail_patterns = ["mineblock failed", "mine attempt failed"]
    if any(p in combined_text for p in inventory_full_patterns):
        relevant_items.append(all_items.get("collectBlock_NoChests"))
    if any(p in combined_text for p in mineblock_fail_patterns):
        relevant_items.append(all_items.get("collectBlock_NoChests"))

    # Deduplicate and filter out None
    seen = set()
    unique_items = []
    for item in relevant_items:
        if item and item.name not in seen:
            seen.add(item.name)
            unique_items.append(item)

    if not unique_items:
        return ""

    lines = ["## Relevant API Behaviors\n"]
    for item in unique_items:
        lines.append(item.to_prompt_text())
        lines.append("")

    return "\n".join(lines)


def get_ore_knowledge(error_message: str) -> str:
    """
    Return relevant ore-spawn knowledge for the error message.

    uses the ORE_SPAWN_RANGES data from game/resources.py.

    Args:
        error_message: error message

    Returns:
        Formatted ore-knowledge text
    """
    error_lower = error_message.lower()

    # Detect specific ore types
    relevant_ores = []  # List of (name, OreSpawnRange) tuples
    for ore_name, ore_info in ORE_SPAWN_RANGES.items():
        # Strip the _ore suffix for matching
        base_name = ore_name.replace("_ore", "").replace("deepslate_", "")
        if base_name in error_lower or ore_name in error_lower:
            relevant_ores.append((ore_name, ore_info))

    # If no specific ore matched but "ore" or "mine" keyword is present, return common ores
    if not relevant_ores and ("ore" in error_lower or "mine" in error_lower):
        common_ores = ["diamond_ore", "iron_ore", "coal_ore", "gold_ore"]
        relevant_ores = [(ore, ORE_SPAWN_RANGES[ore]) for ore in common_ores if ore in ORE_SPAWN_RANGES]

    if not relevant_ores:
        return ""

    lines = ["## Ore Spawn Information\n"]
    for ore_name, ore_info in relevant_ores:
        lines.append(_ore_spawn_to_prompt_text(ore_info, ore_name))
    lines.append("")

    return "\n".join(lines)


# ============================================================
# Resource Knowledge Retrieval
# ============================================================

# Resource association rules: when the error message contains a given keyword, also retrieve related resource knowledge
RESOURCE_ASSOCIATIONS: Dict[str, List[str]] = {
    "obsidian": ["lava_location", "obsidian_creation"],
    "nether_portal": ["obsidian_creation", "lava_location"],
    "nether": ["lava_location"],
    "portal": ["obsidian_creation", "lava_location"],
    "bedrock": ["overworld_vertical_structure"],
    "diamond": ["deepslate_ore_distribution", "diamond_y_level", "ore_vein_density"],
}


def get_relevant_resource_knowledge(error_message: str) -> str:
    """
    retrieve RESOURCE_KNOWLEDGE relevant to the error message.

    Retrieval logic:
    1. Direct keyword match - the error message contains one of the item's keywords
    2. Association match - find related knowledge through RESOURCE_ASSOCIATIONS
    3. Return formatted knowledge text (fact + implications)

    Args:
        error_message: error message (includes feedback content)

    Returns:
        Formatted resource-knowledge text
    """
    if not error_message:
        return ""

    error_lower = error_message.lower()
    relevant_items: List[KnowledgeItem] = []
    seen_names = set()

    # Step 1: direct keyword matching
    for item in RESOURCE_KNOWLEDGE:
        if item.name in seen_names:
            continue

        # Check whether any of the keywords appears in the error message
        for keyword in item.keywords:
            if keyword.lower() in error_lower:
                relevant_items.append(item)
                seen_names.add(item.name)
                break

    # Step 2: association-rule matching
    for trigger, associated_names in RESOURCE_ASSOCIATIONS.items():
        if trigger in error_lower:
            for name in associated_names:
                if name not in seen_names:
                    # Look up the corresponding knowledge item
                    for item in RESOURCE_KNOWLEDGE:
                        if item.name == name:
                            relevant_items.append(item)
                            seen_names.add(name)
                            break

    if not relevant_items:
        return ""

    # Format the output
    lines = ["## Resource Distribution Knowledge", ""]
    for item in relevant_items:
        lines.append(item.to_prompt_text())
        lines.append("")

    return "\n".join(lines)


def get_all_knowledge_for_error(error_message: str, skill_code: str = "") -> str:
    """
    Retrieve all background knowledge relevant to the error.

    adds RESOURCE_KNOWLEDGE retrieval (includes lava_location, etc.)

    Args:
        error_message: error message
        skill_code: skill code (optional)

    Returns:
        Formatted full knowledge text
    """
    parts = []

    api_knowledge = get_relevant_api_knowledge(error_message, skill_code)
    if api_knowledge:
        parts.append(api_knowledge)

    ore_knowledge = get_ore_knowledge(error_message)
    if ore_knowledge:
        parts.append(ore_knowledge)

    # add RESOURCE_KNOWLEDGE retrieval (includes lava_location, obsidian_creation, etc.)
    resource_knowledge = get_relevant_resource_knowledge(error_message)
    if resource_knowledge:
        parts.append(resource_knowledge)

    # add MECHANICS_KNOWLEDGE retrieval (covers placement rules and other game mechanics)
    from .game.mechanics import get_relevant_mechanics_knowledge
    mechanics_knowledge = get_relevant_mechanics_knowledge(error_message, skill_code)
    if mechanics_knowledge:
        parts.append(mechanics_knowledge)

    # add BLOCK_KNOWLEDGE retrieval (block properties: bedrock layer, tool tier, etc.)
    error_lower = error_message.lower()
    try:
        from .game.blocks import get_all_block_knowledge
        block_items = get_all_block_knowledge()
        relevant_blocks = [
            item for item in block_items
            if any(kw.lower() in error_lower for kw in item.keywords)
        ]
        if relevant_blocks:
            block_text = "\n".join(item.to_prompt_text() for item in relevant_blocks)
            parts.append(f"## Block Properties\n{block_text}")
    except ImportError:
        pass

    # add ore-drop-entity knowledge retrieval (raw_iron is an item not a block, etc.)
    try:
        from .game.ores import get_all_ore_drop_knowledge
        ore_drop_items = get_all_ore_drop_knowledge()
        relevant_ore_drops = [
            item for item in ore_drop_items
            if any(kw.lower() in error_lower for kw in item.keywords)
        ]
        if relevant_ore_drops:
            ore_drop_text = "\n".join(item.to_prompt_text() for item in relevant_ore_drops)
            parts.append(f"## Ore Drop Entity Knowledge\n{ore_drop_text}")
    except ImportError:
        pass

    # add PRIMITIVE_KNOWLEDGE retrieval (mineBlock strict matching, craftItem count, etc.)
    try:
        from .primitives.mining import get_all_mining_knowledge
        mining_items = get_all_mining_knowledge()
        relevant_mining = [
            item for item in mining_items
            if any(kw.lower() in error_lower for kw in item.keywords)
        ]
        if relevant_mining:
            mining_text = "\n".join(item.to_prompt_text() for item in relevant_mining)
            parts.append(f"## Primitive Knowledge (Mining)\n{mining_text}")
    except ImportError:
        pass

    # add placement-primitive retrieval (placement failures, entity occupancy, blockUpdate timeouts, etc.)
    try:
        from .primitives.placement import get_all_placement_knowledge
        placement_items = get_all_placement_knowledge()
        relevant_placement = [
            item for item in placement_items
            if any(kw.lower() in error_lower for kw in item.keywords)
        ]
        if relevant_placement:
            placement_text = "\n".join(item.to_prompt_text() for item in relevant_placement)
            parts.append(f"## Primitive Knowledge (Placement)\n{placement_text}")
    except ImportError:
        pass

    if not parts:
        return ""

    return "\n".join(parts)
