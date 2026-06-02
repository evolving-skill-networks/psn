"""
Minecraft Knowledge Retrieval Configuration

Extracted from llm_knowledge_retriever.py to decouple
the generic retrieval framework from Minecraft-specific content.

Contains:
- MINECRAFT_QUERY_PROMPT: LLM prompt for generating knowledge queries
- minecraft_fallback_queries(): Keyword-based fallback when LLM unavailable
"""

from typing import List

from skillnet.agents.optimizer.knowledge.index import KnowledgeQuery


# ============================================================
# LLM Query Generation Prompt
# ============================================================

MINECRAFT_QUERY_PROMPT = """Analyze the following Minecraft skill failure and decide which pieces of game knowledge are needed to diagnose and fix the problem.

## Execution feedback
{feedback_content}

## Game chat log
{chat_log}

## Skill code snippet
```javascript
{skill_code}
```

Based on the above, determine which knowledge domains are relevant. Available knowledge types:
- tool_tier: tool-tier issues (e.g., "need stone_pickaxe to mine X", "wrong tool", "no drops")
- api_behavior: API-usage issues (GoalNear, placeBlock, pathfinder, etc.)
- resource: ore/resource distribution (Y level, biome restrictions)
- game_mechanic: game mechanics (placement, collision, interaction distance, crafting-table requirements)
- primitive: primitive-function knowledge (preconditions and failure modes for craftItem, placeItem, mineBlock, etc.)
- mcdata_naming: mcData item/block-name lookup issues (undefined, wrong item name, registry name)
- reasoning_example: need a reasoning example for reference

Return queries as a JSON array (at most 3):
[
  {{"query_type": "tool_tier", "keywords": ["stone_pickaxe", "deepslate_iron_ore"], "context": "mining deepslate requires stone+ pickaxe"}}
]

Return only the JSON array, no other text. If no additional knowledge is needed, return an empty array []."""


# ============================================================
# Keyword Fallback Queries
# ============================================================

def minecraft_fallback_queries(
    feedback: str,
    chat_log: str,
    skill_code: str,
) -> List[KnowledgeQuery]:
    """
    Minecraft-specific keyword fallback for knowledge retrieval.

    Used when LLM is unavailable or fails to generate queries.
    Returns up to 5 KnowledgeQuery objects based on keyword matching.
    """
    queries: List[KnowledgeQuery] = []
    combined = (feedback + " " + chat_log + " " + skill_code).lower()

    # Tool/mining related
    if "pickaxe" in combined or "mine" in combined or "dig" in combined:
        queries.append(KnowledgeQuery(
            query_type="tool_tier",
            keywords=["pickaxe", "mine", "tool", "tier"],
            context="mining related"
        ))

    # Placement / blockUpdate related
    if "place" in combined or "blockupdate" in combined:
        queries.append(KnowledgeQuery(
            query_type="api_behavior",
            keywords=["place", "block", "goal", "position"],
            context="placement related"
        ))
        queries.append(KnowledgeQuery(
            query_type="primitive",
            keywords=["placeitem", "place", "position", "standing", "target"],
            context="placement position conflict"
        ))

    # Crafting/recipe related
    if "craft" in combined or "recipe" in combined or "3x3" in combined or "crafting_table" in combined:
        queries.append(KnowledgeQuery(
            query_type="primitive",
            keywords=["craftitem", "craft", "crafting_table", "recipe", "3x3", "table"],
            context="crafting related"
        ))
        queries.append(KnowledgeQuery(
            query_type="game_mechanic",
            keywords=["crafting_table", "3x3", "place", "workflow", "placeitem"],
            context="crafting table placement workflow"
        ))

    # Smelting/furnace related
    if "smelt" in combined or "furnace" in combined:
        queries.append(KnowledgeQuery(
            query_type="primitive",
            keywords=["smeltitem", "smelt", "furnace", "fuel"],
            context="smelting related"
        ))

    # Pathfinding related
    if "path" in combined or "goto" in combined or "goal" in combined:
        queries.append(KnowledgeQuery(
            query_type="api_behavior",
            keywords=["pathfinder", "goal", "goto"],
            context="pathfinding related"
        ))

    # Inventory related
    if "inventory" in combined or "slot" in combined or "item" in combined:
        queries.append(KnowledgeQuery(
            query_type="api_behavior",
            keywords=["inventory", "item", "slot", "count"],
            context="inventory related"
        ))

    # Resource/ore related
    # detect specific resource names in context and
    # include them in the query keywords. The original implementation
    # checked for specific names but produced a generic resource query
    # with keywords ["ore", "spawn", "y level", "mine"] only — losing the
    # specific resource name. As a result, searches couldn't hit
    # resource-specific items like `coal_y_level` or `ore_coal_ore`.
    resource_names = [
        "diamond", "iron", "gold", "coal", "copper",
        "lapis", "redstone", "emerald", "quartz", "netherite",
    ]
    detected_resources = [r for r in resource_names if r in combined]
    if detected_resources or "ore" in combined or "y level" in combined or "spawn" in combined:
        query_keywords = ["ore", "spawn", "y level", "mine"]
        # Add specific resource names and their ore/deepslate variants so
        # the index keyword matcher hits resource-specific knowledge items.
        for r in detected_resources:
            query_keywords.append(r)
            query_keywords.append(f"{r}_ore")
            query_keywords.append(f"deepslate_{r}_ore")
            query_keywords.append(f"{r}_y_level")
        queries.append(KnowledgeQuery(
            query_type="resource",
            keywords=query_keywords[:15],  # cap at 15 to reduce noise
            context=f"resource: {','.join(detected_resources) or 'generic'}"
        ))

    # Bedrock / world-boundary related
    if "bedrock" in combined or "y=-6" in combined or "world bottom" in combined or "unbreakable" in combined:
        queries.append(KnowledgeQuery(
            query_type="game_mechanic",
            keywords=["bedrock", "layer", "Y=-64", "Y=-60", "unbreakable"],
            context="world boundary / bedrock layer"
        ))

    # Deepslate ore-variant matching
    if "deepslate" in combined or ("not found" in combined and "ore" in combined):
        queries.append(KnowledgeQuery(
            query_type="resource",
            keywords=["deepslate", "ore", "variant", "findBlock", "matching", "block id"],
            context="deepslate ore variant matching"
        ))

    # null/undefined-access related
    if "cannot read properties" in combined or "undefined" in combined:
        queries.append(KnowledgeQuery(
            query_type="api_behavior",
            keywords=["undefined", "mcData", "itemsByName", "blocksByName", "null", "id"],
            context="undefined property access"
        ))

    # "not a function" errors
    if "not a function" in combined:
        queries.append(KnowledgeQuery(
            query_type="api_behavior",
            keywords=["not a function", "invalid", "emptySlotCount", "inventory"],
            context="calling non-existent API method"
        ))

    # water/bucket interaction
    if any(kw in combined for kw in ["water", "bucket", "fill", "source"]):
        queries.append(KnowledgeQuery(
            query_type="primitive",
            keywords=["bucket", "water", "source", "level", "fill", "activateItem"],
            context="water/bucket interaction"
        ))

    # underwater physics / environment mechanics
    if any(kw in combined for kw in ["underwater", "swim", "drown", "breath", "submerge",
                                      "ocean", "kelp", "float", "buoyancy"]):
        queries.append(KnowledgeQuery(
            query_type="game_mechanic",
            keywords=["water", "underwater", "swim", "drown", "float"],
            context="underwater environment mechanics"
        ))

    return queries[:5]
