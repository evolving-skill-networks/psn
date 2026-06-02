"""
Minecraft Knowledge Index Builder

Builds KnowledgeEntry objects for all Minecraft knowledge sources.
Used by KnowledgeIndex via the knowledge builder registration mechanism.
"""

import logging
from typing import List

from skillnet.agents.optimizer.knowledge.index import KnowledgeEntry

logger = logging.getLogger(__name__)


def build_minecraft_knowledge_entries() -> List[KnowledgeEntry]:
    """Build all Minecraft-specific KnowledgeEntry objects for the index."""
    entries: List[KnowledgeEntry] = []
    _index_api_behaviors(entries)
    _index_game_mechanics(entries)
    _index_resource_knowledge(entries)
    _index_block_knowledge(entries)
    _index_reasoning_examples(entries)
    _index_primitive_knowledge(entries)
    _index_ore_drop_knowledge(entries)
    _index_action_guidance(entries)
    return entries


def _index_api_behaviors(entries: List[KnowledgeEntry]):
    """Index all API knowledge from the api/ directory."""
    try:
        from skillnet.domains.minecraft.knowledge.api import get_all_api_knowledge
        for item in get_all_api_knowledge():
            entries.append(KnowledgeEntry(
                query_type="api_behavior",
                name=item.name,
                keywords=item.keywords,
                content=item.to_prompt_text(),
                priority=5
            ))
    except ImportError as e:
        logger.warning(f"[IndexBuilder] Cannot index API knowledge: {e}")


def _index_game_mechanics(entries: List[KnowledgeEntry]):
    """Index game mechanics knowledge."""
    try:
        from skillnet.domains.minecraft.knowledge.game.mechanics import (
            PLACEMENT_KNOWLEDGE,
            PATHFINDING_KNOWLEDGE,
            CRAFTING_KNOWLEDGE,
            COMBAT_KNOWLEDGE,
            INVENTORY_KNOWLEDGE,
            WATER_MECHANICS_KNOWLEDGE,
        )

        all_mechanics = []
        for knowledge_list in [
            PLACEMENT_KNOWLEDGE,
            PATHFINDING_KNOWLEDGE,
            CRAFTING_KNOWLEDGE,
            COMBAT_KNOWLEDGE,
            INVENTORY_KNOWLEDGE,
            WATER_MECHANICS_KNOWLEDGE,
        ]:
            if knowledge_list:
                all_mechanics.extend(knowledge_list)

        for item in all_mechanics:
            if hasattr(item, 'to_prompt_text'):
                content = item.to_prompt_text()
            elif hasattr(item, 'fact'):
                content = f"**{item.name}**: {item.fact}"
            else:
                content = str(item)

            entries.append(KnowledgeEntry(
                query_type="game_mechanic",
                name=item.name if hasattr(item, 'name') else "unknown",
                keywords=item.keywords if hasattr(item, 'keywords') else [],
                content=content,
                priority=3
            ))

    except ImportError as e:
        logger.warning(f"[IndexBuilder] Cannot index game mechanics: {e}")


def _index_resource_knowledge(entries: List[KnowledgeEntry]):
    """Index resource distribution knowledge."""
    try:
        from skillnet.domains.minecraft.knowledge.game.resources import (
            ORE_SPAWN_RANGES, RESOURCE_KNOWLEDGE,
        )

        for ore_name, ore_info in ORE_SPAWN_RANGES.items():
            content = f"**{ore_name}**: Y range [{ore_info.min_y} to {ore_info.max_y}], best at Y={ore_info.optimal_y}"
            if ore_info.biome_restrictions:
                content += f", only in: {', '.join(ore_info.biome_restrictions)}"

            base_name = ore_name.replace("_ore", "").replace("deepslate_", "")
            entries.append(KnowledgeEntry(
                query_type="resource",
                name=f"ore_{ore_name}",
                keywords=[ore_name, base_name, "ore", "mine", "spawn", "y level"],
                content=content,
                priority=4
            ))

        for item in RESOURCE_KNOWLEDGE:
            if hasattr(item, 'to_prompt_text'):
                content = item.to_prompt_text()
            elif hasattr(item, 'fact'):
                content = f"**{item.name}**: {item.fact}"
            else:
                content = str(item)

            entries.append(KnowledgeEntry(
                query_type="resource",
                name=item.name if hasattr(item, 'name') else "unknown",
                keywords=item.keywords if hasattr(item, 'keywords') else [],
                content=content,
                priority=4
            ))

    except ImportError as e:
        logger.warning(f"[IndexBuilder] Cannot index resource knowledge: {e}")


def _index_block_knowledge(entries: List[KnowledgeEntry]):
    """Index block property knowledge."""
    try:
        from skillnet.domains.minecraft.knowledge.game.blocks import BLOCK_KNOWLEDGE
        for item in BLOCK_KNOWLEDGE:
            entries.append(KnowledgeEntry(
                query_type="game_mechanic",
                name=item.name,
                keywords=item.keywords if hasattr(item, 'keywords') else [],
                content=item.to_prompt_text() if hasattr(item, 'to_prompt_text') else f"**{item.name}**: {item.fact}",
                priority=4
            ))
    except ImportError as e:
        logger.warning(f"[IndexBuilder] Cannot index block knowledge: {e}")


def _index_reasoning_examples(entries: List[KnowledgeEntry]):
    """Index reasoning examples."""
    try:
        from skillnet.domains.minecraft.knowledge.reasoning_examples import (
            ALL_REASONING_EXAMPLES,
        )

        examples = ALL_REASONING_EXAMPLES if ALL_REASONING_EXAMPLES else []

        for example in examples:
            if example and hasattr(example, 'to_prompt_text'):
                entries.append(KnowledgeEntry(
                    query_type="reasoning_example",
                    name=example.scenario if hasattr(example, 'scenario') else "unknown",
                    keywords=example.keywords if hasattr(example, 'keywords') else [],
                    content=example.to_prompt_text(),
                    priority=2
                ))

    except ImportError as e:
        logger.warning(f"[IndexBuilder] Cannot index reasoning examples: {e}")


def _index_primitive_knowledge(entries: List[KnowledgeEntry]):
    """Index primitive function knowledge."""
    try:
        from skillnet.domains.minecraft.knowledge.primitives import (
            get_all_primitive_knowledge,
        )
        for item in get_all_primitive_knowledge():
            entries.append(KnowledgeEntry(
                query_type="primitive",
                name=item.name,
                keywords=item.keywords,
                content=item.to_prompt_text() if hasattr(item, 'to_prompt_text') else f"**{item.name}**: {item.fact}",
                priority=4
            ))
    except ImportError as e:
        logger.warning(f"[IndexBuilder] Cannot index primitive knowledge: {e}")


def _index_action_guidance(entries: List[KnowledgeEntry]):
    """Index action guidance knowledge items for code generation."""
    try:
        from skillnet.domains.minecraft.knowledge.action_guidance import (
            ACTION_GUIDANCE_ITEMS,
        )
        for item in ACTION_GUIDANCE_ITEMS:
            entries.append(KnowledgeEntry(
                query_type="code_generation",
                name=item.name,
                keywords=item.keywords,
                content=item.to_prompt_text(),
                priority=5,
            ))
    except ImportError as e:
        logger.warning(f"[IndexBuilder] Cannot index action guidance: {e}")


def _index_ore_drop_knowledge(entries: List[KnowledgeEntry]):
    """Index ore drop entity knowledge."""
    try:
        from skillnet.domains.minecraft.knowledge.game.ores import ORE_DROP_KNOWLEDGE
        for item in ORE_DROP_KNOWLEDGE:
            entries.append(KnowledgeEntry(
                query_type="game_mechanic",
                name=item.name,
                keywords=item.keywords,
                content=item.to_prompt_text(),
                priority=5
            ))
    except ImportError as e:
        logger.warning(f"[IndexBuilder] Cannot index ore drop knowledge: {e}")
