"""
Minecraft Environment Knowledge - Minecraft-specific environmental knowledge.

v7.0 refactor: this is a skeleton file kept for backward-compatible interfaces.
The actual knowledge data lives in:
  - game/blocks.py: block properties, tool requirements
  - game/resources.py: ore distribution, biome resources
  - game/mechanics.py: game mechanics

Defines the MinecraftEnvironmentKnowledge dataclass and the
create_minecraft_env_knowledge factory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, TYPE_CHECKING

# Import base classes from env_knowledge
from skillnet.agents.optimizer.phases.env_knowledge import (
    EnvironmentKnowledge,
    EnvironmentRule,
)


# import knowledge data from the new game/ module
from .game.blocks import (
    BLOCK_HARDNESS,
    TOOL_EFFECTIVENESS,
    get_tool_for_block as _get_tool_for_block,
)
from .game.resources import (
    ORE_SPAWN_RANGES,
    BIOME_RESOURCES,
    get_ore_y_range as _get_ore_y_range,
)

if TYPE_CHECKING:
    from .primitive_knowledge import PrimitiveKnowledgeBase


@dataclass
class MinecraftEnvironmentKnowledge(EnvironmentKnowledge):
    """
    Minecraft-specific environmental knowledge.

    Extends EnvironmentKnowledge with knowledge categories specific to Minecraft.
    """

    # Minecraft-specific rules
    mining_rules: List[EnvironmentRule] = field(default_factory=list)
    combat_rules: List[EnvironmentRule] = field(default_factory=list)
    inventory_rules: List[EnvironmentRule] = field(default_factory=list)

    # Block knowledge
    block_hardness: Dict[str, float] = field(default_factory=dict)
    tool_effectiveness: Dict[str, List[str]] = field(default_factory=dict)

    # Biome knowledge
    biome_resources: Dict[str, List[str]] = field(default_factory=dict)

    # Primitive knowledge base (lazy initialization)
    _primitive_knowledge: Optional['PrimitiveKnowledgeBase'] = field(
        default=None, repr=False
    )

    @property
    def primitive_knowledge(self) -> 'PrimitiveKnowledgeBase':
        """Get the primitive knowledge base (lazy-loaded)."""
        if self._primitive_knowledge is None:
            from .primitive_knowledge import create_primitive_knowledge_base
            self._primitive_knowledge = create_primitive_knowledge_base()
        return self._primitive_knowledge

    def find_relevant_rules(self, error_message: str) -> List[EnvironmentRule]:
        """Find environment rules relevant to the error (extended version)."""
        relevant = []

        # First call the parent method
        relevant.extend(super().find_relevant_rules(error_message))

        # Add Minecraft-specific rules
        all_mc_rules = (
            self.mining_rules +
            self.combat_rules +
            self.inventory_rules
        )

        error_lower = error_message.lower()
        for rule in all_mc_rules:
            # Simple keyword matching
            condition_keywords = rule.condition.lower().split()
            if any(kw in error_lower for kw in condition_keywords if len(kw) > 3):
                relevant.append(rule)

        return relevant

    def analyze_primitive_failure(
        self,
        error_message: str,
        skill_code: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze primitive-call failures.

        When the error occurs in primitives such as craftItem or mineBlock,
        provides diagnostic information and fix suggestions.

        Args:
            error_message: Error message
            skill_code: Skill code (used for more precise analysis)

        Returns:
            Dict: Contains primitives, matched_patterns, fix_strategies, etc.
        """
        return self.primitive_knowledge.analyze_error(error_message, skill_code)

    def get_primitive_fix_prompt(
        self,
        error_message: str,
        skill_code: str = "",
    ) -> str:
        """
        Generate a fix prompt for primitive failures.

        Returns a formatted prompt that can be appended to an LLM prompt directly.

        Args:
            error_message: Error message
            skill_code: Skill code

        Returns:
            str: Formatted fix prompt
        """
        return self.primitive_knowledge.generate_fix_prompt(error_message, skill_code)


def create_minecraft_env_knowledge() -> MinecraftEnvironmentKnowledge:
    """
    Create Minecraft-specific environmental knowledge.

    Converts the existing ENVIRONMENT_PATTERNS into EnvironmentKnowledge format.

    Returns:
        MinecraftEnvironmentKnowledge: Configured environment knowledge
    """
    return MinecraftEnvironmentKnowledge(
        # Placement rules
        placement_rules=[
            EnvironmentRule(
                rule_type="placement",
                description="Block placement requires adjacent solid block",
                condition="No valid adjacent replaceable positions, cannot place block, No block to place on, placeItem failed, Failed to place",
                consequence="Placement fails, blockUpdate may timeout",
            ),
            EnvironmentRule(
                rule_type="placement",
                description="Interaction distance limit is 4 blocks",
                condition="too far, cannot reach, out of range",
                consequence="Cannot interact with blocks or entities beyond 4 blocks",
            ),
            EnvironmentRule(
                rule_type="placement",
                description="BlockUpdate timeout indicates position conflict",
                condition="blockUpdate did not fire, blockUpdate timeout",
                consequence="Block was not actually placed despite no immediate error",
            ),
            # Added: precise rules based on diagnostic facts
            EnvironmentRule(
                rule_type="placement",
                description="blockUpdate timeout with bot standing at placement target (positions_equal: true)",
                condition="blockUpdate timeout, blockUpdate did not fire, positions_equal: true, DIAGNOSTIC_FACTS",
                consequence="Bot is standing at the exact position where it wants to place a block. The bot itself is an entity occupying the position, causing placeBlock to timeout.",
            ),
            EnvironmentRule(
                rule_type="placement",
                description="blockUpdate timeout with target position already occupied by another block",
                condition="blockUpdate timeout, target_occupied: true, DIAGNOSTIC_FACTS",
                consequence="Another block already exists at the target position, preventing placement.",
            ),
        ],

        # Pathfinding rules
        pathfinding_rules=[
            EnvironmentRule(
                rule_type="pathfinding",
                description="Path blocked by obstacle or terrain",
                condition="Path blocked, Cannot reach, No path found, Goal unreachable, Pathfinder timeout, path timeout",
                consequence="Movement fails, bot cannot reach destination",
            ),
            EnvironmentRule(
                rule_type="pathfinding",
                description="Water and lava block paths",
                condition="nearby water, nearby lava, cannot swim",
                consequence="Standard pathfinding fails near liquids",
            ),
            EnvironmentRule(
                rule_type="pathfinding",
                description="Fences and walls block movement",
                condition="nearby fence, nearby wall",
                consequence="Cannot walk through fences even with gaps",
            ),
        ],

        # Crafting rules
        crafting_rules=[
            EnvironmentRule(
                rule_type="crafting",
                description="Complex recipes require crafting table",
                condition="crafting failed, need crafting table, 3x3 recipe",
                consequence="Cannot craft items requiring 3x3 grid in inventory",
            ),
            EnvironmentRule(
                rule_type="crafting",
                description="Smelting requires furnace",
                condition="need furnace, cannot smelt, smelting failed",
                consequence="Cannot smelt items without furnace",
            ),
            EnvironmentRule(
                rule_type="crafting",
                description="Insufficient materials",
                condition="not enough, missing, need more",
                consequence="Cannot craft due to missing ingredients",
            ),
        ],

        # Mining rules
        mining_rules=[
            EnvironmentRule(
                rule_type="mining",
                description="Block not found nearby",
                condition="No found nearby, Cannot find block, Block not visible, no ore found",
                consequence="Cannot mine target block",
            ),
            EnvironmentRule(
                rule_type="mining",
                description="Wrong tool for block",
                condition="dig failed, Cannot break, tool required",
                consequence="Block breaks slowly or drops nothing",
            ),
            EnvironmentRule(
                rule_type="mining",
                description="Y level affects ore distribution",
                condition="y_level, depth, underground",
                consequence="Some ores only spawn at specific Y levels",
            ),
        ],

        # Combat rules
        combat_rules=[
            EnvironmentRule(
                rule_type="combat",
                description="Entity not found or out of range",
                condition="Entity not found, Target out of range, Cannot attack, Entity too far, no nearby",
                consequence="Combat action fails",
            ),
            EnvironmentRule(
                rule_type="combat",
                description="Attack cooldown",
                condition="attack cooldown, cannot attack yet",
                consequence="Rapid attacks deal reduced damage",
            ),
        ],

        # Inventory rules
        inventory_rules=[
            EnvironmentRule(
                rule_type="inventory",
                description="Inventory full",
                condition="inventory full, no space, cannot pick up",
                consequence="Cannot collect items",
            ),
            EnvironmentRule(
                rule_type="inventory",
                description="Item not in inventory",
                condition="not have, don't have, missing item",
                consequence="Cannot use item that's not in inventory",
            ),
        ],

        # data imported from the game/ module
        block_hardness=BLOCK_HARDNESS,
        tool_effectiveness=TOOL_EFFECTIVENESS,
        biome_resources=BIOME_RESOURCES,
    )


def get_tool_for_block(block_name: str, knowledge: MinecraftEnvironmentKnowledge = None) -> Optional[str]:
    """Get the minimum tier tool required to mine a block. Delegates to game/blocks.py."""
    return _get_tool_for_block(block_name)


def get_ore_y_range(ore_name: str) -> Optional[Dict[str, int]]:
    """Get Y-axis spawn range for an ore. Delegates to game/resources.py."""
    return _get_ore_y_range(ore_name)
