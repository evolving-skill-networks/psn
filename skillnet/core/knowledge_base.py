"""
Knowledge Base - unified knowledge-structure definitions.

Defines knowledge domains (Domain) and categories (Category), along with the
unified KnowledgeItem data structure. All knowledge items are purely factual
descriptions and do not contain solutions.

Design principles:
- Domain separation: GAME (game rules) vs API (mineflayer) vs PRIMITIVE (internal wrappers)
- No solution leakage: only fact + logical_implications; no fix_hint/solution_hint
- Let the LLM reason about solutions from facts
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Any


class KnowledgeDomain(Enum):
    """Knowledge domain — clearly distinguishes the source of knowledge."""
    GAME = "game"           # Minecraft game rules/physics
    API = "api"             # Mineflayer API behaviors
    PRIMITIVE = "primitive" # Our primitive functions


class KnowledgeCategory(Enum):
    """Knowledge category — finer-grained purpose."""
    # ===== Game domain =====
    BLOCK_PROPERTY = "block_property"       # Block properties (hardness, tool requirements)
    RESOURCE_SPAWN = "resource_spawn"       # Resource-spawn rules (ore distribution)
    GAME_MECHANIC = "game_mechanic"         # Game mechanics (water flow, fuels, etc.)
    CRAFTING_RECIPE = "crafting_recipe"     # Crafting/smelting recipes

    # ===== API domain =====
    PATHFINDER = "pathfinder"               # Pathfinder API behavior
    BOT_METHOD = "bot_method"               # Bot method behavior
    INVENTORY = "inventory"                 # Inventory API behavior

    # ===== Primitive domain =====
    PRECONDITION = "precondition"           # Primitive preconditions
    EFFECT = "effect"                       # Primitive effects
    FAILURE = "failure"                     # Primitive failure modes

    # ===== Code Generation domain =====
    CODE_GENERATION = "code_generation"     # Action-agent code generation guidance


@dataclass
class KnowledgeItem:
    """
    Unified knowledge-item structure — purely factual, no solutions.

    Field semantics:
    - fact: core description of the knowledge (purely factual)
    - keywords: keywords used for retrieval
    - logical_implications: pure logical implications (conclusions necessarily derivable from fact)
    - constraints: physical/logical constraints (limiting conditions)
    - available_options: enumeration of available options (not recommendations for specific options)

    Note: the following fields are deliberately absent:
    - solution_hint ❌
    - fix_hint ❌
    - fix_strategy ❌
    - fix_template ❌
    - suggested_fix ❌

    Design principle: let the LLM reason from facts rather than read predefined answers.

    Example (good logical_implications — pure logical implications):
        logical_implications=[
            "No diamonds exist above Y=16",      # Necessarily follows from "spawns Y=-64 to Y=16"
            "No diamonds exist below Y=-64",     # Necessarily follows from "spawns Y=-64 to Y=16"
        ]

    Example (bad implications — strategic suggestions):
        implications=["go deep underground"]     # This is a strategy, not a logical implication
        implications=["mine at Y=-59"]           # This is a recommendation, not a logical implication

    Example (good constraints):
        constraints=[
            "bot.findBlock() search radius ~32 blocks",
            "Blocks must be in loaded chunks",
        ]

    Example (good available_options — list options without recommending):
        available_options=[
            "Search at Y=16 (upper limit)",
            "Search at Y=-59 (highest concentration)",
            "Search at deeper levels",
        ]
    """

    domain: KnowledgeDomain
    category: KnowledgeCategory

    # Core content (purely factual)
    name: str                                   # Knowledge-item name (unique identifier)
    fact: str                                   # Fact description
    keywords: List[str]                         # Retrieval keywords

    # Pure logical implications (necessarily derived from fact)
    logical_implications: List[str] = field(default_factory=list)

    # Decision context (constraints and options)
    constraints: List[str] = field(default_factory=list)        # Physical/logical constraints
    available_options: List[str] = field(default_factory=list)  # Available options (no recommendation)

    # Optional metadata
    conditions: List[str] = field(default_factory=list)     # Applicability conditions
    implications: List[str] = field(default_factory=list)   # [DEPRECATED] use logical_implications
    source: str = ""                            # Source tracking

    def matches(self, text: str) -> bool:
        """Check whether the text matches the keywords of this knowledge item."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.keywords)

    def to_prompt_text(self) -> str:
        """Convert to the LLM prompt text format."""
        lines = [f"**{self.name}**: {self.fact}"]

        if self.conditions:
            lines.append(f"  Conditions: {', '.join(self.conditions)}")

        # Prefer logical_implications; fall back to implications
        implications_to_show = self.logical_implications or self.implications
        if implications_to_show:
            lines.append(f"  Logical Implications: {', '.join(implications_to_show)}")

        if self.constraints:
            lines.append(f"  Constraints: {', '.join(self.constraints)}")

        if self.available_options:
            lines.append(f"  Available Options: {', '.join(self.available_options)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to a dict (for serialization)."""
        return {
            "domain": self.domain.value,
            "category": self.category.value,
            "name": self.name,
            "fact": self.fact,
            "keywords": self.keywords,
            "logical_implications": self.logical_implications,
            "constraints": self.constraints,
            "available_options": self.available_options,
            "conditions": self.conditions,
            "implications": self.implications,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KnowledgeItem':
        """Create from a dict (for deserialization)."""
        return cls(
            domain=KnowledgeDomain(data["domain"]),
            category=KnowledgeCategory(data["category"]),
            name=data["name"],
            fact=data["fact"],
            keywords=data.get("keywords", []),
            logical_implications=data.get("logical_implications", []),
            constraints=data.get("constraints", []),
            available_options=data.get("available_options", []),
            conditions=data.get("conditions", []),
            implications=data.get("implications", []),
            source=data.get("source", ""),
        )


@dataclass
class RetrievedKnowledge:
    """Set of retrieved knowledge items."""
    items: List[KnowledgeItem] = field(default_factory=list)
    relevance_scores: dict = field(default_factory=dict)  # name -> score

    def to_prompt_section(self) -> str:
        """Convert to the knowledge section of a prompt."""
        if not self.items:
            return ""

        lines = ["## Relevant Knowledge\n"]

        # Group by domain
        by_domain: dict[KnowledgeDomain, List[KnowledgeItem]] = {}
        for item in self.items:
            if item.domain not in by_domain:
                by_domain[item.domain] = []
            by_domain[item.domain].append(item)

        domain_names = {
            KnowledgeDomain.GAME: "Minecraft Game Rules",
            KnowledgeDomain.API: "Mineflayer API Behaviors",
            KnowledgeDomain.PRIMITIVE: "Primitive Function Knowledge",
        }

        for domain, domain_items in by_domain.items():
            lines.append(f"### {domain_names.get(domain, domain.value)}\n")
            for item in domain_items:
                lines.append(item.to_prompt_text())
                lines.append("")

        return "\n".join(lines)

    def filter_by_domain(self, domain: KnowledgeDomain) -> 'RetrievedKnowledge':
        """Filter by domain."""
        filtered = [item for item in self.items if item.domain == domain]
        return RetrievedKnowledge(
            items=filtered,
            relevance_scores={k: v for k, v in self.relevance_scores.items()
                            if any(item.name == k for item in filtered)}
        )

    def filter_by_category(self, category: KnowledgeCategory) -> 'RetrievedKnowledge':
        """Filter by category."""
        filtered = [item for item in self.items if item.category == category]
        return RetrievedKnowledge(
            items=filtered,
            relevance_scores={k: v for k, v in self.relevance_scores.items()
                            if any(item.name == k for item in filtered)}
        )


# Exports
__all__ = [
    'KnowledgeDomain',
    'KnowledgeCategory',
    'KnowledgeItem',
    'RetrievedKnowledge',
]
