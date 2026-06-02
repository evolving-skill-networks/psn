"""
Environment Knowledge dataclasses.

These provide structured environment rules and learned strategies.
Used by domain-specific subclasses (e.g., MinecraftEnvironmentKnowledge).

Note: These dataclasses are no longer threaded through the optimizer
pipeline — the LLMAnalyzer uses DomainKnowledge directly.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EnvironmentRule:
    """
    Environment rule - a single piece of environmental knowledge.

    v7.0 refactor: removed the solution_hint field, keeping only the pure factual description.
    The LLM should infer the solution from description + condition + consequence on its own.
    """
    rule_type: str              # placement, pathfinding, interaction, etc.
    description: str            # Rule description
    condition: str              # Trigger condition (keyword matching)
    consequence: str            # Consequence of violating the rule


@dataclass
class EnvironmentKnowledge:
    """
    Environment knowledge - the third argument of psn_reflection.

    Provides knowledge of the world the agent lives in so that psn_reflection can:
    1. Distinguish "code logic errors" from "environment adaptation issues"
    2. Provide environment-specific fix suggestions

    Mathematical intuition: environment knowledge can be viewed as a prior distribution
    that helps psn_reflection "differentiate" more accurately.
    """

    # Domain-specific rules (named for Minecraft backward compat)
    placement_rules: List[EnvironmentRule] = field(default_factory=list)
    pathfinding_rules: List[EnvironmentRule] = field(default_factory=list)
    interaction_rules: List[EnvironmentRule] = field(default_factory=list)
    crafting_rules: List[EnvironmentRule] = field(default_factory=list)

    # Current environment state (optional)
    current_biome: Optional[str] = None
    time_of_day: Optional[str] = None
    weather: Optional[str] = None

    def find_relevant_rules(self, error_message: str) -> List[EnvironmentRule]:
        """Find environment rules relevant to the error."""
        relevant = []
        all_rules = (
            self.placement_rules +
            self.pathfinding_rules +
            self.interaction_rules +
            self.crafting_rules
        )
        for rule in all_rules:
            # Simple keyword matching
            if any(kw in error_message.lower() for kw in rule.condition.lower().split()):
                relevant.append(rule)
        return relevant
