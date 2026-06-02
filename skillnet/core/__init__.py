"""
PSN Core Module

Provides environment-agnostic configuration and abstractions for the
Evolving Programmatic Skill Networks (PSN) framework.
"""

from skillnet.core.config import (
    LLMEndpoint,
    AgentLLMs,
    ActionConfig,
    CurriculumConfig,
    CriticConfig,
    SkillManagerConfig,
    PlannerConfig,
    OptimizationConfig,
    CheckpointConfig,
    RecordingConfig,
    EnvironmentConfig,
    PSNConfig,
)

from skillnet.core.environment import Environment
from skillnet.core.domain import DomainKnowledge, DomainModule
from skillnet.core.curriculum import CurriculumStrategy
from skillnet.core.critic import CriticStrategy
from skillnet.core.skill_language import (
    SkillLanguage,
    FunctionInfo,
    ParseResult,
    ValidationResult,
)
from skillnet.core.knowledge_base import (
    KnowledgeDomain,
    KnowledgeCategory,
    KnowledgeItem,
    RetrievedKnowledge,
)

# Central DK registry
from skillnet.core.dk_registry import (
    get_domain_knowledge,
    set_domain_knowledge as set_global_domain_knowledge,
)

# Domain registry (re-exported for convenience)
from skillnet.domains import register_domain, get_domain, list_domains

__all__ = [
    # Config
    "LLMEndpoint",
    "AgentLLMs",
    "ActionConfig",
    "CurriculumConfig",
    "CriticConfig",
    "SkillManagerConfig",
    "PlannerConfig",
    "OptimizationConfig",
    "CheckpointConfig",
    "RecordingConfig",
    "EnvironmentConfig",
    "PSNConfig",
    # Domain abstractions
    "Environment",
    "DomainKnowledge",
    "DomainModule",
    "CurriculumStrategy",
    "CriticStrategy",
    # Skill language
    "SkillLanguage",
    "FunctionInfo",
    "ParseResult",
    "ValidationResult",
    # Knowledge base types
    "KnowledgeDomain",
    "KnowledgeCategory",
    "KnowledgeItem",
    "RetrievedKnowledge",
    # Central DK registry
    "get_domain_knowledge",
    "set_global_domain_knowledge",
    # Domain registry
    "register_domain",
    "get_domain",
    "list_domains",
]
