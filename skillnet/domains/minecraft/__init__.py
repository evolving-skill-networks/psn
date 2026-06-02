"""
Minecraft Domain for PSN

Provides MinecraftDomain() factory that bundles all Minecraft-specific
components into a DomainModule for use with PSNAgent.from_domain().

Usage:
    from skillnet.domains.minecraft import MinecraftDomain

    domain = MinecraftDomain(mc_port=25565)
    agent = PSNAgent.from_domain(domain, PSNConfig())
"""

from skillnet.domains.minecraft.environment import MinecraftEnv
from skillnet.domains.minecraft.knowledge import MinecraftKnowledge
from skillnet.domains.minecraft.curriculum import MinecraftCurriculum
from skillnet.domains.minecraft.critic import MinecraftCritic
from skillnet.core.domain import DomainModule
from skillnet.languages.javascript import JavaScriptLanguage


def MinecraftDomain(
    *,
    mc_port=None,
    server_host="http://127.0.0.1",
    server_port=3000,
    request_timeout=300,  # > index.js /step wall-clock guardrail (240s)
    log_path="./logs",
    model_name="gpt-5-mini",
    llm=None,
    curriculum_agent=None,
    critic_agent=None,
    kr_llm=None,
) -> DomainModule:
    """
    Factory function that creates a DomainModule for Minecraft.

    Args:
        mc_port: Minecraft server port (auto-detected if None).
        server_host: Mineflayer HTTP server host.
        server_port: Mineflayer HTTP server port.
        request_timeout: HTTP request timeout in seconds.
        log_path: Path for log files.
        model_name: LLM model name (affects control primitive selection).
        llm: Pre-built LLM instance for curriculum/critic (optional).
        curriculum_agent: Pre-built PSNCurriculumAgent (optional).
        critic_agent: Pre-built PSNCriticAgent (optional).

    Returns:
        DomainModule configured for Minecraft.
    """
    env = MinecraftEnv(
        mc_port=mc_port,
        server_host=server_host,
        server_port=server_port,
        request_timeout=request_timeout,
        log_path=log_path,
    )

    knowledge = MinecraftKnowledge(
        model_name=model_name,
        kr_llm=kr_llm,
    )

    curriculum = MinecraftCurriculum(agent=curriculum_agent)
    critic = MinecraftCritic(agent=critic_agent)

    return DomainModule(
        environment=env,
        knowledge=knowledge,
        curriculum=curriculum,
        critic=critic,
        name="minecraft",
        skill_language="javascript",
        skill_language_impl=JavaScriptLanguage(),
    )
