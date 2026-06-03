"""
Minecraft Environment Wrapper

Thin wrapper around SkillNetEnv that implements the Environment ABC.
All logic is delegated to the existing SkillNetEnv implementation.
"""

from typing import Any, Dict, List, Optional, Tuple

from skillnet.core.environment import Environment


class MinecraftEnv(Environment):
    """
    Environment ABC implementation for Minecraft via Mineflayer.

    Wraps SkillNetEnv (gymnasium-based) to provide the PSN Environment
    interface. Construction of the underlying SkillNetEnv is deferred
    until the first reset() call to allow late configuration.
    """

    def __init__(
        self,
        mc_port=None,
        server_host="http://127.0.0.1",
        server_port=3000,
        request_timeout=300,  # > index.js /step wall-clock guardrail (240s)
        log_path="./logs",
    ):
        self._init_kwargs = dict(
            mc_port=mc_port,
            server_host=server_host,
            server_port=server_port,
            request_timeout=request_timeout,
            log_path=log_path,
        )
        self._env = None

    @property
    def mc_port(self):
        """Port of the Minecraft server."""
        return self._init_kwargs.get('mc_port')

    @property
    def server_port(self):
        """Port of the Mineflayer HTTP server."""
        return self._init_kwargs.get('server_port', 3000)

    def _ensure_env(self):
        """Lazily create the underlying SkillNetEnv."""
        if self._env is None:
            from skillnet.domains.minecraft.action_space.env.bridge import SkillNetEnv
            self._env = SkillNetEnv(**self._init_kwargs)
        return self._env

    @property
    def unwrapped(self):
        """Access the underlying SkillNetEnv directly."""
        return self._ensure_env()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        env = self._ensure_env()
        return env.reset(seed=seed, options=options or {})

    def step(
        self,
        code: str,
        programs: str = "",
        skill_names: Optional[List[str]] = None,
        is_iteration: bool = True,
        keep_paused: bool = False,
    ) -> Any:
        env = self._ensure_env()
        return env.step(code, programs=programs, skill_names=skill_names,
                        is_iteration=is_iteration, keep_paused=keep_paused)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()

    def pause(self) -> bool:
        if self._env is not None:
            self._env.pause()
            return True
        return False

    def unpause(self) -> bool:
        if self._env is not None:
            self._env.unpause()
            return True
        return False
