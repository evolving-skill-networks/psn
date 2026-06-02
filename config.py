"""Interactive .env bootstrap, invoked by `./skillnet.sh install`.

Copies .env.example -> .env (preserving the documented vLLM / per-agent /
recording sections), then prompts the user for an OpenAI API key and
substitutes it into the file. The PSN-verified default model is
gpt-5-mini; users can pick a different OpenAI model by passing one when
asked, or edit .env afterwards. Re-runs are idempotent: if .env already
exists, the user is asked before any overwrite.
"""

import os
import pathlib
import re
import shutil
import sys
from dataclasses import dataclass


# ========== Code Bloat Prevention Configuration ==========

@dataclass
class BloatPreventionConfig:
    """Configuration related to code bloat prevention"""

    # Code growth thresholds
    GROWTH_HARD_LIMIT_RATIO: float = 3.0      # 300% growth absolute rejection
    GROWTH_SOFT_LIMIT_RATIO: float = 2.0      # 200% growth triggers warning

    # Line count limits
    WRAPPER_MAX_LINES: int = 50               # max lines for wrapper skill
    ABSOLUTE_MAX_LINES: int = 800             # absolute max lines for any skill (aligned with optimizer/config.py)

    # Wrapper skill detection
    WRAPPER_LINE_THRESHOLD: int = 25          # fewer lines than this is considered a wrapper
    WRAPPER_AWAIT_THRESHOLD: int = 2          # fewer await calls than this is considered a wrapper

    # Retry configuration
    RETRY_ENABLED: bool = True                # whether retry is enabled
    RETRY_MAX_DIFF_LINES: int = 10            # max diff lines allowed for a minimal fix

    # Statistics configuration
    STATS_ENABLED: bool = True                # whether statistics are enabled
    STATS_FILE: str = "bloat_stats.json"      # statistics file name

    # Auto-shrink configuration
    AUTO_SHRINK_ENABLED: bool = False         # whether to auto-shrink (off by default, must be triggered manually)
    AUTO_SHRINK_THRESHOLD: float = 2.0        # only consider shrinking when bloat exceeds 200%

    # Cache configuration
    MCDATA_CACHE_ENABLED: bool = True         # whether minecraft-data cache is enabled
    MCDATA_CACHE_DIR: str = ".cache"          # cache directory

    # Minecraft version
    MC_VERSION: str = "1.19.4"                # Minecraft version PSN is verified against


# Export default configuration instance
BLOAT_CONFIG = BloatPreventionConfig()


# ========== .env bootstrap ==========

ENV_PATH = pathlib.Path(".env")
ENV_EXAMPLE_PATH = pathlib.Path(".env.example")

DEFAULT_MODEL = "gpt-5-mini"


def prompt_api_key():
    """Prompt the user for their OpenAI API key without echoing input."""
    import getpass

    print(
        "PSN can use either the OpenAI API or any OpenAI-compatible endpoint\n"
        "(vLLM, TGI, etc.). This script sets up the OpenAI path. If you only\n"
        "plan to use vLLM, you can press Ctrl-C now and edit .env directly\n"
        "(uncomment the VLLM_* lines)."
    )
    while True:
        api_key = getpass.getpass("\nEnter your OpenAI API key (input hidden): ").strip()
        if api_key:
            return api_key
        print("API key cannot be empty. Please try again.")


def prompt_model():
    """Ask which OpenAI model to use; default to the verified gpt-5-mini."""
    print(
        f"\nOpenAI model to use (PSN is verified against {DEFAULT_MODEL}).\n"
        f"Press Enter to accept the default, or type another OpenAI model name\n"
        f"(e.g. gpt-5, gpt-5-nano)."
    )
    choice = input(f"OpenAI model [{DEFAULT_MODEL}]: ").strip()
    return choice or DEFAULT_MODEL


def write_env(api_key, model):
    """Materialize .env from .env.example, substituting key + model."""
    if not ENV_EXAMPLE_PATH.exists():
        print(f"ERROR: {ENV_EXAMPLE_PATH} not found; cannot bootstrap .env.", file=sys.stderr)
        sys.exit(1)

    shutil.copyfile(ENV_EXAMPLE_PATH, ENV_PATH)

    contents = ENV_PATH.read_text()
    # Substitute the API-key placeholder and the model line. Quote the
    # API key in case it contains shell metacharacters.
    contents = re.sub(
        r'OPENAI_API_KEY="<YOUR OPENAI API KEY>"',
        f'OPENAI_API_KEY="{api_key}"',
        contents,
        count=1,
    )
    contents = re.sub(
        r'OPENAI_MODEL=".*"',
        f'OPENAI_MODEL="{model}"',
        contents,
        count=1,
    )
    ENV_PATH.write_text(contents)
    print(f"\n✅ .env created with model={model}. Edit the file to enable vLLM, per-agent overrides, etc.")


def main():
    if ENV_PATH.exists():
        print(f"{ENV_PATH} already exists.")
        choice = input("Overwrite? [y/N]: ").strip().lower()
        if choice not in ("y", "yes"):
            print("Keeping existing .env. (Run with --force or delete .env to regenerate.)")
            return

    api_key = prompt_api_key()
    model = prompt_model()
    write_env(api_key, model)


if __name__ == "__main__":
    main()
