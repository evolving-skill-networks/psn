"""
Unified LLM construction utility for multi-backend support.

Supports both OpenAI API and OpenAI-compatible endpoints (e.g., vLLM).
"""

import os

import httpx
from langchain_openai.chat_models import ChatOpenAI
from skillnet.utils.model_utils import get_temperature_for_model


def create_chat_llm(
    model_name: str,
    temperature: float,
    request_timeout: int,
    openai_api_base: str = None,
    openai_api_key: str = None,
    component: str = None,
    max_tokens: int = None,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with optional custom endpoint support.

    When openai_api_base is provided, the LLM will connect to that endpoint
    instead of the default OpenAI API. This enables use of vLLM or other
    OpenAI-compatible servers.

    If ``component`` is set and api_base/api_key are None, environment
    variables are checked automatically:
        {COMPONENT}_API_BASE → VLLM_API_BASE → None (OpenAI default)
        {COMPONENT}_API_KEY  → VLLM_API_KEY  → None

    Args:
        model_name: Model name (e.g., "gpt-5-mini", "Qwen/Qwen3-Coder-Next-FP8")
        temperature: Desired temperature value
        request_timeout: API request timeout in seconds
        openai_api_base: Custom API base URL (e.g., "http://<ip>:8000/v1")
        openai_api_key: Custom API key (optional; auto-set to "EMPTY" when api_base is used)
        component: Component name for env var auto-resolution (e.g., "action_agent")

    Returns:
        Configured ChatOpenAI instance
    """
    # Resolve API endpoint: explicit > component env var > global env var
    if openai_api_base is None and component:
        prefix = component.upper()
        openai_api_base = os.getenv(f"{prefix}_API_BASE", os.getenv("VLLM_API_BASE"))
    if openai_api_key is None and component:
        prefix = component.upper()
        openai_api_key = os.getenv(f"{prefix}_API_KEY", os.getenv("VLLM_API_KEY"))

    adjusted_temperature = get_temperature_for_model(model_name, temperature)

    if adjusted_temperature != temperature:
        print(
            f"\033[33mNote: model '{model_name}' temperature adjusted to {adjusted_temperature}\033[0m"
        )

    timeout = httpx.Timeout(
        connect=10.0,           # Fast detection of unreachable endpoints
        read=request_timeout,   # Allow sufficient time for long token generation
        write=30.0,
        pool=30.0,
    )

    kwargs = dict(
        model_name=model_name,
        temperature=adjusted_temperature,
        request_timeout=timeout,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if openai_api_base:
        kwargs["openai_api_base"] = openai_api_base
        # Avoid tiktoken warnings for non-OpenAI model names
        kwargs["tiktoken_model_name"] = "gpt-4"
        if not openai_api_key:
            # OpenAI client requires an API key even for vLLM; use placeholder
            kwargs["openai_api_key"] = "EMPTY"
    if openai_api_key:
        kwargs["openai_api_key"] = openai_api_key

    return ChatOpenAI(**kwargs)
