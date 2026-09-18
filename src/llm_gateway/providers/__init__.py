from typing import Any

from .base import Provider
from .mock import MockClient, MockProvider

__all__ = [
    "GroqClient",
    "MockClient",
    "MockProvider",
    "OpenAICompatibleClient",
    "OpenRouterClient",
    "Provider",
    "groq_provider",
    "openrouter_provider",
    "parse_retry_after",
]

# The HTTP-backed provider layer (http.py, groq.py, openrouter.py) imports
# httpx, which is only installed when the optional `[http]` extra is present.
# Importing those names lazily via module __getattr__ (PEP 562) means
# `import llm_gateway.providers` keeps working without httpx installed, and
# only `from llm_gateway.providers import groq_provider` (or similar) trips
# the friendly ImportError raised inside http.py.
_LAZY_ATTRS = {
    "OpenAICompatibleClient": ("http", "OpenAICompatibleClient"),
    "parse_retry_after": ("http", "parse_retry_after"),
    "GroqClient": ("groq", "GroqClient"),
    "groq_provider": ("groq", "groq_provider"),
    "OpenRouterClient": ("openrouter", "OpenRouterClient"),
    "openrouter_provider": ("openrouter", "openrouter_provider"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(__all__)
