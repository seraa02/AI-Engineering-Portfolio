"""Provider registry and request-class preference lists."""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass


class ProviderError(str, Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    CONTENT_FILTER = "content_filter"
    AUTHENTICATION_FAILURE = "authentication_failure"


class ProviderName(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MOCK = "mock"


@dataclass
class ProviderConfig:
    name: ProviderName
    litellm_model: str  # e.g., "claude-haiku-4-5" or "gpt-3.5-turbo"
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0


PROVIDERS: dict[str, ProviderConfig] = {
    ProviderName.ANTHROPIC: ProviderConfig(
        name=ProviderName.ANTHROPIC,
        litellm_model="claude-haiku-4-5",
        cost_per_input_token=0.80 / 1_000_000,
        cost_per_output_token=4.00 / 1_000_000,
    ),
    ProviderName.OPENAI: ProviderConfig(
        name=ProviderName.OPENAI,
        litellm_model="gpt-3.5-turbo",
        cost_per_input_token=0.50 / 1_000_000,
        cost_per_output_token=1.50 / 1_000_000,
    ),
    ProviderName.MOCK: ProviderConfig(
        name=ProviderName.MOCK,
        litellm_model="mock/mock-model",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
    ),
}

# Preference ordered list per request class.
# The router tries providers in order; skips OPEN circuit breakers.
PROVIDER_PREFERENCES: dict[str, list[str]] = {
    "chat": [ProviderName.ANTHROPIC, ProviderName.OPENAI, ProviderName.MOCK],
    "embeddings": [ProviderName.OPENAI, ProviderName.MOCK],
    "completion": [ProviderName.OPENAI, ProviderName.ANTHROPIC, ProviderName.MOCK],
    "fast": [ProviderName.MOCK, ProviderName.OPENAI, ProviderName.ANTHROPIC],
}

# Maps request class header value -> preference key
REQUEST_CLASS_MAP: dict[str, str] = {
    "interactive": "fast",
    "deferrable": "chat",
}


def get_preference_list(request_class: str) -> list[str]:
    pref_key = REQUEST_CLASS_MAP.get(request_class, "chat")
    return PROVIDER_PREFERENCES.get(pref_key, [ProviderName.MOCK])
