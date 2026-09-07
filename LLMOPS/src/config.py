"""Gateway configuration via pydantic-settings."""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Provider API keys (optional — gateway degrades gracefully)
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Circuit breaker defaults
    cb_error_rate_threshold: float = 0.5
    cb_p95_latency_threshold_ms: float = 5000.0
    cb_window_size: int = 20
    cb_half_open_probes: int = 3
    cb_open_duration_seconds: int = 30

    # Retry
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    # Hedging
    hedge_timeout_seconds: float = 2.0

    # Queue
    deferrable_queue_key: str = "gateway:deferrable"


@lru_cache
def get_settings() -> Settings:
    return Settings()
