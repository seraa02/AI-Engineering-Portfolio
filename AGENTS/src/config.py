"""
Configuration management using pydantic-settings.
All settings are loaded from environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")

    # Tavily
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_ttl_seconds: int = Field(default=86400, env="REDIS_TTL_SECONDS")  # 24h

    # Models
    planner_model: str = Field(default="claude-haiku-4-5", env="PLANNER_MODEL")
    supervisor_model: str = Field(default="claude-haiku-4-5", env="SUPERVISOR_MODEL")
    writer_model: str = Field(default="claude-sonnet-4-6", env="WRITER_MODEL")
    researcher_model: str = Field(default="claude-haiku-4-5", env="RESEARCHER_MODEL")

    # Default budget config
    max_subquestions: int = Field(default=5, env="MAX_SUBQUESTIONS")
    max_searches_per_subquestion: int = Field(default=3, env="MAX_SEARCHES_PER_SUBQUESTION")
    max_total_tokens: int = Field(default=100_000, env="MAX_TOTAL_TOKENS")
    wall_clock_timeout_seconds: int = Field(default=300, env="WALL_CLOCK_TIMEOUT_SECONDS")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    return Settings()
