from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="dev")
    app_name: str = Field(default="FCPO AI Trading System")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)

    secret_key: str
    access_token_expire_minutes: int = 60
    algorithm: str = Field(default="HS256")
    api_key_tradingview: str

    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str

    redis_url: str

    model_path: str
    model_version: str

    log_level: str = Field(default="INFO")

    @property
    def database_url_async(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (Singleton)."""
    return Settings()


settings = get_settings()
