"""Central configuration loaded from environment / .env file."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Database
    db_url: str = Field(
        default="postgresql://bayesbot:bayesbot@localhost:5432/bayesbot",
        alias="DB_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Interactive Brokers
    ib_host: str = Field(default="127.0.0.1", alias="IB_HOST")
    ib_port: int = Field(default=7497, alias="IB_PORT")
    ib_client_id: int = Field(default=1, alias="IB_CLIENT_ID")

    # Trading
    symbol: str = Field(default="MES", alias="SYMBOL")
    initial_capital: float = Field(default=25000.0, alias="INITIAL_CAPITAL")
    point_value: float = Field(default=5.0, description="MES = $5/point")

    # System
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    return Settings()
