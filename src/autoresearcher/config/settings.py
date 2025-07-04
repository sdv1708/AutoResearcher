from functools import lru_cache
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Configuration
    api_title: str = "AutoResearcher API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Development Settings
    debug: bool = True
    environment: str = "development"

    # Data Paths
    data_dir: str = "./data"
    model_cache_dir: str = "./models"

    # External APIs (for mocks)
    use_mock_models: bool = True

    # GCP Configuration (optional override)
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
