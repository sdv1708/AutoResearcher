from functools import lru_cache

from pydantic import BaseSettings


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

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
