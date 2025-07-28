from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Centralised config. Reads from env vars or `.env` file."""

    # --- general ------------------------------------------------
    log_level: str = "INFO"

    # --- embeddings / LLM --------------------------------------
    use_vertex: bool = False                # set via USE_VERTEX
    vertex_project_id: str | None = None    # VERTEX_PROJECT_ID
    vertex_region: str = "us-central1"      # VERTEX_REGION

    # --- auth ---------------------------------------------------
    google_application_credentials: Path | None = None  # GOOGLE_APPLICATION_CREDENTIALS
    service_account_email: str | None = None            # SERVICE_ACCOUNT_EMAIL

    # --- model selection ---------------------------------------
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    lora_adapter_path: Path | None = None  # filled after fine‑tune

    # --- internal pydantic meta --------------------------------
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Import‑safe singleton."""
    return Settings()