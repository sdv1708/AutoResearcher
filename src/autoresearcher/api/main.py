import time
from fastapi import FastAPI
from pydantic import BaseModel

from autoresearcher.core.settings import get_settings
settings = get_settings()  # noqa: E402


class HealthStatus(BaseModel):
    status: str
    uptime_sec: float

boot_time = time.time()

app = FastAPI(
    title="AutoResearcher API",
    version="0.0.1",
    description="Walking‑skeleton service (Iteration 0)",
)

@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Basic welcome route."""
    return {"message": "AutoResearcher API is running — see /docs for usage"}

@app.get("/health", response_model=HealthStatus, tags=["meta"])
async def health() -> HealthStatus:
    """Simple liveness check."""
    return HealthStatus(status="ok", uptime_sec=time.time() - boot_time)
