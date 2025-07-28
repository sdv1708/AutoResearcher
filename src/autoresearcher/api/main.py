import time
from fastapi import FastAPI
from pydantic import BaseModel

class HealthStatus(BaseModel):
    status: str
    uptime_sec: float

boot_time = time.time()

app = FastAPI(
    title="AutoResearcher API",
    version="0.0.1",
    description="Walking‑skeleton service (Iteration 0)",
)

@app.get("/health", response_model=HealthStatus, tags=["meta"])
async def health() -> HealthStatus:
    """Simple liveness check."""
    return HealthStatus(status="ok", uptime_sec=time.time() - boot_time)
