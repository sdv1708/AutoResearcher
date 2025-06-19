from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.api_title, version=settings.api_version, debug=settings.debug
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the AutoResearcher API!",
        "version": settings.api_version,
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.api_version}
