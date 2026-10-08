import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.routes import chat, documents, health
from backend.app.core.config import settings
from backend.app.core.logging import setup_logging
from backend.app.database.init_db import init_db

setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts.
    # Creates any missing PostgreSQL tables (existing tables and data are untouched).
    init_db()
    logger.info("Database tables are ready.")
    yield


app = FastAPI(
    title=settings.app_name,
    description="Advanced Multi-Document AI Knowledge Assistant",
    version=settings.app_version,
    lifespan=lifespan,
)


app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)

app.include_router(
    documents.router,
    prefix="/documents",
    tags=["Documents"],
)

app.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"],
)


@app.get("/")
def root():
    logger.info("Root endpoint requested")

    return {
        "message": f"{settings.app_name} is running",
        "version": settings.app_version,
    }