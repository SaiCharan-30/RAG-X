import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.routes import chat, documents, health
from backend.app.core.config import settings
from backend.app.core.logging import setup_logging
from backend.app.database.init_db import init_db
from backend.app.services.vector_store import vector_store
from backend.app.api.routes import chat, documents, health, search

setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts.
    # Creates any missing PostgreSQL tables (existing tables and data are untouched).
    init_db()
    logger.info("Database tables are ready.")

    # Creates the Qdrant collection if it does not exist yet (safe to repeat).
    vector_store.create_collection()
    logger.info("Qdrant collection is ready.")
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


app.include_router(
    search.router,
    prefix="/search",
    tags=["Search"],
)

@app.get("/")
def root():
    logger.info("Root endpoint requested")

    return {
        "message": f"{settings.app_name} is running",
        "version": settings.app_version,
    }
