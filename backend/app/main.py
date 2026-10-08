import logging

from fastapi import FastAPI

from backend.app.api.routes import health
from backend.app.core.config import settings
from backend.app.core.logging import setup_logging
from backend.app.api.routes import documents
from backend.app.api.routes import chat

setup_logging()

logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.app_name,
    description="Advanced Multi-Document AI Knowledge Assistant",
    version=settings.app_version,
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