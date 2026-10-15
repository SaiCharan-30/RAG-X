"""Shared pytest setup.

Fast tests (test_unit.py) need no AI models and no Docker.

Integration tests (marked "integration") use the real Postgres, Qdrant and AI
models, but ONLY a separate test database and test collection, never your real
documents. Run them with:   uv run pytest -m integration -v
"""
import os

# Avoid slow/failed Hugging Face network checks when models are already cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

TEST_DATABASE = "rag_x_test"
TEST_COLLECTION = "rag_x_test_documents"


def _create_test_database_if_missing(settings):
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    url = make_url(settings.database_url)
    maintenance = create_engine(
        url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
    )
    try:
        with maintenance.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": TEST_DATABASE},
            ).scalar()
            if not exists:
                connection.execute(text(f'CREATE DATABASE "{TEST_DATABASE}"'))
    except Exception as error:  # noqa: BLE001
        pytest.exit(
            "Cannot reach or prepare PostgreSQL for the tests.\n"
            "Is Docker running?  Try:  docker compose up -d\n"
            f"Details: {error}",
            returncode=2,
        )
    finally:
        maintenance.dispose()
    return url.set(database=TEST_DATABASE)


@pytest.fixture(scope="session")
def app_client(tmp_path_factory):
    """The real FastAPI app, wired to the test database and test collection."""
    from fastapi.testclient import TestClient
    from sqlalchemy import inspect, text

    from backend.app.core.config import settings

    test_url = _create_test_database_if_missing(settings)

    # Point the app at the test resources BEFORE anything creates a connection.
    settings.database_url = test_url.render_as_string(hide_password=False)
    settings.qdrant_collection = TEST_COLLECTION
    settings.mock_llm = True
    settings.debug = False  # no SQL echo noise

    from backend.app.database.connection import engine
    from backend.app.services.vector_store import vector_store

    # Safety guard: refuse to run if anything still points at real data.
    assert engine.url.database == TEST_DATABASE, "tests must use the test database"
    assert vector_store.collection_name == TEST_COLLECTION, (
        "tests must use the test collection"
    )

    from backend.app.api.routes import documents as documents_routes
    from backend.app.main import app

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        documents_routes, "UPLOAD_DIR", tmp_path_factory.mktemp("uploads")
    )

    try:
        vector_store.client.delete_collection(TEST_COLLECTION)
    except Exception:  # noqa: BLE001
        pass  # it did not exist yet

    with TestClient(app) as client:
        # App start-up has created the tables; start every run from empty.
        tables = inspect(engine).get_table_names()
        if tables:
            names = ", ".join(f'"{name}"' for name in tables)
            with engine.begin() as connection:
                connection.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))

        yield client

    try:
        vector_store.client.delete_collection(TEST_COLLECTION)
    except Exception:  # noqa: BLE001
        pass
    patcher.undo()