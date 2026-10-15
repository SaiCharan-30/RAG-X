"""First integration tests: prove the test setup is safe and the app starts.

Run:  uv run pytest -m integration -v
"""
import pytest

pytestmark = pytest.mark.integration

SAMPLE_TEXT = (
    "RAG-X is a multi-document knowledge assistant. "
    "It indexes uploaded documents with dense and sparse embeddings. "
    "Qdrant stores the vectors and PostgreSQL stores the metadata. "
) * 4


def test_tests_use_the_test_database_and_collection(app_client):
    from backend.app.database.connection import engine
    from backend.app.services.vector_store import vector_store

    assert engine.url.database == "rag_x_test"
    assert vector_store.collection_name == "rag_x_test_documents"


def test_document_list_starts_empty(app_client):
    response = app_client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_upload_index_list_and_delete_a_text_document(app_client):
    upload = app_client.post(
        "/documents/upload",
        files={"file": ("pytest_sample.txt", SAMPLE_TEXT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["status"] == "indexed"
    assert body["chunks_indexed"] > 0
    assert body["error"] is None
    document_id = body["document_id"]

    listed = app_client.get("/documents").json()
    assert [item["document_id"] for item in listed] == [document_id]
    assert listed[0]["status"] == "indexed"
    assert listed[0]["original_filename"] == "pytest_sample.txt"

    deleted = app_client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 200

    assert app_client.get("/documents").json() == []