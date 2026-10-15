"""API tests with the mock LLM, using the real app, Postgres, Qdrant and models,
but only the separate test database/collection (see conftest.py).

Run:  uv run pytest -m integration -v
"""
import pytest

pytestmark = pytest.mark.integration

VECTOR_TEXT = (
    "Qdrant is a vector database. "
    "Qdrant stores dense and sparse vectors for hybrid search. "
    "Hybrid search combines keyword matching with semantic similarity. "
    "Reranking reorders the retrieved passages. "
) * 4

BREAD_TEXT = (
    "Sourdough bread is baked from flour, water and salt. "
    "The dough ferments slowly with a wild yeast starter. "
    "Bake the loaf in a hot oven until the crust turns golden brown. "
) * 4

NOT_FOUND_ANSWER = "I could not find relevant information in the uploaded documents."
UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"


# ------------------------------------------------------------------- helpers
def upload(client, filename, data, content_type="text/plain"):
    return client.post(
        "/documents/upload",
        files={"file": (filename, data, content_type)},
    )


def search(client, **body):
    return client.post("/search", json=body)


def chat(client, conversation_id, **body):
    return client.post(f"/chat/conversations/{conversation_id}/chat", json=body)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def docs(app_client):
    """Two indexed documents on different topics."""
    first = upload(app_client, "pytest_vectors.txt", VECTOR_TEXT.encode("utf-8"))
    second = upload(app_client, "pytest_bread.txt", BREAD_TEXT.encode("utf-8"))
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["status"] == "indexed"
    assert second.json()["status"] == "indexed"

    ids = {"vectors": first.json()["document_id"], "bread": second.json()["document_id"]}
    yield ids

    for document_id in ids.values():
        app_client.delete(f"/documents/{document_id}")


@pytest.fixture
def failed_document(app_client):
    """A document whose indexing failed (it is not a real PDF)."""
    response = upload(
        app_client, "pytest_corrupt.pdf", b"this is not a real pdf", "application/pdf"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"]
    yield body["document_id"]
    app_client.delete(f"/documents/{body['document_id']}")


@pytest.fixture
def conversation(app_client):
    response = app_client.post("/chat/conversations", json={"title": "pytest conversation"})
    assert response.status_code in (200, 201)
    conversation_id = response.json()["conversation_id"]
    yield conversation_id
    app_client.delete(f"/chat/conversations/{conversation_id}")


# ----------------------------------------------------------------- documents
def test_unsupported_file_type_is_rejected(app_client):
    response = upload(app_client, "program.exe", b"MZ not a document", "application/octet-stream")
    assert 400 <= response.status_code < 500


def test_a_corrupt_pdf_is_kept_as_failed_with_a_reason(app_client, failed_document):
    listed = {item["document_id"]: item for item in app_client.get("/documents").json()}
    assert listed[failed_document]["status"] == "failed"


# -------------------------------------------------------------------- search
def test_search_returns_ranked_results_in_hybrid_order(app_client, docs):
    response = search(app_client, query="What does Qdrant store?", top_k=5)
    assert response.status_code == 200
    body = response.json()

    assert body["reranked"] is True
    assert body["result_count"] == len(body["results"]) > 0
    assert [r["rank"] for r in body["results"]] == list(range(1, len(body["results"]) + 1))

    # Passages keep hybrid-search order and still carry a rerank score.
    retrieval_ranks = [r["retrieval_rank"] for r in body["results"]]
    assert retrieval_ranks == sorted(retrieval_ranks)
    assert all(r["rerank_score"] is not None for r in body["results"])


def test_search_finds_the_matching_document_first(app_client, docs):
    vectors = search(app_client, query="What does Qdrant store?").json()["results"]
    bread = search(app_client, query="How do you bake sourdough bread?").json()["results"]
    assert vectors[0]["document_id"] == docs["vectors"]
    assert bread[0]["document_id"] == docs["bread"]


def test_search_without_reranker_has_no_rerank_score(app_client, docs):
    body = search(app_client, query="What does Qdrant store?", rerank=False).json()
    assert body["reranked"] is False
    assert all(r["rerank_score"] is None for r in body["results"])


def test_search_top_k_limits_the_number_of_results(app_client, docs):
    body = search(app_client, query="Qdrant", top_k=1).json()
    assert body["result_count"] == 1


def test_search_single_document_mode_stays_inside_that_document(app_client, docs):
    body = search(
        app_client, query="bread", mode="single", document_ids=[docs["vectors"]]
    ).json()
    assert body["result_count"] > 0
    assert {r["document_id"] for r in body["results"]} == {docs["vectors"]}


def test_search_selected_mode_uses_only_the_chosen_documents(app_client, docs):
    chosen = [docs["vectors"], docs["bread"]]
    body = search(app_client, query="anything", mode="selected", document_ids=chosen).json()
    assert {r["document_id"] for r in body["results"]} <= set(chosen)


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 21},
        {"query": "x", "mode": "single"},
        {"query": "x", "mode": "single", "document_ids": ["a", "b"]},
        {"query": "x", "mode": "selected"},
        {"query": "x", "mode": "nonsense"},
    ],
)
def test_search_rejects_invalid_requests(app_client, body):
    assert search(app_client, **body).status_code == 422


def test_search_unknown_document_is_404(app_client):
    response = search(app_client, query="x", mode="single", document_ids=[UNKNOWN_ID])
    assert response.status_code == 404


def test_search_not_indexed_document_is_400(app_client, failed_document):
    response = search(app_client, query="x", mode="single", document_ids=[failed_document])
    assert response.status_code == 400


# ---------------------------------------------------------------------- chat
def test_chat_answers_from_the_documents_with_citations(app_client, docs, conversation):
    response = chat(
        app_client,
        conversation,
        query="What does Qdrant store?",
        mode="single",
        document_ids=[docs["vectors"]],
    )
    assert response.status_code == 200
    body = response.json()

    assert body["conversation_id"] == conversation
    assert body["answer"].strip()
    assert body["context_quality"] >= 0.05
    assert len(body["citations"]) >= 1
    assert all(c["filename"] == "pytest_vectors.txt" for c in body["citations"])


def test_chat_says_not_found_when_the_documents_do_not_contain_the_answer(
    app_client, docs, conversation
):
    response = chat(
        app_client,
        conversation,
        query="How do I bake sourdough bread?",
        mode="single",
        document_ids=[docs["vectors"]],
    )
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == NOT_FOUND_ANSWER
    assert body["citations"] == []
    assert body["grounded"] is False
    assert body["context_quality"] < 0.05


def test_chat_messages_are_saved_with_the_conversation(app_client, docs, conversation):
    chat(
        app_client,
        conversation,
        query="What does Qdrant store?",
        mode="single",
        document_ids=[docs["vectors"]],
    )

    messages = app_client.get(f"/chat/conversations/{conversation}/messages").json()
    assert [m["role"] for m in messages["messages"]] == ["user", "assistant"]
    assert messages["messages"][0]["content"] == "What does Qdrant store?"
    assert messages["messages"][1]["citations"]

    summaries = {c["conversation_id"]: c for c in app_client.get("/chat/conversations").json()}
    assert summaries[conversation]["message_count"] == 2


def test_chat_requests_are_validated(app_client, conversation):
    assert chat(app_client, conversation, query="").status_code == 422
    assert chat(app_client, conversation, query="x", mode="single").status_code == 422


def test_chat_unknown_conversation_is_404(app_client):
    assert chat(app_client, UNKNOWN_ID, query="hello").status_code == 404


def test_chat_unknown_document_is_404(app_client, conversation):
    response = chat(app_client, conversation, query="x", mode="single", document_ids=[UNKNOWN_ID])
    assert response.status_code == 404


def test_chat_not_indexed_document_is_400(app_client, conversation, failed_document):
    response = chat(
        app_client, conversation, query="x", mode="single", document_ids=[failed_document]
    )
    assert response.status_code == 400


def test_a_failed_chat_request_does_not_save_messages(app_client, conversation):
    chat(app_client, conversation, query="x", mode="single", document_ids=[UNKNOWN_ID])
    messages = app_client.get(f"/chat/conversations/{conversation}/messages").json()
    assert messages["messages"] == []


# ------------------------------------------------------------- conversations
def test_conversation_lifecycle(app_client):
    created = app_client.post("/chat/conversations", json={"title": "lifecycle"})
    assert created.status_code in (200, 201)
    conversation_id = created.json()["conversation_id"]
    assert created.json()["title"] == "lifecycle"

    listed = [c["conversation_id"] for c in app_client.get("/chat/conversations").json()]
    assert conversation_id in listed

    deleted = app_client.delete(f"/chat/conversations/{conversation_id}")
    assert deleted.status_code == 200

    assert app_client.get(f"/chat/conversations/{conversation_id}/messages").status_code == 404
    assert app_client.delete(f"/chat/conversations/{conversation_id}").status_code == 404