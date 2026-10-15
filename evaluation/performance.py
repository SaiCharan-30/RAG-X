"""Measures how fast RAG-X is and how many Gemini calls each question costs.

Runs the real app inside this process against your real documents (read-only:
chat questions use a temporary conversation that is deleted afterwards).
Does not need the server on port 8001 and does not change your data.

Usage (from the project root):
    uv run python evaluation/performance.py
"""
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOT_FOUND_ANSWER = "I could not find relevant information in the uploaded documents."
FOLLOW_UP = "What are its main advantages?"


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(index, 0)]


def timing_row(label, seconds):
    if not seconds:
        return f"  {label:<34}  (no data)"
    return (
        f"  {label:<34}{statistics.median(seconds):>8.2f}"
        f"{percentile(seconds, 0.95):>8.2f}{max(seconds):>8.2f}   n={len(seconds)}"
    )


def calls_row(label, counts):
    if not counts:
        return f"  {label:<40}  (no data)"
    mean = sum(counts) / len(counts)
    return f"  {label:<40} mean {mean:.1f}   (min {min(counts)}, max {max(counts)}, n={len(counts)})"


def main():
    eval_set = json.loads((ROOT / "evaluation" / "eval_set.json").read_text(encoding="utf-8"))
    answerable = [item["question"] for item in eval_set["in_scope"]][:8]
    unrelated = [item["question"] for item in eval_set["out_of_scope"]][:4]

    print("Loading the app and the AI models ...")
    start = time.perf_counter()
    from fastapi.testclient import TestClient

    from backend.app.core.config import settings
    from backend.app.main import app
    from backend.app.services.gemini_service import gemini_service

    import_seconds = time.perf_counter() - start

    # Count every Gemini call at its single entry point.
    calls = {"count": 0}
    original_generate = gemini_service.generate

    def counting_generate(*args, **kwargs):
        calls["count"] += 1
        return original_generate(*args, **kwargs)

    gemini_service.generate = counting_generate

    start = time.perf_counter()
    client = TestClient(app)
    client.__enter__()
    startup_seconds = time.perf_counter() - start

    try:
        documents = client.get("/documents").json()
        indexed = [d for d in documents if d["status"] == "indexed"]
        if not indexed:
            sys.exit("No indexed documents found. Upload some first, then run this again.")

        def search(query, rerank):
            began = time.perf_counter()
            response = client.post("/search", json={"query": query, "top_k": 5, "rerank": rerank})
            return response.status_code, time.perf_counter() - began

        def ask(conversation_id, query):
            calls["count"] = 0
            began = time.perf_counter()
            response = client.post(
                f"/chat/conversations/{conversation_id}/chat", json={"query": query}
            )
            seconds = time.perf_counter() - began
            body = response.json() if response.status_code == 200 else {}
            return response.status_code, seconds, calls["count"], body

        def new_conversation():
            return client.post("/chat/conversations", json={"title": "performance test"}).json()[
                "conversation_id"
            ]

        # Warm-up: the first call of each kind is slower and not representative.
        search("warm-up", True)
        warm_id = new_conversation()
        ask(warm_id, "warm-up")
        client.delete(f"/chat/conversations/{warm_id}")

        search_with, search_without = [], []
        chat_answerable, chat_not_found = [], []
        calls_answerable, calls_not_found = [], []
        errors = 0

        for query in answerable + unrelated:
            status, seconds = search(query, True)
            (search_with.append(seconds) if status == 200 else None)
            errors += status != 200
            status, seconds = search(query, False)
            (search_without.append(seconds) if status == 200 else None)
            errors += status != 200

            conversation_id = new_conversation()
            try:
                status, seconds, llm_calls, body = ask(conversation_id, query)
            finally:
                client.delete(f"/chat/conversations/{conversation_id}")
            if status != 200:
                errors += 1
                continue
            if body.get("answer") == NOT_FOUND_ANSWER:
                chat_not_found.append(seconds)
                calls_not_found.append(llm_calls)
            else:
                chat_answerable.append(seconds)
                calls_answerable.append(llm_calls)

        # A follow-up question inside one conversation (history makes the
        # assistant rewrite the question first).
        follow_up_calls, follow_up_seconds = [], []
        conversation_id = new_conversation()
        try:
            ask(conversation_id, answerable[0])
            status, seconds, llm_calls, body = ask(conversation_id, FOLLOW_UP)
            if status == 200:
                follow_up_calls.append(llm_calls)
                follow_up_seconds.append(seconds)
            else:
                errors += 1
        finally:
            client.delete(f"/chat/conversations/{conversation_id}")
    finally:
        client.__exit__(None, None, None)

    print()
    print(f"Documents indexed: {len(indexed)}   LLM mode: {'MOCK (no Gemini)' if settings.mock_llm else 'real Gemini'}")
    print()
    print("Startup (seconds)")
    print(f"  importing the app + loading the AI models : {import_seconds:.1f}")
    print(f"  starting the app (database + Qdrant)      : {startup_seconds:.1f}")
    print()
    print(f"Latency in seconds{'':<17}{'median':>8}{'p95':>8}{'max':>8}")
    print(timing_row("POST /search, with reranking", search_with))
    print(timing_row("POST /search, without reranking", search_without))
    print(timing_row("chat, answerable question", chat_answerable))
    print(timing_row("chat, not-found question", chat_not_found))
    print(timing_row("chat, follow-up question", follow_up_seconds))
    print()
    print("Gemini calls per chat question")
    print(calls_row("first question, answerable", calls_answerable))
    print(calls_row("first question, not found", calls_not_found))
    print(calls_row("follow-up question in a conversation", follow_up_calls))
    print()
    if settings.mock_llm:
        print("Note: MOCK mode. With real Gemini, add the time of each call above")
        print("(typically 1-3 seconds each) to the chat latencies.")
    if errors:
        print(f"WARNING: {errors} request(s) failed and were left out of the numbers.")


if __name__ == "__main__":
    main()