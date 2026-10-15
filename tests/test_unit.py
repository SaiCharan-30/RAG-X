"""Fast tests: no AI models, no database, no Docker.

Run:  uv run pytest -m "not integration"
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- text cleaner
from backend.app.services.text_cleaner import clean_text


def test_clean_text_removes_nul_characters():
    assert "\x00" not in clean_text("he\x00llo")


def test_clean_text_normalizes_windows_line_endings():
    assert "\r" not in clean_text("line one\r\nline two")


# ------------------------------------------------------------ context evaluator
from backend.app.services.context_evaluator import (
    MIN_CONTEXT_SCORE,
    evaluate_context,
)


def test_context_score_is_the_best_rerank_score():
    results = [{"rerank_score": 0.2}, {"rerank_score": 0.9}, {"rerank_score": 0.4}]
    assert evaluate_context("q", results) == pytest.approx(0.9)


def test_context_score_is_zero_without_results_or_scores():
    assert evaluate_context("q", []) == 0.0
    assert evaluate_context("q", [{"rerank_score": None}, {}]) == 0.0


def test_context_score_stays_between_zero_and_one():
    assert evaluate_context("q", [{"rerank_score": 1.7}]) == 1.0
    assert evaluate_context("q", [{"rerank_score": -0.3}]) == 0.0


def test_no_answer_threshold_is_the_documented_value():
    # Changing this changes what the assistant refuses to answer.
    # Re-run evaluation/run_eval.py before changing it.
    assert MIN_CONTEXT_SCORE == 0.05


# ------------------------------------------------------------ hybrid ordering
RERANK_SCORES = {"a": 0.10, "b": 0.90, "c": 0.50, "d": 0.02}


class FakeReranker:
    """Scores like a cross-encoder and sorts best-first, like the real one."""

    def rerank(self, query, results, top_k=5):
        scored = []
        for result in results:
            copy = result.copy()
            copy["rerank_score"] = RERANK_SCORES[result["text"]]
            scored.append(copy)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:top_k]


@pytest.fixture
def search_module(monkeypatch):
    """services/search.py loaded with fake models, so no model is downloaded."""
    fakes = {
        "backend.app.services.embeddings": {"embedding_service": None},
        "backend.app.services.vector_store": {"vector_store": None},
        "backend.app.services.reranker": {"reranker": FakeReranker()},
        "backend.app.services.sparse_embeddings": {"sparse_embedding_service": None},
    }
    for name, attributes in fakes.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "search_under_test", ROOT / "backend" / "app" / "services" / "search.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates():
    return [{"text": letter} for letter in "abcd"]


def test_results_keep_hybrid_order_not_reranker_order(search_module):
    results = search_module.rerank_keep_search_order("q", candidates(), top_k=3)
    assert [r["text"] for r in results] == ["a", "b", "c"]


def test_results_keep_their_rerank_score_and_hybrid_rank(search_module):
    results = search_module.rerank_keep_search_order("q", candidates(), top_k=4)
    assert [r["retrieval_rank"] for r in results] == [1, 2, 3, 4]
    assert [r["rerank_score"] for r in results] == [0.10, 0.90, 0.50, 0.02]


def test_best_rerank_score_survives_for_the_no_answer_check(search_module):
    results = search_module.rerank_keep_search_order("q", candidates(), top_k=2)
    # "b" is in the top 2 by hybrid order, so the gate sees its 0.9.
    assert evaluate_context("q", results) == pytest.approx(0.9)


def test_empty_candidates_give_empty_results(search_module):
    assert search_module.rerank_keep_search_order("q", [], top_k=5) == []


def test_search_documents_with_reranker(search_module, monkeypatch):
    monkeypatch.setattr(search_module, "hybrid_search", lambda **kwargs: candidates())
    results = search_module.search_documents("q", top_k=2, use_reranker=True)
    assert [r["text"] for r in results] == ["a", "b"]
    assert all("rerank_score" in r for r in results)


def test_search_documents_without_reranker(search_module, monkeypatch):
    monkeypatch.setattr(search_module, "hybrid_search", lambda **kwargs: candidates())
    results = search_module.search_documents("q", top_k=2, use_reranker=False)
    assert [r["text"] for r in results] == ["a", "b"]
    assert all("rerank_score" not in r for r in results)
    assert [r["retrieval_rank"] for r in results] == [1, 2]

def test_only_the_returned_passages_are_sent_to_the_reranker(search_module, monkeypatch):
    seen = []
    original = search_module.reranker.rerank

    def recording_rerank(query, results, top_k=5):
        seen.append(len(results))
        return original(query, results, top_k)

    monkeypatch.setattr(search_module.reranker, "rerank", recording_rerank)
    results = search_module.rerank_keep_search_order("q", candidates(), top_k=2)
    assert seen == [2]
    assert [r["text"] for r in results] == ["a", "b"]
# ------------------------------------------------------------- evaluation tool
def load_run_eval():
    spec = importlib.util.spec_from_file_location(
        "run_eval_under_test", ROOT / "evaluation" / "run_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_matching_ignores_pdf_ligatures_and_case():
    run_eval = load_run_eval()
    assert run_eval.normalize("The \ufb01nite  Element") == "the finite element"