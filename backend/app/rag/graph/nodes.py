from backend.app.rag.graph.state import RAGState
from backend.app.services.answer_generator import generate_answer
from backend.app.services.citation_generator import generate_citations
from backend.app.services.context_evaluator import (
    MIN_CONTEXT_SCORE,
    evaluate_context,
)
from backend.app.services.grounding_checker import check_grounding
from backend.app.services.query_rewriter import rewrite_query
from backend.app.services.search import (
    hybrid_search,
    rerank_keep_search_order,
)

NOT_FOUND_ANSWER = (
    "I could not find relevant information "
    "in the uploaded documents."
)


def rewrite_query_node(state: RAGState) -> dict:
    rewritten_query = rewrite_query(
        query=state["original_query"],
        conversation_history=state["conversation_history"],
    )

    return {
        "rewritten_query": rewritten_query,
    }


def retrieve_node(state: RAGState) -> dict:
    results = hybrid_search(
        query=state["rewritten_query"],
        limit=20,
        document_ids=state["document_ids"],
    )

    return {
        "retrieved_chunks": results,
    }


def rerank_node(state: RAGState) -> dict:
    # Passages stay in hybrid-search order; each keeps its rerank_score.
    results = rerank_keep_search_order(
        query=state["rewritten_query"],
        candidates=state["retrieved_chunks"],
        top_k=5,
    )

    return {
        "reranked_chunks": results,
    }


def evaluate_context_node(state: RAGState) -> dict:
    score = evaluate_context(
        query=state["rewritten_query"],
        results=state["reranked_chunks"],
    )

    return {
        "context_quality": score,
    }


def decide_next_step(state: RAGState) -> str:
    # Good evidence found: answer the question.
    if state["context_quality"] >= MIN_CONTEXT_SCORE:
        return "generate"

    # Weak evidence: look wider once.
    if state["retry_count"] < 1:
        return "retry"

    # Still weak after the retry: do not ask Gemini to guess.
    return "no_answer"


def retry_retrieval_node(state: RAGState) -> dict:
    results = hybrid_search(
        query=state["rewritten_query"],
        limit=40,
        document_ids=state["document_ids"],
    )

    # "Look wider": keep the top 10 passages instead of the top 5.
    reranked = rerank_keep_search_order(
        query=state["rewritten_query"],
        candidates=results,
        top_k=10,
    )

    return {
        "retrieved_chunks": results,
        "reranked_chunks": reranked,
        "retry_count": state["retry_count"] + 1,
    }


def generate_answer_node(state: RAGState) -> dict:
    answer = generate_answer(
        query=state["original_query"],
        results=state["reranked_chunks"],
    )

    return {
        "answer": answer,
    }


def no_answer_node(state: RAGState) -> dict:
    # No Gemini call: the documents do not contain the answer.
    return {
        "answer": NOT_FOUND_ANSWER,
        "grounded": False,
        "grounding_score": 0.0,
        "citations": [],
    }


def check_grounding_node(state: RAGState) -> dict:
    score = check_grounding(
        query=state["original_query"],
        answer=state["answer"],
        results=state["reranked_chunks"],
    )

    return {
        "grounding_score": score,
        "grounded": score >= 0.7,
    }


def generate_citations_node(state: RAGState) -> dict:
    citations = generate_citations(
        query=state["original_query"],
        answer=state["answer"],
        results=state["reranked_chunks"],
    )

    return {
        "citations": citations,
    }