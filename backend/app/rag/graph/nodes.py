from backend.app.rag.graph.state import RAGState
from backend.app.services.query_rewriter import rewrite_query
from backend.app.services.search import hybrid_search
from backend.app.services.reranker import reranker
from backend.app.services.context_evaluator import evaluate_context
from backend.app.services.answer_generator import generate_answer
from backend.app.services.grounding_checker import check_grounding
from backend.app.services.citation_generator import generate_citations
from backend.app.services.citation_generator import generate_citations
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
    results = reranker.rerank(
        query=state["rewritten_query"],
        results=state["retrieved_chunks"],
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
    if state["context_quality"] >= 0.6:
        return "generate"

    if state["retry_count"] >= 1:
        return "generate"

    return "retry"

def retry_retrieval_node(state: RAGState) -> dict:
    results = hybrid_search(
        query=state["rewritten_query"],
        limit=40,
        document_ids=state["document_ids"],
    )

    reranked = reranker.rerank(
        query=state["rewritten_query"],
        results=results,
        top_k=5,
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