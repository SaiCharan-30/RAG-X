from langgraph.graph import END, START, StateGraph

from backend.app.rag.graph.nodes import (
    decide_next_step,
    evaluate_context_node,
    generate_answer_node,
    retrieve_node,
    rerank_node,
    retry_retrieval_node,
    rewrite_query_node,
    check_grounding_node, 
    generate_citations_node,
)
from backend.app.rag.graph.state import RAGState


def build_rag_graph():
    builder = StateGraph(RAGState)

    # Add nodes
    builder.add_node(
        "rewrite_query",
        rewrite_query_node,
    )

    builder.add_node(
        "retrieve",
        retrieve_node,
    )

    builder.add_node(
        "rerank",
        rerank_node,
    )

    builder.add_node(
        "evaluate_context",
        evaluate_context_node,
    )

    builder.add_node(
        "retry_retrieval",
        retry_retrieval_node,
    )

    builder.add_node(
        "generate_answer",
        generate_answer_node,
    )

    # Normal flow
    builder.add_edge(
        START,
        "rewrite_query",
    )

    builder.add_edge(
        "rewrite_query",
        "retrieve",
    )

    builder.add_edge(
        "retrieve",
        "rerank",
    )

    builder.add_edge(
        "rerank",
        "evaluate_context",
    )

    # Conditional flow
    builder.add_conditional_edges(
        "evaluate_context",
        decide_next_step,
        {
            "generate": "generate_answer",
            "retry": "retry_retrieval",
        },
    )

    # Retry goes back to context evaluation
    builder.add_edge(
        "retry_retrieval",
        "evaluate_context",
    )

    # Answer generation finishes the graph
    builder.add_edge(
        "generate_answer",
        "check_grounding",
    )

    builder.add_edge(
        "check_grounding",
        "generate_citations",
    )

    builder.add_edge(
        "generate_citations",
        END,
    )
    builder.add_node(
    "check_grounding",
    check_grounding_node,
    )

    builder.add_node(
        "generate_citations",
        generate_citations_node,
    )

    return builder.compile()


rag_graph = build_rag_graph()