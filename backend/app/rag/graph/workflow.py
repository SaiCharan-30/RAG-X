from langgraph.graph import END, START, StateGraph

from backend.app.rag.graph.nodes import (
    check_grounding_node,
    decide_next_step,
    evaluate_context_node,
    generate_answer_node,
    generate_citations_node,
    no_answer_node,
    rerank_node,
    retrieve_node,
    retry_retrieval_node,
    rewrite_query_node,
)
from backend.app.rag.graph.state import RAGState


def build_rag_graph():
    builder = StateGraph(RAGState)

    # Nodes
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("evaluate_context", evaluate_context_node)
    builder.add_node("retry_retrieval", retry_retrieval_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("no_answer", no_answer_node)
    builder.add_node("check_grounding", check_grounding_node)
    builder.add_node("generate_citations", generate_citations_node)

    # Normal flow
    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "evaluate_context")

    # Good evidence -> answer, weak -> retry once, still weak -> no answer
    builder.add_conditional_edges(
        "evaluate_context",
        decide_next_step,
        {
            "generate": "generate_answer",
            "retry": "retry_retrieval",
            "no_answer": "no_answer",
        },
    )

    # Retry goes back to context evaluation
    builder.add_edge("retry_retrieval", "evaluate_context")

    # Answer -> grounding check -> citations -> done
    builder.add_edge("generate_answer", "check_grounding")
    builder.add_edge("check_grounding", "generate_citations")
    builder.add_edge("generate_citations", END)

    # No-answer path ends immediately
    builder.add_edge("no_answer", END)

    return builder.compile()


rag_graph = build_rag_graph()