from backend.app.rag.graph.workflow import rag_graph


state = {
    "original_query": "What are the advantages of its architecture?",
    "rewritten_query": "",
    "conversation_history": [
        {
            "role": "user",
            "content": "What is the proposed architecture of RAG-X?",
        },
        {
            "role": "assistant",
            "content": (
                "RAG-X uses hybrid retrieval with dense "
                "and sparse search followed by reranking."
            ),
        },
    ],
    "retrieved_chunks": [],
    "reranked_chunks": [],
    "context_quality": 0.0,
    "answer": "",
    "grounded": False,
    "grounding_score": 0.0,
    "citations": [],
    "retry_count": 0,
}

result = rag_graph.invoke(state)

print("\n=== RAG GRAPH RESULT ===")

print("\nOriginal query:")
print(result["original_query"])

print("\nRewritten query:")
print(result["rewritten_query"])

print("\nRetrieved chunks:")
print(len(result["retrieved_chunks"]))

print("\nReranked chunks:")
print(len(result["reranked_chunks"]))

print("\nContext quality:")
print(result["context_quality"])

print("\nRetry count:")
print(result["retry_count"])

print("\nGenerated answer:")
print(result["answer"])
print("\nGrounded:")
print(result["grounded"])

print("\nGrounding score:")
print(result["grounding_score"])

print("\nCitations:")
for citation in result["citations"]:
    print(citation)