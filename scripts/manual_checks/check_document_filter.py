from backend.app.services.search import hybrid_search


results = hybrid_search(
    query="What is RAG-X?",
    limit=10,
    document_ids=["test-document-001"],
)

print("\n=== FILTERED SEARCH ===")
print("Results:", len(results))

for result in results:
    print(
        result["filename"],
        "|",
        result["document_id"],
        "|",
        result["chunk_id"],
    )