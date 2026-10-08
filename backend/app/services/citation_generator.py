from backend.app.services.gemini_service import gemini_service


def generate_citations(
    query: str,
    answer: str,
    results: list[dict],
) -> list[dict]:
    """
    Generate citation metadata from the chunks that were actually
    retrieved and used by the RAG pipeline.
    """

    if not results:
        return []

    citations = []

    for citation_id, result in enumerate(results, start=1):
        citations.append(
            {
                "citation_id": citation_id,
                "document_id": result.get("document_id"),
                "chunk_id": result.get("chunk_id"),
                "filename": result.get("filename"),
                "page_number": result.get("page_number"),
                "section": result.get("section"),
                "chunk_index": result.get("chunk_index"),
            }
        )

    return citations