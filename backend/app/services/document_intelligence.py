from sqlalchemy.orm import Session

from backend.app.models.chunk import DocumentChunk
from backend.app.services.gemini_service import gemini_service
MAX_GROUP_CHARACTERS = 12000
# MAX_CHUNKS_PER_GROUP = 2
# MAX_GROUP_CHARACTERS = 2000

SUMMARY_INSTRUCTIONS = {
    "quick": (
        "Give a short summary in 3 to 5 sentences. "
        "Focus only on the most important information."
    ),
    "detailed": (
        "Give a detailed summary covering the document's "
        "main ideas, important concepts, methods, and conclusions."
    ),
    "key_points": (
        "Extract the most important points from the document. "
        "Return them as a clear bullet-point list."
    ),
    "methodology": (
        "Describe the methodology, approach, process, or system "
        "described in the document. If methodology is not present, "
        "clearly state that."
    ),
    "findings": (
        "Describe the main findings, results, conclusions, or "
        "observations presented in the document. If these are not "
        "present, clearly state that."
    ),
    "limitations": (
        "Identify the limitations, constraints, assumptions, or "
        "known weaknesses mentioned in the document. If none are "
        "mentioned, clearly state that."
    ),
    "keywords": (
        "Extract the most important technical terms, concepts, "
        "topics, and keywords from the document. Return them as "
        "a concise comma-separated list."
    ),
}

def group_chunks(
    chunks: list[DocumentChunk],
    max_characters: int,
) -> list[list[DocumentChunk]]:
    groups: list[list[DocumentChunk]] = []
    current_group: list[DocumentChunk] = []
    current_characters = 0

    for chunk in chunks:
        chunk_characters = len(chunk.text)

        if (
            current_group
            and current_characters + chunk_characters > max_characters
        ):
            groups.append(current_group)
            current_group = []
            current_characters = 0

        current_group.append(chunk)
        current_characters += chunk_characters

    if current_group:
        groups.append(current_group)

    return groups

def summarize_chunk_group(
    chunks: list[DocumentChunk],
    instruction: str,
) -> str:
    group_text = "\n\n".join(
        chunk.text
        for chunk in chunks
    )

    prompt = f"""
You are a document summarization assistant.

Summarize the following section of a larger document.

Task:
{instruction}

Requirements:
- Preserve the main meaning.
- Use only information provided in the section.
- Do not invent information.
- If the requested information is not present, say so.
- Do not use outside knowledge.

DOCUMENT SECTION:

{group_text}

SUMMARY:
"""

    return gemini_service.generate(prompt)


def combine_partial_summaries(
    partial_summaries: list[str],
    instruction: str,
) -> str:
    summaries_text = "\n\n".join(
        f"PARTIAL SUMMARY {index + 1}:\n{summary}"
        for index, summary in enumerate(partial_summaries)
    )

    prompt = f"""
You are a document summarization assistant.

Create the final summary of a document using the partial
summaries below.

Task:
{instruction}

Requirements:
- Combine the important information.
- Remove unnecessary repetition.
- Preserve the original meaning.
- Do not invent information.
- Use only the information contained in the partial summaries.

PARTIAL SUMMARIES:

{summaries_text}

FINAL SUMMARY:
"""

    return gemini_service.generate(prompt)


def summarize_document(
    document_id: str,
    db: Session,
    summary_type: str = "quick",
) -> dict:
    if summary_type not in SUMMARY_INSTRUCTIONS:
        raise ValueError(
            f"Unsupported summary type: {summary_type}"
        )

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )

    if not chunks:
        raise ValueError(
            "No indexed chunks found for this document."
        )

    instruction = SUMMARY_INSTRUCTIONS[summary_type]

    chunk_groups = group_chunks(
        chunks,
        MAX_GROUP_CHARACTERS,
    )
    partial_summaries = [
        summarize_chunk_group(
            group,
            instruction,
        )
        for group in chunk_groups
    ]

    if len(partial_summaries) == 1:
        summary = partial_summaries[0]
    else:
        summary = combine_partial_summaries(
            partial_summaries,
            instruction,
        )

    return {
        "document_id": document_id,
        "summary_type": summary_type,
        "summary": summary,
        "source_chunks": len(chunks),
    }