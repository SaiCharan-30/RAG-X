import re

# Matches [1], [2] and also [1, 2] / [1,2,3] written by the model.
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

SNIPPET_LENGTH = 300


def extract_cited_numbers(answer: str) -> list[int]:
    """
    Returns the source numbers the answer actually mentions,
    in the order they first appear, without duplicates.
    """
    numbers: list[int] = []

    for match in CITATION_PATTERN.finditer(answer or ""):
        for part in match.group(1).split(","):
            number = int(part.strip())

            if number not in numbers:
                numbers.append(number)

    return numbers


def generate_citations(
    query: str,
    answer: str,
    results: list[dict],
) -> list[dict]:
    """
    Build citations only for sources the answer really cites.

    The answer was written from SOURCE 1..N (the reranked chunks, in order),
    so "[2]" means results[1]. Numbers that do not exist (for example [9]
    when only 5 sources were given) are dropped, so a citation is never
    invented. citation_id is the same number used in the answer text.
    """
    if not results:
        return []

    citations = []

    for number in sorted(extract_cited_numbers(answer)):
        if not 1 <= number <= len(results):
            continue

        result = results[number - 1]
        text = (result.get("text") or "").strip()

        citations.append(
            {
                "citation_id": number,
                "document_id": result.get("document_id"),
                "chunk_id": result.get("chunk_id"),
                "filename": result.get("filename"),
                "page_number": result.get("page_number"),
                "section": result.get("section"),
                "chunk_index": result.get("chunk_index"),
                "snippet": text[:SNIPPET_LENGTH],
            }
        )

    return citations