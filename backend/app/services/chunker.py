from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.services.parser_models import ParsedPage


def create_chunks(
    pages: list[ParsedPage],
) -> list[dict]:
    """
    Split parsed pages into chunks while preserving page metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page.text)

        for chunk_text in page_chunks:
            chunks.append(
                {
                    "text": chunk_text,
                    "page_number": page.page_number,
                    "section": page.section,
                }
            )

    return chunks