import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving meaningful content.
    """

    # Remove NUL characters (PostgreSQL cannot store them; they appear in
    # UTF-16 text files and some PDFs)
    text = text.replace("\x00", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()