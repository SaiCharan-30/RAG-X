from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from backend.app.services.parser_models import ParsedPage
from backend.app.services.text_cleaner import clean_text


def parse_txt(file_path: Path) -> list[ParsedPage]:
    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return [
        ParsedPage(
            page_number=None,
            text=clean_text(text),
        )
    ]


def parse_pdf(file_path: Path) -> list[ParsedPage]:
    pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()

            cleaned = clean_text(text)

            if cleaned:
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=cleaned,
                    )
                )

    return pages


def parse_docx(file_path: Path) -> list[ParsedPage]:
    document = DocxDocument(file_path)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    text = clean_text("\n".join(paragraphs))

    return [
        ParsedPage(
            page_number=None,
            text=text,
        )
    ]


def parse_html(file_path: Path) -> list[ParsedPage]:
    html = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for element in soup(["script", "style"]):
        element.decompose()

    text = soup.get_text(
        separator="\n",
        strip=True,
    )

    text = clean_text(text)

    return [
        ParsedPage(
            page_number=None,
            text=text,
        )
    ]


def parse_document(file_path: Path) -> list[ParsedPage]:
    extension = file_path.suffix.lower()

    if extension == ".txt":
        return parse_txt(file_path)

    if extension == ".pdf":
        return parse_pdf(file_path)

    if extension == ".docx":
        return parse_docx(file_path)

    if extension == ".html":
        return parse_html(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )