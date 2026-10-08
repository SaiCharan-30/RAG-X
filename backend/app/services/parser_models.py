from dataclasses import dataclass


@dataclass
class ParsedPage:
    page_number: int | None
    text: str
    section: str | None = None