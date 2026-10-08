from enum import Enum


class DocumentMode(str, Enum):
    SINGLE = "single"
    SELECTED = "selected"
    ALL = "all"