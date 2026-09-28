"""Small shared domain types used by ingestion and retrieval."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)