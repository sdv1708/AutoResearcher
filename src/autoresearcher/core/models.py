"""
Canonical dataclasses / Pydantic models used across the code‑base.
Keeping them in one module avoids circular imports.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ───────────────────────── Document tree ──────────────────────────
@dataclass
class Section:
    title: Optional[str]
    paragraphs: List[str]
    subsections: List["Section"] = field(default_factory=list)


@dataclass
class Document:
    id: str
    title: str
    abstract: Optional[str]
    sections: List[Section]
    source: str  # "pmc" or "cord19"
    has_full_text: bool
    metadata: Dict[str, Any]


# ───────────────────────── Chunk structure ────────────────────────
@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    chunk_type: str  # "title" | "abstract" | "body"
    metadata: Dict[str, Any]
    char_start: int
    char_end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }

    @staticmethod
    def new_id(doc_id: str, idx: int) -> str:
        # deterministic but unique per chunk
        return f"{doc_id}_chunk_{idx}_{uuid.uuid4().hex[:6]}"