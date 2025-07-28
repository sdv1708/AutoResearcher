"""
Smart chunker – turns a parsed Document into overlapping text chunks.
"""

from __future__ import annotations

from typing import List

from autoresearcher.core.models import Chunk, Document


DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 50


def chunk_document(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0

    # --- helper to slice long strings ----------------------------------
    def _window(text: str, ctype: str) -> None:
        nonlocal idx
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=Chunk.new_id(doc.id, idx),
                    document_id=doc.id,
                    text=chunk_text,
                    chunk_index=idx,
                    chunk_type=ctype,
                    metadata={"source": doc.source, "section": ctype},
                    char_start=start,
                    char_end=end,
                )
            )
            idx += 1
            start = max(end - overlap, end)

    # title chunk
    if doc.title:
        _window(doc.title, "title")
    # abstract chunk
    if doc.abstract:
        _window(doc.abstract, "abstract")
    # body chunks by section / paragraph
    for s in doc.sections:
        for para in s.paragraphs:
            _window(para, "body")

    return chunks