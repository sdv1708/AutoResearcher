"""
CORD19Loader – parses a CORD‑19 single‑paper JSON (document_parses) file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from autoresearcher.core.models import Document, Section


class CORD19Loader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ───────────────────────── Public API ─────────────────────────
    def load(self) -> Document:
        with self.path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        paper_id = data.get("paper_id") or self.path.stem
        metadata = data.get("metadata", {})
        title = metadata.get("title", "Untitled")

        # abstract = list[dict{text, section}]
        abstract_parts = data.get("abstract", [])
        abstract_txt = " ".join(p["text"].strip() for p in abstract_parts)

        sections = self._parse_body(data.get("body_text", []))

        return Document(
            id=paper_id,
            title=title,
            abstract=abstract_txt,
            sections=sections,
            source="cord19",
            has_full_text=bool(sections),
            metadata={
                "doi": metadata.get("doi"),
                "authors": [a["first"] + " " + a["last"] for a in metadata.get("authors", [])],
                "publish_time": metadata.get("publish_time"),
                "journal": metadata.get("journal"),
            },
        )

    # ─────────────────────── private helpers ──────────────────────
    def _parse_body(self, body_sections) -> List[Section]:
        sections: List[Section] = []
        for sec in body_sections:
            title = sec.get("section")
            paragraphs = [sec.get("text", "").strip()]
            sections.append(Section(title=title, paragraphs=paragraphs))
        return sections