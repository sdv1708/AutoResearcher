"""
PMCXMLLoader – parses a single NCBI PMC full‑text XML file.

Only a subset of tags is handled; anything not captured is ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import lxml.etree as ET

from autoresearcher.core.models import Document, Section


class PMCXMLLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ───────────────────────── Public API ─────────────────────────
    def load(self) -> Document:
        tree = ET.parse(self.path)
        root = tree.getroot()

        article_meta = root.find(".//article-meta")
        title = (
            article_meta.findtext("title-group/article-title", default="").strip()
            or "Untitled"
        )

        # abstract may be split into <sec> chunks
        abstract_el = article_meta.find("./abstract")
        abstract_text = self._concat_text(abstract_el) if abstract_el is not None else ""

        pub_date = article_meta.findtext("./pub-date/year")
        keywords = [kw.text for kw in article_meta.findall("./kwd-group/kwd")]

        sections = self._parse_sections(root.findall(".//body/sec"))

        doc_id = self._find_doc_id(root)

        return Document(
            id=doc_id,
            title=title,
            abstract=abstract_text,
            sections=sections,
            source="pmc",
            has_full_text=bool(sections),
            metadata={
                "publication_date": pub_date,
                "keywords": keywords,
            },
        )

    # ─────────────────────── private helpers ──────────────────────
    def _concat_text(self, element) -> str:
        return " ".join(element.itertext()).strip()

    def _parse_sections(self, sec_elements) -> List[Section]:
        sections: List[Section] = []
        for sec in sec_elements:
            title = sec.findtext("title")
            paragraphs = [self._concat_text(p) for p in sec.findall("p")]
            subsecs = self._parse_sections(sec.findall("sec"))
            sections.append(Section(title=title, paragraphs=paragraphs, subsections=subsecs))
        return sections

    def _find_doc_id(self, root) -> str:
        pmcid = root.findtext(".//article-id[@pub-id-type='pmcid']")
        if pmcid:
            return pmcid
        doi = root.findtext(".//article-id[@pub-id-type='doi']")
        if doi:
            return doi
        return self.path.stem  # fallback