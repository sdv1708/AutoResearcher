# src/autoresearcher/data/loaders/cord19_loader.py
"""
CORD-19 Dataset Loader
Handles JSON files from COVID-19 Open Research Dataset
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class CORD19Loader:
    """Load and parse CORD-19 research papers"""

    def __init__(self, data_dir: str = "./data/cord19"):
        self.data_dir = Path(data_dir)
        # TODO: For cloud deployment:
        # self.bucket_name = "your-project-cord19-data"
        # self.storage_client = storage.Client()

    def load_papers(
        self, subset: str = "document_parses", limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Load CORD-19 papers from specified subset

        Args:
            subset: One of 'document_parses', 'metadata', etc.
            limit: Maximum number of papers to load

        TODO: For production:
        - Stream directly from GCS without downloading
        - Add support for loading from multiple subsets
        - Implement parallel processing with Cloud Dataflow
        """
        subset_dir = self.data_dir / subset
        if not subset_dir.exists():
            logger.warning(f"Subset directory {subset_dir} not found")
            return

        json_files = list(subset_dir.glob("**/*.json"))
        if limit:
            json_files = json_files[:limit]

        for json_file in json_files:
            try:
                paper = self._parse_cord19_json(json_file)
                if paper:
                    yield paper
            except Exception as e:
                logger.error(f"Error parsing {json_file}: {e}")
                continue

    def load_metadata(
        self, metadata_file: str = "metadata.csv"
    ) -> Iterator[Dict[str, Any]]:
        """
        Load paper metadata from CSV

        TODO: For production:
        - Use BigQuery for metadata storage and querying
        - Implement incremental updates
        """
        import csv

        metadata_path = self.data_dir / metadata_file
        if not metadata_path.exists():
            logger.warning(f"Metadata file {metadata_path} not found")
            return

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield self._parse_metadata_row(row)

    def _parse_cord19_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a single CORD-19 JSON file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract paper ID (SHA or PMC ID)
            paper_id = data.get("paper_id", "") or data.get("doc_id", "")
            if not paper_id:
                paper_id = file_path.stem

            # Extract title from metadata
            title = data.get("metadata", {}).get("title", "")

            # Extract abstract
            abstract_text = self._extract_abstract_text(data.get("abstract", []))

            # Extract body sections
            body_sections = self._extract_body_sections(data.get("body_text", []))

            # Extract metadata
            metadata = self._extract_paper_metadata(data)

            # Extract references
            references = self._extract_references(data.get("bib_entries", {}))

            return {
                "id": f"CORD19_{paper_id[:12]}",  # Truncate long SHAs
                "title": title,
                "abstract": abstract_text,
                "sections": body_sections,
                "source": "cord19",
                "has_full_text": bool(body_sections),
                "metadata": metadata,
                "references": references,
                # TODO: Add for cloud version:
                # "embedding_status": "pending",
                # "vector_index_version": None,
                # "processing_pipeline": "v1.0"
            }

        except Exception as e:
            logger.error(f"Failed to parse CORD-19 JSON {file_path}: {e}")
            return None

    def _extract_abstract_text(self, abstract_sections: list) -> str:
        """Extract and combine abstract sections"""
        if not abstract_sections:
            return ""

        abstract_parts = []
        for section in abstract_sections:
            text = section.get("text", "").strip()
            if text:
                # Include section name if available
                section_name = section.get("section", "").strip()
                if section_name and section_name.lower() not in ["abstract", "unknown"]:
                    abstract_parts.append(f"{section_name}: {text}")
                else:
                    abstract_parts.append(text)

        return " ".join(abstract_parts)

    def _extract_body_sections(self, body_text: list) -> list:
        """Extract and structure body sections"""
        sections = []
        current_section = None

        for paragraph in body_text:
            section_name = paragraph.get("section", "Unknown")
            text = paragraph.get("text", "").strip()

            if not text:
                continue

            # Check if we need to create a new section
            if current_section is None or current_section["title"] != section_name:
                if current_section:
                    sections.append(current_section)
                current_section = {"title": section_name, "paragraphs": []}

            current_section["paragraphs"].append(text)

        # Don't forget the last section
        if current_section:
            sections.append(current_section)

        return sections

    def _extract_paper_metadata(self, data: dict) -> Dict[str, Any]:
        """Extract comprehensive metadata"""
        metadata = {}
        paper_metadata = data.get("metadata", {})

        # Authors
        authors = []
        for author in paper_metadata.get("authors", []):
            # Handle different author formats
            if isinstance(author, dict):
                first = author.get("first", "")
                middle = " ".join(author.get("middle", []))
                last = author.get("last", "")
                suffix = author.get("suffix", "")

                name_parts = [first, middle, last, suffix]
                full_name = " ".join(part for part in name_parts if part)
                if full_name:
                    authors.append(full_name)
            elif isinstance(author, str):
                authors.append(author)

        metadata["authors"] = authors

        # Other metadata
        if paper_metadata.get("doi"):
            metadata["doi"] = paper_metadata["doi"]

        if paper_metadata.get("pubmed_id"):
            metadata["pubmed_id"] = paper_metadata["pubmed_id"]

        if paper_metadata.get("pmc_id"):
            metadata["pmc_id"] = paper_metadata["pmc_id"]

        # License
        if paper_metadata.get("license"):
            metadata["license"] = paper_metadata["license"]

        # TODO: Add for cloud version:
        # metadata['load_timestamp'] = datetime.utcnow().isoformat()
        # metadata['schema_version'] = '2.0'

        return metadata

    def _extract_references(self, bib_entries: dict) -> list:
        """Extract paper references"""
        references = []

        for ref_id, ref_data in bib_entries.items():
            ref = {
                "ref_id": ref_id,
                "title": ref_data.get("title", ""),
                "venue": ref_data.get("venue", ""),
                "year": ref_data.get("year", 0),
            }

            # Extract authors
            ref_authors = []
            for author in ref_data.get("authors", []):
                if isinstance(author, dict):
                    first = author.get("first", "")
                    last = author.get("last", "")
                    if last:
                        ref_authors.append(f"{first} {last}".strip())

            ref["authors"] = ref_authors
            references.append(ref)

        return references

    def _parse_metadata_row(self, row: dict) -> Dict[str, Any]:
        """Parse a row from metadata.csv"""
        return {
            "id": row.get("cord_uid", ""),
            "title": row.get("title", ""),
            "doi": row.get("doi", ""),
            "pmcid": row.get("pmcid", ""),
            "pubmed_id": row.get("pubmed_id", ""),
            "license": row.get("license", ""),
            "abstract": row.get("abstract", ""),
            "publish_time": row.get("publish_time", ""),
            "authors": row.get("authors", "").split("; ") if row.get("authors") else [],
            "journal": row.get("journal", ""),
            "source": "cord19_metadata",
        }
