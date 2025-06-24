# src/autoresearcher/data/loaders/pmc_loader.py
"""
PMC (PubMed Central) Full-Text Article Loader
Handles XML files from PMC Open Access Subset
"""
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class PMCLoader:
    """Load and parse PMC full-text articles"""

    def __init__(self, data_dir: str = "./data/pmc"):
        self.data_dir = Path(data_dir)
        # TODO: When moving to GCS, replace with:
        # self.bucket_name = "your-project-pmc-data"
        # self.storage_client = storage.Client()

    def load_articles(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Load PMC articles from local directory

        TODO: For production:
        - Replace file iteration with GCS blob listing
        - Add parallel processing for large datasets
        - Implement checkpointing for resume capability
        """
        xml_files = (
            list(self.data_dir.glob("*.xml"))[:limit]
            if limit
            else list(self.data_dir.glob("*.xml"))
        )

        for xml_file in xml_files:
            try:
                article = self._parse_pmc_xml(xml_file)
                if article:
                    yield article
            except Exception as e:
                logger.error(f"Error parsing {xml_file}: {e}")
                continue

    def _parse_pmc_xml(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a single PMC XML file"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract PMC ID
            pmc_id = self._extract_text(root, './/article-id[@pub-id-type="pmc"]')
            if not pmc_id:
                pmc_id = file_path.stem  # Use filename as fallback

            # Extract title
            title = self._extract_text(root, ".//article-title")

            # Extract abstract with sections
            abstract_data = self._extract_abstract(root)

            # Extract full text sections
            sections = self._extract_body_sections(root)

            # Extract metadata
            metadata = self._extract_metadata(root)

            return {
                "id": f"PMC{pmc_id}",
                "title": title,
                "abstract": abstract_data["text"],
                "abstract_sections": abstract_data["sections"],
                "sections": sections,
                "source": "pmc",
                "has_full_text": bool(sections),
                "metadata": metadata,
                # TODO: Add these fields when integrating with cloud:
                # "gcs_path": f"gs://{self.bucket_name}/processed/{pmc_id}.json",
                # "processed_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to parse PMC XML {file_path}: {e}")
            return None

    def _extract_abstract(self, root) -> Dict[str, Any]:
        """Extract structured abstract"""
        abstract_elem = root.find(".//abstract")
        if abstract_elem is None:
            return {"text": "", "sections": []}

        sections = []
        full_text_parts = []

        # Handle structured abstracts (with labeled sections)
        for sec in abstract_elem.findall(".//sec"):
            label = sec.get("sec-type", "") or self._extract_text(sec, "./title")
            text = " ".join([p.text or "" for p in sec.findall(".//p") if p.text])

            if label and text:
                sections.append({"label": label, "text": text})
                full_text_parts.append(f"{label}: {text}")

        # Handle simple abstracts (just paragraphs)
        if not sections:
            for p in abstract_elem.findall(".//p"):
                text = self._get_element_text(p)
                if text:
                    full_text_parts.append(text)

        return {"text": " ".join(full_text_parts), "sections": sections}

    def _extract_body_sections(self, root) -> list:
        """Extract main body sections"""
        sections = []
        body = root.find(".//body")

        if body is not None:
            for sec in body.findall(".//sec"):
                section_data = self._parse_section(sec)
                if section_data:
                    sections.append(section_data)

        return sections

    def _parse_section(self, sec_elem) -> Optional[Dict[str, Any]]:
        """Parse a single section recursively"""
        title = self._extract_text(sec_elem, "./title")

        # Get all paragraphs in this section
        paragraphs = []
        for p in sec_elem.findall("./p"):
            text = self._get_element_text(p)
            if text:
                paragraphs.append(text)

        # Get subsections
        subsections = []
        for subsec in sec_elem.findall("./sec"):
            subsec_data = self._parse_section(subsec)
            if subsec_data:
                subsections.append(subsec_data)

        if title or paragraphs or subsections:
            return {
                "title": title or "Untitled Section",
                "paragraphs": paragraphs,
                "subsections": subsections,
            }

        return None

    def _extract_metadata(self, root) -> Dict[str, Any]:
        """Extract article metadata"""
        metadata = {}

        # Authors
        authors = []
        for author in root.findall('.//contrib[@contrib-type="author"]'):
            name_elem = author.find(".//name")
            if name_elem is not None:
                surname = self._extract_text(name_elem, "./surname")
                given_names = self._extract_text(name_elem, "./given-names")
                if surname:
                    full_name = f"{given_names} {surname}" if given_names else surname
                    authors.append(full_name)
        metadata["authors"] = authors

        # Publication date
        pub_date = self._extract_publication_date(root)
        if pub_date:
            metadata["publication_date"] = pub_date

        # Journal
        journal = self._extract_text(root, ".//journal-title")
        if journal:
            metadata["journal"] = journal

        # Keywords
        keywords = []
        for kwd in root.findall(".//kwd"):
            if kwd.text:
                keywords.append(kwd.text)
        if keywords:
            metadata["keywords"] = keywords

        # TODO: Add when implementing cloud version:
        # metadata['download_source'] = 'pmc-oa-bulk'
        # metadata['processing_version'] = '1.0'

        return metadata

    def _extract_publication_date(self, root) -> Optional[str]:
        """Extract and format publication date"""
        # Try different date locations
        date_elem = root.find('.//pub-date[@date-type="pmc-release"]')
        if date_elem is None:
            date_elem = root.find('.//pub-date[@pub-type="epub"]')
        if date_elem is None:
            date_elem = root.find(".//pub-date")

        if date_elem is not None:
            year = self._extract_text(date_elem, "./year")
            month = self._extract_text(date_elem, "./month")
            day = self._extract_text(date_elem, "./day")

            if year:
                try:
                    year = int(year)
                    month = int(month) if month else 1
                    day = int(day) if day else 1
                    return datetime(year, month, day).isoformat()
                except:
                    return None

        return None

    def _extract_text(self, elem, xpath: str) -> str:
        """Safely extract text from element"""
        found = elem.find(xpath)
        return found.text.strip() if found is not None and found.text else ""

    def _get_element_text(self, elem) -> str:
        """Get all text from element, including nested tags"""
        # This handles mixed content like: <p>This is <italic>important</italic> text</p>
        return "".join(elem.itertext()).strip()
