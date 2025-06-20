"""
================================================================================
  File Name      : pubmed_loader.py
  Created On     : 2025-06-20
  Created By     : Pranav Rajesh Charakondala
  Last Modified  : YYYY-MM-DD (update as needed)
  Description    : 
    This module defines the PubMedLoader class responsible for loading and 
    parsing biomedical abstracts from PubMed XML files (downloaded locally or 
    accessed via GCS). It extracts article metadata such as PMID, title, and 
    abstract and yields structured Python dictionaries for further processing.

  Functionality:
    - Connect to GCS and list available PubMed XML files
    - Parse XML content and extract relevant biomedical metadata
    - Yield documents in a consistent format for downstream processing

  Modification Log:
    - 2025-06-20: File created with XML parsing and GCS integration logic
================================================================================
"""

# src/autoresearcher/data/loaders/pubmed_loader.py
import xml.etree.ElementTree as ET
from typing import Iterator, Dict, Any
from google.cloud import storage
import structlog

logger = structlog.get_logger()

class PubMedLoader:
    def __init__(self, bucket_name: str, client: storage.Client):
        self.bucket = client.bucket(bucket_name)

    def load_abstracts(self, prefix: str = "baseline/") -> Iterator[Dict[str, Any]]:
        blobs = self.bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith('.xml'):
                logger.info("processing_file", file=blob.name)
                content = blob.download_as_text()
                yield from self._parse_xml(content)

    def _parse_xml(self, content: str) -> Iterator[Dict[str, Any]]:
        root = ET.fromstring(content)
        for article in root.findall('.//PubmedArticle'):
            pmid_elem = article.find('.//PMID')
            title_elem = article.find('.//ArticleTitle')
            abstract_elem = article.find('.//AbstractText')

            pmid = pmid_elem.text if pmid_elem is not None else ""
            title = title_elem.text if title_elem is not None else ""
            abstract = abstract_elem.text if abstract_elem is not None else ""
            yield {
                "id": pmid,
                "title": title,
                "abstract": abstract,
                "source": "pubmed"
            }
