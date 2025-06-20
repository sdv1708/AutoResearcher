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
    - 2025-06-20: Added support for parsing vernacular title, language, publication types, MeSH terms, chemical substances, author names and DOIs.
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

          # Optional fields
          vernacular_title_elem = article.find('.//VernacularTitle')
          language_elem = article.find('.//Language')
          pub_type_elems = article.findall('.//PublicationTypeList/PublicationType')
          mesh_heading_elems = article.findall('.//MeshHeadingList/MeshHeading')
          chemical_elems = article.findall('.//ChemicalList/Chemical')
          author_elems = article.findall('.//AuthorList/Author')
          article_id_elems = article.findall('.//ArticleIdList/ArticleId')

          # Base fields
          pmid = pmid_elem.text if pmid_elem is not None else ""
          title = title_elem.text if title_elem is not None else ""
          abstract = abstract_elem.text if abstract_elem is not None else ""

          # Enriched fields
          vernacular_title = vernacular_title_elem.text if vernacular_title_elem is not None else ""
          language = language_elem.text if language_elem is not None else "eng"
          publication_types = [elem.text for elem in pub_type_elems if elem.text]
          mesh_terms = [d.text for d in (elem.find('DescriptorName') for elem in mesh_heading_elems) if d is not None and d.text]
          chemicals = [s.text for s in (elem.find('NameOfSubstance') for elem in chemical_elems) if s is not None and s.text]

          # New: Authors (concatenate LastName + Initials if present)
          authors = []
          for a in author_elems:
              last = a.findtext('LastName') or ""
              initials = a.findtext('Initials') or ""
              full_name = f"{last} {initials}".strip()
              if full_name:
                  authors.append(full_name)

          # New: DOI
          doi = ""
          for aid in article_id_elems:
              if aid.attrib.get("IdType", "").lower() == "doi" and aid.text:
                  doi = aid.text
                  break

          yield {
              "id": pmid,
              "title": title,
              "abstract": abstract,
              "source": "pubmed",
              "vernacular_title": vernacular_title,
              "language": language,
              "publication_types": publication_types,
              "mesh_terms": mesh_terms,
              "chemicals": chemicals,
              "authors": authors,
              "doi": doi
          }


