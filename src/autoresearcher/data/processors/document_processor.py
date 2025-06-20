"""
================================================================================
  File Name      : document_processor.py
  Created On     : 2025-06-20
  Created By     : Pranav Rajesh Charakondala
  Last Modified  : 2025-06-20 (update as needed)
  Description    :
    This module defines the DocumentProcessor class, which provides functionality
    to clean, tokenize, and chunk biomedical text (title + abstract) into 
    manageable sections for embedding and retrieval tasks.

  Functionality:
    - Clean input text (remove noise, normalize casing)
    - Tokenize text using Hugging Face Transformers tokenizer
    - Generate 512-token chunks to be used for embeddings

  Modification Log:
    - 2025-06-20: File created with cleaning and chunking logic
    - 2025-06-20: Added default handling and passthrough for vernacular title, language, publication types, MeSH terms, chemicals, author names and DOIs.
================================================================================
"""
# src/autoresearcher/data/processors/document_processor.py
from typing import Dict, List, Any
import re
from transformers import AutoTokenizer

class DocumentProcessor:
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def process_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        doc["title_clean"] = self._clean_text(doc["title"])
        doc["abstract_clean"] = self._clean_text(doc["abstract"])
        doc["full_text"] = f"{doc['title_clean']} {doc['abstract_clean']}"
        doc["chunks"] = self._chunk_text(doc["full_text"])

        doc["language"] = doc.get("language", "eng")  # fallback to English
        doc["vernacular_title"] = doc.get("vernacular_title", "")
        doc["publication_types"] = doc.get("publication_types", [])
        doc["mesh_terms"] = doc.get("mesh_terms", [])
        doc["chemicals"] = doc.get("chemicals", [])
        doc["authors"] = doc.get("authors", [])
        doc["doi"] = doc.get("doi", "")
        return doc

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s.-]', ' ', text or "")
        return ' '.join(text.split()).lower()

    def _chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        tokens = self.tokenizer.encode(text, truncation=False)
        return [
            self.tokenizer.decode(tokens[i:i + max_length], skip_special_tokens=True)
            for i in range(0, len(tokens), max_length)
        ]
