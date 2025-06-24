# src/autoresearcher/data/chunking.py
"""
Document chunking strategies for optimal embedding and retrieval
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk"""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    chunk_type: str  # 'title', 'abstract', 'section', 'paragraph'
    metadata: Dict[str, Any]
    char_start: int
    char_end: int

    def to_dict(self) -> dict:
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


class DocumentChunker:
    """
    Chunk documents for embedding and retrieval

    TODO: For production:
    - Integrate with sentence-transformers for semantic chunking
    - Add support for sliding window with overlap
    - Implement smart chunking based on document structure
    """

    def __init__(
        self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # TODO: When using real tokenizer:
        # from transformers import AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a document into smaller pieces

        Strategy:
        1. Title as separate chunk
        2. Abstract as separate chunk(s)
        3. Each section chunked by paragraphs
        4. Long paragraphs split further
        """
        chunks = []
        chunk_index = 0
        char_position = 0

        doc_id = document["id"]

        # Chunk 1: Title (always separate)
        if document.get("title"):
            title_chunk = self._create_chunk(
                doc_id=doc_id,
                text=document["title"],
                chunk_index=chunk_index,
                chunk_type="title",
                char_start=char_position,
                metadata={"section": "title"},
            )
            chunks.append(title_chunk)
            chunk_index += 1
            char_position += len(document["title"])

        # Chunk 2: Abstract
        if document.get("abstract"):
            abstract_chunks = self._chunk_text(
                text=document["abstract"],
                doc_id=doc_id,
                chunk_type="abstract",
                chunk_index_start=chunk_index,
                char_position_start=char_position,
                metadata={"section": "abstract"},
            )
            chunks.extend(abstract_chunks)
            chunk_index += len(abstract_chunks)
            char_position += len(document["abstract"])

        # Chunk 3: Body sections
        if document.get("sections"):
            for section in document["sections"]:
                section_chunks = self._chunk_section(
                    section=section,
                    doc_id=doc_id,
                    chunk_index_start=chunk_index,
                    char_position_start=char_position,
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)

                # Update character position
                section_text = self._get_section_text(section)
                char_position += len(section_text)

        # Add document-level metadata to all chunks
        doc_metadata = document.get("metadata", {})
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "source": document.get("source", "unknown"),
                    "has_full_text": document.get("has_full_text", False),
                    "authors": doc_metadata.get("authors", []),
                    "publication_date": doc_metadata.get("publication_date"),
                }
            )

        logger.info(f"Document {doc_id} chunked into {len(chunks)} pieces")
        return chunks

    def _chunk_section(
        self,
        section: Dict[str, Any],
        doc_id: str,
        chunk_index_start: int,
        char_position_start: int,
    ) -> List[Chunk]:
        """Chunk a document section"""
        chunks = []
        chunk_index = chunk_index_start
        char_position = char_position_start

        section_title = section.get("title", "Unknown Section")

        # Process paragraphs
        paragraphs = section.get("paragraphs", [])
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]

        for para in paragraphs:
            if not para or len(para.strip()) < self.min_chunk_size:
                continue

            # Check if paragraph needs splitting
            if len(para) > self.chunk_size:
                # Split large paragraphs
                para_chunks = self._chunk_text(
                    text=para,
                    doc_id=doc_id,
                    chunk_type="paragraph",
                    chunk_index_start=chunk_index,
                    char_position_start=char_position,
                    metadata={"section": section_title},
                )
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)
            else:
                # Keep paragraph as single chunk
                chunk = self._create_chunk(
                    doc_id=doc_id,
                    text=para,
                    chunk_index=chunk_index,
                    chunk_type="paragraph",
                    char_start=char_position,
                    metadata={"section": section_title},
                )
                chunks.append(chunk)
                chunk_index += 1

            char_position += len(para)

        # Process subsections recursively
        for subsection in section.get("subsections", []):
            subsection_chunks = self._chunk_section(
                section=subsection,
                doc_id=doc_id,
                chunk_index_start=chunk_index,
                char_position_start=char_position,
            )
            chunks.extend(subsection_chunks)
            chunk_index += len(subsection_chunks)
            char_position += len(self._get_section_text(subsection))

        return chunks

    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        chunk_type: str,
        chunk_index_start: int,
        char_position_start: int,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Split text into chunks with overlap

        TODO: For production:
        - Use actual tokenizer for accurate splitting
        - Implement sentence boundary detection
        - Add semantic coherence checking
        """
        chunks = []
        chunk_index = chunk_index_start

        # Simple character-based chunking for now
        # TODO: Replace with tokenizer-based chunking
        sentences = self._split_into_sentences(text)

        current_chunk = []
        current_length = 0
        char_start = char_position_start

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk = self._create_chunk(
                    doc_id=doc_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    chunk_type=chunk_type,
                    char_start=char_start,
                    metadata=metadata.copy(),
                )
                chunks.append(chunk)
                chunk_index += 1

                # Handle overlap
                if self.chunk_overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = current_chunk[-2:]  # Keep last 2 sentences
                    current_chunk = overlap_sentences
                    current_length = sum(len(s) for s in overlap_sentences)
                    char_start += len(chunk_text) - current_length
                else:
                    current_chunk = []
                    current_length = 0
                    char_start += len(chunk_text)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = self._create_chunk(
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=chunk_index,
                chunk_type=chunk_type,
                char_start=char_start,
                metadata=metadata.copy(),
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        TODO: For production:
        - Use spaCy or NLTK for better sentence splitting
        - Handle abbreviations and special cases in medical text
        """
        # Simple regex-based splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_chunk(
        self,
        doc_id: str,
        text: str,
        chunk_index: int,
        chunk_type: str,
        char_start: int,
        metadata: Dict[str, Any],
    ) -> Chunk:
        """Create a chunk object"""
        chunk_id = f"{doc_id}_chunk_{chunk_index}"

        return Chunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            text=text.strip(),
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            metadata=metadata,
            char_start=char_start,
            char_end=char_start + len(text),
        )

    def _get_section_text(self, section: Dict[str, Any]) -> str:
        """Get full text from a section"""
        texts = []

        # Add title
        if section.get("title"):
            texts.append(section["title"])

        # Add paragraphs
        paragraphs = section.get("paragraphs", [])
        if isinstance(paragraphs, str):
            texts.append(paragraphs)
        else:
            texts.extend(paragraphs)

        # Add subsections
        for subsection in section.get("subsections", []):
            texts.append(self._get_section_text(subsection))

        return " ".join(texts)


# Convenience function for testing
def test_chunking():
    """Test the chunking logic"""
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)

    # Test document
    test_doc = {
        "id": "PMC123456",
        "title": "Study of COVID-19 Treatment Efficacy",
        "abstract": "This study examines various treatment options for COVID-19. " * 10,
        "sections": [
            {
                "title": "Introduction",
                "paragraphs": [
                    "COVID-19 is a respiratory disease. " * 20,
                    "Treatment options are being explored. " * 15,
                ],
            },
            {
                "title": "Methods",
                "paragraphs": ["We conducted a randomized controlled trial. " * 25],
            },
        ],
        "metadata": {"authors": ["Smith J", "Doe A"], "publication_date": "2023-01-01"},
    }

    chunks = chunker.chunk_document(test_doc)

    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks[:5]:  # Show first 5
        print(f"\nChunk {chunk.chunk_index} ({chunk.chunk_type}):")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Metadata: {chunk.metadata}")


if __name__ == "__main__":
    test_chunking()
