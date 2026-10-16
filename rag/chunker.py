"""
rag/chunker.py
-----------------
Purpose : Split loaded documents into retrieval-sized text chunks, each
          carrying full traceability metadata back to its source document.
Input   : List of document dicts (from rag.document_loader.load_documents()).
Output  : List of chunk dicts.
AI technique used : None -- deterministic fixed-size character chunking with
                     overlap. Simple and easy to explain in a review; the
                     corpus here is small enough that elaborate chunking
                     strategies are not warranted.
"""

from typing import Dict, List

CHUNK_SIZE = 400     # characters
CHUNK_OVERLAP = 60   # characters


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fixed-size character chunking with overlap, on whitespace-normalized text."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_documents(documents: List[Dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Chunk every document into retrieval-sized pieces.

    Output: list of chunk dicts, each with:
        chunk_id (globally unique, "<doc_id>::<index>"), doc_id, title,
        topic, source, source_url, text (the chunk's own text).
    """
    all_chunks: List[Dict] = []
    for doc in documents:
        pieces = _split_text(doc["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}::{i}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "topic": doc["topic"],
                "source": doc["source"],
                "source_url": doc["source_url"],
                "text": piece,
            })
    return all_chunks
