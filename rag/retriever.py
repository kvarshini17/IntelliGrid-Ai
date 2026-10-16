"""
rag/retriever.py
-------------------
Purpose : Cached, simple interface for retrieving relevant knowledge-base
          chunks for a query (Phase 7).
Input   : User query string.
Output  : List of relevant chunk dicts with similarity scores, or an empty
          list if nothing sufficiently relevant was found.
"""

import logging
from typing import Dict, List, Optional

import config
from rag.chunker import chunk_documents
from rag.document_loader import load_documents
from rag.embeddings import EmbeddingIndex, build_index

logger = logging.getLogger(__name__)

_INDEX_CACHE: Optional[EmbeddingIndex] = None


def get_index(force_rebuild: bool = False) -> EmbeddingIndex:
    """Return a cached EmbeddingIndex, building it once per process (avoids
    re-embedding the knowledge base on every query)."""
    global _INDEX_CACHE
    if _INDEX_CACHE is None or force_rebuild:
        try:
            docs = load_documents()
            chunks = chunk_documents(docs)
            _INDEX_CACHE = build_index(chunks)
            logger.info("Retriever index ready (mode=%s, chunks=%d).", _INDEX_CACHE.mode, len(chunks))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to build retriever index: %s", exc)
            _INDEX_CACHE = EmbeddingIndex([])  # safe empty fallback
    return _INDEX_CACHE


def retrieve_relevant_chunks(query: str, top_k: int = None) -> List[Dict]:
    """
    Retrieve the most relevant knowledge-base chunks for a query, filtered
    by config.RAG_SIMILARITY_THRESHOLD (chunks below this score are dropped
    -- they are not relevant enough to ground an answer).

    Output: list of chunk dicts with 'score', possibly empty. Never raises --
    retrieval failure must not crash the calling application (Phase 8/9).
    """
    if not query or not query.strip():
        return []
    top_k = top_k or config.RAG_TOP_K
    try:
        index = get_index()
        results = index.search(query, top_k=top_k)
        return [r for r in results if r["score"] >= config.RAG_SIMILARITY_THRESHOLD]
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval failed for query %r: %s", query, exc)
        return []
