"""
tests/test_knowledge_base.py
-------------------------------
Tests for rag/knowledge_base.py, rag/document_loader.py, rag/chunker.py (Phase 6).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.knowledge_base import DOCUMENTS, get_documents, get_topics
from rag.document_loader import DocumentLoadError, load_documents
from rag.chunker import chunk_documents


def test_corpus_is_small_and_focused():
    """Per the project's instruction to avoid large-scale document scraping."""
    assert 3 <= len(DOCUMENTS) <= 15


def test_every_document_has_required_metadata():
    for doc in DOCUMENTS:
        for field in ("doc_id", "title", "topic", "source", "source_url", "text"):
            assert field in doc


def test_doc_ids_are_unique():
    ids = [doc["doc_id"] for doc in DOCUMENTS]
    assert len(ids) == len(set(ids))


def test_no_document_claims_an_unverified_specific_source_url():
    """Per the project's source policy: no fabricated citation URLs."""
    for doc in DOCUMENTS:
        assert doc["source_url"] is None


def test_source_field_is_honestly_labeled_as_project_authored():
    for doc in DOCUMENTS:
        assert "project-authored" in doc["source"].lower()


def test_topics_cover_required_areas():
    """Per the project spec's required topic list."""
    topics_text = " ".join(get_topics()).lower()
    required_keywords = [
        "hvac", "occupancy", "lighting", "conservation", "appliance",
        "comfort", "tariff",
    ]
    for keyword in required_keywords:
        assert keyword in topics_text


def test_get_documents_returns_a_defensive_copy():
    docs1 = get_documents()
    docs1[0]["title"] = "MUTATED"
    docs2 = get_documents()
    assert docs2[0]["title"] != "MUTATED"


# ---------------------------------------------------------------------------
# Document loader
# ---------------------------------------------------------------------------
def test_load_documents_succeeds_on_valid_corpus():
    docs = load_documents()
    assert len(docs) == len(DOCUMENTS)


def test_load_documents_raises_on_missing_field(monkeypatch):
    def broken_get_documents():
        return [{"doc_id": "x", "title": "t"}]  # missing several required fields
    monkeypatch.setattr("rag.document_loader.get_documents", broken_get_documents)
    with pytest.raises(DocumentLoadError):
        load_documents()


def test_load_documents_raises_on_duplicate_doc_id(monkeypatch):
    dupe = dict(DOCUMENTS[0])
    def broken_get_documents():
        return [dict(DOCUMENTS[0]), dupe]
    monkeypatch.setattr("rag.document_loader.get_documents", broken_get_documents)
    with pytest.raises(DocumentLoadError, match="Duplicate"):
        load_documents()


def test_load_documents_raises_on_empty_text(monkeypatch):
    broken_doc = dict(DOCUMENTS[0])
    broken_doc["text"] = "   "
    def broken_get_documents():
        return [broken_doc]
    monkeypatch.setattr("rag.document_loader.get_documents", broken_get_documents)
    with pytest.raises(DocumentLoadError, match="empty"):
        load_documents()


def test_load_documents_raises_on_empty_corpus(monkeypatch):
    monkeypatch.setattr("rag.document_loader.get_documents", lambda: [])
    with pytest.raises(DocumentLoadError, match="empty"):
        load_documents()


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def test_chunk_documents_produces_chunks_for_every_document():
    docs = load_documents()
    chunks = chunk_documents(docs)
    chunked_doc_ids = {c["doc_id"] for c in chunks}
    assert chunked_doc_ids == {d["doc_id"] for d in docs}


def test_chunk_ids_are_unique():
    docs = load_documents()
    chunks = chunk_documents(docs)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_respect_max_size():
    docs = load_documents()
    chunks = chunk_documents(docs, chunk_size=100, overlap=20)
    for c in chunks:
        assert len(c["text"]) <= 100


def test_chunks_carry_full_traceability_metadata():
    docs = load_documents()
    chunks = chunk_documents(docs)
    for c in chunks:
        assert c["title"]
        assert c["topic"]
        assert c["source"]
        assert c["text"]


def test_small_chunk_size_produces_more_chunks_than_large_chunk_size():
    docs = load_documents()
    small_chunks = chunk_documents(docs, chunk_size=100, overlap=20)
    large_chunks = chunk_documents(docs, chunk_size=2000, overlap=60)
    assert len(small_chunks) > len(large_chunks)


def test_chunking_empty_document_list_returns_empty():
    assert chunk_documents([]) == []
