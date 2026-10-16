"""
tests/test_rag.py
---------------------
Tests for rag/embeddings.py, rag/retriever.py, rag/rag_pipeline.py (Phase 7).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import chunk_documents
from rag.document_loader import load_documents
from rag.embeddings import build_index
from rag.retriever import retrieve_relevant_chunks, get_index
from rag.rag_pipeline import _build_prompt, answer_with_rag


@pytest.fixture(scope="module")
def index():
    docs = load_documents()
    chunks = chunk_documents(docs)
    return build_index(chunks)


# ---------------------------------------------------------------------------
# Embeddings / index
# ---------------------------------------------------------------------------
def test_index_builds_with_a_real_mode(index):
    assert index.mode in ("sentence_transformers+faiss", "tfidf+faiss")


def test_empty_chunks_produce_empty_index():
    empty_index = build_index([])
    assert empty_index.search("anything") == []


def test_search_returns_scored_results(index):
    results = index.search("reduce HVAC energy usage", top_k=3)
    assert len(results) > 0
    for r in results:
        assert "score" in r
        assert "text" in r
        assert "title" in r


def test_search_results_are_sorted_by_relevance(index):
    results = index.search("occupancy-based lighting control", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
def test_retrieve_relevant_chunks_for_on_topic_query():
    results = retrieve_relevant_chunks("Why should HVAC be reduced when a room is unoccupied?")
    assert len(results) > 0
    assert any("HVAC" in r["title"] or "Occupancy" in r["title"] for r in results)


def test_retrieve_relevant_chunks_filters_irrelevant_query():
    """An off-topic query sharing only common stopwords with the corpus
    should return no results (caught and fixed during development -- see
    rag/embeddings.py's stopword filtering)."""
    results = retrieve_relevant_chunks("What is the capital of France?")
    assert results == []


def test_retrieve_relevant_chunks_empty_query_returns_empty():
    assert retrieve_relevant_chunks("") == []
    assert retrieve_relevant_chunks("   ") == []


def test_get_index_is_cached():
    idx1 = get_index()
    idx2 = get_index()
    assert idx1 is idx2  # same cached instance, not rebuilt


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------
def test_answer_with_rag_on_topic_question_is_grounded():
    result = answer_with_rag("Why should lighting be turned off in unoccupied rooms?")
    assert result["grounded"] is True
    assert len(result["sources"]) > 0
    assert result["mode"] in ("null", "ollama", "openai", "fallback")


def test_answer_with_rag_off_topic_question_reports_no_evidence():
    result = answer_with_rag("What is the capital of France?")
    assert result["mode"] == "no_evidence"
    assert result["grounded"] is False
    assert result["sources"] == []


def test_answer_with_rag_empty_question():
    result = answer_with_rag("")
    assert result["mode"] == "no_evidence"


def test_answer_with_rag_never_raises_on_malformed_input():
    # Should not raise even for unusual input.
    result = answer_with_rag("???!!!")
    assert "answer" in result


def test_null_provider_response_is_explicitly_labeled_as_not_llm_generated():
    """With the default (no LLM configured) provider, the answer must be
    honest about not being LLM-generated text."""
    result = answer_with_rag("Why should HVAC be reduced when a room is unoccupied?")
    if result["mode"] == "null":
        assert "not LLM-generated" in result["answer"] or "No LLM provider is configured" in result["answer"]


def test_build_prompt_includes_system_context_when_provided():
    """Verify SYSTEM RESULTS is actually threaded into the prompt sent to
    the provider (important for the AI Assistant page in Phase 9)."""
    chunks = [{"title": "Lighting Energy Management", "text": "Turn off lighting when unoccupied."}]
    prompt = _build_prompt(
        "Why is this flagged?", chunks,
        system_context="Room 101: Unoccupied, Lighting=On, potential wastage detected.",
    )
    assert "SYSTEM RESULTS" in prompt
    assert "Room 101: Unoccupied, Lighting=On" in prompt
    assert "RETRIEVED KNOWLEDGE" in prompt
    assert "USER QUESTION" in prompt


def test_build_prompt_omits_system_results_section_when_not_provided():
    chunks = [{"title": "X", "text": "Y"}]
    prompt = _build_prompt("A question", chunks, system_context=None)
    assert "SYSTEM RESULTS" not in prompt


def test_answer_with_rag_runs_successfully_with_system_context():
    result = answer_with_rag(
        "Why should lighting be turned off in unoccupied rooms?",
        system_context="Room 101: Unoccupied, Lighting=On, potential wastage detected.",
    )
    assert result["sources"]
    assert result["grounded"] is True
