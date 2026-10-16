"""
rag/rag_pipeline.py
----------------------
Purpose : Orchestrate retrieval + LLM-grounded answer generation (Phase 7/8
          boundary). Retrieves relevant knowledge-base chunks, then asks the
          configured LLMProvider (Phase 8) to answer USING ONLY that
          retrieved context -- never inventing sources or measurements.
Input   : User question string.
Output  : dict with answer, mode, sources, grounded flag.

RAG + LLM GROUNDING -- enforced structure:
    The prompt sent to the LLM provider explicitly separates SYSTEM RESULTS
    (not used by this generic knowledge-question pipeline, but supported via
    `system_context` for the AI Assistant page in Phase 9), RETRIEVED
    KNOWLEDGE, and the USER QUESTION, and instructs the model not to invent
    sources, not to invent measurements, and to acknowledge insufficient
    evidence. If retrieval finds nothing above the relevance threshold, the
    LLM is never called with an empty/fabricated context -- a clear
    limitation message is returned directly.
"""

from typing import Dict, List, Optional

from llm.provider import LLMProviderError, get_provider
from rag.retriever import retrieve_relevant_chunks

SYSTEM_PROMPT = (
    "You are an assistant for the IntelliGrid AI smart-building energy project. "
    "Answer the user's question using ONLY the RETRIEVED KNOWLEDGE and, if provided, "
    "the SYSTEM RESULTS sections below. Prioritize the retrieved evidence over any general "
    "knowledge you may have.\n"
    "Rules:\n"
    "1. Do not invent sources. When you rely on a retrieved passage, cite it by its title in "
    "square brackets, exactly as it appears in the RETRIEVED KNOWLEDGE section.\n"
    "2. Do not invent or estimate energy-consumption values, cost savings, energy savings, "
    "percentages or experimental results. Only repeat numbers that appear in the provided text.\n"
    "3. Do not override or contradict the SYSTEM RESULTS -- they come from deterministic "
    "analysis and are authoritative. 'Potential wastage' is a condition-based indicator, not "
    "proven wastage, and an 'anomaly' is a statistical flag, not a confirmed fault; keep that wording.\n"
    "4. Explain recommendations clearly and say why they follow from the evidence. Separate "
    "what the evidence states ('Known from the knowledge base') from anything you are "
    "inferring ('Assumption'), and label each.\n"
    "5. If the retrieved knowledge does not contain enough information to answer, say so "
    "explicitly and state what is missing rather than guessing.\n"
    "Keep answers concise and practical."
)


def _build_prompt(question: str, chunks: List[Dict], system_context: Optional[str]) -> str:
    parts = []
    if system_context:
        parts.append(f"SYSTEM RESULTS (authoritative, from deterministic analysis):\n{system_context}")
    context_text = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)
    parts.append(f"RETRIEVED KNOWLEDGE:\n{context_text}")
    parts.append(f"USER QUESTION:\n{question}")
    return "\n\n".join(parts)


def answer_with_rag(question: str, system_context: Optional[str] = None) -> Dict:
    """
    Answer a question using the RAG pipeline.

    Input:
        question: the user's natural-language question.
        system_context: optional deterministic analysis text (e.g. a
                         specific record's condition/wastage/anomaly/
                         optimization summary) to ground the answer in,
                         used by the AI Assistant page (Phase 9).
    Output:
        {
            "answer": str,
            "mode": "llm" | "fallback" | "no_evidence",
            "sources": [chunk dicts with score],
            "grounded": bool,  # True only when retrieved evidence was used
        }
    Never raises -- any failure degrades to a safe message.
    """
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "mode": "no_evidence", "sources": [], "grounded": False}

    chunks = retrieve_relevant_chunks(question)

    if not chunks:
        return {
            "answer": (
                "No sufficiently relevant information was found in the knowledge base for this "
                "question. Try asking about HVAC, lighting, occupancy-based management, appliances, "
                "temperature/comfort, tariffs, or general building energy conservation."
            ),
            "mode": "no_evidence",
            "sources": [],
            "grounded": False,
        }

    provider = get_provider()
    prompt = _build_prompt(question, chunks, system_context)

    try:
        answer = provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        return {"answer": answer, "mode": provider.name, "sources": chunks, "grounded": True}
    except Exception as exc:  # noqa: BLE001
        # LLMProviderError messages are written to be user-safe (no keys). Anything else is
        # reduced to its class name so no unexpected detail reaches the UI.
        reason = str(exc) if isinstance(exc, LLMProviderError) else type(exc).__name__
        fallback_text = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)
        return {
            "answer": (
                f"The configured LLM provider was unavailable ({reason}). Here is the most relevant "
                f"knowledge retrieved directly from the knowledge base instead:\n\n{fallback_text}"
            ),
            "mode": "fallback",
            "sources": chunks,
            "grounded": True,
        }
