"""
rag/document_loader.py
-------------------------
Purpose : Load and validate the knowledge-base document corpus. Kept as a
          separate module from `knowledge_base.py` (the content) so a future
          phase could swap in file-based or database-backed loading without
          changing any caller -- `load_documents()` is the stable interface.
Input   : None (reads from rag.knowledge_base.DOCUMENTS).
Output  : Validated list of document dicts.
"""

from typing import Dict, List

from rag.knowledge_base import get_documents

REQUIRED_DOCUMENT_FIELDS = ["doc_id", "title", "topic", "source", "source_url", "text"]


class DocumentLoadError(Exception):
    """Raised when the knowledge-base corpus fails metadata validation."""


def load_documents() -> List[Dict]:
    """
    Load the knowledge-base corpus and validate that every document carries
    complete, non-empty metadata (except `source_url`, which may legitimately
    be None -- see rag/knowledge_base.py's SOURCE POLICY).

    Output: list of validated document dicts.
    Raises: DocumentLoadError if any document is missing a required field,
            has empty text, or has a duplicate doc_id.
    """
    documents = get_documents()

    if not documents:
        raise DocumentLoadError("Knowledge base corpus is empty.")

    seen_ids = set()
    for doc in documents:
        missing = [f for f in REQUIRED_DOCUMENT_FIELDS if f not in doc]
        if missing:
            raise DocumentLoadError(f"Document missing required fields {missing}: {doc.get('doc_id', '?')}")

        if doc["doc_id"] in seen_ids:
            raise DocumentLoadError(f"Duplicate doc_id: {doc['doc_id']}")
        seen_ids.add(doc["doc_id"])

        for field in ("doc_id", "title", "topic", "source", "text"):
            if not doc[field] or not str(doc[field]).strip():
                raise DocumentLoadError(f"Document '{doc['doc_id']}' has an empty required field: '{field}'")

    return documents
