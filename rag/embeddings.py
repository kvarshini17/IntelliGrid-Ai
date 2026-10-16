"""
rag/embeddings.py
-------------------
Purpose : Build vector embeddings for knowledge-base chunks and index them
          with FAISS for similarity search (Phase 7).
Input   : List of chunk dicts (from rag.chunker.chunk_documents()).
Output  : An EmbeddingIndex wrapping a real FAISS index + chunk metadata.

===============================================================================
EMBEDDING MODEL -- documented resource-constrained substitution
===============================================================================
Sentence-Transformers (a neural embedding model) was the originally planned
embedding method. In this deployment environment, installing it pulls in a
multi-gigabyte PyTorch+CUDA dependency chain that exceeds available disk
space (confirmed: the install failed with "No space left on device" after
downloading several GB of CUDA packages). Rather than silently falling back
or fabricating a result, this is documented honestly:

    - If `sentence_transformers` IS importable in the running environment,
      it is used automatically (see _SENTENCE_TRANSFORMERS_AVAILABLE below) --
      no code change needed to upgrade later.
    - If it is NOT available, this module uses a lightweight, dependency-free
      TF-IDF-style bag-of-words vectorizer (pure NumPy) as a documented
      substitute. This is a weaker semantic representation than a neural
      sentence embedding -- it captures keyword overlap, not deep semantic
      similarity -- and this limitation is carried into README.md.

FAISS itself IS genuinely installed and used for the vector index in both
cases (faiss-cpu has no heavy dependency chain and installed successfully).
This is not a from-scratch reimplementation of FAISS -- the actual library
performs the similarity search.
===============================================================================
"""

import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

import config

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

import faiss  # genuinely required; no fallback -- see rag/README notes if this ever fails to install


# A small, standard English stopword list. Without filtering these, short
# queries can spuriously match documents purely on shared common words
# (e.g. "what is the capital of France" sharing "is"/"the"/"of" with any
# knowledge-base document), which was caught during manual testing of the
# TF-IDF fallback (see rag/embeddings.py module docstring for why TF-IDF is
# used at all). This is a precision fix for the documented fallback
# embedding method, not a change to the FAISS indexing itself.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "what", "when", "where", "who", "why",
    "how", "this", "these", "those", "i", "you", "your", "we", "do", "does",
})


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class EmbeddingIndex:
    """
    Wraps a real FAISS index (faiss.IndexFlatIP, cosine similarity via L2-
    normalized vectors) over the knowledge-base chunks. Embedding method is
    sentence-transformers if available, otherwise a documented TF-IDF
    fallback (see module docstring).
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.mode = "none"
        self._model = None
        self._vocab: Optional[Dict[str, int]] = None
        self._index = None

        if not chunks:
            return

        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            self._build_sentence_transformer_index()
            self.mode = "sentence_transformers+faiss"
        else:
            self._build_tfidf_index()
            self.mode = "tfidf+faiss"

    # ------------------------------------------------------------------
    def _build_sentence_transformer_index(self):
        self._model = SentenceTransformer(config.EMBEDDING_MODEL_NAME if hasattr(config, "EMBEDDING_MODEL_NAME")
                                           else "all-MiniLM-L6-v2")
        texts = [c["text"] for c in self.chunks]
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vectors)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def _embed_query_sentence_transformer(self, query: str) -> np.ndarray:
        vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    # ------------------------------------------------------------------
    def _build_tfidf_index(self):
        """Pure-NumPy TF-IDF-style vectorization, then indexed with real FAISS."""
        token_lists = [_tokenize(c["text"]) for c in self.chunks]

        vocab: Dict[str, int] = {}
        doc_freq: Counter = Counter()
        for tokens in token_lists:
            for tok in set(tokens):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
                doc_freq[tok] += 1

        n_docs = len(self.chunks)
        idf = np.zeros(len(vocab), dtype="float32")
        for tok, idx in vocab.items():
            idf[idx] = np.log((n_docs + 1) / (doc_freq[tok] + 1)) + 1.0  # smoothed idf

        vectors = np.zeros((n_docs, len(vocab)), dtype="float32")
        for i, tokens in enumerate(token_lists):
            counts = Counter(tokens)
            for tok, cnt in counts.items():
                vectors[i, vocab[tok]] = cnt * idf[vocab[tok]]

        faiss.normalize_L2(vectors)
        self._vocab = vocab
        self._idf = idf
        self._index = faiss.IndexFlatIP(max(len(vocab), 1))
        self._index.add(vectors)

    def _embed_query_tfidf(self, query: str) -> np.ndarray:
        tokens = _tokenize(query)
        vec = np.zeros((1, max(len(self._vocab), 1)), dtype="float32")
        counts = Counter(tokens)
        for tok, cnt in counts.items():
            if tok in self._vocab:
                vec[0, self._vocab[tok]] = cnt * self._idf[self._vocab[tok]]
        faiss.normalize_L2(vec)
        return vec

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Return the top_k most relevant chunks for a query, with similarity scores."""
        if self.mode == "none" or not self.chunks:
            return []

        if self.mode == "sentence_transformers+faiss":
            query_vec = self._embed_query_sentence_transformer(query)
        else:
            query_vec = self._embed_query_tfidf(query)

        scores, indices = self._index.search(query_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({**self.chunks[idx], "score": float(score)})
        return results


def build_index(chunks: List[Dict]) -> EmbeddingIndex:
    """Build the FAISS-backed embedding index for the given chunks."""
    return EmbeddingIndex(chunks)
