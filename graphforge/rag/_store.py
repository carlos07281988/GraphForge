"""Vector store abstraction for RAG.

Provides :class:`VectorStore` ABC and :class:`InMemoryVectorStore`.
"""

from __future__ import annotations

import abc
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from graphforge.rag._embeddings import Embeddings


@dataclass
class Document:
    """A document with content and metadata.

    Parameters
    ----------
    content:
        The text content.
    metadata:
        Optional metadata dict.
    id:
        Optional document ID (auto-generated if not provided).
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class SearchResult:
    """Result of a similarity search.

    Parameters
    ----------
    document:
        The matched document.
    score:
        Similarity score (0-1, higher is more similar).
    """
    document: Document
    score: float


class VectorStore(abc.ABC):
    """Abstract base class for vector stores."""

    @abc.abstractmethod
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Add texts to the store.

        Parameters
        ----------
        texts:
            Texts to add.
        metadatas:
            Optional metadata for each text.
        ids:
            Optional IDs for each text (auto-generated if not provided).

        Returns
        -------
        List of document IDs.
        """
        ...

    @abc.abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[SearchResult]:
        """Search for similar documents.

        Parameters
        ----------
        query:
            Query text.
        k:
            Number of results to return.

        Returns
        -------
        List of search results sorted by relevance.
        """
        ...

    @abc.abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Delete documents by ID."""
        ...


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using cosine similarity.

    Parameters
    ----------
    embeddings:
        Embedding model to use.
    """

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._documents: Dict[str, Document] = {}
        self._embeddings_cache: Dict[str, List[float]] = {}

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        result_ids: List[str] = []
        texts_list = list(texts)
        vectors = self._embeddings.embed_documents(texts_list)

        for i, text in enumerate(texts_list):
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{hash(text)}"
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            self._documents[doc_id] = Document(content=text, metadata=meta, id=doc_id)
            self._embeddings_cache[doc_id] = vectors[i]
            result_ids.append(doc_id)

        return result_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[SearchResult]:
        query_vec = self._embeddings.embed_query(query)
        results: List[SearchResult] = []

        for doc_id, doc in self._documents.items():
            doc_vec = self._embeddings_cache.get(doc_id)
            if doc_vec is None:
                continue
            score = _cosine_similarity(query_vec, doc_vec)
            results.append(SearchResult(document=doc, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def delete(self, ids: Sequence[str]) -> None:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
            self._embeddings_cache.pop(doc_id, None)


__all__ = [
    "Document",
    "InMemoryVectorStore",
    "SearchResult",
    "VectorStore",
]
