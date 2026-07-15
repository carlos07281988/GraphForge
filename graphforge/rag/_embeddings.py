"""Embeddings abstraction for RAG.

Provides :class:`Embeddings` abstract base class and a built-in
:class:`OpenAIEmbeddings` implementation.
"""

from __future__ import annotations

import abc
import hashlib
import json
from typing import Any, Dict, List, Optional


class Embeddings(abc.ABC):
    """Abstract base class for embedding models."""

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (chunks).

        Parameters
        ----------
        texts:
            List of text strings to embed.

        Returns
        -------
        List of embedding vectors (each is a list of floats).
        """
        ...

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Parameters
        ----------
        text:
            Query text to embed.

        Returns
        -------
        Embedding vector as a list of floats.
        """
        ...


class OpenAIEmbeddings(Embeddings):
    """OpenAI embeddings via the ``openai`` Python package.

    Parameters
    ----------
    model:
        Embedding model name (default: ``"text-embedding-3-small"``).
    api_key:
        OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
    **kwargs:
        Additional arguments for the OpenAI client.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._kwargs = kwargs
        self._client: Optional[Any] = None

    def _lazy_init(self) -> None:
        if self._client is not None:
            return
        try:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key, **self._kwargs)
        except ImportError:
            raise ImportError(
                "The ``openai`` package is required for OpenAIEmbeddings. "
                "Install with: pip install openai"
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._lazy_init()
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [r.embedding for r in resp.data]

    def embed_query(self, text: str) -> List[float]:
        self._lazy_init()
        resp = self._client.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding


class DeterministicEmbeddings(Embeddings):
    """Deterministic embeddings for testing. Maps text to hash-based vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self._dim = dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        h = hashlib.md5(text.encode()).hexdigest()
        vals = [int(h[i:i+8], 16) % 1000 / 1000.0 for i in range(0, min(32, self._dim * 8), 8)]
        while len(vals) < self._dim:
            vals.append(0.0)
        return vals[:self._dim]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)


__all__ = [
    "DeterministicEmbeddings",
    "Embeddings",
    "OpenAIEmbeddings",
]
