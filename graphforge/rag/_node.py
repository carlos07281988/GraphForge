"""Retrieval node for RAG — retrieves context into graph state.

Provides :class:`RetrievalNode`, a first-class GraphForge node that
performs similarity search and injects results into state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from graphforge.rag._store import InMemoryVectorStore, VectorStore


class RetrievalNode:
    """A graph node that retrieves context from a vector store.

    Parameters
    ----------
    vector_store:
        The vector store to search.
    query_field:
        State field name containing the query (default: ``"query"``).
    output_field:
        State field name to write retrieved context to (default: ``"context"``).
    top_k:
        Number of results to retrieve (default: 4).
    score_threshold:
        Minimum similarity score (0-1) to include results (default: 0.0).

    Examples
    --------
    .. code-block:: python

        store = InMemoryVectorStore(embeddings)
        store.add_texts(["Paris is the capital of France", ...])

        graph = Graph[MyState]()
        graph.add_node("retrieve", RetrievalNode(store, query_field="question"))
        graph.add_edge("retrieve", "generate")
        graph.set_entry_point("retrieve")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        query_field: str = "query",
        output_field: str = "context",
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> None:
        self._store = vector_store
        self._query_field = query_field
        self._output_field = output_field
        self._top_k = top_k
        self._threshold = score_threshold

    def __call__(self, state: Any) -> Dict[str, Any]:
        """Execute retrieval from the vector store.

        Parameters
        ----------
        state:
            Graph state containing the query field.

        Returns
        -------
        A dict with the output field set to retrieved context.
        """
        query = self._get_field(state, self._query_field)
        if not query:
            return {self._output_field: []}

        results = self._store.similarity_search(query, k=self._top_k)
        filtered = [r for r in results if r.score >= self._threshold]

        context = [
            {
                "content": r.document.content,
                "score": r.score,
                "metadata": r.document.metadata,
            }
            for r in filtered
        ]
        return {self._output_field: context}

    def _get_field(self, state: Any, field: str) -> Any:
        if hasattr(state, field):
            return getattr(state, field)
        if isinstance(state, dict):
            return state.get(field, "")
        return ""


__all__ = [
    "RetrievalNode",
]
