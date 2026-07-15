"""RAG (Retrieval-Augmented Generation) module for GraphForge.

Provides embeddings, vector stores, and retrieval nodes for building
knowledge-augmented agents.

Requires ``numpy`` for vector operations.
"""

from graphforge.rag._embeddings import Embeddings, OpenAIEmbeddings
from graphforge.rag._store import InMemoryVectorStore, VectorStore
from graphforge.rag._node import RetrievalNode
from graphforge.rag._chunking import ChunkingStrategy, chunk_text

__all__ = [
    "Embeddings",
    "OpenAIEmbeddings",
    "VectorStore",
    "InMemoryVectorStore",
    "RetrievalNode",
    "ChunkingStrategy",
    "chunk_text",
]
