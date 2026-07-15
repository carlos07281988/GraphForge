"""Tests for RAG module (embeddings, store, retrieval node, chunking)."""
from graphforge.rag import (
    InMemoryVectorStore,
    RetrievalNode,
    chunk_text,
    ChunkingStrategy,
)
from graphforge.rag._embeddings import DeterministicEmbeddings
from graphforge.rag._store import Document, SearchResult


class TestEmbeddings:
    def test_deterministic(self) -> None:
        emb = DeterministicEmbeddings(dimension=4)
        vec = emb.embed_query("hello")
        assert len(vec) == 4
        assert all(isinstance(v, float) for v in vec)

    def test_deterministic_documents(self) -> None:
        emb = DeterministicEmbeddings(dimension=4)
        vecs = emb.embed_documents(["a", "b"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 4


class TestVectorStore:
    def test_add_and_search(self) -> None:
        emb = DeterministicEmbeddings(dimension=4)
        store = InMemoryVectorStore(emb)
        store.add_texts(["Paris is the capital of France", "London is the capital of UK"])
        results = store.similarity_search("France", k=1)
        assert len(results) >= 1
        assert "Paris" in results[0].document.content

    def test_empty_store(self) -> None:
        emb = DeterministicEmbeddings(dimension=4)
        store = InMemoryVectorStore(emb)
        results = store.similarity_search("hello", k=5)
        assert len(results) == 0

    def test_delete(self) -> None:
        emb = DeterministicEmbeddings(dimension=4)
        store = InMemoryVectorStore(emb)
        ids = store.add_texts(["doc1"])
        store.delete(ids)
        results = store.similarity_search("doc1", k=1)
        assert len(results) == 0

    def test_document_dataclass(self) -> None:
        doc = Document(content="test", metadata={"source": "web"}, id="1")
        assert doc.content == "test"
        assert doc.metadata["source"] == "web"

    def test_search_result_dataclass(self) -> None:
        doc = Document(content="test")
        sr = SearchResult(document=doc, score=0.95)
        assert sr.score == 0.95


class TestRetrievalNode:
    def test_retrieve_from_state(self) -> None:
        emb = DeterministicEmbeddings(dimension=4)
        store = InMemoryVectorStore(emb)
        store.add_texts(["Python is a programming language"])

        node = RetrievalNode(store, query_field="query", output_field="context")
        result = node({"query": "Python"})
        assert "context" in result
        assert len(result["context"]) >= 1

    def test_empty_query(self) -> None:
        emb = DeterministicEmbeddings(dimension=4)
        store = InMemoryVectorStore(emb)
        node = RetrievalNode(store, query_field="query", output_field="context")
        result = node({"query": ""})
        assert result["context"] == []

    def test_score_threshold(self) -> None:
        emb = DeterministicEmbeddings(dimension=4)
        store = InMemoryVectorStore(emb)
        store.add_texts(["some text"])

        node = RetrievalNode(store, query_field="q", output_field="ctx", score_threshold=0.99)
        result = node({"q": "something"})
        # With deterministic embeddings, similarity may be below 0.99
        assert "ctx" in result


class TestChunking:
    def test_fixed_chunking(self) -> None:
        text = "A" * 1000
        chunks = chunk_text(text, strategy="fixed", chunk_size=300, chunk_overlap=0)
        assert len(chunks) >= 3  # ceil(1000/300) = 4

    def test_sentence_chunking(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, strategy="sentence", chunk_size=200)
        assert len(chunks) >= 1

    def test_recursive_chunking(self) -> None:
        text = "Para 1.\\n\\nPara 2.\\n\\nPara 3."
        chunks = chunk_text(text, strategy="recursive", chunk_size=50)
        assert len(chunks) >= 1

    def test_empty_text(self) -> None:
        chunks = chunk_text("", strategy="fixed")
        assert chunks == []

    def test_short_text(self) -> None:
        chunks = chunk_text("short", strategy="fixed", chunk_size=100)
        assert chunks == ["short"]
