"""Tests for the SQLite-backed checkpointer."""

from __future__ import annotations

import os
import tempfile

import pytest

from graphforge._checkpoint_sqlite import SqliteCheckpointer
from graphforge._checkpoint import CheckpointKey


class TestSqliteCheckpointer:
    @pytest.fixture
    def db_path(self) -> str:
        """Create a temporary database file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        yield tmp.name
        os.unlink(tmp.name)

    def test_put_and_get(self, db_path: str) -> None:
        cp = SqliteCheckpointer(db_path)
        key: CheckpointKey = ("thread-1", "node_a", 0)
        cp.put(key, {"count": 1})

        retrieved = cp.get(key)
        assert retrieved is not None
        assert retrieved.key == key
        assert retrieved.state == {"count": 1}

    def test_get_missing(self, db_path: str) -> None:
        cp = SqliteCheckpointer(db_path)
        key: CheckpointKey = ("thread-1", "node_a", 0)
        result = cp.get(key)
        assert result is None

    def test_list_by_thread(self, db_path: str) -> None:
        cp = SqliteCheckpointer(db_path)
        cp.put(("thread-1", "a", 0), {})
        cp.put(("thread-1", "b", 1), {})
        cp.put(("thread-2", "a", 0), {})

        keys = cp.list("thread-1")
        assert len(keys) == 2
        assert keys[0] == ("thread-1", "a", 0)
        assert keys[1] == ("thread-1", "b", 1)

        keys2 = cp.list("thread-2")
        assert len(keys2) == 1

    def test_list_empty(self, db_path: str) -> None:
        cp = SqliteCheckpointer(db_path)
        assert cp.list("nonexistent") == []

    def test_clear(self, db_path: str) -> None:
        cp = SqliteCheckpointer(db_path)
        cp.put(("t", "a", 0), {"x": 1})
        cp.clear()
        assert cp.list("t") == []

    def test_parent_key(self, db_path: str) -> None:
        cp = SqliteCheckpointer(db_path)
        cp.put(("t", "a", 0), {"count": 0})
        cp.put(("t", "b", 1), {"count": 1}, parent_key=("t", "a", 0))

        child = cp.get(("t", "b", 1))
        assert child is not None
        assert child.parent_key == ("t", "a", 0)

    def test_overwrite_existing(self, db_path: str) -> None:
        cp = SqliteCheckpointer(db_path)
        key: CheckpointKey = ("t", "a", 0)
        cp.put(key, {"version": 1})
        cp.put(key, {"version": 2})

        retrieved = cp.get(key)
        assert retrieved is not None
        assert retrieved.state["version"] == 2

    def test_full_state_roundtrip(self, db_path: str) -> None:
        """Test with a realistic nested state dict."""
        cp = SqliteCheckpointer(db_path)
        state = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "count": 42,
            "metadata": {"model": "gpt-4", "temperature": 0.7},
        }
        key: CheckpointKey = ("session-1", "llm_call", 5)
        cp.put(key, state, parent_key=("session-1", "query", 4))

        retrieved = cp.get(key)
        assert retrieved is not None
        assert retrieved.state == state
        assert retrieved.parent_key == ("session-1", "query", 4)
        assert len(retrieved.state["messages"]) == 2
        assert retrieved.state["count"] == 42
        assert retrieved.state["metadata"]["model"] == "gpt-4"

    def test_persistence_across_instances(self, db_path: str) -> None:
        """Data persists across SqliteCheckpointer instances with the same DB."""
        cp1 = SqliteCheckpointer(db_path)
        cp1.put(("t", "a", 0), {"data": "persistent"})
        cp1.close()

        cp2 = SqliteCheckpointer(db_path)
        retrieved = cp2.get(("t", "a", 0))
        assert retrieved is not None
        assert retrieved.state == {"data": "persistent"}
        cp2.close()

    def test_close_and_reopen(self, db_path: str) -> None:
        cp = SqliteCheckpointer(db_path)
        cp.put(("t", "a", 0), {"val": 1})
        cp.close()

        cp2 = SqliteCheckpointer(db_path)
        result = cp2.get(("t", "a", 0))
        assert result is not None
        assert result.state == {"val": 1}
        cp2.close()

    def test_metadata(self, db_path: str) -> None:
        cp = SqliteCheckpointer(db_path)
        cp.put(("t", "a", 0), {"x": 1}, metadata={"paused": True, "reason": "awaiting_input"})
        retrieved = cp.get(("t", "a", 0))
        assert retrieved is not None
        assert retrieved.metadata.get("paused") is True
        assert retrieved.metadata.get("reason") == "awaiting_input"
