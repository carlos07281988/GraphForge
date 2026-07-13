"""Tests for the checkpoint/persistence module."""

from __future__ import annotations

from graphforge import Checkpoint, CheckpointKey, InMemoryCheckpointer


class TestInMemoryCheckpointer:
    def test_put_and_get(self) -> None:
        cp = InMemoryCheckpointer()
        key: CheckpointKey = ("test-thread", "node_a", 0)
        cp.put(key, {"count": 1})
        retrieved = cp.get(key)
        assert retrieved is not None
        assert retrieved.key == key
        assert retrieved.state == {"count": 1}

    def test_get_missing(self) -> None:
        cp = InMemoryCheckpointer()
        key: CheckpointKey = ("test-thread", "node_a", 0)
        result = cp.get(key)
        assert result is None

    def test_list_by_thread(self) -> None:
        cp = InMemoryCheckpointer()
        cp.put(("thread-1", "a", 0), {})
        cp.put(("thread-1", "b", 1), {})
        cp.put(("thread-2", "a", 0), {})

        keys = cp.list("thread-1")
        assert len(keys) == 2
        assert all(k[0] == "thread-1" for k in keys)

    def test_clear(self) -> None:
        cp = InMemoryCheckpointer()
        cp.put(("t", "a", 0), {})
        cp.clear()
        assert cp.list("t") == []

    def test_parent_key(self) -> None:
        cp = InMemoryCheckpointer()
        cp.put(("t", "a", 0), {"count": 0})
        cp.put(("t", "b", 1), {"count": 1}, parent_key=("t", "a", 0))
        child = cp.get(("t", "b", 1))
        assert child is not None
        assert child.parent_key == ("t", "a", 0)
