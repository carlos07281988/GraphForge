"""Tests for RedisCheckpointer using a mock Redis client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from graphforge._checkpoint_redis import RedisCheckpointer


@pytest.fixture
def pipe_mock() -> MagicMock:
    """A mock Redis pipeline."""
    p = MagicMock()
    p.execute.return_value = [True]
    return p


@pytest.fixture
def mock_redis(pipe_mock: MagicMock) -> MagicMock:
    client = MagicMock()
    client.pipeline.return_value = pipe_mock
    client.get.return_value = None
    client.zrange.return_value = []
    client.scan.return_value = (0, [])
    return client


@pytest.fixture
def cp(mock_redis: MagicMock) -> RedisCheckpointer:
    return RedisCheckpointer(mock_redis, key_prefix="test:")


class TestPut:
    def test_put_creates_state_entry(self, cp: RedisCheckpointer, mock_redis: MagicMock, pipe_mock: MagicMock) -> None:
        key = ("thread-a", "node_1", 0)
        state = {"x": 1, "y": "hello"}
        cp.put(key, state)
        # Pipeline was created
        mock_redis.pipeline.assert_called_once()
        # State was set on the pipeline
        pipe_mock.set.assert_called_once_with("test:thread-a:node_1:0", json.dumps(state))
        # Pipeline was executed
        pipe_mock.execute.assert_called_once()

    def test_put_with_parent(self, cp: RedisCheckpointer, mock_redis: MagicMock, pipe_mock: MagicMock) -> None:
        key = ("thread-a", "node_2", 1)
        parent = ("thread-a", "node_1", 0)
        state = {"x": 2}
        cp.put(key, state, parent_key=parent)
        assert pipe_mock.set.call_count >= 2  # state + parent
        # One of the set calls should be the parent key
        parent_key = "test:parent:thread-a:node_2:1"
        parent_set_calls = [
            c for c in pipe_mock.set.call_args_list
            if parent_key in str(c)
        ]
        assert len(parent_set_calls) >= 1

    def test_put_with_metadata(self, cp: RedisCheckpointer, pipe_mock: MagicMock) -> None:
        key = ("thread-a", "node_1", 0)
        state = {"x": 1}
        meta = {"user": "test", "ts": 1234567890}
        cp.put(key, state, metadata=meta)
        meta_key = "test:meta:thread-a:node_1:0"
        meta_set_calls = [
            c for c in pipe_mock.set.call_args_list
            if meta_key in str(c)
        ]
        assert len(meta_set_calls) >= 1


class TestGet:
    def test_get_returns_none_when_missing(self, cp: RedisCheckpointer, mock_redis: MagicMock) -> None:
        mock_redis.get.return_value = None
        result = cp.get(("thread-x", "node_1", 99))
        assert result is None

    def test_get_returns_checkpoint(self, cp: RedisCheckpointer, mock_redis: MagicMock) -> None:
        key = ("thread-a", "node_1", 0)
        state = {"x": 42, "y": [1, 2, 3]}
        mock_redis.get.side_effect = [
            json.dumps(state),    # state
            None,                 # parent
            None,                 # metadata
        ]
        result = cp.get(key)
        assert result is not None
        assert result.key == key
        assert result.state == state

    def test_get_with_parent_and_metadata(self, cp: RedisCheckpointer, mock_redis: MagicMock) -> None:
        key = ("thread-a", "node_2", 1)
        state = {"x": 5}
        mock_redis.get.side_effect = [
            json.dumps(state),
            json.dumps(["thread-a", "node_1", 0]),
            json.dumps({"user": "alice"}),
        ]
        result = cp.get(key)
        assert result is not None
        assert result.parent_key == ("thread-a", "node_1", 0)
        assert result.metadata == {"user": "alice"}


class TestList:
    def test_list_empty(self, cp: RedisCheckpointer, mock_redis: MagicMock) -> None:
        mock_redis.zrange.return_value = []
        keys = cp.list("thread-empty")
        assert keys == []

    def test_list_returns_keys(self, cp: RedisCheckpointer, mock_redis: MagicMock) -> None:
        mock_redis.zrange.return_value = [
            b"test:thread-a:node_1:0",
            b"test:thread-a:node_2:1",
        ]
        keys = cp.list("thread-a")
        assert len(keys) == 2
        assert keys[0] == ("thread-a", "node_1", 0)
        assert keys[1] == ("thread-a", "node_2", 1)


class TestClear:
    def test_clear_deletes_all_prefixed_keys(self, cp: RedisCheckpointer, mock_redis: MagicMock) -> None:
        mock_redis.scan.return_value = (0, [b"test:key1", b"test:key2"])
        cp.clear()
        mock_redis.scan.assert_called_with(0, match="test:*")
        mock_redis.delete.assert_called_once()

    def test_clear_scans_loop(self, cp: RedisCheckpointer, mock_redis: MagicMock) -> None:
        mock_redis.scan.side_effect = [
            (1, [b"test:k1"]),
            (0, [b"test:k2"]),
        ]
        cp.clear()
        assert mock_redis.scan.call_count == 2
        assert mock_redis.delete.call_count == 2


class TestKeyPrefix:
    def test_custom_prefix(self) -> None:
        client = MagicMock()
        pipe = MagicMock()
        client.pipeline.return_value = pipe
        cp = RedisCheckpointer(client, key_prefix="app:")
        cp.put(("t1", "n1", 0), {"x": 1})
        pipe.set.assert_called_with("app:t1:n1:0", json.dumps({"x": 1}))

    def test_default_prefix(self) -> None:
        mock = MagicMock()
        cp = RedisCheckpointer(mock)
        assert cp._prefix == "gf:"
