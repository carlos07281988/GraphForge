# Copyright 2026 GraphForge Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Store and InMemoryStore."""

from __future__ import annotations

import pytest
from graphforge.store import Store, InMemoryStore


class TestStoreABC:
    def test_abstract_instantiation(self) -> None:
        """Store ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Store()  # type: ignore[abstract]


class TestInMemoryStore:
    @pytest.fixture
    def store(self) -> InMemoryStore:
        return InMemoryStore()

    def test_put_and_get(self, store: InMemoryStore) -> None:
        store.put("ns1", "k1", {"value": 42})
        result = store.get("ns1", "k1")
        assert result == {"value": 42}

    def test_get_missing_key(self, store: InMemoryStore) -> None:
        result = store.get("ns1", "nonexistent")
        assert result is None

    def test_get_missing_namespace(self, store: InMemoryStore) -> None:
        result = store.get("nonexistent_ns", "k1")
        assert result is None

    def test_put_overwrites(self, store: InMemoryStore) -> None:
        store.put("ns1", "k1", {"value": 1})
        store.put("ns1", "k1", {"value": 2})
        result = store.get("ns1", "k1")
        assert result == {"value": 2}

    def test_delete_existing(self, store: InMemoryStore) -> None:
        store.put("ns1", "k1", {"value": 42})
        assert store.delete("ns1", "k1") is True
        assert store.get("ns1", "k1") is None

    def test_delete_nonexistent(self, store: InMemoryStore) -> None:
        assert store.delete("ns1", "nonexistent") is False

    def test_list_keys(self, store: InMemoryStore) -> None:
        store.put("ns1", "a", {})
        store.put("ns1", "b", {})
        store.put("ns2", "c", {})
        keys = store.list_keys("ns1")
        assert sorted(keys) == ["a", "b"]

    def test_list_keys_empty(self, store: InMemoryStore) -> None:
        assert list(store.list_keys("empty_ns")) == []

    def test_clear_namespace(self, store: InMemoryStore) -> None:
        store.put("ns1", "k1", {})
        store.put("ns1", "k2", {})
        store.clear_namespace("ns1")
        assert list(store.list_keys("ns1")) == []

    def test_clear_namespace_preserves_others(self, store: InMemoryStore) -> None:
        store.put("ns1", "k1", {})
        store.put("ns2", "k2", {})
        store.clear_namespace("ns1")
        assert list(store.list_keys("ns2")) == ["k2"]

    def test_clear_all(self, store: InMemoryStore) -> None:
        store.put("ns1", "k1", {})
        store.put("ns2", "k2", {})
        store.clear_all()
        assert list(store.list_keys("ns1")) == []
        assert list(store.list_keys("ns2")) == []

    def test_namespace_isolation(self, store: InMemoryStore) -> None:
        store.put("ns1", "key", {"v": 1})
        store.put("ns2", "key", {"v": 2})
        assert store.get("ns1", "key") == {"v": 1}
        assert store.get("ns2", "key") == {"v": 2}

    def test_store_json_values(self, store: InMemoryStore) -> None:
        complex_value = {
            "string": "hello",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"a": 1},
        }
        store.put("ns1", "complex", complex_value)
        assert store.get("ns1", "complex") == complex_value


class TestRedisStoreImport:
    def test_import_redis_store(self) -> None:
        """RedisStore should be importable (without redis, it raises on init)."""
        from graphforge.store_redis import RedisStore
        assert RedisStore is not None

    def test_redis_store_init_fails_without_redis(self) -> None:
        from graphforge.store_redis import RedisStore
        with pytest.raises(ImportError):
            RedisStore()  # tries to import redis
