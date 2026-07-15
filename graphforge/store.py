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

"""Persistent key-value store abstraction for agent memory.

Provides a :class:`Store` abstract base class and :class:`InMemoryStore`
for cross-thread, cross-session agent memory that is independent of
checkpoint state.

The Store is designed for **semantic** memory — facts, preferences,
knowledge — as opposed to checkpoint state, which is **execution** state.
"""

from __future__ import annotations

import abc
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from graphforge._logging import get_logger

logger = get_logger("store")


# ---------------------------------------------------------------------------
# Base Store
# ---------------------------------------------------------------------------


class Store(abc.ABC):
    """Abstract interface for persistent key-value storage.

    Stores are namespace-scoped for isolation. Each ``namespace``
    represents a logical partition (e.g. a thread_id, agent_id, or
    user_id).

    All values are JSON-serializable dicts.
    """

    @abc.abstractmethod
    def get(
        self,
        namespace: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a value by key within a namespace.

        Returns ``None`` if the key does not exist.
        """
        ...

    @abc.abstractmethod
    def put(
        self,
        namespace: str,
        key: str,
        value: Dict[str, Any],
    ) -> None:
        """Store a value by key within a namespace.

        If the key already exists, the value is **replaced**.
        Values must be JSON-serializable.
        """
        ...

    @abc.abstractmethod
    def delete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """Delete a key from a namespace.

        Returns ``True`` if the key existed, ``False`` otherwise.
        """
        ...

    @abc.abstractmethod
    def list_keys(self, namespace: str) -> Sequence[str]:
        """Return all keys within a namespace."""
        ...

    @abc.abstractmethod
    def clear_namespace(self, namespace: str) -> None:
        """Remove all keys within a namespace."""
        ...


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------


class InMemoryStore(Store):
    """Ephemeral store backed by an in-memory dictionary.

    Useful for development, testing, and single-process deployments.
    Data is **not** persisted across process restarts.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def get(
        self,
        namespace: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        ns = self._data.get(namespace)
        if ns is None:
            logger.debug("InMemoryStore.get(%r, %r): MISS (no namespace)", namespace, key)
            return None
        value = ns.get(key)
        logger.debug(
            "InMemoryStore.get(%r, %r): %s", namespace, key, "HIT" if value else "MISS"
        )
        return value

    def put(
        self,
        namespace: str,
        key: str,
        value: Dict[str, Any],
    ) -> None:
        if namespace not in self._data:
            self._data[namespace] = {}
        self._data[namespace][key] = value
        logger.debug("InMemoryStore.put(%r, %r): %d field(s)", namespace, key, len(value))

    def delete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        ns = self._data.get(namespace)
        if ns is None or key not in ns:
            return False
        del ns[key]
        return True

    def list_keys(self, namespace: str) -> Sequence[str]:
        ns = self._data.get(namespace)
        if ns is None:
            return []
        return list(ns.keys())

    def clear_namespace(self, namespace: str) -> None:
        self._data.pop(namespace, None)

    def clear_all(self) -> None:
        """Remove all namespaces and keys."""
        self._data.clear()


__all__ = [
    "Store",
    "InMemoryStore",
]
