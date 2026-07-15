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

"""Redis-backed implementation of :class:`~graphforge.store.Store`.

Requires ``redis`` package (install with ``pip install graphforge[store-redis]``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from graphforge._logging import get_logger
from graphforge.store import Store

logger = get_logger("store.redis")

try:
    import redis as _redis
    from redis import Redis

    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False
    Redis = None  # type: ignore[assignment,misc]


class RedisStore(Store):
    """Redis-backed persistent store for agent memory.

    Stores JSON-serializable values keyed by ``namespace:key`` in Redis.

    Parameters
    ----------
    redis_client:
        An existing ``redis.Redis`` client instance. If ``None``,
        creates one from ``**kwargs``.
    key_prefix:
        Optional prefix for all Redis keys (default: ``"graphforge:"``).
    **kwargs:
        Passed to ``redis.Redis()`` if ``redis_client`` is ``None``.

    Examples
    --------
    .. code-block:: python

        from graphforge.store_redis import RedisStore

        store = RedisStore(host="localhost", port=6379, db=0)

        store.put("session-123", "user_prefs", {"theme": "dark"})
        prefs = store.get("session-123", "user_prefs")
        print(prefs)  # {"theme": "dark"}
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        key_prefix: str = "graphforge:",
        **kwargs: Any,
    ) -> None:
        if not _HAS_REDIS:
            raise ImportError(
                "The ``redis`` package is required. "
                "Install with: pip install graphforge[store-redis]"
            )
        self._client: Redis = redis_client or Redis(**kwargs)
        self._prefix = key_prefix

    def _make_key(self, namespace: str, key: str) -> str:
        return f"{self._prefix}{namespace}:{key}"

    def get(
        self,
        namespace: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        redis_key = self._make_key(namespace, key)
        raw = self._client.get(redis_key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("RedisStore.get(%r, %r): failed to decode", namespace, key)
            return None

    def put(
        self,
        namespace: str,
        key: str,
        value: Dict[str, Any],
    ) -> None:
        redis_key = self._make_key(namespace, key)
        raw = json.dumps(value, ensure_ascii=False, default=str)
        self._client.set(redis_key, raw)

    def delete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        redis_key = self._make_key(namespace, key)
        return bool(self._client.delete(redis_key))

    def list_keys(self, namespace: str) -> Sequence[str]:
        pattern = self._make_key(namespace, "*")
        keys = self._client.keys(pattern)
        prefix_len = len(self._make_key(namespace, ""))
        return [k.decode() if isinstance(k, bytes) else k[prefix_len:] for k in keys]

    def clear_namespace(self, namespace: str) -> None:
        pattern = self._make_key(namespace, "*")
        keys = self._client.keys(pattern)
        if keys:
            self._client.delete(*keys)

    def clear_all(self) -> None:
        pattern = f"{self._prefix}*"
        keys = self._client.keys(pattern)
        if keys:
            self._client.delete(*keys)


__all__ = [
    "RedisStore",
]
