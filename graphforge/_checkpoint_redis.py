"""Redis-backed checkpointer for distributed state persistence.

The :class:`RedisCheckpointer` stores checkpoints in Redis, making it suitable
for multi-process and distributed deployments where state must be shared.

Key schema
----------
- ``{prefix}{thread_id}:{node_name}:{step}`` → JSON-encoded state dict
- ``{prefix}thread:{thread_id}`` → sorted set of checkpoint keys (by step)
- ``{prefix}parent:{thread_id}:{node_name}:{step}`` → parent key JSON
- ``{prefix}meta:{thread_id}:{node_name}:{step}`` → metadata JSON
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import logging

from graphforge._checkpoint import Checkpoint, CheckpointKey, Checkpointer

logger = logging.getLogger("graphforge.checkpoint.redis")


class RedisCheckpointer(Checkpointer[Any]):
    """Persistent checkpointer backed by Redis.

    Parameters
    ----------
    client:
        An open ``redis.Redis`` client instance.
    key_prefix:
        Prefix for all Redis keys to avoid collisions.
    """

    __slots__ = ("_client", "_prefix")

    def __init__(
        self,
        client: Any,
        key_prefix: str = "gf:",
    ) -> None:
        self._client = client
        self._prefix = key_prefix

    def _ckey(self, key: CheckpointKey) -> str:
        """Redis key for a checkpoint's state."""
        return f"{self._prefix}{key[0]}:{key[1]}:{key[2]}"

    def _pkey(self, key: CheckpointKey) -> str:
        """Redis key for a checkpoint's parent link."""
        return f"{self._prefix}parent:{key[0]}:{key[1]}:{key[2]}"

    def _mkey(self, key: CheckpointKey) -> str:
        """Redis key for a checkpoint's metadata."""
        return f"{self._prefix}meta:{key[0]}:{key[1]}:{key[2]}"

    def _tkey(self, thread_id: str) -> str:
        """Redis key for a thread's sorted set of checkpoint keys."""
        return f"{self._prefix}thread:{thread_id}"

    # -- Checkpointer interface --------------------------------------------

    def put(
        self,
        key: CheckpointKey,
        state: Dict[str, Any],
        parent_key: Optional[CheckpointKey] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        thread_id, node_name, step = key
        ckey = self._ckey(key)
        score = float(step)

        pipe = self._client.pipeline()
        pipe.set(ckey, json.dumps(state))
        pipe.zadd(self._tkey(thread_id), {ckey: score})

        if parent_key:
            pipe.set(self._pkey(key), json.dumps(list(parent_key)))

        if metadata:
            pipe.set(self._mkey(key), json.dumps(metadata))

        pipe.execute()
        logger.debug(
            "RedisCheckpointer.put: thread=%r node=%r step=%d",
            thread_id, node_name, step,
        )

    def get(self, key: CheckpointKey) -> Optional[Checkpoint[Any]]:
        thread_id, node_name, step = key
        ckey = self._ckey(key)

        state_json = self._client.get(ckey)
        if state_json is None:
            return None

        state = json.loads(state_json)

        parent_key: Optional[CheckpointKey] = None
        parent_json = self._client.get(self._pkey(key))
        if parent_json:
            parts = json.loads(parent_json)
            if len(parts) == 3:
                parent_key = (parts[0], parts[1], parts[2])

        metadata: Dict[str, Any] = {}
        meta_json = self._client.get(self._mkey(key))
        if meta_json:
            metadata = json.loads(meta_json)

        return Checkpoint(
            key=key,
            state=state,
            parent_key=parent_key,
            metadata=metadata,
        )

    def list(self, thread_id: str) -> List[CheckpointKey]:
        tkey = self._tkey(thread_id)
        members = self._client.zrange(tkey, 0, -1)
        keys: List[CheckpointKey] = []
        for member in members:
            member_str = member.decode("utf-8") if isinstance(member, bytes) else member
            parts = member_str.split(":")
            if len(parts) >= 3:
                node_name = parts[-2]
                step = int(parts[-1])
                keys.append((thread_id, node_name, step))
        return keys

    def clear(self) -> None:
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=f"{self._prefix}*")
            if keys:
                self._client.delete(*keys)

            if cursor == 0:
                break
        logger.debug("RedisCheckpointer: cleared all keys with prefix %r", self._prefix)


__all__ = ["RedisCheckpointer"]
