"""Checkpoint / persistence abstractions.

Checkpoints allow graph execution state to be saved and resumed.
This is essential for long-running agents, human-in-the-loop workflows,
and fault tolerance.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Generic, List, Optional, Tuple

from graphforge._logging import get_logger
from graphforge._types import NodeName, StateT, StateUpdate

logger = get_logger("checkpoint")


CheckpointKey = Tuple[str, NodeName, int]


class Checkpoint(Generic[StateT]):
    """A snapshot of graph state at a particular point in execution."""

    __slots__ = ("key", "state", "parent_key", "metadata")

    def __init__(
        self,
        key: CheckpointKey,
        state: Dict[str, Any],
        parent_key: Optional[CheckpointKey] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.key = key
        self.state = state
        self.parent_key = parent_key
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"Checkpoint(thread={self.key[0]!r}, "
            f"node={self.key[1]!r}, "
            f"step={self.key[2]})"
        )


class Checkpointer(abc.ABC, Generic[StateT]):
    """Abstract interface for state persistence."""

    @abc.abstractmethod
    def put(
        self,
        key: CheckpointKey,
        state: Dict[str, Any],
        parent_key: Optional[CheckpointKey] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a checkpoint."""
        ...

    @abc.abstractmethod
    def get(self, key: CheckpointKey) -> Optional[Checkpoint[StateT]]:
        """Retrieve a checkpoint by key. Returns ``None`` if not found."""
        ...

    @abc.abstractmethod
    def list(
        self,
        thread_id: str,
    ) -> List[CheckpointKey]:
        """Return all checkpoint keys for a thread, in execution order."""
        ...

    def clear(self) -> None:
        """Remove all checkpoints (testing convenience)."""
        ...


class InMemoryCheckpointer(Checkpointer[StateT]):
    """Ephemeral checkpointer that stores state in a dictionary."""

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: Dict[CheckpointKey, Checkpoint[StateT]] = {}

    def put(
        self,
        key: CheckpointKey,
        state: Dict[str, Any],
        parent_key: Optional[CheckpointKey] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        logger.debug(
            "Checkpoint.put: thread=%r node=%r step=%d",
            key[0], key[1], key[2],
        )
        self._store[key] = Checkpoint(
            key=key, state=state, parent_key=parent_key, metadata=metadata
        )

    def get(self, key: CheckpointKey) -> Optional[Checkpoint[StateT]]:
        result = self._store.get(key)
        logger.debug(
            "Checkpoint.get: thread=%r node=%r step=%d -> %s",
            key[0], key[1], key[2], "HIT" if result else "MISS",
        )
        return result

    def list(self, thread_id: str) -> List[CheckpointKey]:
        keys = [key for key in self._store if key[0] == thread_id]
        logger.debug(
            "Checkpoint.list: thread=%r -> %d checkpoint(s)", thread_id, len(keys)
        )
        return keys

    def clear(self) -> None:
        self._store.clear()


__all__ = [
    "Checkpoint",
    "Checkpointer",
    "CheckpointKey",
    "InMemoryCheckpointer",
]
