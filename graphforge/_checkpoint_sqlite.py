# Copyright 2024 GraphForge Contributors
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

"""SQLite-backed checkpointer for persistent state storage.

The :class:`SqliteCheckpointer` stores checkpoints in a local SQLite database,
making it suitable for single-process production deployments, development
workflows, and any scenario where checkpoint data should survive process restarts.

Schema
------
.. code-block:: sql

    CREATE TABLE checkpoints (
        thread_id  TEXT NOT NULL,
        node_name  TEXT NOT NULL,
        step       INTEGER NOT NULL,
        state      TEXT NOT NULL,           -- JSON-encoded state dict
        parent_thread_id TEXT,
        parent_node_name TEXT,
        parent_step      INTEGER,
        metadata   TEXT DEFAULT '{}',       -- JSON-encoded metadata dict
        created_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (thread_id, node_name, step)
    );
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from graphforge._checkpoint import Checkpoint, CheckpointKey, Checkpointer

logger = logging.getLogger("graphforge.checkpoint.sqlite")


class SqliteCheckpointer(Checkpointer[Any]):
    """Persistent checkpointer backed by SQLite.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Use ``":memory:"`` for an
        in-memory database (not persisted across restarts).
    table_name:
        Name of the checkpoints table.
    """

    __slots__ = ("_db_path", "_table", "_local")

    def __init__(
        self,
        db_path: str = "graphforge_checkpoints.db",
        table_name: str = "checkpoints",
    ) -> None:
        self._db_path = db_path
        self._table = table_name
        self._local = threading.local()
        self._init_db()

    # -- connection management (thread-safe) --------------------------------

    @property
    def _conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self) -> None:
        """Create the checkpoints table if it does not exist."""
        conn = self._conn
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {self._table} (
                thread_id  TEXT NOT NULL,
                node_name  TEXT NOT NULL,
                step       INTEGER NOT NULL,
                state      TEXT NOT NULL,
                parent_thread_id TEXT,
                parent_node_name TEXT,
                parent_step      INTEGER,
                metadata   TEXT DEFAULT '{{}}',
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (thread_id, node_name, step)
            )"""
        )
        conn.commit()

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
        parent_thread = parent_key[0] if parent_key else None
        parent_node = parent_key[1] if parent_key else None
        parent_step = parent_key[2] if parent_key else None

        self._conn.execute(
            f"""INSERT OR REPLACE INTO {self._table}
                (thread_id, node_name, step, state,
                 parent_thread_id, parent_node_name, parent_step, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thread_id,
                node_name,
                step,
                json.dumps(state),
                parent_thread,
                parent_node,
                parent_step,
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()
        logger.debug(
            "SqliteCheckpointer.put: thread=%r node=%r step=%d",
            thread_id, node_name, step,
        )

    def get(self, key: CheckpointKey) -> Optional[Checkpoint[Any]]:
        thread_id, node_name, step = key
        cursor = self._conn.execute(
            f"""SELECT state, parent_thread_id, parent_node_name,
                       parent_step, metadata
                FROM {self._table}
                WHERE thread_id=? AND node_name=? AND step=?""",
            (thread_id, node_name, step),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        parent_key: Optional[CheckpointKey] = None
        if row["parent_thread_id"]:
            parent_key = (
                row["parent_thread_id"],
                row["parent_node_name"],
                row["parent_step"],
            )

        checkpoint = Checkpoint(
            key=key,
            state=json.loads(row["state"]),
            parent_key=parent_key,
            metadata=json.loads(row["metadata"]),
        )
        return checkpoint

    def list(self, thread_id: str) -> List[CheckpointKey]:
        cursor = self._conn.execute(
            f"""SELECT node_name, step FROM {self._table}
                WHERE thread_id=?
                ORDER BY step ASC""",
            (thread_id,),
        )
        return [(thread_id, row["node_name"], row["step"]) for row in cursor.fetchall()]

    def clear(self) -> None:
        self._conn.execute(f"DELETE FROM {self._table}")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


__all__ = ["SqliteCheckpointer"]
