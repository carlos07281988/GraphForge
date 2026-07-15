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

"""PostgreSQL-backed checkpointer for production persistence.

Requires ``psycopg2`` or ``asyncpg`` (install with ``pip install psycopg2-binary``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from graphforge._checkpoint import Checkpoint, Checkpointer, CheckpointKey
from graphforge._logging import get_logger

logger = get_logger("checkpoint.postgres")

try:
    import psycopg2
    import psycopg2.pool
    from psycopg2.extras import RealDictCursor

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False
    psycopg2 = None  # type: ignore[assignment]


class PostgresCheckpointer(Checkpointer[Any]):
    """PostgreSQL-backed checkpointer for production persistence.

    Uses a connection pool for thread-safe concurrent access.
    Creates the ``graphforge_checkpoints`` table automatically on first use.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string (e.g. ``"postgresql://user:pass@localhost:5432/db"``).
    table_name:
        Table name for checkpoints (default: ``"graphforge_checkpoints"``).
    min_conn:
        Minimum connections in pool (default: 1).
    max_conn:
        Maximum connections in pool (default: 10).
    **kwargs:
        Additional keyword arguments for ``psycopg2.pool.ThreadedConnectionPool``.

    Examples
    --------
    .. code-block:: python

        from graphforge._checkpoint_postgres import PostgresCheckpointer

        checkpointer = PostgresCheckpointer(
            "postgresql://user:pass@localhost:5432/graphforge"
        )
        compiled = graph.compile(checkpointer=checkpointer)
    """

    def __init__(
        self,
        dsn: str,
        *,
        table_name: str = "graphforge_checkpoints",
        min_conn: int = 1,
        max_conn: int = 10,
        **kwargs: Any,
    ) -> None:
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "The ``psycopg2`` package is required. "
                "Install with: pip install psycopg2-binary"
            )
        self._dsn = dsn
        self._table = table_name
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_conn, maxconn=max_conn, dsn=dsn, **kwargs
        )
        self._init_table()

    def _get_conn(self):
        return self._pool.getconn()

    def _put_conn(self, conn):
        self._pool.putconn(conn)

    def _init_table(self) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        thread_id TEXT NOT NULL,
                        node_name TEXT NOT NULL,
                        step INTEGER NOT NULL,
                        state JSONB NOT NULL DEFAULT '{{}}',
                        parent_thread_id TEXT,
                        parent_node_name TEXT,
                        parent_step INTEGER,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (thread_id, node_name, step)
                    )
                """)
                conn.commit()
        finally:
            self._put_conn(conn)

    def put(
        self,
        key: CheckpointKey,
        state: Dict[str, Any],
        parent_key: Optional[CheckpointKey] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                thread_id, node_name, step = key
                parent_thread = parent_key[0] if parent_key else None
                parent_node = parent_key[1] if parent_key else None
                parent_step = parent_key[2] if parent_key else None
                meta = json.dumps(metadata or {})
                state_json = json.dumps(state, default=str)
                cur.execute(f"""
                    INSERT INTO {self._table}
                        (thread_id, node_name, step, state, parent_thread_id,
                         parent_node_name, parent_step, metadata)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (thread_id, node_name, step)
                    DO UPDATE SET state = EXCLUDED.state,
                                  metadata = EXCLUDED.metadata
                """, (thread_id, node_name, step, state_json,
                      parent_thread, parent_node, parent_step, meta))
                conn.commit()
        finally:
            self._put_conn(conn)
        logger.debug("PostgresCheckpointer.put(%r, %r, %d)", thread_id, node_name, step)

    def get(self, key: CheckpointKey) -> Optional[Checkpoint[Any]]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT thread_id, node_name, step, state, parent_thread_id,
                           parent_node_name, parent_step, metadata
                    FROM {self._table}
                    WHERE thread_id = %s AND node_name = %s AND step = %s
                """, (key[0], key[1], key[2]))
                row = cur.fetchone()
                if row is None:
                    return None
                parent_key = None
                if row["parent_thread_id"]:
                    parent_key = (row["parent_thread_id"],
                                  row["parent_node_name"],
                                  row["parent_step"])
                return Checkpoint(
                    key=(row["thread_id"], row["node_name"], row["step"]),
                    state=row["state"] if isinstance(row["state"], dict) else json.loads(row["state"]),
                    parent_key=parent_key,
                    metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
                )
        finally:
            self._put_conn(conn)

    def list(self, thread_id: str) -> List[CheckpointKey]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT thread_id, node_name, step
                    FROM {self._table}
                    WHERE thread_id = %s
                    ORDER BY step ASC
                """, (thread_id,))
                return [(row[0], row[1], row[2]) for row in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def clear(self) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._table}")
                conn.commit()
        finally:
            self._put_conn(conn)


__all__ = [
    "PostgresCheckpointer",
]
