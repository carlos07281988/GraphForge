"""Background task execution for running graphs in separate threads.

Provides :class:`BackgroundTaskRunner` for executing compiled graphs
asynchronously in a thread pool with status tracking.

Usage::

    from graphforge.background import BackgroundTaskRunner

    runner = BackgroundTaskRunner(max_workers=4)
    task = runner.submit(compiled_graph, input_state)
    # ... do other work ...
    result = task.result()  # blocks until done
    print(task.status)  # "completed" | "failed" | "cancelled"
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from graphforge._graph import CompiledGraph
from graphforge._logging import get_logger

logger = get_logger("background")

T = TypeVar("T")


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask(Generic[T]):
    """A background task representing a graph execution.

    Parameters
    ----------
    task_id:
        Unique identifier for the task.
    status:
        Current task status.
    future:
        Concurrent.futures Future object.
    created_at:
        Creation timestamp.
    """

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    future: Optional[Future] = None
    created_at: float = field(default_factory=time.time)
    _result: Optional[T] = None
    _error: Optional[str] = None

    @property
    def result(self) -> Optional[T]:
        return self._result

    @property
    def error(self) -> Optional[str]:
        return self._error

    def wait(self, timeout: Optional[float] = None) -> Optional[T]:
        """Wait for the task to complete and return the result."""
        if self.future is not None:
            try:
                self._result = self.future.result(timeout=timeout)
                self.status = TaskStatus.COMPLETED
            except Exception as e:
                self.status = TaskStatus.FAILED
                self._error = str(e)
        return self._result

    def cancel(self) -> bool:
        """Attempt to cancel the task."""
        if self.future is not None and self.future.cancel():
            self.status = TaskStatus.CANCELLED
            return True
        self.status = TaskStatus.CANCELLED
        return False


class BackgroundTaskRunner:
    """Run compiled graphs in background threads.

    Parameters
    ----------
    max_workers:
        Maximum number of concurrent threads (default: 4).
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, BackgroundTask] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def submit(
        self,
        graph: CompiledGraph,
        state: Any,
        *,
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[Any] = None,
        store: Optional[Any] = None,
        task_id: Optional[str] = None,
    ) -> BackgroundTask:
        """Submit a graph for background execution.

        Parameters
        ----------
        graph:
            Compiled graph to execute.
        state:
            Input state.
        config:
            Runtime configuration.
        callbacks:
            Optional callback manager.
        store:
            Optional store instance.
        task_id:
            Optional custom task ID (auto-generated if not provided).

        Returns
        -------
        A :class:`BackgroundTask` that can be monitored.
        """
        with self._lock:
            self._counter += 1
            tid = task_id or f"task_{self._counter}"
            task = BackgroundTask(tid)
            self._tasks[tid] = task

        def _run() -> Any:
            task.status = TaskStatus.RUNNING
            try:
                result = graph.invoke(
                    state,
                    config=config or {},
                    callbacks=callbacks,
                    store=store,
                )
                task.status = TaskStatus.COMPLETED
                task._result = result
                return result
            except Exception as e:
                task.status = TaskStatus.FAILED
                task._error = str(e)
                raise

        task.future = self._pool.submit(_run)
        logger.info("BackgroundTask %r submitted", tid)
        return task

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[BackgroundTask]:
        """List all tasks."""
        return list(self._tasks.values())

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task by ID."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        # Cancel via the graph's cancellation API
        thread_id = config.get("thread_id", task_id) if hasattr(task, '_config') else task_id
        return task.cancel()

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=wait)


__all__ = [
    "BackgroundTask",
    "BackgroundTaskRunner",
    "TaskStatus",
]
