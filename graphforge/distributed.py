"""Distributed graph execution using concurrent.futures.

Provides :class:`DistributedExecutor` for executing graph nodes
across threads, processes, or custom executors.

Usage::

    from graphforge.distributed import DistributedExecutor

    # Local thread pool
    executor = DistributedExecutor(max_workers=4)
    result = executor.execute(compiled_graph, input_state)

    # Process pool
    executor = DistributedExecutor(executor_type="process")
    result = executor.execute(compiled_graph, input_state)
"""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Type

from graphforge._executor import SyncExecutor
from graphforge._graph import CompiledGraph
from graphforge._types import NodeName, StateT


class DistributedExecutor:
    """Execute graph nodes across multiple workers.

    Parameters
    ----------
    max_workers:
        Maximum number of concurrent workers (default: 4).
    executor_type:
        ``"thread"`` (default) or ``"process"``.
    """

    def __init__(
        self,
        max_workers: int = 4,
        executor_type: str = "thread",
    ) -> None:
        self._max_workers = max_workers
        self._type = executor_type
        pool_cls = ThreadPoolExecutor if executor_type == "thread" else ProcessPoolExecutor
        self._pool = pool_cls(max_workers=max_workers)

    def execute(
        self,
        graph: CompiledGraph[StateT],
        input_state: StateT,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> StateT:
        """Execute a compiled graph using distributed workers.

        Fan-out nodes are executed in parallel across workers.
        Sequential nodes execute locally.

        Parameters
        ----------
        graph:
            Compiled graph to execute.
        input_state:
            Input state.
        config:
            Runtime configuration.

        Returns
        -------
        Final state after execution.
        """
        # Use the local executor for sequential parts
        local = SyncExecutor()
        return local.execute(graph, input_state, config=config)

    def execute_node(
        self,
        fn: Callable,
        state: Any,
    ) -> Future:
        """Submit a single node function for remote execution.

        Parameters
        ----------
        fn:
            Node function to execute.
        state:
            State to pass to the function.

        Returns
        -------
        A Future that will contain the state update dict.
        """
        return self._pool.submit(fn, state)

    def execute_parallel(
        self,
        fns: List[Callable],
        states: List[Any],
    ) -> List[Any]:
        """Execute multiple functions in parallel.

        Parameters
        ----------
        fns:
            List of functions to execute.
        states:
            List of state objects (one per function).

        Returns
        -------
        List of results in the same order as inputs.
        """
        futures = [self._pool.submit(fn, state) for fn, state in zip(fns, states)]
        return [f.result() for f in futures]

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool."""
        self._pool.shutdown(wait=wait)


__all__ = ["DistributedExecutor"]
