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

"""Map-Reduce node for parallel processing of list-structured state fields.

The :class:`MapReduce` class is a first-class GraphForge node (usable with
``Graph.add_node()``) that applies a *map* function to each element of a
list field in parallel, then applies a *reduce* function to combine results.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from graphforge._logging import get_logger

logger = get_logger("map_reduce")

T = TypeVar("T")
U = TypeVar("U")


class MapReduce:
    """A node that applies map-reduce over a list state field.

    The map phase applies ``map_func`` to each element of the input list
    **in parallel** via a thread pool. The reduce phase calls ``reduce_func``
    on the collected results.

    Parameters
    ----------
    map_func:
        Callable ``(item, **kwargs) -> result`` applied to each element.
    reduce_func:
        Callable ``(results: list, **kwargs) -> value`` that combines
        mapped outputs into a single value.
    input_field:
        Name of the state field containing the input list (default: ``"items"``).
    output_field:
        Name of the state field to write the reduced result to (default: ``"result"``).
    max_workers:
        Maximum number of parallel workers (default: 4).
    map_kwargs:
        Additional keyword arguments passed to every ``map_func`` call.

    Examples
    --------
    .. code-block:: python

        from graphforge import Graph, GraphState, node_field
        from graphforge._map_reduce import MapReduce

        class ChunkState(GraphState):
            chunks: List[str] = node_field(default=[], merge="overwrite")
            summary: str = ""

        def analyze(chunk: str) -> str:
            return f"Analyzed: {chunk}"

        def combine(results: List[str]) -> str:
            return "\\n".join(results)

        mr = MapReduce(analyze, combine, input_field="chunks", output_field="summary")

        graph = Graph[ChunkState]()
        graph.add_node("process", mr)
        graph.add_edge("process", "__end__")
        graph.set_entry_point("process")
        compiled = graph.compile()

        result = compiled.invoke(ChunkState(chunks=["doc1", "doc2", "doc3"]))
        print(result.summary)
    """

    def __init__(
        self,
        map_func: Callable[..., U],
        reduce_func: Callable[[List[U]], Any],
        *,
        input_field: str = "items",
        output_field: str = "result",
        max_workers: int = 4,
        **map_kwargs: Any,
    ) -> None:
        self._map_func = map_func
        self._reduce_func = reduce_func
        self._input_field = input_field
        self._output_field = output_field
        self._max_workers = max_workers
        self._map_kwargs = map_kwargs

    def __call__(self, state: Any) -> Dict[str, Any]:
        """Execute the map-reduce operation.

        Parameters
        ----------
        state:
            The current graph state.

        Returns
        -------
        A dict with the output field set to the reduced result.
        """
        items: List[Any] = self._get_items(state)

        if not items:
            logger.debug("MapReduce: input field %r is empty", self._input_field)
            return {self._output_field: self._reduce_func([])}

        # Map phase (parallel)
        logger.info(
            "MapReduce: mapping %d items with %d workers",
            len(items), self._max_workers,
        )
        results: List[U] = [None] * len(items)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_map = {
                pool.submit(self._map_func, item, **self._map_kwargs): i
                for i, item in enumerate(items)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                results[idx] = future.result()

        # Reduce phase
        logger.debug("MapReduce: reducing %d results", len(results))
        reduced = self._reduce_func(results)

        return {self._output_field: reduced}

    def _get_items(self, state: Any) -> List[Any]:
        """Extract the input list from state."""
        if isinstance(state, dict):
            items = state.get(self._input_field, [])
        elif hasattr(state, self._input_field):
            items = getattr(state, self._input_field)
        else:
            items = []

        if items is None:
            items = []
        if not isinstance(items, list):
            items = [items]
        return items


__all__ = [
    "MapReduce",
]
