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

"""Tests for MapReduce node."""

from __future__ import annotations

import time
from typing import List

import pytest

from graphforge import Append, Graph, GraphState, node_field
from graphforge._map_reduce import MapReduce


# ── Fixtures ──────────────────────────────────────────────────────────


class AnalysisState(GraphState):
    chunks: List[str] = node_field(default=[], merge="overwrite")
    summary: str = ""
    counts: List[int] = node_field(default=[], merge="append")


def slow_map(item: str) -> str:
    time.sleep(0.01)
    return f"Processed: {item}"


def join_results(results: List[str]) -> str:
    return "\n".join(results)


def count_length(item: str) -> int:
    return len(item)


def sum_counts(results: List[int]) -> int:
    return sum(results)


# ── Tests ─────────────────────────────────────────────────────────────


class TestMapReduce:
    def test_basic_map_reduce(self) -> None:
        mr = MapReduce(slow_map, join_results, input_field="chunks", output_field="summary")
        state = AnalysisState(chunks=["a", "b", "c"])
        result = mr(state)
        assert "Processed: a" in result["summary"]
        assert "Processed: b" in result["summary"]
        assert "Processed: c" in result["summary"]

    def test_empty_input_list(self) -> None:
        mr = MapReduce(slow_map, join_results, input_field="chunks", output_field="summary")
        state = AnalysisState(chunks=[])
        result = mr(state)
        assert result["summary"] == ""

    def test_map_reduce_in_graph(self) -> None:
        mr = MapReduce(
            lambda x: str(len(x)),
            lambda results: " | ".join(sorted(set(results))),
            input_field="chunks", output_field="summary",
        )

        graph = Graph[AnalysisState]()
        graph.add_node("process", mr)
        graph.add_edge("process", "__end__")
        graph.set_entry_point("process")
        compiled = graph.compile()

        result = compiled.invoke(AnalysisState(chunks=["one", "three", "five"]))
        # lengths: one=3, three=5, five=4
        assert "3" in result.summary
        assert "4" in result.summary
        assert "5" in result.summary

    def test_custom_workers(self) -> None:
        mr = MapReduce(
            slow_map, join_results,
            input_field="chunks", output_field="summary",
            max_workers=2,
        )
        state = AnalysisState(chunks=["x", "y"])
        t0 = time.time()
        result = mr(state)
        elapsed = time.time() - t0
        assert "Processed: x" in result["summary"]
        # With 2 workers for 2 items, should be fast
        assert elapsed < 1.0

    def test_state_dict_compatibility(self) -> None:
        mr = MapReduce(
            lambda x: x.upper(),
            lambda results: " ".join(results),
            input_field="data",
            output_field="result",
        )
        result = mr({"data": ["a", "b", "c"]})
        assert result["result"] == "A B C"

    def test_non_list_field(self) -> None:
        mr = MapReduce(
            lambda x: x,
            lambda results: results,
            input_field="value",
            output_field="result",
        )
        # When the field isn't a list, it gets wrapped
        result = mr({"value": "single"})
        assert result["result"] == ["single"]

    def test_map_kwargs(self) -> None:
        def map_with_prefix(item: str, prefix: str = "") -> str:
            return f"{prefix}{item}"

        mr = MapReduce(
            map_with_prefix,
            lambda results: " ".join(results),
            input_field="chunks", output_field="summary",
            prefix="PREFIX_",
        )
        state = AnalysisState(chunks=["a", "b"])
        result = mr(state)
        assert "PREFIX_a" in result["summary"]
        assert "PREFIX_b" in result["summary"]
