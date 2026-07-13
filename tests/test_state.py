"""Tests for the state management module."""

from __future__ import annotations

import pytest
from graphforge.state import (
    Append,
    GraphState,
    MergeStrategy,
    ReducerDescriptor,
    merge_state,
    node_field,
)


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class SimpleState(GraphState):
    name: str = ""
    count: int = 0


class AppendState(GraphState):
    items: list = node_field(default=[], merge="append")


class CustomReducerState(GraphState):
    total: int = node_field(
        default=0,
        merge="reduce",
        reducer=lambda old, new: (old or 0) + new,
    )


# ---------------------------------------------------------------------------
# GraphState.apply
# ---------------------------------------------------------------------------


class TestGraphState:
    def test_apply_overwrite(self) -> None:
        s = SimpleState(name="hello", count=1)
        s2 = s.apply(name="world")
        assert s2.name == "world"
        assert s2.count == 1  # unchanged
        assert s.name == "hello"  # original unchanged (immutable)

    def test_apply_empty(self) -> None:
        s = SimpleState(name="hello", count=1)
        s2 = s.apply()
        assert s2.name == "hello"
        assert s2.count == 1

    def test_apply_append(self) -> None:
        s = AppendState()
        s2 = s.apply(items=["a"])
        s3 = s2.apply(items=["b"])
        assert s3.items == ["a", "b"]

    def test_apply_append_from_none(self) -> None:
        s = AppendState()
        s2 = s.apply(items=["a"])
        assert s2.items == ["a"]

    def test_apply_custom_reducer(self) -> None:
        s = CustomReducerState()
        s2 = s.apply(total=5)
        assert s2.total == 5
        s3 = s2.apply(total=3)
        assert s3.total == 8

    def test_immutability(self) -> None:
        s = SimpleState(name="hello")
        s2 = s.apply(name="world")
        assert s.name == "hello"  # original unchanged
        assert s2.name == "world"

    def test_model_dump(self) -> None:
        s = SimpleState(name="test", count=42)
        d = s.model_dump()
        assert d == {"name": "test", "count": 42}


# ---------------------------------------------------------------------------
# merge_state standalone
# ---------------------------------------------------------------------------


class TestMergeState:
    def test_empty_updates(self) -> None:
        s = SimpleState(name="x", count=1)
        result = merge_state(s, {})
        assert result.name == "x"

    def test_overwrite(self) -> None:
        s = SimpleState(name="x", count=1)
        result = merge_state(s, {"name": "y"})
        assert result.name == "y"
        assert result.count == 1

    def test_append_new_key(self) -> None:
        s = AppendState()
        result = merge_state(s, {"items": ["a"]})
        assert result.items == ["a"]

    def test_append_existing(self) -> None:
        s = AppendState(items=["a"])
        result = merge_state(s, {"items": ["b"]})
        assert result.items == ["a", "b"]


# ---------------------------------------------------------------------------
# node_field
# ---------------------------------------------------------------------------


class TestNodeField:
    def test_default_overwrite(self) -> None:
        field = node_field(default=0)
        desc = field.json_schema_extra["reducer"]
        assert desc.strategy == MergeStrategy.OVERWRITE

    def test_append(self) -> None:
        field = node_field(default=[], merge="append")
        desc = field.json_schema_extra["reducer"]
        assert desc.strategy == MergeStrategy.APPEND

    def test_reduce_needs_func(self) -> None:
        with pytest.raises(ValueError, match="reducer"):
            node_field(merge="reduce")

    def test_reduce_with_func(self) -> None:
        field = node_field(default=0, merge="reduce", reducer=lambda o, n: o + n)
        desc = field.json_schema_extra["reducer"]
        assert desc.strategy == MergeStrategy.REDUCE
        assert desc.func is not None


# ---------------------------------------------------------------------------
# Append marker
# ---------------------------------------------------------------------------


class TestAppend:
    def test_is_list(self) -> None:
        a = Append([1, 2, 3])
        assert list(a) == [1, 2, 3]
        assert isinstance(a, list)

    def test_add(self) -> None:
        a = Append([1])
        b = list(a) + [2]
        assert b == [1, 2]


class TestReducerDescriptor:
    def test_defaults(self) -> None:
        r = ReducerDescriptor()
        assert r.strategy == MergeStrategy.OVERWRITE
        assert r.func is None

    def test_custom(self) -> None:
        r = ReducerDescriptor(strategy=MergeStrategy.APPEND)
        assert r.strategy == MergeStrategy.APPEND
        assert r.func is None

    def test_with_func(self) -> None:
        fn = lambda o, n: o + n
        r = ReducerDescriptor(strategy=MergeStrategy.REDUCE, func=fn)
        assert r.strategy == MergeStrategy.REDUCE
        assert r.func is fn
