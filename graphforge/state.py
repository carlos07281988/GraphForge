"""State schema and merge engine for GraphForge.

Every graph execution revolves around a **state object** that flows through
nodes. Nodes emit *partial updates* that the framework folds into the current
state using a configurable merge strategy.

Core design decisions (vs LangGraph):

1. **Pydantic v2 base** — validation, serialization, and schema generation
   are built-in; no half-baked TypedDict + ``__reducers__`` magic.
2. **Explicit merge strategies** — each field declares how updates are folded
   (overwrite, append, or custom reducer). No implicit ``add_messages``-style
   surprises.
3. **Immutable snapshots** — each execution step produces a new state instance;
   the framework never mutates state in place.
4. **Type-safe** — :meth:`apply` is fully typed; a node's return type flows
   through the merge engine.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
)

from pydantic import BaseModel, Field
from graphforge._compat import IS_PYDANTIC_V2, PydanticConfigDict, model_copy as _compat_model_copy
from typing_extensions import Self

from graphforge._logging import get_logger

logger = get_logger("state")


# ---------------------------------------------------------------------------
# Field merge strategy
# ---------------------------------------------------------------------------


class MergeStrategy(str, Enum):
    """How a state field should be folded when a node returns an update.

    ``OVERWRITE`` — the old value is replaced entirely (default).
    ``APPEND`` — the old *list* value is extended with the new iterable.
    ``REDUCE`` — a user-supplied callable folds ``(old, new) -> value``.
    """

    OVERWRITE = "overwrite"
    APPEND = "append"
    REDUCE = "reduce"


# ---------------------------------------------------------------------------
# Reducer descriptor — stored as ``json_schema_extra`` on Pydantic fields
# ---------------------------------------------------------------------------


class ReducerDescriptor:
    """Carries merge-strategy metadata for a single state field.

    Attached to the Pydantic field via ``json_schema_extra={"reducer": ...}``.
    """

    __slots__ = ("strategy", "func")

    def __init__(
        self,
        strategy: MergeStrategy = MergeStrategy.OVERWRITE,
        func: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self.strategy = strategy
        self.func = func


# ---------------------------------------------------------------------------
# Append marker — signals that a list field uses APPEND semantics
# ---------------------------------------------------------------------------


class Append(list):
    """Marker type for list fields that should use append-merge semantics.

    When a node returns ``{"messages": Append([new_msg])}``, the framework
    *extends* the existing list rather than replacing it.
    """

    pass


# ---------------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------------

_ReducerMap = Dict[str, ReducerDescriptor]


def _build_reducer_map(state_cls: Type[BaseModel]) -> _ReducerMap:
    """Introspect a state model and build the reducer lookup table."""
    reducers: _ReducerMap = {}
    for field_name, field_info in state_cls.model_fields.items():
        extra = field_info.json_schema_extra or {}
        desc = extra.get("reducer")
        if isinstance(desc, ReducerDescriptor):
            reducers[field_name] = desc
    logger.debug("Built reducer map for %s: %d fields", state_cls.__name__, len(reducers))
    return reducers


def merge_state(
    state: BaseModel,
    updates: Dict[str, Any],
    *,
    reducers: Optional[_ReducerMap] = None,
) -> BaseModel:
    """Apply *updates* to *state*, returning a new state instance.

    Parameters
    ----------
    state:
        Current state snapshot.
    updates:
        Partial dictionary of field -> new value.
    reducers:
        Pre-built reducer map (cached from :func:`_build_reducer_map`).
        Built on the fly if omitted (slower).

    Returns
    -------
    A new state instance with updates folded in.
    """
    if not updates:
        logger.debug("merge_state: no updates, returning unchanged")
        return state

    if reducers is None:
        reducers = _build_reducer_map(type(state))

    resolved: Dict[str, Any] = {}
    logger.debug(
        "merge_state: processing %d fields: %s", len(updates), list(updates.keys())
    )

    for key, new_val in updates.items():
        reducer = reducers.get(key)
        if reducer is None or reducer.strategy == MergeStrategy.OVERWRITE:
            resolved[key] = new_val
        elif reducer.strategy == MergeStrategy.APPEND:
            old_val = getattr(state, key, None)
            if old_val is None:
                resolved[key] = list(new_val) if new_val is not None else []
            elif isinstance(old_val, list):
                if isinstance(new_val, Append):
                    resolved[key] = old_val + list(new_val)
                elif isinstance(new_val, list):
                    resolved[key] = [*old_val, *new_val]
                else:
                    resolved[key] = [*old_val, new_val]
            else:
                existing = [old_val] if old_val is not None else []
                if isinstance(new_val, list):
                    resolved[key] = existing + new_val
                else:
                    resolved[key] = existing + [new_val]
        elif reducer.strategy == MergeStrategy.REDUCE and reducer.func is not None:
            old_val = getattr(state, key, None)
            resolved[key] = reducer.func(old_val, new_val)

    logger.debug("merge_state: resolved %d fields: %s", len(resolved), list(resolved.keys()))
    return _compat_model_copy(state, update=resolved, deep=True)


# ---------------------------------------------------------------------------
# node_field — declarative field factory
# ---------------------------------------------------------------------------


def node_field(
    default: Any = None,
    *,
    merge: Union[MergeStrategy, str] = MergeStrategy.OVERWRITE,
    reducer: Optional[Callable[[Any, Any], Any]] = None,
    description: str = "",
    **extra: Any,
) -> Any:
    """Declare a graph-state field with an explicit merge strategy.

    Parameters
    ----------
    default:
        Default value for the field.
    merge:
        One of ``"overwrite"``, ``"append"``, or ``"reduce"``.
    reducer:
        Custom reducer callable ``(old, new) -> value``. Required when
        ``merge="reduce"``; ignored otherwise.
    description:
        Human-readable field description.
    **extra:
        Additional keyword arguments forwarded to Pydantic's :func:`Field`.

    Example::

        class ChatState(GraphState):
            messages: list[Message] = node_field(
                default=[],
                merge="append",
                description="Conversation history",
            )
            turn_count: int = node_field(default=0, merge="overwrite")
    """
    if isinstance(merge, str):
        merge = MergeStrategy(merge)

    if merge == MergeStrategy.REDUCE and reducer is None:
        raise ValueError(
            "A callable `reducer` is required when merge='reduce'."
        )

    desc = ReducerDescriptor(strategy=merge, func=reducer)
    return Field(
        default=default,
        description=description,
        json_schema_extra={"reducer": desc},
        **extra,
    )


# ---------------------------------------------------------------------------
# GraphState — base model for all graph state schemas
# ---------------------------------------------------------------------------


class GraphState(BaseModel):
    """Base class for all graph execution state.

    Subclass this to define your graph's state schema. Each field declared
    on the subclass participates in the merge system via :func:`node_field`.
    Fields declared with a plain Pydantic :func:`Field` default to
    ``overwrite`` semantics.

    Examples
    --------
    >>> class SearchState(GraphState):
    ...     query: str
    ...     results: list[str] = node_field(default=[], merge="append")
    ...
    >>> state = SearchState(query="hello")
    >>> merged = state.apply(results=["first result"])
    >>> merged.results
    ['first result']
    >>> merged2 = merged.apply(results=["second result"])
    >>> merged2.results
    ['first result', 'second result']
    """

    #: Cache of reducer descriptors keyed by field name.
    _reducers: ClassVar[Optional[_ReducerMap]] = None

    model_config = PydanticConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
    arbitrary_types_allowed=True,
)

    if not IS_PYDANTIC_V2:
        class Config:
            extra = "forbid"
            validate_assignment = True
            arbitrary_types_allowed = True

    def apply(self, **updates: Any) -> Self:
        """Return a new state instance with *updates* merged in.

        This is the primary way nodes produce new state. Instead of mutating
        the state object, a node returns the fields it wants to change and
        the framework creates a new snapshot.
        """
        cls = type(self)
        if cls._reducers is None:
            cls._reducers = _build_reducer_map(cls)
        result = merge_state(self, updates, reducers=cls._reducers)
        logger.debug("GraphState.apply(%s): %d updates", cls.__name__, len(updates))
        return cast(Self, result)


__all__ = [
    "Append",
    "GraphState",
    "MergeStrategy",
    "ReducerDescriptor",
    "merge_state",
    "node_field",
]
