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

"""Tool definition utilities — ``@tool`` decorator for auto-generating
OpenAI-compatible ToolDef from Python functions.

Usage::

    from graphforge.tools import tool

    @tool
    def search(query: str) -> str:
        \"\"\"Search the web for information.\"\"\"
        return f\"Results for: {query}\"

    # search.tool_def => OpenAI-compatible ToolDef dict
    # search.fn => the original function
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_type_hints,
)


# ---------------------------------------------------------------------------
# Tool descriptor — wraps a function with its JSON schema
# ---------------------------------------------------------------------------


class Tool:
    """Descriptor wrapping a callable with its auto-generated tool definition.

    Parameters
    ----------
    fn:
        The underlying Python function.
    name:
        Tool name (defaults to ``fn.__name__``).
    description:
        Tool description (defaults to ``fn.__doc__``).
    schema:
        JSON Schema for parameters (auto-generated from type hints).
    tool_def:
        Full OpenAI-compatible tool definition dict.
    """

    __slots__ = ("fn", "name", "description", "schema", "tool_def")

    def __init__(
        self,
        fn: Callable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or fn.__doc__ or ""
        self.schema = schema or _generate_schema(fn)
        self.tool_def = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
            "_func": self.fn,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Decorator that converts a Python function into a ``Tool``.

    The resulting ``Tool`` instance has a ``.tool_def`` property returning
    an OpenAI-compatible tool definition dict ready for use with ``ToolNode``.

    Parameters
    ----------
    fn:
        Function to decorate (optional, for ``@tool`` syntax).
    name:
        Override tool name (defaults to ``fn.__name__``).
    description:
        Override tool description (defaults to ``fn.__doc__``).

    Examples
    --------
    .. code-block:: python

        @tool
        def search(query: str) -> str:
            \"\"\"Search the web for information.\"\"\"
            return f\"Results for {query}\"

        # Use with ToolNode
        from graphforge.agents import ToolNode
        graph.add_node("agent", ToolNode(llm, tools=[search.tool_def]))

        # Or call directly
        result = search(query="hello")
    """
    if fn is not None:
        return _make_tool(fn, name=name, description=description)

    def decorator(f: Callable) -> Tool:
        return _make_tool(f, name=name, description=description)

    return decorator


def _make_tool(
    fn: Callable,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Tool:
    schema = _generate_schema(fn)
    return Tool(
        fn,
        name=name or fn.__name__,
        description=description or fn.__doc__ or "",
        schema=schema,
    )


# ---------------------------------------------------------------------------
# JSON Schema generation from type hints
# ---------------------------------------------------------------------------


def _generate_schema(fn: Callable) -> Dict[str, Any]:
    """Generate a JSON Schema dict from a function's type annotations."""
    sig = inspect.signature(fn)
    hints = _safe_get_type_hints(fn)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "return":
            continue
        if param_name == "self" or param_name == "cls":
            continue

        ptype = hints.get(param_name, str)
        schema_type = _type_to_json_schema(ptype)

        # Extract description from Annotated type or default
        description = ""
        if hasattr(ptype, "__metadata__"):
            meta = ptype.__metadata__
            if meta and isinstance(meta[0], str):
                description = meta[0]

        prop: Dict[str, Any] = {"type": schema_type}
        if description:
            prop["description"] = description

        # Default value
        if param.default is not inspect.Parameter.empty:
            prop["default"] = _serialize_default(param.default)
        else:
            required.append(param_name)

        properties[param_name] = prop

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _type_to_json_schema(tp: Any) -> str:
    """Map a Python type to a JSON Schema type string."""
    origin = getattr(tp, "__origin__", None)

    if origin is list or origin is List:
        return "array"
    if origin is dict or origin is Dict:
        return "object"
    if tp is str or tp is Optional[str]:
        return "string"
    if tp is int or tp is float or tp is Optional[int] or tp is Optional[float]:
        return "number"
    if tp is bool or tp is Optional[bool]:
        return "boolean"
    if tp is type(None):
        return "null"

    # Handle Optional[X]
    args = getattr(tp, "__args__", None)
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _type_to_json_schema(non_none[0])

    return "string"


def _safe_get_type_hints(fn: Callable) -> Dict[str, Any]:
    """Get type hints, safely falling back to empty dict on error."""
    try:
        return get_type_hints(fn) or {}
    except Exception:
        return {}


def _serialize_default(value: Any) -> Any:
    """Serialize a default parameter value."""
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if value is None:
        return None
    return str(value)


__all__ = [
    "Tool",
    "tool",
]
