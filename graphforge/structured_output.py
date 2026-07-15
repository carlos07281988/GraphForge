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

"""Structured output utilities — enforce LLM outputs conform to a Pydantic
model schema, with retry and validation.

Usage::

    from pydantic import BaseModel
    from graphforge.structured_output import with_structured_output

    class SearchResult(BaseModel):
        query: str
        results: list[str]
        confidence: float

    # Wrap any LLM callable
    llm = with_structured_output(my_llm_func, SearchResult)
    result = llm(messages)  # returns validated SearchResult instance
"""

from __future__ import annotations

import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from graphforge._logging import get_logger

try:
    from pydantic import BaseModel, ValidationError

    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False
    BaseModel = None  # type: ignore[assignment,misc]
    ValidationError = None  # type: ignore[assignment,misc]

logger = get_logger("structured_output")

T = TypeVar("T", bound="BaseModel")


# ---------------------------------------------------------------------------
# StructuredOutputNode — wraps any LLM callable
# ---------------------------------------------------------------------------


class StructuredOutputWrapper(Generic[T]):
    """Wraps an LLM callable to return validated Pydantic model instances.

    Parameters
    ----------
    llm_func:
        Callable ``(messages, **kwargs) -> str`` that returns text output.
    schema:
        Pydantic model class to validate against.
    max_retries:
        Maximum number of retries on validation failure (default: 3).
    json_mode:
        If ``True``, tells the LLM to output JSON (default: ``True``).
    prompt_template:
        Custom prompt template for JSON mode. Must contain ``{schema}``.
    """

    def __init__(
        self,
        llm_func: Callable[..., str],
        schema: Type[T],
        *,
        max_retries: int = 3,
        json_mode: bool = True,
        prompt_template: Optional[str] = None,
    ) -> None:
        self._llm_func = llm_func
        self._schema = schema
        self._max_retries = max_retries
        self._json_mode = json_mode
        self._prompt_template = prompt_template or _DEFAULT_PROMPT

    @property
    def schema(self) -> Type[T]:
        return self._schema

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Invoke the LLM and parse the output.

        Returns
        -------
        A validated instance of ``schema``.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                raw = self._invoke_llm(*args, **kwargs)
                data = self._parse(raw)
                return self._schema.model_validate(data)
            except (ValidationError, json.JSONDecodeError, TypeError) as e:
                last_error = e
                logger.warning(
                    "Structured output validation failed (attempt %d/%d): %s",
                    attempt + 1, self._max_retries, e,
                )
                if attempt < self._max_retries - 1:
                    # Append error feedback for retry
                    kwargs["_feedback"] = f"Validation error: {e}. Please fix the output."
                    kwargs["_last_raw"] = raw if 'raw' in dir() else ""

        raise ValueError(
            f"Failed to produce valid output after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _invoke_llm(self, *args: Any, **kwargs: Any) -> str:
        """Call the underlying LLM function."""
        if self._json_mode:
            schema_json = self._schema.model_json_schema()
            prompt = self._prompt_template.format(schema=json.dumps(schema_json, indent=2))

            # Inject prompt into messages
            if args and isinstance(args[0], list):
                messages = list(args[0])
                messages.append({"role": "user", "content": prompt})
                args = (messages,) + args[1:]

        return self._llm_func(*args, **kwargs)

    def _parse(self, raw: str) -> Dict[str, Any]:
        """Parse LLM output string to dict."""
        # Try JSON parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())

        raise json.JSONDecodeError(f"Could not parse LLM output as JSON", raw, 0)


_DEFAULT_PROMPT = """You must respond with valid JSON that matches this schema:
{schema}

Respond with ONLY valid JSON. No markdown formatting, no explanation."""


def with_structured_output(
    llm_func: Callable[..., str],
    schema: Type[T],
    *,
    max_retries: int = 3,
    json_mode: bool = True,
) -> StructuredOutputWrapper[T]:
    """Wrap an LLM callable to return validated Pydantic model instances.

    Parameters
    ----------
    llm_func:
        Callable ``(messages, **kwargs) -> str`` that returns text output.
    schema:
        Pydantic model class to validate against.
    max_retries:
        Maximum retries on validation failure (default: 3).
    json_mode:
        If ``True``, appends a JSON prompt to force structured output.

    Returns
    -------
    A callable that returns validated ``schema`` instances.

    Examples
    --------
    .. code-block:: python

        from pydantic import BaseModel
        from graphforge.structured_output import with_structured_output

        class Weather(BaseModel):
            city: str
            temperature: float
            conditions: str

        # Wrap your LLM function
        llm = with_structured_output(my_llm_func, Weather)

        # Use as a graph node
        def weather_node(state):
            result = llm(state.messages)
            return {"weather": result.model_dump()}
    """
    return StructuredOutputWrapper(
        llm_func,
        schema,
        max_retries=max_retries,
        json_mode=json_mode,
    )


__all__ = [
    "StructuredOutputWrapper",
    "with_structured_output",
]
