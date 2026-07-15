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

"""Guardrails — input/output safety validation at graph boundaries.

Guardrails allow you to define policies that run **before** a graph
executes (input validation) and **after** a graph executes (output
validation). They are the first line of defense for:

* Prompt injection detection
* PII / secret leakage prevention
* Content safety checks
* Business rule enforcement
* Format / schema validation

Usage::

    from graphforge.guardrails import InputGuardian, OutputGuardian, GuardrailResult

    # Define a custom guardrail
    class NoPII(Guardrail):
        def check_input(self, state: dict) -> GuardrailResult:
            text = str(state)
            if "ssn:" in text:
                return GuardrailResult.block("PII detected in input")
            return GuardrailResult.allow()

    # Apply to a compiled graph
    guardian = InputGuardian([NoPII()])
    compiled.graph.add_guardrail(guardian)
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from graphforge._logging import get_logger

logger = get_logger("guardrails")


# ---------------------------------------------------------------------------
# GuardrailResult
# ---------------------------------------------------------------------------


class GuardrailAction(str, enum.Enum):
    """Action to take after a guardrail check."""

    ALLOW = "allow"
    """Proceed with execution normally."""

    BLOCK = "block"
    """Block execution and raise an error."""

    REPLACE = "replace"
    """Replace the content and proceed (e.g., mask PII)."""


@dataclass
class GuardrailResult:
    """The result of a guardrail check.

    Parameters
    ----------
    action:
        What to do after the check.
    message:
        Human-readable message (e.g., reason for blocking).
    replacement:
        Replacement content when action is ``REPLACE``.
    metadata:
        Optional additional data for logging or debugging.
    """

    action: GuardrailAction = GuardrailAction.ALLOW
    message: str = ""
    replacement: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, message: str = "") -> "GuardrailResult":
        return cls(action=GuardrailAction.ALLOW, message=message)

    @classmethod
    def block(cls, message: str = "") -> "GuardrailResult":
        return cls(action=GuardrailAction.BLOCK, message=message)

    @classmethod
    def replace(
        cls, replacement: Dict[str, Any], message: str = ""
    ) -> "GuardrailResult":
        return cls(
            action=GuardrailAction.REPLACE,
            message=message,
            replacement=replacement,
        )


# ---------------------------------------------------------------------------
# Guardrail Protocol
# ---------------------------------------------------------------------------


class Guardrail(Protocol):
    """Protocol for guardrail implementations.

    Implement one or both of ``check_input`` and ``check_output``.
    All methods are optional — a guardrail that only validates input
    can omit ``check_output``.
    """

    def check_input(self, state: Dict[str, Any]) -> GuardrailResult:
        """Validate state **before** graph execution.

        Parameters
        ----------
        state:
            The input state as a dictionary.

        Returns
        -------
        A :class:`GuardrailResult` indicating the action to take.
        """
        return GuardrailResult.allow()

    def check_output(self, state: Dict[str, Any]) -> GuardrailResult:
        """Validate state **after** graph execution.

        Parameters
        ----------
        state:
            The output state as a dictionary.

        Returns
        -------
        A :class:`GuardrailResult` indicating the action to take.
        """
        return GuardrailResult.allow()


# ---------------------------------------------------------------------------
# Built-in guardrails
# ---------------------------------------------------------------------------


class NoOpGuardrail:
    """A guardrail that always allows. Useful as a base for testing."""

    def check_input(self, state: Dict[str, Any]) -> GuardrailResult:
        return GuardrailResult.allow()

    def check_output(self, state: Dict[str, Any]) -> GuardrailResult:
        return GuardrailResult.allow()


class FieldLengthGuardrail:
    """Guardrail that checks a string field does not exceed max length.

    Parameters
    ----------
    field_name:
        The state field to check.
    max_length:
        Maximum allowed length.
    action_on_exceed:
        Action when length is exceeded (default: ``"block"``).
    """

    def __init__(
        self,
        field_name: str,
        max_length: int = 10000,
        action_on_exceed: str = "block",
    ) -> None:
        self._field = field_name
        self._max = max_length
        self._action = action_on_exceed

    def check_input(self, state: Dict[str, Any]) -> GuardrailResult:
        value = state.get(self._field, "")
        if isinstance(value, str) and len(value) > self._max:
            msg = f"Field {self._field!r} exceeds max length ({len(value)} > {self._max})"
            if self._action == "block":
                return GuardrailResult.block(msg)
            return GuardrailResult.replace({}, msg)
        return GuardrailResult.allow()

    def check_output(self, state: Dict[str, Any]) -> GuardrailResult:
        return self.check_input(state)


# ---------------------------------------------------------------------------
# Guardian containers
# ---------------------------------------------------------------------------


class InputGuardian:
    """Container for input-side guardrails.

    Runs all registered guardrails in order before graph execution.
    If any guardrail returns ``BLOCK``, execution is prevented.

    Parameters
    ----------
    guardrails:
        List of guardrail instances.
    raise_on_block:
        If ``True`` (default), raise ``GuardrailError`` on block.
        If ``False``, return the blocked result.
    """

    def __init__(
        self,
        guardrails: Optional[Sequence[Guardrail]] = None,
        *,
        raise_on_block: bool = True,
    ) -> None:
        self._guardrails = list(guardrails or [])
        self._raise_on_block = raise_on_block

    def add(self, guardrail: Guardrail) -> None:
        self._guardrails.append(guardrail)

    def check(self, state: Dict[str, Any]) -> GuardrailResult:
        """Run all input guardrails against *state*.

        Returns the first non-ALLOW result, or the last result if all pass.
        """
        last_result = GuardrailResult.allow()
        for g in self._guardrails:
            if hasattr(g, "check_input"):
                result = g.check_input(state)
                last_result = result
                if result.action == GuardrailAction.BLOCK:
                    logger.warning(
                        "Input guardrail %s blocked: %s",
                        type(g).__name__, result.message,
                    )
                    if self._raise_on_block:
                        raise GuardrailError(result.message)
                    return result
                if result.action == GuardrailAction.REPLACE and result.replacement:
                    state.update(result.replacement)
        return last_result


class OutputGuardian:
    """Container for output-side guardrails.

    Runs all registered guardrails in order after graph execution.
    If any guardrail returns ``BLOCK``, the output is rejected.

    Parameters
    ----------
    guardrails:
        List of guardrail instances.
    raise_on_block:
        If ``True`` (default), raise ``GuardrailError`` on block.
        If ``False``, return the blocked result.
    """

    def __init__(
        self,
        guardrails: Optional[Sequence[Guardrail]] = None,
        *,
        raise_on_block: bool = True,
    ) -> None:
        self._guardrails = list(guardrails or [])
        self._raise_on_block = raise_on_block

    def add(self, guardrail: Guardrail) -> None:
        self._guardrails.append(guardrail)

    def check(self, state: Dict[str, Any]) -> GuardrailResult:
        """Run all output guardrails against *state*."""
        last_result = GuardrailResult.allow()
        for g in self._guardrails:
            if hasattr(g, "check_output"):
                result = g.check_output(state)
                last_result = result
                if result.action == GuardrailAction.BLOCK:
                    logger.warning(
                        "Output guardrail %s blocked: %s",
                        type(g).__name__, result.message,
                    )
                    if self._raise_on_block:
                        raise GuardrailError(result.message)
                    return result
                if result.action == GuardrailAction.REPLACE and result.replacement:
                    state.update(result.replacement)
        return last_result


# ---------------------------------------------------------------------------
# GuardrailError
# ---------------------------------------------------------------------------


class GuardrailError(Exception):
    """Raised when a guardrail blocks execution."""
    pass


__all__ = [
    "Guardrail",
    "GuardrailAction",
    "GuardrailError",
    "GuardrailResult",
    "InputGuardian",
    "NoOpGuardrail",
    "FieldLengthGuardrail",
    "OutputGuardian",
]
