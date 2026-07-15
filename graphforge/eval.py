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

"""Agent evaluation framework — test agent behavior programmatically.

Provides :class:`EvalCase`, :func:`evaluate`, and built-in metrics for
testing compiled graphs against expected outcomes.

Usage::

    from graphforge.eval import evaluate, EvalCase, exact_match

    cases = [
        EvalCase(
            input={"messages": [{"role": "user", "content": "Hello"}]},
            expected={"output": "Hi there!"},
            metrics=[exact_match("output")],
        ),
    ]

    results = evaluate(compiled, cases, state_type=MyState)
    print(results.summary())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from graphforge._graph import CompiledGraph
from graphforge._logging import get_logger

logger = get_logger("eval")


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


MetricFn = Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, str]]
"""
A metric function ``(actual_state, expected_state) -> (passed, message)``.
Return ``(True, "")`` on success or ``(False, "reason")`` on failure.
"""


def exact_match(field: str) -> MetricFn:
    """Create a metric that checks if a field matches exactly.

    Parameters
    ----------
    field:
        State field name to compare.

    Returns
    -------
    A metric function.
    """

    def _metric(actual: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str]:
        actual_val = actual.get(field)
        expected_val = expected.get(field)
        if actual_val == expected_val:
            return (True, "")
        return (False, f"Field {field!r}: expected {expected_val!r}, got {actual_val!r}")

    return _metric


def contains(field: str, substring: str) -> MetricFn:
    """Create a metric that checks if a string field contains a substring.

    Parameters
    ----------
    field:
        State field name.
    substring:
        Expected substring.

    Returns
    -------
    A metric function.
    """

    def _metric(actual: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str]:
        actual_val = str(actual.get(field, ""))
        if substring in actual_val or (expected.get(field) and str(expected.get(field, "")) in actual_val):
            return (True, "")
        target = substring or str(expected.get(field, ""))
        return (False, f"Field {field!r} does not contain {target!r}")

    return _metric


def json_match(field: str) -> MetricFn:
    """Create a metric that compares serialized JSON fields.

    Useful for comparing list/dict fields where order may vary.
    """

    def _metric(actual: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str]:
        actual_val = actual.get(field)
        expected_val = expected.get(field)
        try:
            a_json = json.dumps(actual_val, sort_keys=True, default=str)
            e_json = json.dumps(expected_val, sort_keys=True, default=str)
            if a_json == e_json:
                return (True, "")
            return (False, f"Field {field!r}: JSON mismatch")
        except (TypeError, ValueError) as e:
            return (False, f"Field {field!r}: JSON compare error: {e}")

    return _metric


# ---------------------------------------------------------------------------
# EvalCase — single test case
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """A single evaluation case for testing a compiled graph.

    Parameters
    ----------
    input:
        Input state data (will be passed to ``graph.invoke()``).
    expected:
        Expected output state data (compared against actual output).
    metrics:
        List of metric functions to evaluate (default: ``[exact_match]`` for
        each key in ``expected``).
    name:
        Optional human-readable name for the case.
    config:
        Optional runtime config passed to ``graph.invoke()``.
    """

    input: Dict[str, Any]
    expected: Dict[str, Any]
    metrics: Optional[List[MetricFn]] = None
    name: str = ""
    config: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# EvalResult — result of evaluating a single case
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result of evaluating a single :class:`EvalCase`.

    Parameters
    ----------
    case:
        The original test case.
    passed:
        Whether all metrics passed.
    actual:
        The actual output state dict.
    metric_results:
        List of ``(metric_name, passed, message)`` tuples.
    error:
        Optional error message if execution failed.
    """

    case: EvalCase
    passed: bool
    actual: Dict[str, Any]
    metric_results: List[Tuple[str, bool, str]] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# EvalResults — collection of evaluation results
# ---------------------------------------------------------------------------


@dataclass
class EvalResults:
    """Results from evaluating multiple test cases.

    Parameters
    ----------
    results:
        List of individual :class:`EvalResult` instances.
    """

    results: List[EvalResult]

    @property
    def passed(self) -> int:
        """Number of passing cases."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        """Number of failing cases."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        """Total number of cases."""
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        """Pass rate as a float between 0.0 and 1.0."""
        if self.total == 0:
            return 1.0
        return self.passed / self.total

    def summary(self) -> str:
        """Return a human-readable summary of results."""
        return (
            f"EvalResults: {self.passed}/{self.total} passed "
            f"({self.pass_rate * 100:.1f}%)"
        )

    def failures(self) -> List[EvalResult]:
        """Return only the failing results."""
        return [r for r in self.results if not r.passed]


# ---------------------------------------------------------------------------
# evaluate — main entry point
# ---------------------------------------------------------------------------


def evaluate(
    graph: CompiledGraph,
    cases: Sequence[EvalCase],
    state_type: Type,
    *,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> EvalResults:
    """Run evaluation cases against a compiled graph.

    For each case:
    1. Reconstruct state from ``case.input`` using ``state_type``
    2. Invoke the graph
    3. Run each metric against the actual output
    4. Collect results

    Parameters
    ----------
    graph:
        The compiled graph to evaluate.
    cases:
        List of test cases.
    state_type:
        Pydantic model class for state reconstruction.
    config:
        Optional runtime config passed to each invocation.
    verbose:
        If ``True``, log each case result.

    Returns
    -------
    An :class:`EvalResults` instance with all results.

    Examples
    --------
    .. code-block:: python

        from graphforge.eval import evaluate, EvalCase, exact_match

        cases = [
            EvalCase(
                input={"messages": [{"role": "user", "content": "Hi"}]},
                expected={"messages": [...]},
                metrics=[exact_match("messages")],
            ),
        ]

        results = evaluate(graph, cases, state_type=ChatState)
        print(results.summary())
    """
    eval_results: List[EvalResult] = []

    for i, case in enumerate(cases):
        name = case.name or f"case_{i}"
        try:
            # Reconstruct state
            state = state_type.model_validate(case.input)

            # Execute
            cfg = {**(config or {}), **(case.config or {})}
            actual_state = graph.invoke(state, config=cfg)

            # Serialize
            actual_dict: Dict[str, Any] = {}
            if hasattr(actual_state, "model_dump"):
                actual_dict = actual_state.model_dump()
            else:
                actual_dict = dict(actual_state)

            # Run metrics
            metrics = case.metrics or []
            if not metrics:
                # Default: exact_match for each key in expected
                metrics = [exact_match(k) for k in case.expected]

            metric_results: List[Tuple[str, bool, str]] = []
            all_passed = True

            for metric_fn in metrics:
                metric_name = getattr(metric_fn, "__name__", str(metric_fn))
                try:
                    passed, msg = metric_fn(actual_dict, case.expected)
                except Exception as e:
                    passed, msg = False, f"Metric error: {e}"
                metric_results.append((metric_name, passed, msg))
                if not passed:
                    all_passed = False

            result = EvalResult(
                case=case,
                passed=all_passed,
                actual=actual_dict,
                metric_results=metric_results,
            )

        except Exception as e:
            logger.exception("Eval case %r failed with exception", name)
            result = EvalResult(
                case=case,
                passed=False,
                actual={},
                error=str(e),
            )

        eval_results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            logger.info("Eval %s: %s", status, name)

    return EvalResults(results=eval_results)


__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalResults",
    "MetricFn",
    "contains",
    "evaluate",
    "exact_match",
    "json_match",
]
