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

"""Tests for guardrails module."""

from __future__ import annotations

import pytest
from graphforge.guardrails import (
    FieldLengthGuardrail,
    GuardrailAction,
    GuardrailError,
    GuardrailResult,
    InputGuardian,
    NoOpGuardrail,
    OutputGuardian,
)


class TestGuardrailResult:
    def test_allow(self) -> None:
        r = GuardrailResult.allow()
        assert r.action == GuardrailAction.ALLOW

    def test_allow_with_message(self) -> None:
        r = GuardrailResult.allow("all good")
        assert r.action == GuardrailAction.ALLOW
        assert r.message == "all good"

    def test_block(self) -> None:
        r = GuardrailResult.block("blocked")
        assert r.action == GuardrailAction.BLOCK
        assert r.message == "blocked"

    def test_replace(self) -> None:
        r = GuardrailResult.replace({"key": "new_value"}, "replaced")
        assert r.action == GuardrailAction.REPLACE
        assert r.replacement == {"key": "new_value"}
        assert r.message == "replaced"


class TestNoOpGuardrail:
    def test_check_input_allows(self) -> None:
        g = NoOpGuardrail()
        result = g.check_input({"key": "value"})
        assert result.action == GuardrailAction.ALLOW

    def test_check_output_allows(self) -> None:
        g = NoOpGuardrail()
        result = g.check_output({"key": "value"})
        assert result.action == GuardrailAction.ALLOW


class TestFieldLengthGuardrail:
    def test_short_field_passes(self) -> None:
        g = FieldLengthGuardrail("text", max_length=100)
        result = g.check_input({"text": "short"})
        assert result.action == GuardrailAction.ALLOW

    def test_long_field_blocked(self) -> None:
        g = FieldLengthGuardrail("text", max_length=5)
        result = g.check_input({"text": "too long text"})
        assert result.action == GuardrailAction.BLOCK

    def test_long_field_replace(self) -> None:
        g = FieldLengthGuardrail("text", max_length=5, action_on_exceed="replace")
        result = g.check_input({"text": "too long text"})
        assert result.action == GuardrailAction.REPLACE

    def test_check_output(self) -> None:
        g = FieldLengthGuardrail("text", max_length=5)
        result = g.check_output({"text": "too long text"})
        assert result.action == GuardrailAction.BLOCK

    def test_missing_field_passes(self) -> None:
        g = FieldLengthGuardrail("text", max_length=5)
        result = g.check_input({"other": "value"})
        assert result.action == GuardrailAction.ALLOW


class TestInputGuardian:
    def test_empty_guardrails_allows(self) -> None:
        guardian = InputGuardian([])
        result = guardian.check({"key": "value"})
        assert result.action == GuardrailAction.ALLOW

    def test_allow_when_all_pass(self) -> None:
        guardian = InputGuardian([NoOpGuardrail()])
        result = guardian.check({"key": "value"})
        assert result.action == GuardrailAction.ALLOW

    def test_block_raises_by_default(self) -> None:
        g = FieldLengthGuardrail("text", max_length=5)
        guardian = InputGuardian([g])
        with pytest.raises(GuardrailError, match="exceeds max length"):
            guardian.check({"text": "a" * 100})

    def test_block_returns_result_when_not_raise(self) -> None:
        g = FieldLengthGuardrail("text", max_length=5)
        guardian = InputGuardian([g], raise_on_block=False)
        result = guardian.check({"text": "a" * 100})
        assert result.action == GuardrailAction.BLOCK

    def test_multiple_guardrails_all_pass(self) -> None:
        guardian = InputGuardian([
            NoOpGuardrail(),
            FieldLengthGuardrail("text", max_length=100),
        ])
        result = guardian.check({"text": "hello"})
        assert result.action == GuardrailAction.ALLOW

    def test_add_guardrail(self) -> None:
        guardian = InputGuardian([])
        guardian.add(NoOpGuardrail())
        result = guardian.check({})
        assert result.action == GuardrailAction.ALLOW


class TestOutputGuardian:
    def test_empty_output_allows(self) -> None:
        guardian = OutputGuardian([])
        result = guardian.check({"output": "value"})
        assert result.action == GuardrailAction.ALLOW

    def test_block_raises(self) -> None:
        g = FieldLengthGuardrail("output", max_length=5)
        guardian = OutputGuardian([g])
        with pytest.raises(GuardrailError):
            guardian.check({"output": "a" * 100})

    def test_block_returns_result_when_not_raise(self) -> None:
        g = FieldLengthGuardrail("output", max_length=5)
        guardian = OutputGuardian([g], raise_on_block=False)
        result = guardian.check({"output": "a" * 100})
        assert result.action == GuardrailAction.BLOCK

    def test_add_output_guardrail(self) -> None:
        guardian = OutputGuardian([])
        guardian.add(NoOpGuardrail())
        result = guardian.check({"output": "test"})
        assert result.action == GuardrailAction.ALLOW
