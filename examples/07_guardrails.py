"""
Example 07: Guardrails
Input/output safety validation at graph boundaries.
"""
from graphforge import Graph, GraphState, configure_logging
from graphforge.guardrails import (
    InputGuardian, OutputGuardian,
    FieldLengthGuardrail, GuardrailResult,
    GuardrailAction, GuardrailError,
)

configure_logging()


class SafeState(GraphState):
    prompt: str = ""
    output: str = ""
    safe: bool = False


# Custom guardrail: block PII-like patterns
class PIIGuardrail:
    def check_input(self, state):
        text = str(state)
        patterns = ["ssn:", "credit card:", "password="]
        for pattern in patterns:
            if pattern in text.lower():
                return GuardrailResult.block(f"Blocked: {pattern} detected in input")
        return GuardrailResult.allow()

    def check_output(self, state):
        text = str(state)
        if "secret:" in text.lower():
            return GuardrailResult.replace({"output": "[REDACTED]"}, "Output contained secrets")
        return GuardrailResult.allow()


# Built-in guardrail: limit prompt length
max_length = FieldLengthGuardrail("prompt", max_length=200)


# Define nodes
def process_safe(state: SafeState) -> dict:
    return {"output": f"Processed: {state.prompt}", "safe": True}


# Build graph
graph = Graph[SafeState]()
graph.add_node("process", process_safe)
graph.add_edge("process", "__end__")
graph.set_entry_point("process")
compiled = graph.compile(state_type=SafeState)

# Test 1: Normal input passes guardrails
print("=== Test 1: Normal Input ===")
guardian = InputGuardian([PIIGuardrail(), max_length], raise_on_block=False)
result = guardian.check({"prompt": "What is the weather today?"})
print(f"Guardrail result: {result.action.value} — {result.message}")

if result.action != GuardrailAction.BLOCK:
    state = SafeState(prompt="What is the weather today?")
    final = compiled.invoke(state)
    print(f"Output: {final.output}")

# Test 2: PII input is blocked
print("\n=== Test 2: PII Input ===")
try:
    guardian_blocking = InputGuardian([PIIGuardrail()], raise_on_block=True)
    guardian_blocking.check({"prompt": "My SSN: 123-45-6789"})
    print("PASSED (unexpected)")
except GuardrailError as e:
    print(f"Correctly blocked: {e}")

# Test 3: Field too long
print("\n=== Test 3: Field Too Long ===")
try:
    guardian_long = InputGuardian([max_length], raise_on_block=True)
    guardian_long.check({"prompt": "A" * 500})
except GuardrailError as e:
    print(f"Correctly blocked: {e}")

# Test 4: Output guardrail
print("\n=== Test 4: Output Guardrail ===")
output_guardian = OutputGuardian([PIIGuardrail()], raise_on_block=False)
result = output_guardian.check({"output": "The secret: my password is 123"})
print(f"Output guardrail: {result.action}")
if result.replacement:
    print(f"  Replaced with: {result.replacement}")
