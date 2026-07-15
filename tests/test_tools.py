"""Tests for @tool decorator."""
from graphforge.tools import tool, Tool


class TestToolDecorator:
    def test_basic_tool(self) -> None:
        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"results for {query}"

        assert isinstance(search, Tool)
        assert search.name == "search"
        assert "web" in search.description
        assert search.tool_def["function"]["name"] == "search"

    def test_call_tool(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            return a + b

        assert add(a=1, b=2) == 3

    def test_tool_def_format(self) -> None:
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello {name}"

        td = greet.tool_def
        assert td["type"] == "function"
        assert "_func" in td
        assert td["_func"]("World") == "Hello World"

    def test_custom_name(self) -> None:
        @tool(name="custom_name")
        def my_func(x: str) -> str:
            return x

        assert my_func.name == "custom_name"

    def test_schema_generation(self) -> None:
        @tool
        def process(query: str, limit: int = 10) -> str:
            """Process query."""
            return query

        params = tool.fn.__code__ if False else {}
        schema = process.tool_def["function"]["parameters"]
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]

    def test_no_annotations(self) -> None:
        @tool
        def simple(x) -> str:  # type: ignore
            return str(x)

        t = simple  # the tool itself
        assert t.tool_def["function"]["name"] == "simple"
