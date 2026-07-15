"""Tests for unified serve() function."""
from graphforge.serve import serve, UnifiedServer
from graphforge import Graph, GraphState


class ServeState(GraphState):
    value: str = ""


def dummy_fn(state):
    return {"value": "ok"}


class TestServe:
    def test_serve_import(self) -> None:
        assert callable(serve)

    def test_unified_server_import(self) -> None:
        assert UnifiedServer is not None

    def test_unified_server_creates(self) -> None:
        g = Graph[ServeState]()
        g.add_node("a", dummy_fn).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=ServeState)

        server = UnifiedServer(compiled, host="127.0.0.1", port=9091)
        assert server is not None

    def test_unified_server_start_stop(self) -> None:
        import asyncio

        g = Graph[ServeState]()
        g.add_node("a", dummy_fn).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=ServeState)

        async def test():
            server = UnifiedServer(compiled, host="127.0.0.1", port=9092)
            await server.start()
            await server.stop()

        asyncio.run(test())

    def test_top_level_import(self) -> None:
        from graphforge.serve import serve
        assert callable(serve)
