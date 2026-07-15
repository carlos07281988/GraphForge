"""Tests for state middleware."""
from graphforge._middleware import MiddlewarePipeline


class TestMiddleware:
    def test_empty_pipeline(self) -> None:
        p = MiddlewarePipeline()
        result = p.pre_update("node", {}, {"key": "val"})
        assert result == {"key": "val"}

    def test_pre_update_modifies(self) -> None:
        class AddField:
            def pre_update(self, node, state, updates):
                updates["modified"] = True
                return updates

        p = MiddlewarePipeline([AddField()])
        result = p.pre_update("n", {}, {"k": "v"})
        assert result["modified"] is True
        assert result["k"] == "v"

    def test_post_update_called(self) -> None:
        calls = []

        class Tracker:
            def post_update(self, node, old_state, new_state):
                calls.append((node, old_state, new_state))

        p = MiddlewarePipeline([Tracker()])
        p.post_update("n", {"a": 1}, {"a": 2})
        assert len(calls) == 1
        assert calls[0][0] == "n"

    def test_multiple_stages(self) -> None:
        stages = []

        class Stage1:
            def pre_update(self, node, state, updates):
                updates["s1"] = True
                return updates

        class Stage2:
            def pre_update(self, node, state, updates):
                updates["s2"] = True
                return updates

        p = MiddlewarePipeline([Stage1(), Stage2()])
        result = p.pre_update("n", {}, {})
        assert result["s1"] is True
        assert result["s2"] is True

    def test_imports(self) -> None:
        from graphforge._middleware import MiddlewarePipeline, StateMiddleware
        assert MiddlewarePipeline is not None
        assert StateMiddleware is not None
