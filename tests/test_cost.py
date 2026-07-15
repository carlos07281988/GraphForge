"""Tests for CostCallback."""
from graphforge._callbacks import CostCallback


class TestCostCallback:
    def test_track_basic(self) -> None:
        cb = CostCallback()
        cb.track("gpt-4", prompt_tokens=100, completion_tokens=50, node="llm")
        stats = cb.get_stats()
        assert "llm" in stats
        assert stats["llm"]["total_tokens"] == 150
        assert stats["llm"]["cost"] > 0

    def test_total_cost(self) -> None:
        cb = CostCallback()
        cb.track("gpt-4", prompt_tokens=1000, completion_tokens=0, node="a")
        assert cb.total_cost() > 0

    def test_total_tokens(self) -> None:
        cb = CostCallback()
        cb.track("gpt-4", prompt_tokens=100, completion_tokens=50, node="a")
        assert cb.total_tokens() == 150

    def test_multiple_calls(self) -> None:
        cb = CostCallback()
        cb.track("gpt-4", prompt_tokens=100, completion_tokens=50, node="llm")
        cb.track("gpt-4", prompt_tokens=200, completion_tokens=100, node="llm")
        stats = cb.get_stats()["llm"]
        assert stats["total_tokens"] == 450  # 150 + 300

    def test_multiple_nodes(self) -> None:
        cb = CostCallback()
        cb.track("gpt-4", 100, 50, node="a")
        cb.track("gpt-3.5-turbo", 50, 10, node="b")
        assert len(cb.get_stats()) == 2

    def test_custom_pricing(self) -> None:
        cb = CostCallback()
        cb.set_pricing("my-model", 0.001, 0.002)
        cb.track("my-model", 1000, 1000, node="test")
        cost = cb.get_stats()["test"]["cost"]
        assert cost == (1000/1000 * 0.001) + (1000/1000 * 0.002)

    def test_reset(self) -> None:
        cb = CostCallback()
        cb.track("gpt-4", 100, 0, node="a")
        cb.reset()
        assert cb.total_cost() == 0.0
        assert cb.total_tokens() == 0

    def test_default_model_pricing(self) -> None:
        cb = CostCallback()
        cb.track("unknown-model", 1000, 0, node="a")
        # Falls back to "default" pricing
        assert cb.total_cost() > 0
