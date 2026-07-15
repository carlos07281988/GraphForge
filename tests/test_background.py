"""Tests for background task execution."""
import time
from graphforge import Graph, GraphState
from graphforge.background import BackgroundTaskRunner, TaskStatus


class BgState(GraphState):
    value: int = 0


def slow_node(state):
    time.sleep(0.05)
    return {"value": 42}


class TestBackgroundRunner:
    def test_submit_and_wait(self) -> None:
        g = Graph[BgState]()
        g.add_node("a", slow_node).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=BgState)

        runner = BackgroundTaskRunner(max_workers=2)
        task = runner.submit(compiled, BgState())
        result = task.wait(timeout=5)
        assert task.status == TaskStatus.COMPLETED
        assert result.value == 42
        runner.shutdown()

    def test_task_list(self) -> None:
        runner = BackgroundTaskRunner()
        assert len(runner.list_tasks()) == 0
        runner.shutdown()

    def test_task_status_pending(self) -> None:
        g = Graph[BgState]()
        g.add_node("a", slow_node).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=BgState)

        runner = BackgroundTaskRunner(max_workers=1)
        task = runner.submit(compiled, BgState())
        assert task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        task.wait(timeout=5)
        runner.shutdown()

    def test_runner_imports(self) -> None:
        from graphforge.background import BackgroundTask, BackgroundTaskRunner, TaskStatus
        assert BackgroundTask is not None
        assert BackgroundTaskRunner is not None
        assert TaskStatus is not None
