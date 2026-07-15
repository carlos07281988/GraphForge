"""Tests for TimelineRecorder."""
from graphforge import Graph, GraphState, node_field, Append
from graphforge._callbacks import CallbackManager
from graphforge._timeline import TimelineRecorder, TimelineFrame
import json
import os


class TState(GraphState):
    value: int = 0
    path: list = node_field(default=[], merge="append")


def inc(state):
    return {"value": state.value + 1, "path": Append(["inc"])}


class TestTimelineRecorder:
    def test_records_frames(self) -> None:
        g = Graph[TState]()
        g.add_node("a", inc).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=TState)

        recorder = TimelineRecorder()
        compiled.invoke(TState(), callbacks=CallbackManager([recorder]))

        timeline = recorder.get_timeline()
        assert len(timeline) >= 1
        assert timeline[0].node == "a"

    def test_frame_structure(self) -> None:
        g = Graph[TState]()
        g.add_node("a", inc).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=TState)

        recorder = TimelineRecorder()
        compiled.invoke(TState(), callbacks=CallbackManager([recorder]))

        frame = recorder.get_timeline()[0]
        assert frame.node == "a"
        assert frame.step >= 0
        assert frame.duration >= 0
        assert "value" in frame.updates

    def test_get_node_frames(self) -> None:
        g = Graph[TState]()
        g.add_node("a", inc).add_node("b", inc)
        g.add_edge("a", "b").add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(state_type=TState)

        recorder = TimelineRecorder()
        compiled.invoke(TState(), callbacks=CallbackManager([recorder]))

        a_frames = recorder.get_node_frames("a")
        b_frames = recorder.get_node_frames("b")
        assert len(a_frames) >= 1
        assert len(b_frames) >= 1

    def test_export_json(self) -> None:
        g = Graph[TState]()
        g.add_node("a", inc).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=TState)

        recorder = TimelineRecorder()
        compiled.invoke(TState(), callbacks=CallbackManager([recorder]))

        path = "/tmp/test_timeline.json"
        recorder.export_json(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "frames" in data
        assert "summary" in data
        os.unlink(path)

    def test_replay(self) -> None:
        g = Graph[TState]()
        g.add_node("a", inc).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=TState)

        recorder = TimelineRecorder()
        compiled.invoke(TState(), callbacks=CallbackManager([recorder]))

        frames = list(recorder.replay())
        assert len(frames) >= 1

    def test_total_duration(self) -> None:
        g = Graph[TState]()
        g.add_node("a", inc).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=TState)

        recorder = TimelineRecorder()
        compiled.invoke(TState(), callbacks=CallbackManager([recorder]))

        assert recorder.total_duration() >= 0

    def test_timeline_frame_dataclass(self) -> None:
        frame = TimelineFrame(node="test", step=0)
        assert frame.node == "test"
        assert frame.step == 0
        assert frame.duration == 0.0
