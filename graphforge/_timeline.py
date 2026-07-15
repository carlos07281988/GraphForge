"""Execution timeline recording and replay debugging.

Provides :class:`TimelineRecorder` that captures every state transition
during graph execution for post-mortem debugging and replay.

Usage::

    from graphforge._timeline import TimelineRecorder
    from graphforge import CallbackManager

    recorder = TimelineRecorder()
    cm = CallbackManager([recorder])
    compiled.invoke(state, callbacks=cm)

    # Export for debugging
    recorder.export_json("execution.json")

    # Replay step by step
    for frame in recorder.replay():
        print(frame.node, frame.updates)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Generator, List, Optional


@dataclass
class TimelineFrame:
    """A single frame in the execution timeline.

    Parameters
    ----------
    node:
        Name of the node that executed.
    step:
        Step number in the execution.
    state_before:
        State dict before the node ran.
    updates:
        Updates produced by the node.
    state_after:
        State dict after the node ran.
    duration:
        Wall-clock time in seconds for the node execution.
    timestamp:
        Unix timestamp when the frame was recorded.
    metadata:
        Optional additional metadata.
    """
    node: str
    step: int
    state_before: Dict[str, Any] = field(default_factory=dict)
    updates: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimelineRecorder:
    """Callback that records every state transition during execution.

    Provides methods to inspect, export, and replay the recorded timeline.

    Usage::

        recorder = TimelineRecorder()
        compiled.invoke(state, callbacks=CallbackManager([recorder]))

        # Get full timeline
        for frame in recorder.get_timeline():
            print(f"{frame.node}: {list(frame.updates.keys())}")

        # Export to JSON
        recorder.export_json("trace.json")

        # Replay
        for frame in recorder.replay():
            print(frame.node, frame.state_after)
    """

    def __init__(self) -> None:
        self._frames: List[TimelineFrame] = []
        self._current_node: Optional[str] = None
        self._current_state: Dict[str, Any] = {}
        self._current_start: float = 0.0
        self._graph_name: str = ""

    # -- Callback hooks ---------------------------------------------------

    def on_graph_start(self, graph_name: str, input_state: Dict[str, Any]) -> None:
        self._graph_name = graph_name
        self._frames.clear()

    def on_node_start(self, node: str, state: Dict[str, Any]) -> None:
        self._current_node = node
        self._current_state = dict(state)
        self._current_start = time.time()

    def on_state_update(
        self, node: str, updates: Dict[str, Any], new_state: Dict[str, Any]
    ) -> None:
        duration = time.time() - self._current_start
        frame = TimelineFrame(
            node=node,
            step=len(self._frames),
            state_before=dict(self._current_state),
            updates=dict(updates),
            state_after=dict(new_state),
            duration=duration,
            timestamp=time.time(),
            metadata={"graph": self._graph_name},
        )
        self._frames.append(frame)
        self._current_state = dict(new_state)

    def on_node_end(self, node: str, state: Dict[str, Any]) -> None:
        self._current_node = None

    def on_node_error(self, node: str, error: Exception) -> None:
        if self._current_node is not None and self._frames:
            last = self._frames[-1]
            last.metadata["error"] = str(error)
        self._current_node = None

    # -- Query API --------------------------------------------------------

    def get_timeline(self) -> List[TimelineFrame]:
        """Return all recorded frames in execution order."""
        return list(self._frames)

    def get_node_frames(self, node: str) -> List[TimelineFrame]:
        """Return all frames for a specific node."""
        return [f for f in self._frames if f.node == node]

    def total_duration(self) -> float:
        """Return total execution duration across all frames."""
        return sum(f.duration for f in self._frames)

    # -- Export / Import --------------------------------------------------

    def export_json(self, path: str) -> None:
        """Export timeline to a JSON file for offline analysis.

        Parameters
        ----------
        path:
            File path for the JSON output.
        """
        data = {
            "graph": self._graph_name,
            "frames": [asdict(f) for f in self._frames],
            "summary": {
                "total_frames": len(self._frames),
                "total_duration": self.total_duration(),
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # -- Replay -----------------------------------------------------------

    def replay(self) -> Generator[TimelineFrame, None, None]:
        """Replay the execution frame by frame.

        Yields
        ------
        Each :class:`TimelineFrame` in execution order.
        """
        for frame in self._frames:
            yield frame

    def reset(self) -> None:
        """Clear all recorded frames."""
        self._frames.clear()
        self._current_node = None
        self._current_state = {}
        self._graph_name = ""


__all__ = [
    "TimelineFrame",
    "TimelineRecorder",
]
