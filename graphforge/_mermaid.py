"""Mermaid diagram export for graph visualisation.

Provides :func:`export_mermaid` which converts a :class:`~graphforge._graph.CompiledGraph`
into a `Mermaid <https://mermaid.js.org/>`_ flowchart string, suitable for
embedding in Markdown or rendering in any Mermaid-compatible viewer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from graphforge._graph import CompiledGraph
from graphforge._types import NodeName


def export_mermaid(
    compiled: CompiledGraph[Any],
    *,
    direction: str = "LR",
    show_kind: bool = True,
) -> str:
    """Export a compiled graph to a Mermaid flowchart string.

    Parameters
    ----------
    compiled:
        The compiled graph to visualise.
    direction:
        Flowchart direction (``"LR"`` = left-to-right, ``"TB"`` = top-to-bottom).
    show_kind:
        If ``True``, append the node kind label to node names.

    Returns
    -------
    A Mermaid flowchart definition string.
    """
    lines: List[str] = []
    lines.append(f"graph {direction}")

    # ------------------------------------------------------------------
    # Style definitions
    # ------------------------------------------------------------------
    lines.append("  classDef start fill:#e1f5fe,stroke:#0288d1,stroke-width:2px")
    lines.append("  classDef end fill:#fce4ec,stroke:#d32f2f,stroke-width:2px")
    lines.append("  classDef function fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px")
    lines.append("  classDef subgraph fill:#e8f5e9,stroke:#388e3c,stroke-width:1px")
    lines.append("  classDef error fill:#fff3e0,stroke:#f57c00,stroke-width:1px")
    lines.append("")

    # ------------------------------------------------------------------
    # Entry-point marker
    # ------------------------------------------------------------------
    entry = compiled.entry_point
    lines.append(f"  __start__([__start__]):::start --> {entry}")

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    for name, node in compiled.nodes.items():
        label = name
        if show_kind and node.kind.value != "function":
            label += f" ({node.kind.value})"

        kind_class = _kind_css(node.kind.value)
        is_error_source = name in compiled.error_map
        if is_error_source:
            kind_class = "error"

        lines.append(f"  {name}[[{label}]]:::{kind_class}")

    lines.append("")

    # ------------------------------------------------------------------
    # Direct edges
    # ------------------------------------------------------------------
    edges_drawn: set = set()
    for name in compiled.nodes:
        for succ in compiled.successors(name):
            target = "__end__" if succ is None else succ
            edge_id = f"{name}->{target}"
            if edge_id not in edges_drawn:
                lines.append(f"  {name} --> {target}")
                edges_drawn.add(edge_id)

    # ------------------------------------------------------------------
    # Conditional edges
    # ------------------------------------------------------------------
    if hasattr(compiled, "_conditionals"):
        for source, cond_edge in compiled._conditionals.items():
            for key, target in cond_edge.path_map.items():
                target_name = "__end__" if target == "__end__" else target
                edge_id = f"{source}->{target_name}[{key}]"
                if edge_id not in edges_drawn:
                    lines.append(f"  {source} -->|{key}| {target_name}")
                    edges_drawn.add(edge_id)

    # ------------------------------------------------------------------
    # Fan-out edges
    # ------------------------------------------------------------------
    if hasattr(compiled, "_fanout_map"):
        for source, fan_edge in compiled._fanout_map.items():
            for target in fan_edge.targets:
                edge_id = f"{source}->{target}[fan]"
                if edge_id not in edges_drawn:
                    lines.append(f"  {source} -.->|fan| {target}")
                    edges_drawn.add(edge_id)

    # ------------------------------------------------------------------
    # Error edges
    # ------------------------------------------------------------------
    for source, fallback in compiled.error_map.items():
        edge_id = f"{source}->{fallback}[error]"
        if edge_id not in edges_drawn:
            lines.append(f"  {source} -.->|error| {fallback}")
            edges_drawn.add(edge_id)

    # ------------------------------------------------------------------
    # End-point marker
    # ------------------------------------------------------------------
    lines.append("  __end__([__end__]):::end")

    return "\n".join(lines)


def _kind_css(kind: str) -> str:
    """Map a node kind to a Mermaid CSS class name."""
    mapping = {
        "function": "function",
        "subgraph": "subgraph",
        "pipeline": "subgraph",
    }
    return mapping.get(kind, "function")


__all__ = ["export_mermaid"]
