"""Graph visualization and export utilities.

Provides ``export_dot()`` to convert a :class:`~graphforge._graph.CompiledGraph`
into Graphviz DOT format, and ``render_graph()`` to render it to an image
if the ``graphviz`` package is installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from graphforge._graph import CompiledGraph
from graphforge._node import Node
from graphforge._types import NodeName


def export_dot(
    compiled: CompiledGraph[Any],
    *,
    show_kind: bool = True,
    show_metadata: bool = False,
) -> str:
    """Export a compiled graph to Graphviz DOT format.

    Parameters
    ----------
    compiled:
        The compiled graph to export.
    show_kind:
        If ``True``, append the node kind (function/async/subgraph) to labels.
    show_metadata:
        If ``True``, include node metadata as tooltips.

    Returns
    -------
    A DOT format string.
    """
    lines: List[str] = []
    lines.append("digraph GraphForge {")
    lines.append("    rankdir=LR;")
    lines.append("    splines=ortho;")
    lines.append("    node [shape=box, style=rounded, fontname=Helvetica];")
    lines.append("    edge [fontname=Helvetica, fontsize=10];")
    lines.append("")

    # Entry point marker
    lines.append(f"    __start__ [shape=point, width=0.2, label=\"\"];")
    lines.append(f"    __start__ -> {compiled.entry_point};")
    lines.append("")

    # Nodes
    for name, node in compiled.nodes.items():
        label = name
        if show_kind and node.kind.value != "function":
            label += f"\\n({node.kind.value})"
        tooltip = ""
        if show_metadata and node.metadata:
            tooltip = f", tooltip={node.metadata!r}"
        lines.append(f"    \"{name}\" [label=\"{label}\"{tooltip}];")

    lines.append("")

    # Direct edges
    edges_drawn: Set[str] = set()
    for name in compiled.nodes:
        for succ in compiled.successors(name):
            if succ is None:
                target = "__end__"
            else:
                target = succ
            edge_id = f"{name}->{target}"
            if edge_id not in edges_drawn:
                lines.append(f"    \"{name}\" -> \"{target}\";")
                edges_drawn.add(edge_id)

    # Conditional edges
    if hasattr(compiled, "_conditionals"):
        for source, cond_edge in compiled._conditionals.items():
            for key, target in cond_edge.path_map.items():
                edge_id = f"{source}->{target}[{key}]"
                if edge_id not in edges_drawn:
                    target_name = target if target != "__end__" else "__end__"
                    lines.append(f"    \"{source}\" -> \"{target_name}\" [label=\"{key}\"];")
                    edges_drawn.add(edge_id)

    # Fan-out edges
    if hasattr(compiled, "_fanout_map"):
        for source, fan_edge in compiled._fanout_map.items():
            for target in fan_edge.targets:
                edge_id = f"{source}->{target}[fan]"
                if edge_id not in edges_drawn:
                    target_name = target if target != "__end__" else "__end__"
                    lines.append(f"    \"{source}\" -> \"{target_name}\" [style=dashed, label=\"fan\"];")
                    edges_drawn.add(edge_id)

            if fan_edge.join:
                lines.append(f"    \"{fan_edge.join}\" [style=bold, peripheries=2];")

    # End point
    lines.append("")
    lines.append("    __end__ [shape=point, width=0.2, label=\"\"];")
    lines.append("}")

    return "\n".join(lines)


def render_graph(
    compiled: CompiledGraph[Any],
    output_path: str = "graph.png",
    *,
    format: str = "png",
    engine: str = "dot",
    **kwargs: Any,
) -> Optional[str]:
    """Render a compiled graph to an image file.

    Requires the ``graphviz`` Python package.

    Parameters
    ----------
    compiled:
        The graph to render.
    output_path:
        Output file path.
    format:
        Output format (png, svg, pdf, etc.).
    engine:
        Graphviz layout engine (dot, neato, fdp, etc.).

    Returns
    -------
    The output path on success, ``None`` if graphviz is not available.
    """
    try:
        import graphviz
    except ImportError:
        return None

    dot_source = export_dot(compiled)
    graph = graphviz.Source(dot_source)
    graph.render(outfile=output_path, format=format, engine=engine, cleanup=True)
    return output_path


__all__ = ["export_dot", "render_graph"]
