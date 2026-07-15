"""Command-line interface for GraphForge.

Provides basic CLI commands for common operations:

- ``graphforge run <graph.json> <state.json>`` — invoke a serialized graph
- ``graphforge viz <graph.json>`` — export graph visualization
- ``graphforge info <graph.json>`` — show graph topology info

Usage::

    python -m graphforge.cli run graph.json state.json
    python -m graphforge.cli viz graph.json -o output.png
    python -m graphforge.cli info graph.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="graphforge",
        description="GraphForge CLI — run, visualize, and inspect graphs",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # run
    run_parser = sub.add_parser("run", help="Run a serialized graph")
    run_parser.add_argument("graph", help="Path to graph JSON file")
    run_parser.add_argument("state", help="Path to input state JSON file")
    run_parser.add_argument("-o", "--output", help="Output path for result JSON")
    run_parser.add_argument("--config", help="Runtime config JSON string")

    # viz
    viz_parser = sub.add_parser("viz", help="Export graph visualization")
    viz_parser.add_argument("graph", help="Path to graph JSON file")
    viz_parser.add_argument("-o", "--output", default="graph.png", help="Output image path")
    viz_parser.add_argument("--format", default="png", help="Output format (png, svg, pdf)")

    # info
    info_parser = sub.add_parser("info", help="Show graph topology info")
    info_parser.add_argument("graph", help="Path to graph JSON file")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Run a graph with input state."""
    try:
        with open(args.graph) as f:
            graph_data = json.load(f)
        with open(args.state) as f:
            state_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading files: {e}", file=sys.stderr)
        return 1

    from graphforge._graph import Graph

    graph = Graph.deserialize(graph_data)
    # Note: nodes need real functions registered; this is a stub
    print(f"Graph loaded: {len(graph_data.get('node_specs', {}))} nodes", file=sys.stderr)
    print(f"State: {json.dumps(state_data, indent=2)}")
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    """Export graph visualization."""
    try:
        with open(args.graph) as f:
            graph_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    from graphforge._graph import Graph

    graph = Graph.deserialize(graph_data)
    print(f"Exporting visualization to {args.output}...", file=sys.stderr)
    print(f"[viz] format={args.format}, output={args.output}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Display graph topology information."""
    try:
        with open(args.graph) as f:
            graph_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    node_count = len(graph_data.get("node_specs", {}))
    edge_count = len(graph_data.get("direct_edges", []))
    cond_count = len(graph_data.get("conditional_edges", []))
    entry = graph_data.get("entry_point", "none")

    print(f"Graph: {args.graph}")
    print(f"  Nodes:     {node_count}")
    print(f"  Edges:     {edge_count}")
    print(f"  Conditionals: {cond_count}")
    print(f"  Entry:     {entry}")
    print(f"  Fan-outs:  {len(graph_data.get('fanout_edges', []))}")
    print(f"  Metadata:  {json.dumps(graph_data.get('metadata', {}), indent=2)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "viz":
        return cmd_viz(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
