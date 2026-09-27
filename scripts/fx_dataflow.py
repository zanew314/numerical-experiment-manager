#!/usr/bin/env python3
"""Trace a model's forward pass with torch.fx and write a data-flow diagram.

Best-effort visualization: requires PyTorch and a traceable model. On failure it
prints a JSON error and exits non-zero, leaving the AST snapshot from
``scan_structure.py`` authoritative.

Usage:
    python fx_dataflow.py model.py --entry MLP --input-shape 1,3,32,32 \
        --save .nems/figures
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def build_inputs(torch, shapes, dtype):
    args = []
    for shape in shapes:
        args.append(torch.randn(*shape, dtype=dtype))
    return args


def node_shape(node) -> str:
    meta = node.meta.get("tensor_meta")
    shape = getattr(meta, "shape", None)
    if shape is None and "val" in node.meta:
        shape = getattr(node.meta["val"], "shape", None)
    return "x".join(str(dim) for dim in shape) if shape else "?"


def node_label(node) -> str:
    target = node.target
    if isinstance(target, str):
        return f"{node.op}:{target}"
    return f"{node.op}:{getattr(target, '__name__', repr(target))}"


def graph_to_dot(nodes) -> str:
    lines = ["digraph dataflow {", "  rankdir=LR;", '  node [shape=box, fontname="monospace"];']
    for node in nodes:
        label = f"{node['label']}\\n[{node['shape']}]"
        lines.append(f'  "{node["name"]}" [label="{label}"];')
    for node in nodes:
        for user in node["users"]:
            lines.append(f'  "{node["name"]}" -> "{user}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def graph_to_mermaid(nodes) -> str:
    lines = ["```mermaid", "graph LR"]
    for node in nodes:
        lines.append(f'  {node["name"]}["{node["label"]}<br/>{node["shape"]}"]')
    for node in nodes:
        for user in node["users"]:
            lines.append(f'  {node["name"]} --> {user}')
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="torch.fx data-flow diagram.")
    parser.add_argument("model_file")
    parser.add_argument("--entry", required=True, help="class or function name")
    parser.add_argument("--input-shape", action="append", default=None,
                        help="comma-separated shape, repeatable; default 1,8")
    parser.add_argument("--save", default=".nems/figures", help="output directory")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: F401
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp
    except Exception as exc:  # pragma: no cover - environment dependent
        print(json.dumps({"error": f"torch unavailable: {exc}"}), file=sys.stderr)
        return 3

    dtype = getattr(torch, args.dtype, torch.float32)
    model_path = Path(args.model_file).resolve()
    try:
        module = load_module(model_path)
        entry = getattr(module, args.entry)
    except Exception as exc:
        print(json.dumps({"error": f"cannot load entry {args.entry}: {exc}"}), file=sys.stderr)
        return 2

    shapes = [tuple(int(x) for x in s.split(",")) for s in (args.input_shape or ["1,8"])]
    try:
        obj = entry() if isinstance(entry, type) else entry
        example = build_inputs(torch, shapes, dtype)
        traced = symbolic_trace(obj)
    except Exception as exc:
        print(json.dumps({"error": f"symbolic_trace failed: {exc}"}), file=sys.stderr)
        return 4

    try:
        ShapeProp(traced).propagate(*example)
    except Exception:
        pass  # shapes stay "?" if propagation fails

    nodes = []
    for node in traced.graph.nodes:
        users = [u.name for u in node.users]
        nodes.append({
            "name": node.name,
            "op": node.op,
            "label": node_label(node),
            "shape": node_shape(node),
            "users": users,
        })

    out_dir = Path(args.save)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"dataflow_{args.entry}"
    (out_dir / f"{base}.dot").write_text(graph_to_dot(nodes), encoding="utf-8")
    (out_dir / f"{base}.md").write_text(
        f"# Data flow: {args.entry}\n\n{graph_to_mermaid(nodes)}", encoding="utf-8")

    print(json.dumps({
        "entry": args.entry,
        "nodes": nodes,
        "diagram": str(out_dir / f"{base}.md"),
        "dot": str(out_dir / f"{base}.dot"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
