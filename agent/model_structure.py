"""Save a whole model's structure without copying its function bodies.

Two layers, both optional at call time:

1. **AST skeleton** (always available, standard library only). Every ``.py`` file
   is parsed with ``ast`` and reduced to imports, module constants, classes
   (bases, attributes, method signatures, and ``self.<layer> = Module(...)``
   assignments), and top-level function signatures with the calls they make. No
   function body statements are copied, so the skeleton is a structural card,
   not a source copy. It works on partially generated or injected sources and
   never executes the project.

2. **Dynamic graph** (opt-in, best-effort). When PyTorch is importable and the
   agent can supply a model instance plus example inputs, ``augment_with_torch_fx``
   traces the module and records the operator graph, tensor shapes, and
   parameter counts. A failure here never invalidates the AST skeleton.

The result is written to ``.nems/model_structure.json`` (data) and
``.nems/model_structure.md`` (human-readable). ``diff_model_structures`` compares
two snapshots so a model update can be summarized incrementally.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional

from . import iso_timestamp, nems_dir, save_json
from .code_reader import CodeReader, hash_text

SCHEMA_VERSION = "nems-model-structure-v1"

# Module calls that usually define part of a model architecture. A ``self.x =``
# assignment whose right-hand side is one of these is treated as a layer.
LAYER_CALL_HINTS = (
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm",
    "Dropout",
    "Embedding",
    "LSTM",
    "GRU",
    "RNN",
    "MultiheadAttention",
    "TransformerEncoderLayer",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
)

MODEL_BASE_HINTS = ("Module", "ModuleList", "ModuleDict", "Model")


# ----------------------------------------------------------------- primitives
def _source_text(node, source: str) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.get_source_segment(source, node)
    except (ValueError, TypeError):
        return None


def _constant_value(node):
    """Return ``(is_constant, value)`` for a literal node."""
    if isinstance(node, ast.Constant):
        return True, node.value
    return False, None


def _dotted_name(node) -> Optional[str]:
    """Return the dotted name of a Name/Attribute chain, else None."""
    parts: List[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _is_layer_call(dotted: Optional[str]) -> bool:
    if not dotted:
        return False
    last = dotted.split(".")[-1]
    return any(last.endswith(hint) for hint in LAYER_CALL_HINTS)


def _arg_summary(arg: ast.arg, default_node, source: str) -> Dict[str, object]:
    is_constant, value = _constant_value(default_node)
    return {
        "name": arg.arg,
        "annotation": _source_text(arg.annotation, source),
        "default": value if is_constant else _source_text(default_node, source),
        "has_default": default_node is not None,
    }


def _signature_args(func: ast.AST, source: str) -> List[Dict[str, object]]:
    args = list(func.args.posonlyargs) + list(func.args.args)
    defaults = [None] * (len(args) - len(func.args.defaults)) + list(func.args.defaults)
    summary = [_arg_summary(arg, default, source) for arg, default in zip(args, defaults)]
    if func.args.vararg is not None:
        summary.append({"name": "*" + func.args.vararg.arg, "default": None, "has_default": False, "annotation": None})
    for arg, default in zip(func.args.kwonlyargs, func.args.kw_defaults):
        summary.append(_arg_summary(arg, default, source))
    if func.args.kwarg is not None:
        summary.append({"name": "**" + func.args.kwarg.arg, "default": None, "has_default": False, "annotation": None})
    return summary


def _first_docstring_line(node) -> Optional[str]:
    doc = ast.get_docstring(node)
    if not doc:
        return None
    return doc.strip().splitlines()[0].strip()


def _collect_calls(node, source: str) -> List[str]:
    calls: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            dotted = _dotted_name(child.func)
            if dotted and dotted not in calls:
                calls.append(dotted)
    return calls


def _target_is_self_attr(target) -> Optional[str]:
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
        return target.attr
    return None


def _layer_assignment(stmt, source: str) -> Optional[Dict[str, object]]:
    """Return a layer record when ``stmt`` is ``self.x = SomeModule(...)``."""
    value = None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = _target_is_self_attr(stmt.targets[0])
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        target = _target_is_self_attr(stmt.target)
        value = stmt.value
    else:
        return None
    if target is None or not isinstance(value, ast.Call):
        return None
    dotted = _dotted_name(value.func)
    if not _is_layer_call(dotted):
        return None
    return {
        "attr": target,
        "call": dotted,
        "args": [_source_text(arg, source) for arg in value.args],
        "keywords": {kw.arg: _source_text(kw.value, source) for kw in value.keywords if kw.arg},
        "line": getattr(stmt, "lineno", None),
    }


def _class_attribute(stmt, source: str) -> Optional[Dict[str, object]]:
    name = None
    value_node = None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        name = stmt.targets[0].id
        value_node = stmt.value
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        name = stmt.target.id
        value_node = stmt.value
    if name is None:
        return None
    is_constant, value = _constant_value(value_node)
    return {
        "name": name,
        "value": value if is_constant else _source_text(value_node, source),
        "is_literal": is_constant,
        "line": getattr(stmt, "lineno", None),
    }


def _module_constant(stmt, source: str) -> Optional[Dict[str, object]]:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        name, value_node = stmt.targets[0].id, stmt.value
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        name, value_node = stmt.target.id, stmt.value
    else:
        return None
    is_constant, value = _constant_value(value_node)
    if not is_constant:
        return None
    return {"name": name, "value": value, "line": getattr(stmt, "lineno", None)}


def _parse_class(node: ast.ClassDef, source: str) -> Dict[str, object]:
    attributes: List[Dict[str, object]] = []
    methods: List[Dict[str, object]] = []
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            layer_assignments: List[Dict[str, object]] = []
            for inner in stmt.body:
                record = _layer_assignment(inner, source)
                if record is not None:
                    layer_assignments.append(record)
            methods.append({
                "name": stmt.name,
                "line": getattr(stmt, "lineno", None),
                "args": _signature_args(stmt, source),
                "returns": _source_text(stmt.returns, source),
                "docstring": _first_docstring_line(stmt),
                "calls": _collect_calls(stmt, source),
                "layer_assignments": layer_assignments,
            })
        else:
            record = _class_attribute(stmt, source)
            if record is not None:
                attributes.append(record)
    return {
        "name": node.name,
        "bases": [_dotted_name(base) or _source_text(base, source) for base in node.bases],
        "docstring": _first_docstring_line(node),
        "line": getattr(node, "lineno", None),
        "attributes": attributes,
        "methods": methods,
    }


def _parse_function(node, source: str) -> Dict[str, object]:
    return {
        "name": node.name,
        "line": getattr(node, "lineno", None),
        "args": _signature_args(node, source),
        "returns": _source_text(node.returns, source),
        "docstring": _first_docstring_line(node),
        "calls": _collect_calls(node, source),
    }


def _parse_imports(tree: ast.AST, source: str) -> List[str]:
    imports: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = _source_text(node, source)
            if text:
                imports.append(text.strip())
    return imports


def parse_source(rel_path: str, source: str) -> Dict[str, object]:
    """Parse one module into a body-free structural record.

    Returns ``{"error": ...}`` when the source cannot be parsed, without copying
    any of the offending source text.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"error": f"{rel_path}: {exc.__class__.__name__}: {exc.msg}"}

    module_constants: List[Dict[str, object]] = []
    classes: List[Dict[str, object]] = []
    functions: List[Dict[str, object]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(_parse_class(node, source))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_parse_function(node, source))
        else:
            record = _module_constant(node, source)
            if record is not None:
                module_constants.append(record)

    return {
        "imports": _parse_imports(tree, source),
        "module_constants": module_constants,
        "classes": classes,
        "functions": functions,
    }


# -------------------------------------------------------------- model classes
def _model_classes(modules: Dict[str, dict]) -> List[Dict[str, object]]:
    found: List[Dict[str, object]] = []
    for rel, module in modules.items():
        for cls in module.get("classes", []):
            bases = cls.get("bases") or []
            is_model = any(
                any(str(base).endswith(hint) for hint in MODEL_BASE_HINTS) for base in bases
            )
            methods = {m["name"]: m for m in cls.get("methods", [])}
            layers = methods.get("__init__", {}).get("layer_assignments", [])
            is_model = is_model or "forward" in methods or bool(layers)
            if not is_model:
                continue
            found.append({
                "module": rel,
                "name": cls["name"],
                "bases": bases,
                "layer_count": len(layers),
                "layers": layers,
            })
    return found


def extract_model_structure(project_root, files: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Build the structural snapshot for a project.

    ``files`` is an optional ``rel_path -> source`` mapping; when omitted the
    project is read with :class:`agent.code_reader.CodeReader`.
    """
    root = Path(project_root)
    reader = CodeReader(root)
    if files is None:
        scan = reader.scan()
        files = {code_file.rel_path: code_file.content for code_file in scan}
        hashes = {code_file.rel_path: code_file.sha256 for code_file in scan}
    else:
        hashes = {rel: hash_text(content) for rel, content in files.items()}

    modules: Dict[str, dict] = {}
    errors: List[str] = []
    for rel, content in files.items():
        parsed = parse_source(rel, content)
        if "error" in parsed:
            errors.append(parsed["error"])
            continue
        modules[rel] = parsed

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "project_root": str(root),
        "file_count": len(files),
        "source_hashes": hashes,
        "modules": modules,
        "model_classes": _model_classes(modules),
        "dynamic_graph": None,
        "parse_errors": errors,
    }


def summarize_model_structure(structure: Dict[str, object]) -> Dict[str, object]:
    """Return a compact structural summary for prompts or incremental diffs."""
    modules = structure.get("modules", {}) or {}
    return {
        "file_count": structure.get("file_count"),
        "modules": sorted(modules),
        "model_classes": structure.get("model_classes", []),
        "class_count": sum(len(m.get("classes", [])) for m in modules.values()),
        "function_count": sum(len(m.get("functions", [])) for m in modules.values()),
        "dynamic_graph": bool(structure.get("dynamic_graph")),
    }


def diff_model_structures(old: Dict[str, object], new: Dict[str, object]) -> Dict[str, object]:
    """Compare two snapshots and report structural changes only."""
    changes: List[Dict[str, object]] = []
    old_classes = {
        f"{c['module']}::{c['name']}": c for c in (old.get("model_classes", []) or [])
    }
    new_classes = {
        f"{c['module']}::{c['name']}": c for c in (new.get("model_classes", []) or [])
    }
    for key in sorted(set(old_classes) | set(new_classes)):
        before = old_classes.get(key)
        after = new_classes.get(key)
        if before is None:
            changes.append({"field": "model_class", "target": key, "before": None,
                            "after": after.get("layer_count"), "reason": "class added"})
        elif after is None:
            changes.append({"field": "model_class", "target": key, "before": before.get("layer_count"),
                            "after": None, "reason": "class removed"})
        elif before.get("layer_count") != after.get("layer_count"):
            changes.append({"field": "layer_count", "target": key,
                            "before": before.get("layer_count"), "after": after.get("layer_count"),
                            "reason": "layer count changed"})
    old_g = old.get("dynamic_graph") or {}
    new_g = new.get("dynamic_graph") or {}
    if old_g.get("total_parameters") != new_g.get("total_parameters"):
        changes.append({"field": "total_parameters", "target": "dynamic_graph",
                        "before": old_g.get("total_parameters"), "after": new_g.get("total_parameters"),
                        "reason": "parameter count changed"})
    changed_files = sorted(
        rel for rel, digest in (new.get("source_hashes", {}) or {}).items()
        if (old.get("source_hashes", {}) or {}).get(rel) != digest
    )
    return {"changes": changes, "changed_files": changed_files}


# ------------------------------------------------------------ torch.fx (opt-in)
def augment_with_torch_fx(structure: Dict[str, object], model, example_inputs) -> Dict[str, object]:
    """Trace ``model`` and merge an operator graph into ``structure``.

    Best-effort: on any failure the structure is returned unchanged with a
    ``dynamic_graph_error`` note, so the AST skeleton remains the source of
    truth. Requires PyTorch and a model that supports ``symbolic_trace``.
    """
    try:
        import torch
        from torch.fx import symbolic_trace
    except Exception as exc:  # noqa: BLE001 - optional dependency
        structure["dynamic_graph_error"] = f"torch.fx unavailable: {exc}"
        return structure

    try:
        graph_module = symbolic_trace(model)
        try:
            from torch.fx.passes.shape_prop import ShapeProp

            ShapeProp(graph_module).propagate(*example_inputs)
        except Exception:  # shape propagation is optional
            pass
        nodes: List[Dict[str, object]] = []
        for node in graph_module.graph.nodes:
            meta = {}
            tensor_meta = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
            if tensor_meta is not None:
                shape = getattr(tensor_meta, "shape", None)
                if shape is not None:
                    meta["shape"] = [int(dim) for dim in shape]
            nodes.append({
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "args": [str(arg) for arg in node.args],
                "meta": meta,
            })
        total_parameters = None
        try:
            total_parameters = int(sum(p.numel() for p in model.parameters()))
        except Exception:  # pragma: no cover - defensive
            total_parameters = None
        structure["dynamic_graph"] = {
            "framework": f"torch.fx ({getattr(torch, '__version__', 'unknown')})",
            "total_parameters": total_parameters,
            "nodes": nodes,
        }
        structure.pop("dynamic_graph_error", None)
    except Exception as exc:  # noqa: BLE001 - tracing must never be fatal
        structure["dynamic_graph_error"] = f"{type(exc).__name__}: {exc}"
    return structure


# ----------------------------------------------------------------- rendering
def render_markdown(structure: Dict[str, object]) -> str:
    lines: List[str] = [
        "# Model Structure",
        "",
        f"- schema: `{structure.get('schema_version')}`",
        f"- generated: {structure.get('generated_at')}",
        f"- files analysed: {structure.get('file_count')}",
        "",
    ]
    for error in structure.get("parse_errors", []) or []:
        lines.append(f"> parse warning: {error}")
    if structure.get("parse_errors"):
        lines.append("")

    for rel in sorted((structure.get("modules") or {}).keys()):
        module = structure["modules"][rel]
        lines.append(f"## {rel}")
        lines.append("")
        if module.get("imports"):
            lines.append("Imports: " + ", ".join(f"`{item}`" for item in module["imports"]))
            lines.append("")
        if module.get("module_constants"):
            lines += ["| constant | value | line |", "| --- | --- | --- |"]
            for const in module["module_constants"]:
                lines.append(f"| {const['name']} | `{const['value']}` | {const.get('line')} |")
            lines.append("")
        for cls in module.get("classes", []):
            bases = ", ".join(str(b) for b in (cls.get("bases") or [])) or "-"
            lines.append(f"### class {cls['name']}({bases})")
            lines.append("")
            if cls.get("attributes"):
                lines += ["| attribute | value | line |", "| --- | --- | --- |"]
                for attr in cls["attributes"]:
                    lines.append(f"| {attr['name']} | `{attr['value']}` | {attr.get('line')} |")
                lines.append("")
            for method in cls.get("methods", []):
                args = ", ".join(arg["name"] for arg in method.get("args", []))
                lines.append(f"- `{method['name']}({args})` (line {method.get('line')})")
                if method.get("layer_assignments"):
                    for layer in method["layer_assignments"]:
                        args_text = ", ".join(str(a) for a in layer.get("args", []))
                        lines.append(f"  - layer `{layer['attr']} = {layer['call']}({args_text})`")
                if method.get("calls"):
                    lines.append(f"  - calls: {', '.join(method['calls'])}")
            lines.append("")
        for func in module.get("functions", []):
            args = ", ".join(arg["name"] for arg in func.get("args", []))
            lines.append(f"### function {func['name']}({args}) (line {func.get('line')})")
            if func.get("calls"):
                lines.append(f"- calls: {', '.join(func['calls'])}")
            lines.append("")

    graph = structure.get("dynamic_graph")
    if graph:
        lines += ["## Dynamic graph (torch.fx)", "",
                  f"- framework: {graph.get('framework')}",
                  f"- total parameters: {graph.get('total_parameters')}",
                  f"- nodes: {len(graph.get('nodes', []))}", "",
                  "| node | op | target | shape |", "| --- | --- | --- | --- |"]
        for node in graph.get("nodes", []):
            shape = (node.get("meta") or {}).get("shape")
            lines.append(f"| {node['name']} | {node['op']} | `{node['target']}` | {shape} |")
        lines.append("")
    elif structure.get("dynamic_graph_error"):
        lines += ["## Dynamic graph (torch.fx)", "",
                  f"_not captured: {structure['dynamic_graph_error']}_", ""]
    return "\n".join(lines)


def write_model_structure(project_root, structure: Optional[Dict[str, object]] = None,
                          files: Optional[Dict[str, str]] = None):
    """Write ``.nems/model_structure.json`` and ``.nems/model_structure.md``."""
    if structure is None:
        structure = extract_model_structure(project_root, files=files)
    nd = nems_dir(project_root)
    json_path = save_json(nd / "model_structure.json", structure)
    md_path = nd / "model_structure.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(structure), encoding="utf-8")
    return {"json": json_path, "markdown": str(md_path), "structure": structure}
