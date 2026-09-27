#!/usr/bin/env python3
"""Scan a Python project's structure into a versioned snapshot under .nems/.

Reads project ``.py`` files with the standard-library ``ast`` and stores a
body-free structural snapshot (imports, classes, bases, method signatures,
``self.<attr>`` assignments, module constants). A new version is created only
when the normalized structure hash changes.

Usage:
    python scan_structure.py <project_root> [--track main.py model.py] [--force]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SKIP_DIRS = {
    ".nems", "__pycache__", ".git", ".hg", ".svn", ".venv", "venv", "env",
    "node_modules", ".idea", ".vscode", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".demo_workspace",
}
PRIORITY = ("main.py", "run.py", "train.py", "model.py", "trainer.py", "solver.py")
SCHEMA_VERSION = 1


def iter_py_files(root: Path, track=None):
    if track:
        for name in track:
            path = root / name
            if path.is_file() and path.suffix == ".py":
                yield path
        return
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(Path(dirpath) / filename)

    def sort_key(path: Path):
        try:
            return (PRIORITY.index(path.name.lower()), str(path))
        except ValueError:
            return (len(PRIORITY), str(path))

    yield from sorted(files, key=sort_key)


def self_attrs(node: ast.AST) -> list:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.Assign, ast.AnnAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    names.add(target.attr)
    return sorted(names)


def arg_names(node: ast.AST) -> list:
    args = node.args
    names = [a.arg for a in getattr(args, "posonlyargs", [])]
    names += [a.arg for a in args.args]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return names


def parse_module(rel: str, source: str) -> dict:
    tree = ast.parse(source, filename=rel)
    imports, classes, functions, constants = [], [], [], {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imports.extend(f"{base}.{alias.name}".strip(".") for alias in node.names)
        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "bases": [ast.unparse(b) for b in node.bases],
                "methods": [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ],
                "attributes": self_attrs(node),
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({"name": node.name, "args": arg_names(node)})
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    constants[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and isinstance(node.value, ast.Constant):
                constants[node.target.id] = node.value.value
    return {
        "path": rel,
        "imports": sorted(set(imports)),
        "classes": classes,
        "functions": functions,
        "constants": constants,
    }


def structure_hash(modules: list) -> str:
    blob = json.dumps(modules, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_md(snapshot: dict) -> str:
    lines = [f"# Structure {snapshot['version']}", ""]
    lines.append(f"- hash: `{snapshot['hash']}`")
    lines.append(f"- files: {', '.join(snapshot['files']) or '(none)'}")
    lines.append("")
    for module in snapshot["modules"]:
        lines.append(f"## `{module['path']}`")
        if module["imports"]:
            lines.append(f"- imports: {', '.join(module['imports'])}")
        for cls in module["classes"]:
            bases = f" ({', '.join(cls['bases'])})" if cls["bases"] else ""
            lines.append(f"- class `{cls['name']}`{bases}")
            lines.append(f"  - methods: {', '.join(cls['methods']) or '-'}")
            lines.append(f"  - attributes: {', '.join(cls['attributes']) or '-'}")
        for fn in module["functions"]:
            lines.append(f"- def `{fn['name']}({', '.join(fn['args'])})`")
        if module["constants"]:
            consts = ", ".join(f"{k}={v!r}" for k, v in module["constants"].items())
            lines.append(f"- constants: {consts}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_index(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": SCHEMA_VERSION, "current": None, "versions": []}


def next_version(index: dict) -> str:
    number = len(index["versions"]) + 1
    return f"v{number:04d}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Versioned AST structure scan.")
    parser.add_argument("project_root", nargs="?", default=".", help="project directory")
    parser.add_argument("--track", nargs="*", default=None,
                        help="only these project-relative .py files")
    parser.add_argument("--force", action="store_true", help="re-snapshot even if unchanged")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    if not root.is_dir():
        print(json.dumps({"error": f"not a directory: {root}"}), file=sys.stderr)
        return 2

    modules, files, errors = [], [], []
    for path in iter_py_files(root, args.track):
        rel = path.relative_to(root).as_posix()
        try:
            modules.append(parse_module(rel, path.read_text(encoding="utf-8")))
            files.append(rel)
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            errors.append({"path": rel, "error": str(exc)})

    digest = structure_hash(modules)
    structure_dir = root / ".nems" / "structure"
    index_path = structure_dir / "index.json"
    index = load_index(index_path)
    current = next((v for v in index["versions"] if v["version"] == index.get("current")), None)

    if current and current["hash"] == digest and not args.force:
        print(json.dumps({
            "status": "unchanged", "version": current["version"], "hash": digest,
            "files": files, "errors": errors,
        }, ensure_ascii=False))
        return 0

    structure_dir.mkdir(parents=True, exist_ok=True)
    version = next_version(index)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": digest,
        "files": files,
        "modules": modules,
        "summary": None,
    }
    (structure_dir / f"{version}.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (structure_dir / f"{version}.md").write_text(render_md(snapshot), encoding="utf-8")

    index["current"] = version
    index["versions"].append({
        "version": version,
        "created_at": snapshot["created_at"],
        "parent": current["version"] if current else None,
        "hash": digest,
        "files": files,
        "summary": None,
    })
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "created", "version": version, "parent": index["versions"][-1]["parent"],
        "hash": digest, "files": files, "snapshot": str(structure_dir / f"{version}.json"),
        "errors": errors,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
