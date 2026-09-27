#!/usr/bin/env python3
"""Diff two structure versions under .nems/structure/ and print JSON.

Works on the snapshots written by ``scan_structure.py``. The output is designed
to be pasted into ``prompts/structure_diff.md`` for LLM attribution.

Usage:
    python diff_structure.py <project_root> [--from v0001] [--to v0002]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_index(structure_dir: Path) -> dict:
    path = structure_dir / "index.json"
    if not path.is_file():
        raise FileNotFoundError(f"no index at {path}; run scan_structure.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_versions(structure_dir: Path, index: dict, frm, to):
    versions = [v["version"] for v in index["versions"]]
    if not versions:
        raise ValueError("no structure versions recorded")
    to = to or index.get("current") or versions[-1]
    if frm is None:
        idx = versions.index(to)
        if idx == 0:
            raise ValueError(f"{to} has no previous version to diff against")
        frm = versions[idx - 1]
    return frm, to


def load_snapshot(structure_dir: Path, version: str) -> dict:
    path = structure_dir / f"{version}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing snapshot {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def by_name(items):
    return {item["name"]: item for item in items}


def diff_class(old: dict, new: dict) -> dict:
    changes = {}
    if old["bases"] != new["bases"]:
        changes["bases"] = {"from": old["bases"], "to": new["bases"]}
    added_m = [m for m in new["methods"] if m not in old["methods"]]
    removed_m = [m for m in old["methods"] if m not in new["methods"]]
    if added_m:
        changes["added_methods"] = added_m
    if removed_m:
        changes["removed_methods"] = removed_m
    added_a = [a for a in new["attributes"] if a not in old["attributes"]]
    removed_a = [a for a in old["attributes"] if a not in new["attributes"]]
    if added_a:
        changes["added_attributes"] = added_a
    if removed_a:
        changes["removed_attributes"] = removed_a
    return changes


def diff_modules(old: dict, new: dict) -> dict:
    changes = {"path": old["path"]}
    added_i = [i for i in new["imports"] if i not in old["imports"]]
    removed_i = [i for i in old["imports"] if i not in new["imports"]]
    if added_i:
        changes["added_imports"] = added_i
    if removed_i:
        changes["removed_imports"] = removed_i

    old_classes, new_classes = by_name(old["classes"]), by_name(new["classes"])
    added_c = [c for c in new_classes if c not in old_classes]
    removed_c = [c for c in old_classes if c not in new_classes]
    if added_c:
        changes["added_classes"] = sorted(added_c)
    if removed_c:
        changes["removed_classes"] = sorted(removed_c)
    changed_c = []
    for name in sorted(set(old_classes) & set(new_classes)):
        delta = diff_class(old_classes[name], new_classes[name])
        if delta:
            changed_c.append({"name": name, **delta})
    if changed_c:
        changes["changed_classes"] = changed_c

    old_fns, new_fns = by_name(old["functions"]), by_name(new["functions"])
    added_f = [f for f in new_fns if f not in old_fns]
    removed_f = [f for f in old_fns if f not in new_fns]
    if added_f:
        changes["added_functions"] = sorted(added_f)
    if removed_f:
        changes["removed_functions"] = sorted(removed_f)
    changed_f = [
        name for name in set(old_fns) & set(new_fns)
        if old_fns[name]["args"] != new_fns[name]["args"]
    ]
    if changed_f:
        changes["changed_function_signatures"] = sorted(changed_f)

    constant_delta = {}
    for key in sorted(set(old["constants"]) | set(new["constants"])):
        if old["constants"].get(key) != new["constants"].get(key):
            constant_delta[key] = {
                "from": old["constants"].get(key),
                "to": new["constants"].get(key),
            }
    if constant_delta:
        changes["changed_constants"] = constant_delta

    return changes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Diff two structure versions.")
    parser.add_argument("project_root", nargs="?", default=".")
    parser.add_argument("--from", dest="frm", default=None, help="older version, e.g. v0001")
    parser.add_argument("--to", dest="to", default=None, help="newer version, e.g. v0002")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    structure_dir = root / ".nems" / "structure"
    try:
        index = load_index(structure_dir)
        frm, to = resolve_versions(structure_dir, index, args.frm, args.to)
        old = load_snapshot(structure_dir, frm)
        new = load_snapshot(structure_dir, to)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2

    old_modules = {m["path"]: m for m in old["modules"]}
    new_modules = {m["path"]: m for m in new["modules"]}
    added_files = sorted(set(new_modules) - set(old_modules))
    removed_files = sorted(set(old_modules) - set(new_modules))
    common = sorted(set(old_modules) & set(new_modules))
    file_changes = [
        delta for delta in (diff_modules(old_modules[p], new_modules[p]) for p in common)
        if len(delta) > 1
    ]

    summary = {
        "files_added": len(added_files),
        "files_removed": len(removed_files),
        "files_changed": len(file_changes),
        "classes_added": sum(len(c.get("added_classes", [])) for c in file_changes),
        "classes_removed": sum(len(c.get("removed_classes", [])) for c in file_changes),
        "methods_added": sum(
            len(cc.get("added_methods", []))
            for c in file_changes for cc in c.get("changed_classes", [])
        ),
        "methods_removed": sum(
            len(cc.get("removed_methods", []))
            for c in file_changes for cc in c.get("changed_classes", [])
        ),
        "structural": bool(added_files or removed_files or any(
            any(k in c for k in (
                "added_classes", "removed_classes", "changed_classes",
                "added_imports", "removed_imports",
            ))
            for c in file_changes
        )),
    }

    print(json.dumps({
        "from": frm,
        "to": to,
        "added_files": added_files,
        "removed_files": removed_files,
        "changes": file_changes,
        "summary": summary,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
