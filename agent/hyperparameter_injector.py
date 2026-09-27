"""Hyperparameter injection: turn selected literals into ``get_param`` lookups.

Steps:

1. Resolve a non-conflicting adapter module name at the project root.
2. Snapshot every original ``.py`` file under ``.nems/code_snapshots/<stamp>/``.
3. Replace only the selected parameters with ``NAME = get_param("NAME", default=...)``.
4. Generate the adapter module that reads ``.nems/current_config.json``.
5. Record structured and human-readable injection history plus a diff.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from . import ensure_dir, iso_timestamp, nems_dir, save_json, timestamp
from .code_reader import CodeReader

CONFIG_MODULE_CANDIDATES = ["nems_config.py", "nems_param_loader.py", "_nems_config.py"]


def resolve_config_module_name(project_root) -> str:
    """Return a config module filename that does not collide with the project."""
    root = Path(project_root)
    for candidate in CONFIG_MODULE_CANDIDATES:
        if not (root / candidate).exists():
            return candidate
    return CONFIG_MODULE_CANDIDATES[-1]


def _format_default(value, type_name=None) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _insert_import(source: str, module_name: str) -> str:
    import_line = f"from {module_name} import get_param\n"
    if re.search(rf"^\s*from\s+{re.escape(module_name)}\s+import\s+get_param", source, re.M):
        return source
    lines = source.splitlines(keepends=True)
    idx = 0
    if idx < len(lines) and lines[idx].startswith("#!"):
        idx += 1
    if idx < len(lines) and re.match(r"^#.*coding[:=]", lines[idx]):
        idx += 1
    # skip a leading module docstring
    if idx < len(lines) and re.match(r'^\s*(?:"""|\'\'\')', lines[idx]):
        quote = lines[idx].lstrip()[:3]
        if lines[idx].count(quote) >= 2:
            idx += 1
        else:
            idx += 1
            while idx < len(lines) and quote not in lines[idx]:
                idx += 1
            idx += 1
    # keep __future__ imports first
    future = idx
    while future < len(lines) and re.match(r"^\s*from\s+__future__\s+import", lines[future]):
        future += 1
    insert = future
    lines.insert(insert, import_line)
    if insert + 1 < len(lines) and lines[insert + 1].strip():
        lines.insert(insert + 1, "\n")
    return "".join(lines)


def _inject_text(source: str, entry: dict) -> tuple:
    """Replace one parameter assignment; return ``(new_source, changed, before, after)``."""
    name = entry["name"]
    form = entry.get("form")
    default = _format_default(entry.get("value"), entry.get("type"))

    if form == "self_attr":
        pattern = re.compile(
            rf"^(\s*)self\.{re.escape(name)}(\s*=\s*)(.+?)(\s*(?:#.*)?)$", re.M
        )
        replacement = (
            lambda m: f'{m.group(1)}self.{name}{m.group(2)}'
            f'get_param("{name}", default={default}){m.group(4)}'
        )
    else:
        pattern = re.compile(
            rf"^(\s*){re.escape(name)}(\s*=\s*)(.+?)(\s*(?:#.*)?)$", re.M
        )
        replacement = (
            lambda m: f'{m.group(1)}{name}{m.group(2)}'
            f'get_param("{name}", default={default}){m.group(4)}'
        )

    match = pattern.search(source)
    if not match:
        return source, False, None, None
    before = match.group(0)
    if "get_param(" in before:
        return source, False, before, before
    after = replacement(match)
    new_source = source[:match.start()] + after + source[match.end():]
    return new_source, True, before, after


def backup_sources(project_root, snapshot_dir) -> dict:
    """Copy all project ``.py`` files into the snapshot directory."""
    root = Path(project_root)
    snap = ensure_dir(snapshot_dir)
    reader = CodeReader(root)
    manifest = {}
    for code_file in reader.scan():
        dest = snap / code_file.rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(code_file.content, encoding="utf-8")
        manifest[code_file.rel_path] = code_file.sha256
    save_json(snap / "manifest.json", manifest)
    return manifest


def generate_config_module(project_root, module_name=None) -> str:
    """Write the adapter module at the project root and return its path."""
    root = Path(project_root)
    module_name = module_name or resolve_config_module_name(root)
    content = CONFIG_MODULE_TEMPLATE
    path = root / module_name
    path.write_text(content, encoding="utf-8")
    return str(path)


def inject_hyperparameters(project_root, params, module_name=None, snapshot_dir=None):
    """Inject the selected hyperparameters into the project sources.

    ``params`` is a list of entries shaped like the output of
    ``llm_analyzer.extract_hyperparameters`` (plus ``file``/``form``).
    Returns a report dict.
    """
    root = Path(project_root)
    module_name = module_name or resolve_config_module_name(root)
    import_module = Path(module_name).stem
    snap_id = timestamp()
    snapshot_dir = Path(snapshot_dir) if snapshot_dir else nems_dir(root) / "code_snapshots" / snap_id
    manifest = backup_sources(root, snapshot_dir)

    by_file = {}
    for entry in params:
        rel = entry.get("file")
        if not rel and entry.get("location"):
            rel = str(entry["location"]).split(":")[0]
        if not rel:
            continue
        by_file.setdefault(rel, []).append(entry)

    diffs = []
    records = []
    for rel, entries in by_file.items():
        path = root / rel
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8", errors="replace")
        current = original
        for entry in entries:
            current, changed, before, after = _inject_text(current, entry)
            records.append({
                "name": entry["name"],
                "value": entry.get("value"),
                "type": entry.get("type"),
                "file": rel,
                "form": entry.get("form"),
                "location": entry.get("location"),
                "changed": changed,
                "before": before,
                "after": after,
            })
        if current != original:
            current = _insert_import(current, import_module)
            # move the import to the top of the snapshot diff base
            path.write_text(current, encoding="utf-8")
            diff_text = "\n".join(difflib.unified_diff(
                original.splitlines(),
                current.splitlines(),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            ))
            diffs.append({"file": rel, "diff": diff_text})

    config_path = generate_config_module(root, module_name)
    diff_md = snapshot_dir / "diff.md"
    lines = [f"# Source Injection Diff ({snap_id})", ""]
    if diffs:
        for item in diffs:
            lines.append(f"## {item['file']}")
            lines.append("")
            lines.append("```diff")
            lines.append(item["diff"])
            lines.append("```")
            lines.append("")
    else:
        lines.append("_No source changes._")
    diff_md.write_text("\n".join(lines), encoding="utf-8")

    report = {
        "snapshot_id": snap_id,
        "snapshot_dir": str(snapshot_dir),
        "config_module": module_name,
        "config_module_path": config_path,
        "injected": [r for r in records if r["changed"]],
        "skipped": [r for r in records if not r["changed"]],
        "files_changed": [d["file"] for d in diffs],
        "diff_path": str(diff_md),
        "manifest": manifest,
    }
    save_json(snapshot_dir / "injection_report.json", report)
    return report


def write_injection_history(project_root, report, confirmed_params) -> tuple:
    """Append an injection event to JSON+Markdown history; return their paths."""
    root = Path(project_root)
    nd = ensure_dir(nems_dir(root))
    history_path = nd / "hyperparameter_history.json"
    history = []
    if history_path.exists():
        try:
            import json
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            history = []
    event = {
        "timestamp": iso_timestamp(),
        "snapshot_id": report["snapshot_id"],
        "config_module": report["config_module"],
        "injected": [{"name": r["name"], "value": r["value"], "file": r["file"]} for r in report["injected"]],
        "skipped": [r["name"] for r in report["skipped"]],
        "confirmed": [p.get("name") for p in confirmed_params],
        "diff_path": report["diff_path"],
    }
    history.append(event)
    save_json(history_path, history)

    md_path = nd / "hyperparameter_history.md"
    md_lines = ["# Hyperparameter Injection History", ""]
    for evt in history:
        md_lines.append(f"## {evt['timestamp']} ({evt['snapshot_id']})")
        md_lines.append("")
        md_lines.append(f"- adapter module: `{evt['config_module']}`")
        md_lines.append(f"- diff: `{evt['diff_path']}`")
        md_lines.append("")
        md_lines.append("| injected | value | file |")
        md_lines.append("| --- | --- | --- |")
        for item in evt["injected"]:
            md_lines.append(f"| {item['name']} | `{item['value']}` | {item['file']} |")
        if evt["skipped"]:
            md_lines.append("")
            md_lines.append(f"Excluded (unchanged): {', '.join(evt['skipped'])}")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return str(history_path), str(md_path)


CONFIG_MODULE_TEMPLATE = '''"""Auto-generated parameter index for the AI Numerical Experiment Manager.

Reads ``.nems/current_config.json`` at call time. If the file is missing or a
parameter is absent, ``get_param`` returns the supplied default, so the source
stays runnable with no experiment manager present.
"""

import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, ".nems", "current_config.json")
_CACHE = None


def _load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    data = {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8-sig") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            data = payload.get("params", payload)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    _CACHE = data
    return _CACHE


def get_param(name, default=None, cast=None):
    """Return the injected value for ``name`` or ``default``."""
    data = _load()
    value = data.get(name, None)
    if value is None:
        value = default
    if cast is not None:
        try:
            return cast(value)
        except (TypeError, ValueError):
            return value
    if value is default or value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, (int, float)) and not isinstance(value, bool):
        try:
            if isinstance(default, int):
                return int(value)
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def get_all_params():
    """Return a copy of the current parameter mapping."""
    return dict(_load())


def reload_config():
    """Drop the cache so the next ``get_param`` re-reads the file."""
    global _CACHE
    _CACHE = None
'''
