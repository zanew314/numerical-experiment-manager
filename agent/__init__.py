"""AI Numerical Experiment Manager Skill - agent tools.

These modules are deterministic helpers the coding agent calls during the
lifecycle workflow. They do not replace the agent's reasoning or approvals.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

NEMS_DIRNAME = ".nems"

__all__ = [
    "NEMS_DIRNAME",
    "nems_dir",
    "ensure_dir",
    "save_json",
    "load_json",
    "append_jsonl",
    "read_jsonl",
    "timestamp",
    "iso_timestamp",
]


def nems_dir(project_root) -> Path:
    """Return the ``.nems`` directory path for a project root."""
    return Path(project_root) / NEMS_DIRNAME


def ensure_dir(path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path, data) -> str:
    """Write ``data`` as UTF-8 JSON, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return str(p)


def load_json(path, default=None):
    """Load JSON from ``path``; return ``default`` on any failure."""
    p = Path(path)
    if not p.exists():
        return default
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def append_jsonl(path, record) -> None:
    """Append a single JSON record as a line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    """Read a JSONL file into a list, skipping malformed lines."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def timestamp() -> str:
    """Filesystem-friendly timestamp."""
    return time.strftime("%Y%m%d_%H%M%S")


def iso_timestamp() -> str:
    """Human-readable ISO-like timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")
