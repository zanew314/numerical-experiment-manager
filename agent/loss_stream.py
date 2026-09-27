"""Pluggable sources for an experiment's wall-clock loss stream.

The instrumentation contract asks every managed program to write
``loss_time.jsonl``. Real projects do not always do that: some only print a
per-epoch line, some write a sidecar CSV, and some report a single final metric.
A :class:`LossSource` turns whatever a program emits at run time into the same
list of ``{"wall_clock", "loss", "epoch"}`` points, so the early stopper,
analyzer, and reports work unchanged.

Sources are chosen per project with a small spec stored at
``.nems/stream_adapter.json``:

```json
{"type": "stdout_regex",
 "config": {"pattern": "epoch (?P<epoch>\\\\d+) loss=(?P<loss>[0-9.eE+-]+)"}}
```

Built-in sources:

- ``jsonl`` (default): read ``loss_time.jsonl``;
- ``stdout_regex``: parse the program's captured stdout / log lines;
- ``sidecar``: parse a CSV / JSON / JSONL file the program already writes;
- ``completion_only``: no live curve; use the final ``metrics.json`` only.

Register a custom source with :func:`register_source`, or address one on disk
with ``{"type": "custom", "module": "my_adapters", "class": "MySource"}``.
"""

from __future__ import annotations

import csv
import importlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from . import load_json, nems_dir, read_jsonl, save_json

STREAM_SPEC_FILENAME = "stream_adapter.json"
DEFAULT_SPEC = {"type": "jsonl", "config": {}}


# ------------------------------------------------------------------ base type
class LossSource:
    """Base class: turn a program's output into wall-clock loss points."""

    name = "base"
    live = True

    def __init__(self, config: Optional[dict] = None):
        self.config = dict(config or {})
        self.synthetic_time = False
        self.error: Optional[str] = None

    def prepare(self, exp_dir) -> None:
        """Hook called once before a run (create dirs, clear stale output)."""

    def read_points(self, exp_dir, log_path=None) -> List[dict]:
        """Return the ``{"wall_clock", "loss", "epoch"}`` points seen so far."""
        return []

    def read_final(self, exp_dir, log_text: str = "") -> dict:
        """Return a final metrics mapping when no curve is available."""
        return {}

    def describe(self) -> dict:
        return {
            "type": self.name,
            "live": self.live,
            "synthetic_time": self.synthetic_time,
            "config": self.config,
            "error": self.error,
        }


def normalize_point(raw: dict) -> Optional[dict]:
    """Coerce one raw mapping into a point, or return None when unusable."""
    if not isinstance(raw, dict) or "loss" not in raw:
        return None
    try:
        loss = float(raw["loss"])
    except (TypeError, ValueError):
        return None
    try:
        wall_clock = float(raw.get("wall_clock", 0.0) or 0.0)
    except (TypeError, ValueError):
        wall_clock = 0.0
    epoch = raw.get("epoch")
    if epoch is not None:
        try:
            epoch = int(epoch)
        except (TypeError, ValueError):
            epoch = None
    return {"wall_clock": wall_clock, "loss": loss, "epoch": epoch}


def normalize_points(points) -> List[dict]:
    cleaned = [normalize_point(point) for point in (points or [])]
    return [point for point in cleaned if point is not None]


# ---------------------------------------------------------------- jsonl source
class JsonlSource(LossSource):
    """Read the default ``loss_time.jsonl`` instrumentation file."""

    name = "jsonl"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.filename = self.config.get("filename", "loss_time.jsonl")

    def read_points(self, exp_dir, log_path=None) -> List[dict]:
        return normalize_points(read_jsonl(Path(exp_dir) / self.filename))


# --------------------------------------------------------------- stdout source
class StdoutRegexSource(LossSource):
    """Parse per-step lines printed by the program (captured to the run log).

    ``pattern`` must contain a named group ``loss`` and may contain
    ``wall_clock`` and ``epoch``. When the pattern has no ``wall_clock`` group
    the axis is synthesized as ``index * time_step`` and ``synthetic_time`` is
    set, so reports can note that the wall-clock axis is approximate.
    """

    name = "stdout_regex"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        pattern = self.config.get("pattern")
        try:
            self.regex = re.compile(pattern) if pattern else None
        except re.error as exc:
            self.regex = None
            self.error = f"invalid pattern: {exc}"
        self.time_step = float(self.config.get("time_step", 1.0))
        self.synthetic_time = bool(self.regex) and "wall_clock" not in (self.regex.groupindex or {})

    def _log_text(self, exp_dir, log_path) -> str:
        candidate = Path(log_path) if log_path else Path(exp_dir) / "logs" / "run.log"
        if not candidate.exists():
            return ""
        try:
            return candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def read_points(self, exp_dir, log_path=None) -> List[dict]:
        if self.regex is None:
            return []
        points: List[dict] = []
        for index, line in enumerate(self._log_text(exp_dir, log_path).splitlines()):
            match = self.regex.search(line)
            if not match:
                continue
            groups = match.groupdict()
            if groups.get("loss") is None:
                continue
            point = {"loss": groups.get("loss")}
            if groups.get("wall_clock") is not None:
                point["wall_clock"] = groups.get("wall_clock")
            else:
                point["wall_clock"] = index * self.time_step
            if groups.get("epoch") is not None:
                point["epoch"] = groups.get("epoch")
            else:
                point["epoch"] = index
            points.append(point)
        return normalize_points(points)


# -------------------------------------------------------------- sidecar source
class SidecarFileSource(LossSource):
    """Parse a CSV / JSON / JSONL file the program already writes."""

    name = "sidecar"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.path = self.config.get("path", "metrics_history.csv")
        self.format = str(self.config.get("format", "csv")).lower()
        self.columns = {
            "wall_clock": self.config.get("columns", {}).get("wall_clock", "wall_clock"),
            "loss": self.config.get("columns", {}).get("loss", "loss"),
            "epoch": self.config.get("columns", {}).get("epoch", "epoch"),
        }

    def _resolve(self, exp_dir) -> List[Path]:
        base = Path(exp_dir)
        raw = self.path
        if any(token in raw for token in ("*", "?", "[")):
            return sorted(base.glob(raw))
        candidate = base / raw
        return [candidate] if candidate.exists() else []

    def _rows_from(self, path: Path) -> List[dict]:
        try:
            if self.format == "jsonl":
                return read_jsonl(path)
            if self.format == "json":
                payload = load_json(path, default=[]) or []
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    return payload.get("points", [])
                return []
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
                return list(csv.DictReader(handle))
        except (OSError, ValueError, csv.Error):
            return []

    def read_points(self, exp_dir, log_path=None) -> List[dict]:
        points: List[dict] = []
        for path in self._resolve(exp_dir):
            for row in self._rows_from(path):
                if not isinstance(row, dict):
                    continue
                points.append({
                    "wall_clock": row.get(self.columns["wall_clock"]),
                    "loss": row.get(self.columns["loss"]),
                    "epoch": row.get(self.columns["epoch"]),
                })
        return normalize_points(points)


# ---------------------------------------------------------- completion source
class CompletionOnlySource(LossSource):
    """No live curve: only the final ``metrics.json`` is available."""

    name = "completion_only"
    live = False

    def read_points(self, exp_dir, log_path=None) -> List[dict]:
        return []

    def read_final(self, exp_dir, log_text: str = "") -> dict:
        return load_json(Path(exp_dir) / "metrics.json", default={}) or {}


# ------------------------------------------------------------------- registry
_SOURCES: Dict[str, type] = {}


def register_source(name: str, source_cls: type) -> None:
    """Register a :class:`LossSource` subclass under ``name``."""
    if not isinstance(name, str) or not name:
        raise ValueError("source name must be a non-empty string")
    if not (isinstance(source_cls, type) and issubclass(source_cls, LossSource)):
        raise ValueError("source_cls must be a LossSource subclass")
    _SOURCES[name] = source_cls


def available_sources() -> List[str]:
    """Return the sorted names of the registered sources."""
    return sorted(_SOURCES)


for _builtin in (JsonlSource, StdoutRegexSource, SidecarFileSource, CompletionOnlySource):
    register_source(_builtin.name, _builtin)


def _load_custom(spec: dict) -> type:
    module_name = spec.get("module")
    class_name = spec.get("class")
    if not module_name or not class_name:
        raise ValueError("a custom source needs both 'module' and 'class'")
    module = importlib.import_module(module_name)
    source_cls = getattr(module, class_name, None)
    if not (isinstance(source_cls, type) and issubclass(source_cls, LossSource)):
        raise ValueError(f"{module_name}.{class_name} is not a LossSource subclass")
    return source_cls


def build_source(spec: Optional[dict] = None) -> LossSource:
    """Instantiate a source from a spec mapping (or the default ``jsonl``)."""
    if spec is None:
        spec = DEFAULT_SPEC
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        raise ValueError("stream spec must be a mapping or a source name")
    spec_type = spec.get("type", "jsonl")
    config = spec.get("config", {}) or {}
    if spec_type == "custom":
        source_cls = _load_custom(spec)
    else:
        if spec_type not in _SOURCES:
            names = ", ".join(available_sources()) or "(none)"
            raise ValueError(f"unknown stream source '{spec_type}'; available: {names}")
        source_cls = _SOURCES[spec_type]
    return source_cls(config)


def load_stream_spec(project_root) -> dict:
    """Read ``.nems/stream_adapter.json``; default to the jsonl contract."""
    spec = load_json(nems_dir(project_root) / STREAM_SPEC_FILENAME, default=None)
    if not isinstance(spec, dict) or not spec.get("type"):
        return dict(DEFAULT_SPEC)
    spec.setdefault("config", {})
    return spec


def save_stream_spec(project_root, spec: dict) -> str:
    """Persist a stream spec under ``.nems/`` and return its path."""
    normalized = {"type": spec.get("type", "jsonl"), "config": spec.get("config", {}) or {}}
    return save_json(nems_dir(project_root) / STREAM_SPEC_FILENAME, normalized)


def read_points(spec: Optional[dict], exp_dir, log_path=None) -> List[dict]:
    """Convenience: build a source from ``spec`` and read its points."""
    return build_source(spec).read_points(exp_dir, log_path=log_path)
