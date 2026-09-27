"""Maintain ``.nems/history.json`` and the experiment index."""

from __future__ import annotations

from pathlib import Path

from . import iso_timestamp, nems_dir, save_json, load_json
from . import report_generator
from .early_stopper import align_series


def load_history(project_root) -> list:
    return load_json(nems_dir(project_root) / "history.json", default=[]) or []


def curve_from_series(series, max_points=60):
    """Downsample an aligned error curve for compact storage/plotting."""
    aligned = align_series(series or [])
    items = sorted(aligned.items())
    if not items:
        return []
    if len(items) > max_points:
        stride = max(1, len(items) // max_points)
        items = items[::stride] + [items[-1]]
    return [{"sec": sec, "error": error} for sec, error in items]


def add_experiment(project_root, record, analysis=None, baseline_curve=None):
    """Insert or replace an experiment entry in ``history.json``."""
    history = load_history(project_root)
    analysis = analysis or {}
    metrics = record.get("metrics", {}) or {}
    exp_id = record.get("exp_id")
    entry = {
        "exp_id": exp_id,
        "status": record.get("status"),
        "config": record.get("config"),
        "metrics": {
            "final_loss": metrics.get("final_loss"),
            "final_error": metrics.get("final_error"),
            "total_wall_clock": metrics.get("total_wall_clock"),
            "epochs": metrics.get("epochs"),
        },
        "early_stopped": record.get("early_stopped", False),
        "early_stop": record.get("early_stop"),
        "verdict": analysis.get("verdict"),
        "final_error_ratio": analysis.get("final_error_ratio"),
        "time_to_target": analysis.get("time_to_target"),
        "timestamp": iso_timestamp(),
        "curve": curve_from_series(record.get("series", [])),
    }
    history = [h for h in history if h.get("exp_id") != exp_id]
    history.append(entry)
    save_json(nems_dir(project_root) / "history.json", history)
    return entry


def rebuild_index(project_root, baseline_record=None) -> str:
    """Regenerate ``.nems/INDEX.md`` from the current history."""
    return report_generator.write_index(project_root, load_history(project_root), baseline_record)
