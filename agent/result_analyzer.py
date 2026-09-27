"""Compare an experiment against the baseline and history at equal wall-clock."""

from __future__ import annotations

from pathlib import Path

from . import load_json, nems_dir
from .early_stopper import aligned_comparison, align_series, loss_to_error
from .llm_analyzer import LLMAnalyzer


def load_baseline(project_root) -> dict:
    path = nems_dir(project_root) / "baseline" / "baseline.json"
    return load_json(path, default={}) or {}


def load_history(project_root) -> list:
    path = nems_dir(project_root) / "history.json"
    return load_json(path, default=[]) or []


def time_to_target(series, target_error):
    """First integer second at which ``error <= target_error`` (or None)."""
    if target_error is None:
        return None
    aligned = align_series(series)
    for sec in sorted(aligned):
        error = aligned[sec]
        if error is not None and error <= target_error:
            return sec
    return None


def final_error(series):
    if not series:
        return None
    return loss_to_error(series[-1].get("loss"))


def analyze(experiment_record, baseline_record, history=None, target_error=None,
            prompts_dir=None, analyzer=None) -> dict:
    """Return a structured comparison plus an LLM analysis."""
    exp_series = experiment_record.get("series", [])
    base_series = baseline_record.get("series", [])
    aligned = aligned_comparison(exp_series, base_series)

    exp_final = final_error(exp_series)
    base_final = final_error(base_series)
    if target_error is None:
        target_error = base_final

    base_time = time_to_target(base_series, target_error)
    exp_time = time_to_target(exp_series, target_error)

    finite_ratios = [
        row["ratio"] for row in aligned
        if row["ratio"] is not None and row["ratio"] == row["ratio"]
        and row["ratio"] not in (float("inf"), float("-inf"))
    ]
    final_ratio = None
    if exp_final is not None and base_final not in (None, 0):
        final_ratio = exp_final / base_final

    if experiment_record.get("early_stopped"):
        verdict = "failed"
    elif finite_ratios and max(finite_ratios) > 1.10:
        verdict = "worse"
    elif final_ratio is not None and final_ratio < 0.95:
        verdict = "improved"
    else:
        verdict = "similar"

    analysis = {
        "exp_id": experiment_record.get("exp_id"),
        "verdict": verdict,
        "final_experiment_error": exp_final,
        "final_baseline_error": base_final,
        "final_error_ratio": final_ratio,
        "target_error": target_error,
        "time_to_target": {
            "target_error": target_error,
            "baseline_sec": base_time,
            "experiment_sec": exp_time,
        },
        "max_ratio": max(finite_ratios) if finite_ratios else None,
        "aligned": aligned,
    }

    llm = analyzer or LLMAnalyzer(prompts_dir=prompts_dir)
    try:
        llm_report = llm.analyze_report(experiment_record, baseline_record, history or [], aligned)
    except Exception as exc:  # never let analysis break a run
        llm_report = {"verdict": verdict, "summary": f"LLM analysis unavailable: {exc}",
                      "possible_reasons": [], "recommendations": []}
    analysis["llm_analysis"] = llm_report
    if llm_report.get("verdict"):
        analysis["verdict"] = llm_report["verdict"] if verdict != "failed" else "failed"
    return analysis
