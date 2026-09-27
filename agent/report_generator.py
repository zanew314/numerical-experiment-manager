"""Generate Markdown reports and figures for the managed project."""

from __future__ import annotations

from pathlib import Path

from . import ensure_dir, nems_dir, save_json
from .early_stopper import align_series


def _fmt(value, digits=3):
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    if value is None:
        return "null"
    return str(value)


def write_project_overview(project_root, summary, data_flow=None, confirmed=None):
    nd = ensure_dir(nems_dir(project_root))
    arch = summary.get("architecture", {}) or {}
    optimizer = summary.get("optimizer", {}) or {}
    training = summary.get("training", {}) or {}
    lines = [
        "# Project Overview",
        "",
        f"- model type: **{_fmt(summary.get('model_type'))}**",
        f"- entry point: `{_fmt(summary.get('entry_point'))}`",
        f"- layers: {_fmt(arch.get('layers'))}",
        f"- neurons per layer: {_fmt(arch.get('neurons_per_layer'))}",
        f"- activation: {_fmt(arch.get('activation'))}",
        f"- optimizer: {_fmt(optimizer.get('type'))} (lr={_fmt(optimizer.get('lr'))}, special={_fmt(optimizer.get('special'))})",
        f"- scheduler: {_fmt(optimizer.get('scheduler'))}",
        f"- test function: {_fmt(summary.get('test_function'))}",
        f"- epochs: {_fmt(training.get('epochs'))}, batch: {_fmt(training.get('batch_size'))}, sampling: {_fmt(training.get('sampling_strategy'))}",
        "",
        "## Loss",
        "",
    ]
    for term in (summary.get("loss", {}) or {}).get("terms", []):
        lines.append(f"- {term.get('name')} (weight {term.get('weight')}): `{term.get('formula')}`")
    lines += ["", "## Data Flow", "", "| from | to | content |", "| --- | --- | --- |"]
    for edge in (data_flow if data_flow is not None else summary.get("data_flow", [])):
        lines.append(f"| {edge.get('from')} | {edge.get('to')} | `{edge.get('content')}` |")
    lines += ["", "## Confirmed Hyperparameters", ""]
    params = confirmed if confirmed is not None else summary.get("hyperparameters", [])
    if params:
        lines += ["| name | value | type | location |", "| --- | --- | --- | --- |"]
        for param in params:
            lines.append(
                f"| {param.get('name')} | `{param.get('value')}` | {param.get('type')} | {param.get('location')} |"
            )
    else:
        lines.append("_none_")
    lines.append("")
    path = nd / "PROJECT_OVERVIEW.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_environment_setup(project_root, summary):
    nd = ensure_dir(nems_dir(project_root))
    lines = [
        "# Environment Setup",
        "",
        "This project is managed by the ai-numerical-experiment-manager skill.",
        "Only Python projects are supported.",
        "",
        "## Detected stack",
        "",
        f"- entry point: `{_fmt(summary.get('entry_point'))}`",
        f"- model type: {_fmt(summary.get('model_type'))}",
        "",
        "## Reproduce",
        "",
        "1. Create/activate a Python environment.",
        "2. Install the dependencies imported by the project (e.g. `torch`, `numpy`).",
        "3. Run the entry point once to verify it works before experiments:",
        "",
        "   ```bash",
        f"   python {summary.get('entry_point') or 'main.py'}",
        "   ```",
        "",
        "## Experiment hooks",
        "",
        "The manager sets these environment variables when running experiments:",
        "",
        "| variable | meaning |",
        "| --- | --- |",
        "| `NEMS_OUTPUT_DIR` | directory for `loss_time.jsonl` and `metrics.json` |",
        "| `NEMS_MAX_WALL_CLOCK` | optional soft wall-clock limit in seconds |",
        "| `NEMS_EXP_ID` | experiment identifier |",
        "",
        "Configured parameters are read from `.nems/current_config.json` by the generated adapter module.",
        "",
        "If the program does not write `loss_time.jsonl`, the manager reads its run-time",
        "output through a source configured in `.nems/stream_adapter.json` (stdout regex,",
        "a sidecar CSV/JSON, or the final `metrics.json` only).",
        "",
    ]
    path = nd / "ENVIRONMENT_SETUP.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_baseline_report(project_root, baseline_record):
    nd = ensure_dir(nems_dir(project_root) / "baseline")
    metrics = baseline_record.get("metrics", {}) or {}
    lines = [
        "# Baseline Report",
        "",
        f"- source: {baseline_record.get('source', 'model code (default config)')}",
        f"- status: {baseline_record.get('status')}",
        f"- final loss: {_fmt(metrics.get('final_loss'), 6)}",
        f"- final error (RMSE): {_fmt(metrics.get('final_error'), 6)}",
        f"- total wall-clock: {_fmt(metrics.get('total_wall_clock'), 6)} s",
        f"- epochs: {_fmt(metrics.get('epochs'))}",
        "",
        "## Wall-clock / error curve",
        "",
        "| second | error (RMSE) |",
        "| --- | --- |",
    ]
    aligned = align_series(baseline_record.get("series", []))
    for sec in sorted(aligned):
        lines.append(f"| {sec} | {_fmt(aligned[sec], 6)} |")
    lines.append("")
    path = nd / "BASELINE_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_experiment_report(project_root, record, analysis):
    exp_dir = ensure_dir(nems_dir(project_root) / "experiments" / record["exp_id"])
    llm = analysis.get("llm_analysis", {}) or {}
    metrics = record.get("metrics", {}) or {}
    lines = [
        f"# Experiment Report: {record['exp_id']}",
        "",
        f"- status: **{record.get('status')}**",
        f"- verdict: **{analysis.get('verdict')}**",
        f"- config: `{record.get('config')}`",
        f"- final error (RMSE): {_fmt(analysis.get('final_experiment_error'), 6)}",
        f"- baseline error (RMSE): {_fmt(analysis.get('final_baseline_error'), 6)}",
        f"- final error ratio: {_fmt(analysis.get('final_error_ratio'), 4)}",
        f"- max aligned ratio: {_fmt(analysis.get('max_ratio'), 4)}",
        f"- wall-clock: {_fmt(metrics.get('total_wall_clock'), 6)} s",
        "",
        "## Summary",
        "",
        str(llm.get("summary", "")),
        "",
    ]
    early = record.get("early_stop") or {}
    if early.get("triggered"):
        lines += [
            "## Early Stop",
            "",
            f"- triggered at second {early.get('trigger_second')}",
            f"- ratio: {_fmt(early.get('trigger_ratio'), 4)} (threshold {_fmt(early.get('threshold_ratio'), 4)})",
            f"- reason: {early.get('trigger_reason')}",
            "",
        ]
    lines += ["## Aligned comparison (same wall-clock)", "",
              "| second | baseline error | experiment error | ratio |",
              "| --- | --- | --- | --- |"]
    for row in analysis.get("aligned", []):
        lines.append(
            f"| {row['wall_clock_sec']} | {_fmt(row['baseline_error'], 6)} | "
            f"{_fmt(row['experiment_error'], 6)} | {_fmt(row['ratio'], 4)} |"
        )
    lines += ["", "## Possible reasons", ""]
    for reason in llm.get("possible_reasons", []):
        lines.append(f"- {reason}")
    lines += ["", "## Recommendations", ""]
    for rec in llm.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.append("")
    path = exp_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_index(project_root, history, baseline_record=None):
    nd = ensure_dir(nems_dir(project_root))
    lines = [
        "# Experiment Index",
        "",
        "| exp_id | status | verdict | final error | baseline error | wall-clock (s) | notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    base_error = None
    if baseline_record:
        base_error = (baseline_record.get("metrics", {}) or {}).get("final_error")
    lines.append(
        f"| baseline | {('ok' if baseline_record else 'missing')} | reference | "
        f"{_fmt(base_error, 6)} | - | "
        f"{_fmt((baseline_record or {}).get('metrics', {}).get('total_wall_clock'), 6)} | fixed baseline |"
    )
    for rec in history:
        metrics = rec.get("metrics", {}) or {}
        notes = ""
        if rec.get("early_stopped"):
            notes = "early stopped"
        lines.append(
            f"| {rec.get('exp_id')} | {rec.get('status')} | {rec.get('verdict', '')} | "
            f"{_fmt(metrics.get('final_error'), 6)} | {_fmt(base_error, 6)} | "
            f"{_fmt(metrics.get('total_wall_clock'), 6)} | {notes} |"
        )
    lines.append("")
    path = nd / "INDEX.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_project_summary_diff(project_root, diff_result, snapshot_id):
    nd = ensure_dir(nems_dir(project_root))
    lines = [
        "# Project Summary Diff",
        "",
        f"- snapshot: `{snapshot_id}`",
        "",
        "## Changed fields",
        "",
        "| field | before | after | reason |",
        "| --- | --- | --- | --- |",
    ]
    for change in diff_result.get("changes", []):
        lines.append(
            f"| {change['field']} | `{_fmt(change.get('before'))}` | "
            f"`{_fmt(change.get('after'))}` | {change.get('reason')} |"
        )
    if not diff_result.get("changes"):
        lines.append("| _none_ | | | |")
    lines += ["", "## Hyperparameter changes", "",
              "| name | old | new | location |", "| --- | --- | --- | --- |"]
    for change in diff_result.get("hyperparameter_changes", []):
        lines.append(
            f"| {change['name']} | `{change.get('old')}` | `{change.get('new')}` | {change.get('location')} |"
        )
    if not diff_result.get("hyperparameter_changes"):
        lines.append("| _none_ | | | |")
    lines += ["", "## Unchanged", ""]
    for key, value in (diff_result.get("unchanged", {}) or {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    path = nd / "project_summary_diff.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_overall_report(project_root, baseline_record=None, history=None, tuning=None):
    """Write ``.nems/REPORT.md`` summarizing the whole run history."""
    nd = ensure_dir(nems_dir(project_root))
    history = history or []
    base_metrics = (baseline_record or {}).get("metrics", {}) or {}
    lines = [
        "# Numerical Experiment Analysis Report",
        "",
        "Generated by the ai-numerical-experiment-manager skill.",
        "",
        "## Baseline",
        "",
        f"- status: {(baseline_record or {}).get('status', 'missing')}",
        f"- final error (RMSE): {_fmt(base_metrics.get('final_error'), 6)}",
        f"- total wall-clock: {_fmt(base_metrics.get('total_wall_clock'), 6)} s",
        "",
        "## Experiments",
        "",
        "| exp_id | status | verdict | final error | error ratio | wall-clock (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rec in history:
        metrics = rec.get("metrics", {}) or {}
        lines.append(
            f"| {rec.get('exp_id')} | {rec.get('status')} | {rec.get('verdict', '')} | "
            f"{_fmt(metrics.get('final_error'), 6)} | {_fmt(rec.get('final_error_ratio'), 4)} | "
            f"{_fmt(metrics.get('total_wall_clock'), 6)} |"
        )
    if not history:
        lines.append("| _none_ | | | | | |")

    lines += ["", "## Early stops", ""]
    stopped = [r for r in history if r.get("early_stopped")]
    if stopped:
        for rec in stopped:
            early = rec.get("early_stop") or {}
            lines.append(
                f"- {rec['exp_id']}: stopped at second {early.get('trigger_second')} "
                f"(ratio {_fmt(early.get('trigger_ratio'), 4)})"
            )
    else:
        lines.append("- none")

    lines += ["", "## Small-parameter tuning", ""]
    if tuning and tuning.get("results"):
        lines += ["| parameter | value | verdict |", "| --- | --- | --- |"]
        for row in tuning["results"]:
            lines.append(f"| {row['param']} | {row['trial_value']} | {row['verdict']} |")
        lines += ["", f"- keep: {[r['trial_id'] for r in tuning.get('keep', [])]}",
                  f"- revert: {[r['trial_id'] for r in tuning.get('revert', [])]}"]
    else:
        lines.append("- not run in this session")

    lines.append("")
    path = nd / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_metrics_evolution(project_root, baseline_record=None, history=None, out_path=None):
    """Plot error vs wall-clock for the baseline and the experiments."""
    history = history or []
    nd = ensure_dir(nems_dir(project_root))
    out_path = Path(out_path) if out_path else nd / "figures" / "metrics_evolution.png"
    ensure_dir(out_path.parent)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted = False

    def _curve_points(record_or_curve):
        if not record_or_curve:
            return []
        if isinstance(record_or_curve, dict) and "curve" in record_or_curve:
            return [(item["sec"], item["error"]) for item in record_or_curve["curve"]]
        if isinstance(record_or_curve, dict) and "series" in record_or_curve:
            aligned = align_series(record_or_curve["series"])
            return sorted(aligned.items())
        return []

    base_points = _curve_points(baseline_record or {})
    if base_points:
        xs, ys = zip(*base_points)
        ax.plot(xs, ys, label="baseline", linewidth=2, color="black")
        plotted = True

    for rec in history:
        points = _curve_points(rec)
        if not points:
            continue
        xs, ys = zip(*points)
        style = "--" if rec.get("early_stopped") else "-"
        ax.plot(xs, ys, style, label=str(rec.get("exp_id")), linewidth=1.5)
        plotted = True

    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel("error (RMSE)")
    ax.set_title("Metrics evolution (same wall-clock)")
    ax.set_yscale("log")
    if plotted:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)
