"""Small-parameter micro-tuning.

Chooses at most 3 small numeric parameters (default 2 candidate values each),
runs one short experiment per value, compares against the baseline, and marks
each candidate "建议保留" (keep) or "建议回退" (revert).
"""

from __future__ import annotations

import re
from pathlib import Path

from . import ensure_dir, nems_dir, save_json
from . import result_analyzer

EXCLUDE_KEYWORDS = ("epoch", "iter", "batch", "seed", "step")
RANK_KEYWORDS = ("lr", "learning_rate", "learn_rate", "hidden", "dim", "width", "layer", "depth", "num_layer")


def _unique(values, base):
    out = []
    for value in values:
        if value == base or value in out:
            continue
        out.append(value)
    return out


def _rank(name: str) -> int:
    lower = name.lower()
    for i, key in enumerate(RANK_KEYWORDS):
        if key in lower:
            return i
    return len(RANK_KEYWORDS)


def select_params(confirmed_params, max_params=3, values_per_param=2):
    """Pick numeric parameters and their candidate values."""
    candidates = []
    for param in confirmed_params:
        value = param.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        lower = str(param.get("name", "")).lower()
        if any(key in lower for key in EXCLUDE_KEYWORDS):
            continue
        candidates.append(param)
    candidates.sort(key=lambda p: _rank(str(p.get("name", ""))))

    selected = []
    for param in candidates[:max_params]:
        name = param["name"]
        value = param["value"]
        if isinstance(value, int):
            if "layer" in name.lower() or "depth" in name.lower():
                raw = [value + 2, max(2, value - 2)]
            else:
                raw = [value * 2, max(1, value // 2)]
            values = _unique(raw, value)
        else:
            raw = [round(value * 3.0, 8), round(value / 3.0, 8)]
            values = _unique(raw, value)
        selected.append({
            "name": name,
            "base": value,
            "values": values[:values_per_param],
            "type": param.get("type"),
        })
    return selected


def _verdict(analysis) -> str:
    ttt = analysis.get("time_to_target", {})
    exp_time = ttt.get("experiment_sec")
    base_time = ttt.get("baseline_sec")
    ratio = analysis.get("final_error_ratio")
    if exp_time is not None and (base_time is None or exp_time < base_time):
        return "建议保留"
    if ratio is not None and ratio < 0.98:
        return "建议保留"
    if analysis.get("early_stopped") or (ratio is not None and ratio > 1.02):
        return "建议回退"
    return "建议回退"


def run_tuning(
    project_root,
    exp_id,
    base_config,
    baseline_record,
    confirmed_params,
    runner,
    max_wall_clock=30,
    max_params=3,
    values_per_param=2,
    analyzer=None,
):
    """Run the tuning sweep and persist the plan/results."""
    root = Path(project_root)
    tuning_dir = ensure_dir(nems_dir(root) / "experiments" / exp_id / "param_tuning")
    selected = select_params(confirmed_params, max_params=max_params, values_per_param=values_per_param)

    base_config_path = nems_dir(root) / "experiments" / exp_id / "current_config.json"
    results = []
    for item in selected:
        for idx, value in enumerate(item["values"]):
            config = dict(base_config)
            config[item["name"]] = value
            trial_id = f"tune_{re.sub(r'[^A-Za-z0-9]+', '_', item['name'])}_v{idx}"
            record = runner.run(
                trial_id,
                config,
                baseline_series=baseline_record.get("series", []),
                early_stop=True,
                max_wall_clock=max_wall_clock,
                metadata={"tuning_for": exp_id, "param": item["name"], "value": value},
            )
            analysis = result_analyzer.analyze(
                record, baseline_record, history=[], analyzer=analyzer
            )
            results.append({
                "param": item["name"],
                "base_value": item["base"],
                "trial_value": value,
                "trial_id": trial_id,
                "status": record["status"],
                "early_stopped": record["early_stopped"],
                "final_error": analysis.get("final_experiment_error"),
                "baseline_error": analysis.get("final_baseline_error"),
                "final_error_ratio": analysis.get("final_error_ratio"),
                "time_to_target": analysis.get("time_to_target"),
                "verdict": _verdict(analysis),
            })

    # restore the base experiment config pointer
    runner.point_current_config(base_config_path)

    summary = {
        "exp_id": exp_id,
        "max_wall_clock": max_wall_clock,
        "selected_params": selected,
        "results": results,
        "keep": [r for r in results if r["verdict"] == "建议保留"],
        "revert": [r for r in results if r["verdict"] == "建议回退"],
    }
    save_json(tuning_dir / "results.json", summary)
    (tuning_dir / "TUNING.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _render_markdown(summary) -> str:
    lines = [
        "# Small-Parameter Auto-Tuning",
        "",
        f"- experiment: `{summary['exp_id']}`",
        f"- max wall-clock per trial: {summary['max_wall_clock']}s",
        "",
        "## Plan",
        "",
        "| parameter | base | candidate values |",
        "| --- | --- | --- |",
    ]
    for item in summary["selected_params"]:
        values = ", ".join(str(v) for v in item["values"])
        lines.append(f"| {item['name']} | {item['base']} | {values} |")
    lines += ["", "## Results", "", "| parameter | value | status | final error | error ratio | verdict |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in summary["results"]:
        ratio = row["final_error_ratio"]
        ratio_text = f"{ratio:.3f}" if isinstance(ratio, (int, float)) else str(ratio)
        final = row["final_error"]
        final_text = f"{final:.3e}" if isinstance(final, (int, float)) else str(final)
        lines.append(
            f"| {row['param']} | {row['trial_value']} | {row['status']} | {final_text} | {ratio_text} | {row['verdict']} |"
        )
    lines.append("")
    return "\n".join(lines)
