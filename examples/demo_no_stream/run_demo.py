#!/usr/bin/env python3
"""Demonstrate managing a project that has NO loss_time.jsonl stream.

The demo copies ``main.py`` (which only *prints* progress) into a sandbox, then
teaches the manager to read that printed output with the ``stdout_regex``
loss-stream source. It then runs a baseline, a good experiment, and a bad one
that early-stops, and writes the usual reports.

Run from the package directory:

    python examples/demo_no_stream/run_demo.py
    python examples/demo_no_stream/run_demo.py --workspace D:\\tmp\\nems_nostream
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL_ROOT))

from agent import (  # noqa: E402
    ensure_dir,
    experiment_runner,
    history_manager,
    llm_analyzer,
    loss_stream,
    model_structure,
    nems_dir,
    report_generator,
    result_analyzer,
    save_json,
)

DEMO_DIR = Path(__file__).resolve().parent
BASE_CONFIG = {"LR": 0.01, "HIDDEN_DIM": 64, "EPOCHS": 400}
# The program prints: epoch=<i> loss=<v> elapsed=<seconds>
STDOUT_SPEC = {
    "type": "stdout_regex",
    "config": {
        "pattern": r"epoch=(?P<epoch>\d+) loss=(?P<loss>[-+0-9.eEnanif]+) elapsed=(?P<wall_clock>[0-9.]+)"
    },
}


def banner(step, title):
    print()
    print("=" * 72)
    print(f"STEP {step}: {title}")
    print("=" * 72)


def prepare_workspace(workspace: Path) -> Path:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    shutil.copyfile(DEMO_DIR / "main.py", workspace / "main.py")
    return workspace


def step1_model_structure(project_root):
    banner(1, "Model structure snapshot (AST skeleton, no code copy)")
    result = model_structure.write_model_structure(project_root)
    structure = result["structure"]
    classes = [f"{c['module']}::{c['name']}" for c in structure["model_classes"]]
    print(f"  files analysed : {structure['file_count']}")
    print(f"  model classes  : {classes}")
    for cls in structure["model_classes"]:
        print(f"    {cls['name']}: {cls['layer_count']} layer(s)")
    print(f"  -> {result['json']}")
    print(f"  -> {result['markdown']}")
    return structure


def step2_configure_stream(project_root):
    banner(2, "Configure the no-stream adapter (read stdout)")
    path = loss_stream.save_stream_spec(project_root, STDOUT_SPEC)
    print(f"  source : {STDOUT_SPEC['type']}")
    print(f"  pattern: {STDOUT_SPEC['config']['pattern']}")
    print(f"  -> {path}")
    return loss_stream.build_source(STDOUT_SPEC)


def step3_baseline(project_root, runner):
    banner(3, "Baseline via printed output")
    record = runner.run("baseline", BASE_CONFIG, early_stop=False, max_wall_clock=30,
                        stream_spec=STDOUT_SPEC)
    baseline = {
        "exp_id": "baseline",
        "status": record["status"],
        "source": "no loss_time.jsonl; stdout_regex source",
        "metrics": record["metrics"],
        "series": record["series"],
        "stream": record.get("stream"),
    }
    nd = ensure_dir(nems_dir(project_root) / "baseline")
    save_json(nd / "baseline.json", baseline)
    report_generator.write_baseline_report(project_root, baseline)
    print(f"  status           : {record['status']}")
    print(f"  points from log  : {record['num_points']}")
    print(f"  final error      : {baseline['metrics'].get('final_error')}")
    print("  -> wrote .nems/baseline/baseline.json and BASELINE_REPORT.md")
    return baseline


def run_and_report(project_root, runner, analyzer, exp_id, config, baseline, title):
    banner(5 if exp_id == "exp_good" else 6, title)
    print(f"  config: {config}")
    history = history_manager.load_history(project_root)
    record = runner.run(exp_id, config, baseline_series=baseline["series"], early_stop=True,
                        max_wall_clock=30, stream_spec=STDOUT_SPEC)
    analysis = result_analyzer.analyze(record, baseline, history=history, analyzer=analyzer)
    report_generator.write_experiment_report(project_root, record, analysis)
    history_manager.add_experiment(project_root, record, analysis)
    print(f"  sink type        : {(record.get('stream') or {}).get('type')}")
    print(f"  points           : {record['num_points']}")
    print(f"  status           : {record['status']}")
    print(f"  early stopped    : {record['early_stopped']}")
    print(f"  final error      : {analysis.get('final_experiment_error')}")
    print(f"  verdict          : {analysis.get('verdict')}")
    print(f"  -> wrote .nems/experiments/{exp_id}/REPORT.md")
    return record


def step7_reports(project_root, baseline):
    banner(7, "Overall REPORT.md, INDEX.md and figure")
    history = history_manager.load_history(project_root)
    report_path = report_generator.write_overall_report(project_root, baseline, history)
    index_path = report_generator.write_index(project_root, history, baseline)
    figure_path = report_generator.write_metrics_evolution(project_root, baseline, history)
    print(f"  -> {report_path}")
    print(f"  -> {index_path}")
    print(f"  -> {figure_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="no-loss-stream demo")
    parser.add_argument("--workspace", default=str(DEMO_DIR / ".demo_workspace"),
                        help="sandbox directory for the demo project")
    args = parser.parse_args()

    project_root = prepare_workspace(Path(args.workspace))
    print(f"demo project root: {project_root}")

    analyzer = llm_analyzer.LLMAnalyzer(prompts_dir=SKILL_ROOT / "prompts")
    runner = experiment_runner.ExperimentRunner(project_root, entry="main.py", timeout=120,
                                                monitor_interval=0.2)

    step1_model_structure(project_root)
    step2_configure_stream(project_root)
    baseline = step3_baseline(project_root, runner)

    run_and_report(project_root, runner, analyzer, "exp_good", BASE_CONFIG, baseline,
                   "Good experiment (LR=0.01)")
    bad = run_and_report(project_root, runner, analyzer, "exp_bad",
                         {**BASE_CONFIG, "LR": 10.0}, baseline,
                         "Bad experiment (LR=10.0, early stop)")
    if not bad["early_stopped"]:
        raise RuntimeError("early stop did NOT trigger for LR=10.0; the demo needs review")

    step7_reports(project_root, baseline)

    print()
    print("DEMO COMPLETE: no-loss-stream lifecycle ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
