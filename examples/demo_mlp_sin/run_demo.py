#!/usr/bin/env python3
"""End-to-end demonstration of the ai-numerical-experiment-manager skill.

It copies ``main.py`` into a sandbox workspace and walks the full lifecycle:

    Step 1  first scan (LLM-style structure summary)
    Step 2  command-line confirmation (simulated or interactive)
    Step 3  hyperparameter injection (backup + get_param rewrite)
    Step 4  baseline (default config)
    Step 5  first experiment (LR=0.01)
    Step 6  early-stop demo (LR=10.0 triggers the 20% rule)
    Step 7  model update (NUM_LAYERS=6, incremental diff)
    Step 8  overall REPORT.md + INDEX.md + figure

Run:

    python run_demo.py
    python run_demo.py --interactive          # really prompt for selections
    python run_demo.py --with-tuning          # also run Phase 7 micro-tuning
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL_ROOT))

from agent import (  # noqa: E402
    code_reader,
    ensure_dir,
    experiment_runner,
    history_manager,
    hyperparameter_injector,
    iso_timestamp,
    llm_analyzer,
    load_json,
    model_structure,
    nems_dir,
    param_tuner,
    report_generator,
    result_analyzer,
    save_json,
)

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_KEEP = ["LR", "NUM_LAYERS", "HIDDEN_DIM"]
# The demo's MLP defines its layers explicitly (fc1..fc4), so NUM_LAYERS is
# informational and does not change the architecture; architecture changes are
# demonstrated through the Phase 6 source edit. Tuning therefore only sweeps
# parameters that actually affect this model.
DEMO_TUNING_EXCLUDE = {"NUM_LAYERS"}


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


def step1_first_scan(project_root, analyzer, reader):
    banner(1, "First project scan (LLM reads code)")
    files = reader.read_map()
    print(f"  collected {len(files)} python file(s): {list(files)}")
    summary = analyzer.analyze_structure(files)
    nd = ensure_dir(nems_dir(project_root))
    save_json(nd / "project_summary.json", summary)
    save_json(nd / "hyperparameters.json", summary["hyperparameters"])
    save_json(nd / "source_hashes.json", reader.hashes())
    structure = model_structure.write_model_structure(project_root)
    print(f"  analysis source : {summary.get('_source')}")
    print(f"  model_type      : {summary.get('model_type')}")
    print(f"  architecture    : {summary.get('architecture')}")
    print(f"  optimizer       : {summary.get('optimizer')}")
    print(f"  candidates      : {[h['name'] for h in summary['hyperparameters']]}")
    print(f"  -> wrote .nems/project_summary.json and .nems/hyperparameters.json")
    print(f"  -> wrote .nems/model_structure.json and .nems/model_structure.md "
          f"({len(structure['structure']['model_classes'])} model class(es))")
    return summary


def confirm_hyperparameters(candidates, keep_names, interactive=False):
    if interactive:
        print("  Candidate hyperparameters:")
        for idx, param in enumerate(candidates, start=1):
            print(f"    [{idx}] {param['name']} = {param['value']}  ({param['location']})")
        raw = input("  Enter numbers to KEEP (comma separated, empty=all): ").strip()
        if raw:
            keep_names = [candidates[int(tok) - 1]["name"] for tok in raw.replace(" ", "").split(",") if tok]
    selected = [p for p in candidates if p["name"] in keep_names]
    excluded = [p["name"] for p in candidates if p["name"] not in keep_names]
    return selected, excluded


def step2_confirmation(project_root, summary, keep_names, interactive=False):
    banner(2, "Command-line confirmation (keep / exclude)")
    candidates = summary["hyperparameters"]
    if not interactive:
        print("  simulated user dialogue:")
        for param in candidates:
            mark = "keep" if param["name"] in keep_names else "exclude"
            print(f"    user -> {param['name']:<12} {mark}")
    selected, excluded = confirm_hyperparameters(candidates, keep_names, interactive=interactive)
    nd = ensure_dir(nems_dir(project_root))
    save_json(nd / "hyperparameters_confirmed.json", {
        "confirmed_at": iso_timestamp(),
        "params": selected,
        "excluded": excluded,
    })
    save_json(nd / "data_flow_confirmed.json", {
        "confirmed_at": iso_timestamp(),
        "data_flow": summary.get("data_flow", []),
        "note": "simulated confirmation; a real session lets the user correct the graph",
    })
    report_generator.write_project_overview(project_root, summary, summary.get("data_flow"), selected)
    report_generator.write_environment_setup(project_root, summary)
    print(f"  kept    : {[p['name'] for p in selected]}")
    print(f"  excluded: {excluded}")
    print("  -> wrote hyperparameters_confirmed.json, data_flow_confirmed.json,")
    print("     PROJECT_OVERVIEW.md, ENVIRONMENT_SETUP.md")
    return selected


def step3_injection(project_root, selected):
    banner(3, "Hyperparameter injection (backup + get_param rewrite)")
    report = hyperparameter_injector.inject_hyperparameters(project_root, selected)
    hyperparameter_injector.write_injection_history(project_root, report, selected)
    print(f"  snapshot     : {report['snapshot_dir']}")
    print(f"  adapter      : {Path(report['config_module_path']).name}")
    print(f"  diff         : {report['diff_path']}")
    for item in report["injected"]:
        print(f"    injected {item['name']}: {item['before']}  ->  {item['after']}")
    for item in report["skipped"]:
        print(f"    skipped  {item['name']} (already injected or not found)")
    print("  -> wrote hyperparameter_history.json and hyperparameter_history.md")
    return report


def step4_baseline(project_root, runner, base_config):
    banner(4, "Baseline (default config)")
    record = runner.run("baseline", base_config, early_stop=False, max_wall_clock=60)
    baseline = {
        "exp_id": "baseline",
        "status": record["status"],
        "source": "original model code (injected defaults)",
        "metrics": record["metrics"],
        "series": record["series"],
    }
    nd = ensure_dir(nems_dir(project_root) / "baseline")
    save_json(nd / "baseline.json", baseline)
    shutil.copyfile(
        Path(record["paths"]["loss_time"]), nd / "loss_time.jsonl"
    )
    report_generator.write_baseline_report(project_root, baseline)
    print(f"  status          : {record['status']}")
    print(f"  final error     : {baseline['metrics'].get('final_error')}")
    print(f"  total wall-clock: {baseline['metrics'].get('total_wall_clock'):.3f}s")
    print("  -> wrote .nems/baseline/baseline.json and BASELINE_REPORT.md")
    return baseline


def run_and_report(project_root, runner, analyzer, exp_id, config, baseline, title):
    banner(5 if exp_id == "exp_001" else 6, title)
    print(f"  config: {config}")
    history = history_manager.load_history(project_root)
    record = runner.run(
        exp_id, config, baseline_series=baseline["series"], early_stop=True, max_wall_clock=60
    )
    analysis = result_analyzer.analyze(record, baseline, history=history, analyzer=analyzer)
    report_generator.write_experiment_report(project_root, record, analysis)
    history_manager.add_experiment(project_root, record, analysis)
    print(f"  status           : {record['status']}")
    print(f"  early stopped    : {record['early_stopped']}")
    retrain = record.get("retrain", {})
    if retrain.get("attempted"):
        print(f"  retrained        : {retrain.get('count')}x "
              f"(attempts: {[a['status'] for a in retrain.get('attempts', [])]})")
    if record["early_stop"]:
        early = record["early_stop"]
        print(f"  trigger second   : {early.get('trigger_second')} "
              f"(ratio {early.get('trigger_ratio')}, threshold {early.get('threshold_ratio')})")
    print(f"  final error      : {analysis.get('final_experiment_error')}")
    print(f"  baseline error   : {analysis.get('final_baseline_error')}")
    print(f"  verdict          : {analysis.get('verdict')}")
    print(f"  -> wrote .nems/experiments/{exp_id}/REPORT.md")
    return record, analysis


def apply_model_update(main_py: Path) -> None:
    """Simulate a user editing the model: 4 layers -> 6 layers."""
    text = main_py.read_text(encoding="utf-8")
    for old, new in (
        ('get_param("NUM_LAYERS", default=4)', 'get_param("NUM_LAYERS", default=6)'),
        ("NUM_LAYERS = 4", "NUM_LAYERS = 6"),
    ):
        if old in text:
            text = text.replace(old, new, 1)
            break
    else:
        raise RuntimeError("could not locate NUM_LAYERS assignment for the model update")

    old_fc4 = "        self.fc4 = nn.Linear(config.HIDDEN_DIM, 1)\n"
    new_fc = (
        "        self.fc4 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)\n"
        "        self.fc5 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)\n"
        "        self.fc6 = nn.Linear(config.HIDDEN_DIM, 1)\n"
    )
    if old_fc4 not in text:
        raise RuntimeError("could not locate fc4 definition for the model update")
    text = text.replace(old_fc4, new_fc, 1)

    old_forward = (
        "        x = torch.tanh(self.fc3(x))\n"
        "        x = torch.tanh(self.fc4(x))\n"
        "        return x\n"
    )
    new_forward = (
        "        x = torch.tanh(self.fc3(x))\n"
        "        x = torch.tanh(self.fc4(x))\n"
        "        x = torch.tanh(self.fc5(x))\n"
        "        x = torch.tanh(self.fc6(x))\n"
        "        return x\n"
    )
    if old_forward not in text:
        raise RuntimeError("could not locate forward body for the model update")
    text = text.replace(old_forward, new_forward, 1)
    main_py.write_text(text, encoding="utf-8")


def step7_model_update(project_root, analyzer, reader, snapshot_id="exp_001"):
    banner(7, "Incremental summary after model update (NUM_LAYERS=6)")
    nd = nems_dir(project_root)
    old_summary = load_json(nd / "project_summary.json", default={}) or {}
    old_hashes = load_json(nd / "source_hashes.json", default={}) or {}

    apply_model_update(project_root / "main.py")
    new_hashes = reader.hashes()
    changed = sorted(f for f in new_hashes if old_hashes.get(f) != new_hashes[f])
    added = sorted(f for f in new_hashes if f not in old_hashes)
    changed = sorted(set(changed) | set(added))
    unchanged = sorted(f for f in new_hashes if f not in changed)
    changed_code = reader.read_matching(changed)
    print(f"  changed files  : {changed}")
    print(f"  unchanged files: {unchanged}")

    diff = analyzer.analyze_diff(old_summary, changed, unchanged, changed_code, snapshot_id)
    report_generator.write_project_summary_diff(project_root, diff, snapshot_id)

    updated = dict(old_summary)
    for key, value in (diff.get("changed_summary") or {}).items():
        updated[key] = value
    updated["_unchanged"] = diff.get("unchanged", {})
    updated["_updated_from"] = snapshot_id
    save_json(nd / "project_summary.json", updated)
    save_json(nd / "source_hashes.json", new_hashes)

    hp_history_path = nd / "hyperparameter_history.json"
    hp_history = load_json(hp_history_path, default=[]) or []
    hp_history.append({
        "timestamp": iso_timestamp(),
        "type": "model_update",
        "snapshot_id": snapshot_id,
        "files_changed": changed,
        "changes": diff.get("changes", []),
        "hyperparameter_changes": diff.get("hyperparameter_changes", []),
    })
    save_json(hp_history_path, hp_history)
    with open(nd / "hyperparameter_history.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n## Model update ({snapshot_id})\n\n")
        for change in diff.get("changes", []):
            fh.write(f"- {change['field']}: `{change.get('before')}` -> `{change.get('after')}`\n")
        for change in diff.get("hyperparameter_changes", []):
            fh.write(f"- {change['name']}: `{change.get('old')}` -> `{change.get('new')}`\n")
        fh.write("\n")
    print(f"  changed fields : {[c['field'] for c in diff.get('changes', [])]}")
    print(f"  hp changes     : {diff.get('hyperparameter_changes')}")
    print(f"  unchanged      : {diff.get('unchanged')}")
    print("  -> wrote .nems/project_summary_diff.md, updated project_summary.json")
    return diff


def step8_reports(project_root, baseline, tuning=None):
    banner(8, "Generate REPORT.md, INDEX.md and figure")
    history = history_manager.load_history(project_root)
    report_path = report_generator.write_overall_report(project_root, baseline, history, tuning=tuning)
    index_path = report_generator.write_index(project_root, history, baseline)
    figure_path = report_generator.write_metrics_evolution(project_root, baseline, history)
    print(f"  -> {report_path}")
    print(f"  -> {index_path}")
    print(f"  -> {figure_path}")
    return history


def list_artifacts(project_root):
    nd = nems_dir(project_root)
    print()
    print(".nems/ artifact tree:")
    for path in sorted(nd.rglob("*")):
        rel = path.relative_to(nd)
        depth = len(rel.parts) - 1
        if depth > 2:
            continue
        indent = "  " * depth
        name = rel.parts[-1] + ("/" if path.is_dir() else "")
        print(f"  {indent}{name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ai-numerical-experiment-manager demo")
    parser.add_argument("--workspace", default=str(DEMO_DIR / ".demo_workspace"),
                        help="sandbox directory for the demo project")
    parser.add_argument("--interactive", action="store_true",
                        help="prompt for hyperparameter selection instead of simulating")
    parser.add_argument("--with-tuning", action="store_true",
                        help="also run the Phase 7 small-parameter micro-tuning")
    parser.add_argument("--keep", default=",".join(DEFAULT_KEEP),
                        help="comma-separated hyperparameter names to keep")
    args = parser.parse_args()

    project_root = prepare_workspace(Path(args.workspace))
    keep_names = [name.strip() for name in args.keep.split(",") if name.strip()]
    reader = code_reader.CodeReader(project_root)
    analyzer = llm_analyzer.LLMAnalyzer(prompts_dir=SKILL_ROOT / "prompts")
    runner = experiment_runner.ExperimentRunner(project_root, entry="main.py", timeout=180,
                                                monitor_interval=0.2)

    print(f"demo project root: {project_root}")

    summary = step1_first_scan(project_root, analyzer, reader)
    selected = step2_confirmation(project_root, summary, keep_names, interactive=args.interactive)
    step3_injection(project_root, selected)
    base_config = {p["name"]: p["value"] for p in selected}

    print()
    print("(warm-up run to stabilize wall-clock timing; not recorded)")
    runner.run("_warmup", base_config, early_stop=False, max_wall_clock=60)
    shutil.rmtree(nems_dir(project_root) / "experiments" / "_warmup", ignore_errors=True)

    baseline = step4_baseline(project_root, runner, base_config)

    run_and_report(project_root, runner, analyzer, "exp_001",
                   {**base_config, "LR": 0.01}, baseline,
                   "First experiment (LR=0.01)")

    record2, _ = run_and_report(project_root, runner, analyzer, "exp_002_earlystop",
                                {**base_config, "LR": 10.0}, baseline,
                                "Early-stop demo (LR=10.0)")
    if not record2["early_stopped"]:
        raise RuntimeError("early stop did NOT trigger for LR=10.0; the criterion needs review")

    step7_model_update(project_root, analyzer, reader)

    tuning = None
    if args.with_tuning:
        print()
        print("(optional) Phase 7 small-parameter micro-tuning")
        baseline = load_json(nems_dir(project_root) / "baseline" / "baseline.json", default={}) or baseline
        tuning = param_tuner.run_tuning(
            project_root,
            exp_id="exp_001",
            base_config={**base_config, "LR": 0.01},
            baseline_record=baseline,
            confirmed_params=[p for p in selected if p["name"] not in DEMO_TUNING_EXCLUDE],
            runner=runner,
            max_wall_clock=20,
            analyzer=analyzer,
        )
        print(f"  keep  : {[r['trial_id'] for r in tuning['keep']]}")
        print(f"  revert: {[r['trial_id'] for r in tuning['revert']]}")

    step8_reports(project_root, baseline, tuning=tuning)
    list_artifacts(project_root)

    print()
    print("DEMO COMPLETE: all 8 steps ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
