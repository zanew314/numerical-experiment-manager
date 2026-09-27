#!/usr/bin/env python3
"""Demonstrate the manager on the 2D Poisson problem (PINN + ParticleWNN + FD).

It copies ``main.py`` (a PINN), ``particle_wnn.py`` (a weak-form ParticleWNN) and
``traditional_solver.py`` (a classical 5-point finite-difference solver) into a
sandbox, then:

    1. saves the model-structure snapshot (AST skeleton);
    2. runs the traditional finite-difference solver as a numerical reference;
    3. runs a PINN baseline, a good experiment, and a bad experiment that
       early-stops;
    4. runs the ParticleWNN directly and records its accuracy;
    5. writes a traditional / PINN / ParticleWNN note plus the usual reports.

Run from the package directory:

    python examples/demo_poisson/run_demo.py
    python examples/demo_poisson/run_demo.py --workspace D:\\tmp\\nems_poisson
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL_ROOT))

from agent import (  # noqa: E402
    ensure_dir,
    experiment_runner,
    history_manager,
    llm_analyzer,
    load_json,
    model_structure,
    nems_dir,
    report_generator,
    result_analyzer,
    save_json,
)

import traditional_solver  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
# The repository notebook uses freq=4*pi. That target needs a much larger network
# and a longer budget than a CPU demo allows, so the demo uses freq=pi (also a
# valid choice in the repository's own Problem class). See PROBLEM.md.
FREQ = math.pi
BASE_CONFIG = {"FREQ": FREQ, "HIDDEN": 32, "DEPTH": 3, "LR": 1e-3,
               "EPOCHS": 600, "N_COLLOCATION": 600, "N_TEST": 31}
GOOD_CONFIG = {**BASE_CONFIG, "LR": 1e-2}
BAD_CONFIG = {**BASE_CONFIG, "LR": 1.0}


def banner(step, title):
    print()
    print("=" * 72)
    print(f"STEP {step}: {title}")
    print("=" * 72)


def _num(value):
    return f"{value:.3e}" if isinstance(value, (int, float)) else "n/a"


def prepare_workspace(workspace: Path) -> Path:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    shutil.copyfile(DEMO_DIR / "main.py", workspace / "main.py")
    shutil.copyfile(DEMO_DIR / "traditional_solver.py", workspace / "traditional_solver.py")
    shutil.copyfile(DEMO_DIR / "particle_wnn.py", workspace / "particle_wnn.py")
    return workspace


def step1_model_structure(project_root):
    banner(1, "Model structure snapshot (AST skeleton)")
    result = model_structure.write_model_structure(project_root)
    structure = result["structure"]
    for rel in sorted(structure["modules"]):
        models = [c["name"] for c in structure["model_classes"] if c["module"] == rel]
        print(f"  {rel}: classes={[c['name'] for c in structure['modules'][rel]['classes']]}"
              f"{' model=' + str(models) if models else ''}")
    print(f"  -> {result['json']}")
    print(f"  -> {result['markdown']}")


def step2_traditional(project_root):
    banner(2, "Traditional finite-difference solver (numerical reference)")
    out_dir = ensure_dir(nems_dir(project_root) / "traditional")
    coarse = traditional_solver.solve_poisson_2d(n=31, freq=FREQ)
    medium = traditional_solver.solve_poisson_2d(n=63, freq=FREQ)
    fine = traditional_solver.solve_poisson_2d(n=127, freq=FREQ)
    report = {
        "problem": {"pde": "-Delta u = f on [-1,1]^2, u=0 on boundary",
                    "exact": f"u*(x)=sin({FREQ:.4g}*x1) sin({FREQ:.4g}*x2)",
                    "source": f"f=2*({FREQ:.4g})^2*u*"},
        "primary": fine,
        "convergence": [
            {"n": coarse["n"], "h": coarse["h"], "l2_rel_error": coarse["l2_rel_error"]},
            {"n": medium["n"], "h": medium["h"], "l2_rel_error": medium["l2_rel_error"]},
            {"n": fine["n"], "h": fine["h"], "l2_rel_error": fine["l2_rel_error"]},
        ],
        "observed_order": traditional_solver.observed_order(
            medium["l2_rel_error"], coarse["l2_rel_error"], 2),
        "observed_order_fine": traditional_solver.observed_order(
            fine["l2_rel_error"], medium["l2_rel_error"], 2),
    }
    save_json(out_dir / "poisson_fd.json", report)
    print(f"  PDE           : {report['problem']['pde']}")
    print(f"  exact         : {report['problem']['exact']}")
    print(f"  n={fine['n']:>3}  h={fine['h']:.4f}  L2_rel_err={fine['l2_rel_error']:.3e}  "
          f"CG_iters={fine['iterations']}")
    print(f"  observed order: {report['observed_order']:.2f} (n=31->63), "
          f"{report['observed_order_fine']:.2f} (n=63->127)")
    print(f"  -> {out_dir / 'poisson_fd.json'}")
    return report


def step3_baseline(project_root, runner):
    banner(3, "PINN baseline (default config)")
    record = runner.run("baseline", BASE_CONFIG, early_stop=False, max_wall_clock=120)
    baseline = {
        "exp_id": "baseline",
        "status": record["status"],
        "source": "PINN main.py with default config",
        "metrics": record["metrics"],
        "series": record["series"],
    }
    nd = ensure_dir(nems_dir(project_root) / "baseline")
    save_json(nd / "baseline.json", baseline)
    shutil.copyfile(Path(record["paths"]["loss_time"]), nd / "loss_time.jsonl")
    report_generator.write_baseline_report(project_root, baseline)
    metrics = baseline["metrics"]
    print(f"  status          : {record['status']}")
    print(f"  pde loss        : {metrics.get('final_loss')}")
    print(f"  L2 error vs u*  : {metrics.get('final_l2_error')}")
    print(f"  total wall-clock: {metrics.get('total_wall_clock'):.3f}s")
    print("  -> wrote .nems/baseline/ and BASELINE_REPORT.md")
    return baseline


def run_and_report(project_root, runner, analyzer, exp_id, config, baseline, title):
    banner(4 if exp_id == "exp_lr1e-2" else 5, title)
    print(f"  config: {config}")
    history = history_manager.load_history(project_root)
    record = runner.run(exp_id, config, baseline_series=baseline["series"], early_stop=True,
                        max_wall_clock=120)
    analysis = result_analyzer.analyze(record, baseline, history=history, analyzer=analyzer)
    report_generator.write_experiment_report(project_root, record, analysis)
    history_manager.add_experiment(project_root, record, analysis)
    print(f"  status         : {record['status']}  (early stopped: {record['early_stopped']})")
    print(f"  pde loss       : {record['metrics'].get('final_loss')}")
    print(f"  L2 error vs u* : {record['metrics'].get('final_l2_error')}")
    print(f"  verdict        : {analysis.get('verdict')}")
    print(f"  -> wrote .nems/experiments/{exp_id}/REPORT.md")
    return record


PARTICLE_CONFIG = {"FREQ": FREQ, "HIDDEN": 40, "DEPTH": 3, "ACTIVATION": "Tanh_Sin",
                   "LR": 1e-3, "EPOCHS": 300, "N_CENTER": 256, "RADIUS": 1e-3,
                   "N_MESH": 9, "N_TEST": 31}


def step_particlewnn(project_root):
    banner(6, "ParticleWNN weak-form solver (direct run)")
    out_dir = ensure_dir(nems_dir(project_root) / "particlewnn")
    save_json(out_dir / "current_config.json", {"params": PARTICLE_CONFIG})
    env = dict(os.environ)
    env["NEMS_OUTPUT_DIR"] = str(out_dir)
    env["NEMS_SEED"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, "particle_wnn.py"], cwd=str(project_root), env=env,
        capture_output=True, text=True, timeout=300,
    )
    print("  " + proc.stdout.strip().replace("\n", "\n  "))
    if proc.returncode != 0:
        raise RuntimeError("particle_wnn.py failed:\n" + proc.stderr)
    metrics = load_json(out_dir / "metrics.json", default={}) or {}
    print(f"  L2 error vs u* : {metrics.get('final_l2_error')}")
    print(f"  -> {out_dir / 'metrics.json'}")
    return metrics


def step_reference_note(project_root, fd_report, records, particle_metrics=None):
    banner(7, "Traditional / PINN / ParticleWNN reference note")
    lines = [
        "# Traditional vs learned solvers: 2D Poisson",
        "",
        "PDE: `-Delta u = f` on `[-1,1]^2`, `u = 0` on the boundary,",
        f"`u*(x) = sin({FREQ:.4g}*x1) sin({FREQ:.4g}*x2)`, "
        f"`f = 2*({FREQ:.4g})^2 u*`.",
        "",
        "## Traditional finite-difference reference",
        "",
        "| n | h | L2 rel. error | CG iterations |",
        "| --- | --- | --- | --- |",
    ]
    for row in fd_report["convergence"]:
        lines.append(f"| {row['n']} | {row['h']:.4f} | {row['l2_rel_error']:.3e} | - |")
    lines += [
        "",
        f"Observed convergence order: {fd_report['observed_order']:.2f} "
        f"(n=31->63), {fd_report['observed_order_fine']:.2f} (n=63->127); "
        "the 5-point scheme is second order.",
        "",
        "## PINN experiments (strong form, managed)",
        "",
        "| exp_id | status | pde loss | L2 rel. error | verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        metrics = record.get("metrics", {}) or {}
        loss = metrics.get("final_loss")
        l2 = metrics.get("final_l2_error")
        lines.append(
            f"| {record['exp_id']} | {record['status']} | "
            f"{_num(loss)} | {_num(l2)} | {record.get('_verdict', '')} |"
        )
    if particle_metrics:
        lines += [
            "",
            "## ParticleWNN (weak form, direct run)",
            "",
            "| solver | final loss | L2 rel. error | wall-clock (s) |",
            "| --- | --- | --- | --- |",
            f"| ParticleWNN | {_num(particle_metrics.get('final_loss'))} | "
            f"{_num(particle_metrics.get('final_l2_error'))} | "
            f"{_num(particle_metrics.get('total_wall_clock'))} |",
        ]
    lines += [
        "",
        "The finite-difference solver solves the same PDE to second order and",
        "provides a reference accuracy the learned solutions can be checked",
        "against at a fixed wall-clock budget.",
        "",
    ]
    path = nems_dir(project_root) / "traditional" / "TRADITIONAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Poisson (PINN + finite difference) demo")
    parser.add_argument("--workspace", default=str(DEMO_DIR / ".demo_workspace"))
    args = parser.parse_args()

    project_root = prepare_workspace(Path(args.workspace))
    print(f"demo project root: {project_root}")

    analyzer = llm_analyzer.LLMAnalyzer(prompts_dir=SKILL_ROOT / "prompts")
    runner = experiment_runner.ExperimentRunner(project_root, entry="main.py", timeout=180,
                                                monitor_interval=0.2)

    step1_model_structure(project_root)
    fd_report = step2_traditional(project_root)
    baseline = step3_baseline(project_root, runner)

    records = []
    good = run_and_report(project_root, runner, analyzer, "exp_lr1e-2", GOOD_CONFIG, baseline,
                          "Good experiment (LR=1e-2)")
    good["_verdict"] = "see REPORT.md"
    records.append(good)
    bad = run_and_report(project_root, runner, analyzer, "exp_lr1.0", BAD_CONFIG, baseline,
                         "Bad experiment (LR=1.0, early stop)")
    records.append(bad)
    if not bad["early_stopped"]:
        raise RuntimeError("early stop did NOT trigger for LR=1.0; the demo needs review")

    particle_metrics = step_particlewnn(project_root)
    step_reference_note(project_root, fd_report, records, particle_metrics)

    banner(8, "Overall REPORT.md, INDEX.md and figure")
    history = history_manager.load_history(project_root)
    print(f"  -> {report_generator.write_overall_report(project_root, baseline, history)}")
    print(f"  -> {report_generator.write_index(project_root, history, baseline)}")
    print(f"  -> {report_generator.write_metrics_evolution(project_root, baseline, history)}")

    print()
    print("DEMO COMPLETE: Poisson (PINN + ParticleWNN + finite difference) ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
