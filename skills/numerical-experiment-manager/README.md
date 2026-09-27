<div align="center">

# Numerical Experiment Manager

A guidance-first skill that keeps a versioned structure record of a Python
numerical/ML experiment project and, on request, tests it: torch.fx data flow,
wall-clock experiments against a fixed baseline, early stopping, and micro-tuning.

[中文说明](README.zh-CN.md) · [SKILL](SKILL.md) · [Repository](https://github.com/zanew314/numerical-experiment-manager) · [References](references/) · [Scripts](scripts/)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![scripts](https://img.shields.io/badge/scripts-4-2ea44f)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

## What This Skill Is

This package teaches a coding agent to manage a numerical/ML experiment project
as a reproducible campaign. The **guidance lives in [`SKILL.md`](SKILL.md)**;
four small, dependency-free scripts do the deterministic bookkeeping. There is no
standalone pipeline — the agent drives every step in conversation.

| Workflow | Trigger | What happens |
| --- | --- | --- |
| A. Structure versioning | project first seen, or the model file changes | An AST snapshot is versioned (`v0001`, `v0002`, …); the author's structural edits become versions; after a run, the structural diff and metrics are handed to the LLM to explain the performance change. |
| B. Automated testing | the user says "帮我测试" / "run this" | The program runs, its data flow is optionally traced with `torch.fx`, a precise structure diagram and experiment report are produced, the run is early-stopped if its relative error stays above 0.10, and a few small parameters can be tuned. |

Evidence discipline: preserve the raw loss series, compare **best-so-far** at
equal wall-clock, and never replace a measurement with a plausible summary.

## Scripts

| Script | Purpose |
| --- | --- |
| [`scripts/scan_structure.py`](scripts/scan_structure.py) | AST scan → versioned `.nems/structure/` JSON + Markdown. |
| [`scripts/diff_structure.py`](scripts/diff_structure.py) | Structural diff between two versions → JSON for the LLM. |
| [`scripts/fx_dataflow.py`](scripts/fx_dataflow.py) | `torch.fx` trace → Mermaid/DOT data-flow diagram + shapes. |
| [`scripts/watch_experiment.py`](scripts/watch_experiment.py) | Run a program, record wall-clock loss, early-stop, retrain once, write run artifacts. |

## Quick Start

```bash
# 1. Version the project structure (first scan creates v0001).
python scripts/scan_structure.py /path/to/project

# 2. After the author edits the model, scan again (creates v0002) and diff.
python scripts/scan_structure.py /path/to/project
python scripts/diff_structure.py /path/to/project --from v0001 --to v0002

# 3. Trace the data flow (optional, needs PyTorch).
python scripts/fx_dataflow.py /path/to/project/model.py \
    --entry MLP --input-shape 1,2 --save .nems/figures

# 4. Record a baseline once, then run and monitor an experiment.
python scripts/watch_experiment.py /path/to/project --run-id baseline \
    --cmd "python main.py" --loss-source jsonl
python scripts/watch_experiment.py /path/to/project --run-id run_001 \
    --cmd "python main.py" --baseline .nems/baseline/loss_time.jsonl \
    --tolerance 0.10 --sustain 3
```

Then fill [`prompts/structure_diff.md`](prompts/structure_diff.md) and
[`prompts/experiment_report.md`](prompts/experiment_report.md) and let the agent
write the change report and `REPORT.md`.

## Artifacts

Everything durable is written under `.nems/` at the managed project root:

```text
.nems/
├── structure/   # index.json + vNNNN.json/md (versioned AST snapshots)
├── baseline/    # fixed wall-clock baseline curve
├── runs/<id>/   # loss_time.jsonl, metrics.json, early_stop.json, REPORT.md
├── reports/     # vNNNN_summary.md, vNNNN_to_vNNNN_change.md
└── figures/     # dataflow_*.md/dot
```

`.nems/` is evidence, never source. Add it to `.gitignore`. The full contract is
in [`references/artifacts.md`](references/artifacts.md).

## License and Scope

MIT (see the repository [`LICENSE`](../../LICENSE)). Do not commit generated run
outputs, private datasets, or secrets. This skill targets Python projects; Julia
and MATLAB are out of scope.
