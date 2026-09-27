<div align="center">

# Numerical Experiment Manager

A guidance-first skill for reproducible Python numerical/ML experiments: versioned
structure tracking, torch.fx data flow, wall-clock baseline comparison, early
stopping, and small-parameter tuning.

[中文说明](README.zh-CN.md) · [Contributors](CONTRIBUTORS.md) · [Workflows](#workflows) · [Scripts](#scripts) · [Installation](#installation) · [License and scope](#license-and-scope)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![scripts](https://img.shields.io/badge/scripts-4-2ea44f)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

<p align="center">
  If this project helps your work, please consider giving the repository a Star ⭐
  <a href="https://github.com/zanew314/numerical-experiment-manager"><img alt="GitHub stars" src="https://img.shields.io/github/stars/zanew314/numerical-experiment-manager?style=social"></a>
</p>

## What This Skill Is

This repository **is** the skill. It teaches a coding agent to manage a Python
numerical/ML experiment project as a reproducible campaign. The guidance lives in
[`SKILL.md`](SKILL.md); four small, dependency-free scripts do the deterministic
bookkeeping. There is no standalone pipeline — the agent drives every interactive
step in conversation.

## Workflows

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
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager

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

## Installation

The recommended path is AI-assisted installation: ask your coding agent to clone
the repository, read the Skill instructions, link the repository root into the
local skills directory, and verify discovery.

```text
Please install the Numerical Experiment Manager Skill for me.

Repository: https://github.com/zanew314/numerical-experiment-manager.git
Branch: main

Steps:
1. Clone or update the repository locally.
2. Read README.md and SKILL.md.
3. If this environment supports local Skill discovery, link the repository root (the directory that contains SKILL.md) into the local skills directory.
4. Verify that the installed Skill is discoverable.
5. Tell me the installed path, whether a restart is needed, and give me one test prompt.
```

Manual fallback for opencode and Codex-style local discovery:

```bash
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager

# opencode skill directory
mkdir -p ~/.config/opencode/skill
ln -s "$PWD" ~/.config/opencode/skill/numerical-experiment-manager

# Codex-style local discovery
mkdir -p ~/.codex/skills
ln -s "$PWD" ~/.codex/skills/numerical-experiment-manager
```

On Windows, create a junction instead of a symlink:

```powershell
New-Item -ItemType Junction `
  -Path "$env:USERPROFILE\.config\opencode\skill\numerical-experiment-manager" `
  -Target "E:\path\to\numerical-experiment-manager"
```

If your agent uses a different local Skill directory, replace the path above with
that configured path.

## Validation

There is no build step. The scripts use only the Python standard library
(`fx_dataflow.py` additionally needs PyTorch). Run the smoke test from the
repository root:

```bash
python .github/smoke_test.py
```

## License and Scope

MIT (see [`LICENSE`](LICENSE)). Do not commit generated run outputs, private
datasets, `.env` files, or secrets. `.nems/` is generated evidence and should be
gitignored in managed projects. This skill targets Python projects; Julia and
MATLAB are out of scope.
