<div align="center">

# Numerical Experiment Manager

Reproducible wall-clock management of Python numerical and machine-learning
experiments: one-time scan, confirmed hyperparameter injection, fixed-baseline
comparison, early stopping with one retrain, reports, and micro-tuning.

[中文说明](README.zh-CN.md) · [Contributors](CONTRIBUTORS.md) · [Capabilities](#capabilities) · [Installation](#installation) · [Quick start](#quick-start) · [License and scope](#license-and-scope)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![modules](https://img.shields.io/badge/modules-11-2ea44f)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

<p align="center">
  If this project helps your work, please consider giving the repository a Star ⭐
  <a href="https://github.com/zanew314/numerical-experiment-manager"><img alt="GitHub stars" src="https://img.shields.io/github/stars/zanew314/numerical-experiment-manager?style=social"></a>
</p>

## What This Repository Is

This repository is the home of the Numerical Experiment Manager, a single
agent-native skill package for experimental Python projects. It turns a coding
agent into a reproducible experiment manager: the agent scans the project once,
injects confirmed hyperparameters, runs wall-clock-aligned experiments against a
fixed baseline, stops clearly bad runs early, writes reports, and micro-tunes
small parameters.

The skill is **agent-native**. The coding agent performs the interactive steps
(confirming hyperparameters, approving experiments, approving tuning) in
conversation. The Python modules under `agent/` are deterministic tools the agent
calls to do backup, injection, subprocess execution, wall-clock alignment,
reporting, and history maintenance. They are not a standalone CLI.

Treat every recorded comparison as evidence. At equal wall-clock time, compare
the **best-so-far** error, preserve the raw `loss_time.jsonl`, and never replace
a measurement with a plausible-looking summary.

Route elsewhere when the project is not Python (Julia and MATLAB are out of
scope), when the program produces no usable per-run output at all (no
`loss_time.jsonl`, no parseable stdout, no sidecar file), or when the task is
documentation or packaging only with no experiment to run.

## Capabilities

| Capability | Use it for | Entry point |
| --- | --- | --- |
| Project scan | One-time LLM structure summary and candidate hyperparameters. | `agent/code_reader.py`, `agent/llm_analyzer.py` |
| Model structure | A body-free AST skeleton plus optional `torch.fx` graph. | `agent/model_structure.py` |
| Injection | Back up source, confirm parameters, and inject `get_param(...)`. | `agent/hyperparameter_injector.py` |
| Loss streams | Read `jsonl`, `stdout_regex`, `sidecar`, `completion_only`, or custom sources. | `agent/loss_stream.py` |
| Baseline and runner | Wall-clock-aligned runs against a fixed baseline, with one retrain. | `agent/experiment_runner.py` |
| Early stopping | Per-second best-so-far comparison and trigger classification. | `agent/early_stopper.py` |
| Analysis | Compare experiment, baseline, and history at equal wall-clock. | `agent/result_analyzer.py` |
| Reports | Write `REPORT.md`, `INDEX.md`, `BASELINE_REPORT.md`, and figures. | `agent/report_generator.py` |
| Micro-tuning | Adjust at most 3 small parameters with keep/revert verdicts. | `agent/param_tuner.py` |
| History | Maintain `.nems/history.json` and the experiment index. | `agent/history_manager.py` |

Bundled resources:

| Path | What it is |
| --- | --- |
| [`SKILL.md`](SKILL.md) | The agent-facing workflow and approval rules. |
| `prompts/` | LLM prompt templates the agent fills. |
| `references/` | Early-stop, hyperparameter, model-structure, and loss-stream notes. |
| `scripts/` | Dependency-light JSON command-line checks. |
| `examples/` | Three self-contained CPU demos. |
| `tests/` | A `unittest` suite that does not require pytest. |

## Installation

The recommended path is AI-assisted installation: ask your coding agent to clone
or update this repository, read the Skill instructions, install the entrypoint,
and verify discovery.

```text
Please install the Numerical Experiment Manager Skill for me.

Repository: https://github.com/zanew314/numerical-experiment-manager.git
Branch: main
Skill path:
- . (the repository root contains SKILL.md)

Steps:
1. Clone or update the repository locally.
2. Read README.md, SKILL.md, and AGENTS.md if present.
3. If this environment supports local Skill discovery, link the directory that contains SKILL.md into the local skills directory.
4. Keep shared sibling support directories in place when the Skill depends on them.
5. Verify that the installed Skill is discoverable.
6. Tell me the installed path, whether a restart is needed, and give me one test prompt.
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
New-Item -ItemType Junction -Path "$env:USERPROFILE\.config\opencode\skill\numerical-experiment-manager" -Target "E:\path\to\numerical-experiment-manager"
```

If your agent uses a different local Skill directory, replace the path above with
that configured path. To register the package through `opencode.json` instead,
add its directory to `skills.paths`:

```json
{
  "skills": {
    "paths": ["skills"]
  }
}
```

Create an isolated environment and install the test extras before running the
demos:

```bash
python3 -m venv .venv-nems
source .venv-nems/bin/activate
python -m pip install -r requirements-test.txt
```

Obtain approval before installing into an existing environment.

## Quick Start

Clone the repository and run the self-contained, CPU-only demonstration (well
under three minutes):

```bash
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager
python examples/demo_mlp_sin/run_demo.py
```

It copies `main.py` into a sandbox at `examples/demo_mlp_sin/.demo_workspace` and
walks Phase 0 through Phase 8 while printing where each artifact is written.
Optional flags:

```bash
python examples/demo_mlp_sin/run_demo.py --interactive   # prompt for keep/exclude
python examples/demo_mlp_sin/run_demo.py --with-tuning   # also run Phase 7 tuning
python examples/demo_mlp_sin/run_demo.py --workspace D:\tmp\nems_demo
```

Expected outcome:

- `exp_001` (LR=0.01) completes and is **improved** against the baseline;
- `exp_002_earlystop` (LR=10.0) is **early stopped** after its single retrain also
  fails;
- the model update re-reads only the changed `main.py` and writes
  `project_summary_diff.md` with `unchanged (see exp_001)` for untouched fields.

A second, shorter demo manages a project that has **no** `loss_time.jsonl` and
reads its printed progress instead:

```bash
python examples/demo_no_stream/run_demo.py
```

A scientific demo solves the 2D Poisson problem `-Delta u = f` (exact solution
`u* = sin(freq x1) sin(freq x2)`, `freq = pi`) three ways: a PINN in `main.py`, a
weak-form ParticleWNN in `particle_wnn.py`, and a classical 5-point
finite-difference solver in `traditional_solver.py`:

```bash
python examples/demo_poisson/run_demo.py
python examples/demo_poisson/traditional_solver.py --n 127
python examples/demo_poisson/particle_wnn.py
```

The finite-difference reference converges at second order (L2 relative error
`~2e-4` at `n=127`); the manager then runs PINN experiments against a fixed
baseline. See `examples/demo_poisson/PROBLEM.md`.

## Instrumentation Contract

The managed program reads `NEMS_OUTPUT_DIR` and writes:

- `loss_time.jsonl` — one JSON object per epoch:
  `{"wall_clock": <sec>, "loss": <float>, "epoch": <int>}`;
- `metrics.json` — `{"final_loss": ..., "final_error": ...,
  "total_wall_clock": ..., "epochs": ...}`.

The program should also honor:

- `NEMS_MAX_WALL_CLOCK` — a soft seconds limit for one run;
- `NEMS_SEED` (default `0`) and `NEMS_RETRAIN_ATTEMPT` (`0` on the first run,
  `1` on the retrain) so an automatic retrain can vary its initialization.

The agent's early-stopper converts each recorded `loss` to an RMSE-style error
with `sqrt`, so a managed program must log a non-negative loss. The generated
adapter module reads `.nems/current_config.json` and falls back to the source
defaults when the file is missing, so the project still runs standalone.

When a program does not write `loss_time.jsonl`, select a loss source in
`.nems/stream_adapter.json` (or pass `stream_spec=` to `ExperimentRunner.run`):

| type | reads | live |
| --- | --- | --- |
| `jsonl` (default) | `loss_time.jsonl` | yes |
| `stdout_regex` | the captured stdout / `logs/run.log` | yes |
| `sidecar` | a CSV / JSON / JSONL file the program writes | yes |
| `completion_only` | final `metrics.json` only | no |

Every source yields the same `{"wall_clock", "loss", "epoch"}` points, so the
early stopper and reports are unchanged. When a source has no `wall_clock`
group, the axis is synthesized as `index * time_step` and flagged as
`synthetic_time`. Register a custom source with
`loss_stream.register_source`, or point at one on disk with
`{"type": "custom", "module": ..., "class": ...}`. See
`references/loss_stream_adapters.md`.

## Runner Contract and Limits

- Accuracy is RMSE; at each integer wall-clock second the **best-so-far** error
  is used, so transient loss spikes do not create false regressions.
- A run shorter than one second still occupies second `1` (ceil rule).
- Early stop: `ratio > 1.20` after a 2-second warm-up, plus an immediate stop on
  a non-finite error or a catastrophic ratio (`> 2.0`).
- A speed guard skips the ratio rule when the experiment is far behind the
  baseline's epoch progress, so machine-speed jitter is not mistaken for a bad
  configuration.
- Retrain once: a run stopped for a too-large relative error is retrained once
  (fresh process, shifted `NEMS_SEED`, same or overridden config) before being
  recorded as early-stopped. Attempt 1 artifacts are kept under
  `experiments/<exp_id>/attempts/`; set `max_retrains=0` to disable.
- Original sources are snapshotted under `.nems/code_snapshots/<timestamp>/`
  before any edit; excluded hyperparameters keep their original literals.
- The adapter module name is adapted to avoid collisions:
  `nems_config.py` -> `nems_param_loader.py` -> `_nems_config.py`.
- Small-parameter tuning defaults to at most 3 parameters and at most 2 values
  each.

## Repository Layout

```text
numerical-experiment-manager/
├── README.md
├── README.zh-CN.md
├── SKILL.md
├── CONTRIBUTORS.md
├── LICENSE
├── pyproject.toml
├── requirements-test.txt
├── agent/
├── prompts/
├── references/
├── scripts/
├── examples/
│   ├── demo_mlp_sin/
│   ├── demo_no_stream/
│   └── demo_poisson/
└── tests/
```

Keep demos, generated run evidence, and `.nems/` state inside the project that
owns them; run outputs are gitignored and never committed.

## Validation

There is no root build step. From this package directory:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The end-to-end demo tests are opt-in and run the full lifecycle:

```powershell
# PowerShell
$env:NEMS_RUN_INTEGRATION="1"; python -m unittest tests.test_demo_pipeline -v
```

```bash
# bash
NEMS_RUN_INTEGRATION=1 python -m unittest tests.test_demo_pipeline -v
```

The `Validate skill` GitHub Actions workflow runs the dependency-free tests with
zero skips on Python 3.10 and 3.13, then the full suite with CPU PyTorch. Do not
treat skipped integration tests as release evidence; run the opt-in test before
publishing.

## License and Scope

This package is released under the MIT License; see `LICENSE`. Do not commit
solver licenses, API keys, private datasets, `.env` files, generated logs with
sensitive data, or local run outputs. Public examples should include source notes
for benchmark data and be safe to redistribute.

The demo's dependency on PyTorch does not vendor or relicense PyTorch code. The
2D Poisson example (`examples/demo_poisson/`) adapts the problem setup and the
ParticleWNN method from
[`yaohua32/Physics-Driven-Deep-Learning-for-PDEs`](https://github.com/yaohua32/Physics-Driven-Deep-Learning-for-PDEs)
(Apache-2.0). The upstream code was **not** copied: `main.py`, `particle_wnn.py`
and `traditional_solver.py` are independent re-implementations of the same
mathematics. Attribution is recorded in [`CONTRIBUTORS.md`](CONTRIBUTORS.md);
contributors must be named factually in public releases.
