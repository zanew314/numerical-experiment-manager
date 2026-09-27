# AI Numerical Experiment Manager Skill

Chinese guide: [README.zh-CN.md](README.zh-CN.md)

This AI4Math skill manages the full lifecycle of a Python numerical/ML
experiment project with a coding agent. The agent scans the project once, injects
confirmed hyperparameters, runs wall-clock-aligned experiments against a fixed
baseline, stops bad runs early, reports, and micro-tunes small parameters. It is
a release-ready local candidate for
[`VeryMath/AI4Math-Optimization`](https://github.com/VeryMath/AI4Math-Optimization);
it is not an official upstream package until that repository accepts and merges
it.

## Scope

Use the skill for Python projects that can record a per-epoch wall-clock loss
stream. The agent drives the workflow in conversation; the modules under
`agent/` are deterministic tools the agent calls, not a standalone CLI.

Supported:

- one-time LLM structure summary and incremental diff after a model update;
- a body-free model-structure snapshot (AST skeleton + optional torch.fx graph);
- hyperparameter confirmation, injection, and structured history;
- wall-clock baseline comparison at equal elapsed seconds;
- early stopping with at most one automatic retrain;
- loss-stream adapters for programs that do not write `loss_time.jsonl`;
- per-experiment and overall reports, `INDEX.md`, and evolution figures;
- small-parameter auto-tuning with keep/revert verdicts.

Do not route Julia, MATLAB, non-Python, or pure-documentation tasks to this
package. The managed program should emit `loss_time.jsonl`; when it cannot, use a
loss-stream adapter so the manager can still read its output (see below).

## Requirements and isolated installation

- Python 3.10+
- `numpy` for the agent tools; `torch` (CPU) and `matplotlib` for the demo
- no external LLM SDK: remote calls use `urllib`, and the analyzer falls back to
  a deterministic offline mode when no API key is set

Create an isolated environment and install the test extras:

```bash
python3 -m venv .venv-nems
source .venv-nems/bin/activate
python -m pip install -r requirements-test.txt
```

Obtain approval before installing into an existing environment.

## Quick start

Run the self-contained, CPU-only demonstration (well under three minutes) from
this package directory:

```bash
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
- `exp_002_earlystop` (LR=10.0) is **early stopped** and recorded as
  early-stopped after its single retrain also fails;
- the model update re-reads only the changed `main.py` and writes
  `project_summary_diff.md` with `unchanged (see exp_001)` for untouched fields.

The demo recreates its sandbox on each run, so re-running it overwrites
`.demo_workspace` (gitignored; no run outputs are committed).

A second, shorter demo manages a project that has **no** `loss_time.jsonl` and
reads its printed progress instead:

```bash
python examples/demo_no_stream/run_demo.py
```

It writes `.nems/model_structure.json`/`.md`, a `.nems/stream_adapter.json` spec
for the `stdout_regex` source, a baseline, one good and one early-stopped
experiment, and the usual reports.

A scientific demo solves the 2D Poisson problem `-Delta u = f` (exact solution
`u* = sin(freq x1) sin(freq x2)`, `freq = pi`) three ways: a PINN in `main.py`,
a weak-form ParticleWNN in `particle_wnn.py`, and a classical 5-point
finite-difference solver in `traditional_solver.py`:

```bash
python examples/demo_poisson/run_demo.py
python examples/demo_poisson/traditional_solver.py --n 127
python examples/demo_poisson/particle_wnn.py
```

The finite-difference reference converges at second order (L2 relative error
`~2e-4` at `n=127`); the manager then runs PINN experiments against a fixed
baseline and the demo also runs the ParticleWNN directly. See
`examples/demo_poisson/PROBLEM.md`.

## Instrumentation contract

The managed program reads `NEMS_OUTPUT_DIR` and writes:

- `loss_time.jsonl` - one JSON object per epoch:
  `{"wall_clock": <sec>, "loss": <float>, "epoch": <int>}`;
- `metrics.json` - `{"final_loss": ..., "final_error": ...,
  "total_wall_clock": ..., "epochs": ...}`.

The program should also honor:

- `NEMS_MAX_WALL_CLOCK` - a soft seconds limit for one run;
- `NEMS_SEED` (default `0`) and `NEMS_RETRAIN_ATTEMPT` (`0` on the first run,
  `1` on the retrain) so an automatic retrain can vary its initialization.

The agent's early-stopper converts each recorded `loss` to an RMSE-style error
with `sqrt`, so a managed program must log a non-negative loss. The generated
adapter module reads `.nems/current_config.json` and falls back to the source
defaults when the file is missing, so the project still runs standalone.

## Model-structure snapshot

`agent/model_structure.py` saves what a model *is* without copying function
bodies. It always writes a standard-library AST skeleton (imports, constants,
class bases, method signatures, and `self.<layer> = Module(...)` assignments) to
`.nems/model_structure.json` and `.nems/model_structure.md`. When PyTorch is
available and a model instance plus example inputs can be supplied,
`augment_with_torch_fx` adds an operator graph with shapes and parameter counts;
tracing is best-effort and never invalidates the skeleton. `diff_model_structures`
compares two snapshots for incremental summaries. See
`references/model_structure_methods.md`.

## Loss-stream adapters

When a program does not write `loss_time.jsonl`, select a source in
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
`synthetic_time`. Register a custom source with `loss_stream.register_source`, or
point at one on disk with `{"type": "custom", "module": ..., "class": ...}`. See
`references/loss_stream_adapters.md`.

## Runner contract and limits

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

### LLM configuration

The coding agent is the primary LLM. To let the Python tool call an
OpenAI-compatible endpoint instead, set:

```bash
NEMS_LLM_API_KEY=...
NEMS_LLM_BASE_URL=https://api.openai.com/v1   # optional
NEMS_LLM_MODEL=gpt-4o-mini                    # optional
```

Without a key, `LLMAnalyzer` uses the offline fallback and sets
`_source = "offline_fallback"`.

## Command-line entry points

Two deterministic, dependency-light scripts back the platform-style checks and
the agent workflow:

```bash
python scripts/check_environment.py        # JSON: required/optional packages
python scripts/analyze_project.py <dir>    # JSON: offline structure + hyperparameters
```

Both print a single JSON document to stdout. `analyze_project.py` exits `2` and
prints a JSON error to stderr on a bad input.

## Validation

From this package directory:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The end-to-end demo test is opt-in and runs the full lifecycle:

```bash
# PowerShell
$env:NEMS_RUN_INTEGRATION="1"; python -m unittest tests.test_demo_pipeline -v
# bash
NEMS_RUN_INTEGRATION=1 python -m unittest tests.test_demo_pipeline -v
```

Do not treat skipped integration tests as release evidence; run the opt-in test
before publishing.

## Registration

Add this package directory to `skills.paths` in `opencode.json`:

```json
{
  "skills": {
    "paths": ["skills"]
  }
}
```

## License and provenance

This package is intended to follow the AI4Math repository's MIT license after
maintainer authorization and upstream acceptance. The demo's dependency on
PyTorch does not vendor or relicense PyTorch code.

The 2D Poisson example (`examples/demo_poisson/`) adapts the problem setup and
the ParticleWNN method from
[`yaohua32/Physics-Driven-Deep-Learning-for-PDEs`](https://github.com/yaohua32/Physics-Driven-Deep-Learning-for-PDEs)
(Apache-2.0). The upstream code was **not** copied: `main.py`, `particle_wnn.py`
and `traditional_solver.py` are independent re-implementations of the same
mathematics.

Before public release, a human maintainer must confirm publication authorization
and record factual contributor attribution; packaging by an AI agent is not that
authorization.
