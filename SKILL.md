---
name: numerical-experiment-manager
description: >-
  Manage the full lifecycle of a Python numerical/ML experiment project with a
  coding agent: one-time LLM code scan, hyperparameter confirmation and
  injection, wall-clock accuracy tracking against a fixed baseline, early
  stopping with at most one automatic retrain, report generation, incremental
  re-summarization after a model update, and small-parameter auto-tuning. Use
  when a numerical, scientific-computing, or machine-learning project needs a
  reproducible baseline, wall-clock experiment comparison, early stopping, or
  hyperparameter tuning driven by a coding agent (opencode, Codex, Claude Code).
  Do not use for Julia or MATLAB projects, for pure documentation tasks, or as a
  standalone command-line pipeline.
---

# AI Numerical Experiment Manager

## Scope

Turn a coding agent into a reproducible experiment manager for a Python project.
The agent scans the project once, keeps a durable structured summary, injects
confirmed hyperparameters as a parameter index, runs wall-clock-aligned
experiments against a fixed baseline, stops clearly bad runs early, writes
reports, and micro-tunes small parameters.

The skill is **agent-native**. The coding agent performs the interactive steps
(confirming hyperparameters, approving experiments, approving tuning) in
conversation. The Python modules under `agent/` are tools the agent calls to do
deterministic work: backup, injection, subprocess execution, wall-clock
alignment, reporting, and history maintenance. They are not a standalone CLI.

Two capabilities are built for projects that do not fit the narrow default path:

- a **model-structure snapshot** (`agent.model_structure`): an AST skeleton with
  optional `torch.fx` augmentation saves the whole model structure to
  `.nems/model_structure.json`/`.md` without copying function bodies;
- **loss-stream adapters** (`agent.loss_stream`): when a program does not write
  `loss_time.jsonl`, an adapter reads its run-time output (stdout lines, a
  sidecar CSV/JSON, or only the final metric) and feeds the same aligned
  comparison. See `references/loss_stream_adapters.md`.

Treat every recorded comparison as evidence. At equal wall-clock time, compare
the **best-so-far** error, preserve the raw `loss_time.jsonl`, and never replace
a measurement with a plausible-looking summary.

Route elsewhere when:

- the project is not Python (Julia and MATLAB are out of scope);
- the program produces no usable per-run output at all (no `loss_time.jsonl`,
  no parseable stdout, no sidecar file);
- the task is documentation or packaging only, with no experiment to run.

## Workflow

### 1. First scan (Phase 0)

Run `agent.code_reader.CodeReader` to collect project `.py` files (main-first
priority) and compute hashes. Read `prompts/structure_analysis.md`, fill
`{{CODE_CONTENT}}`, and produce `.nems/project_summary.json` plus
`.nems/hyperparameters.json`. `agent.llm_analyzer.LLMAnalyzer` can call an
OpenAI-compatible endpoint, or run the deterministic offline fallback when no
key is set. Do not edit source files during the scan.

Also call `agent.model_structure.write_model_structure` to save a body-free
structural card to `.nems/model_structure.json` and `.nems/model_structure.md`.
Optionally augment it with `augment_with_torch_fx(structure, model, example_inputs)`
when PyTorch is available and the model can be instantiated; this is best-effort
and never required. See `references/model_structure_methods.md`.

### 2. Confirm in conversation

Show candidate hyperparameters and the data-flow graph. Ask the user to
keep/exclude parameters by number. Save only what the user confirmed to
`.nems/hyperparameters_confirmed.json` and `.nems/data_flow_confirmed.json`.
Never inject an unconfirmed parameter.

### 3. Inject (Phase 1)

Call `agent.hyperparameter_injector.inject_hyperparameters`. It backs up every
edited source to `.nems/code_snapshots/<stamp>/`, replaces only the selected
parameters with `get_param("NAME", default=...)`, and generates the parameter
index module at the project root. Only selected parameters change; excluded
parameters keep their original literals. Source edits require approval.

### 4. Baseline (Phase 2)

First choose how the run's loss curve is read. Default to the `jsonl`
instrumentation contract; when the program only prints progress or writes a
sidecar file, build a `loss_stream` adapter and save its spec with
`agent.loss_stream.save_stream_spec` (or pass `stream_spec=` to each run). Confirm
the source with the user and record whether the wall-clock axis is synthesized.

Ask the user for the baseline source. Run it once with
`agent.experiment_runner.ExperimentRunner.run(..., early_stop=False)` and store
the wall-clock loss-time curve under `.nems/baseline/`. Compute the baseline
once and reuse it; do not silently recompute it per experiment.

### 5. Experiment and early stop (Phase 3-4)

Write the experiment config, run it, and record `loss_time.jsonl`, logs, and
`metrics.json`. At each integer second compare the experiment's best-so-far RMSE
with the baseline's. Stop early when the relative ratio exceeds the threshold
(default `0.20`, i.e. `ratio > 1.20`) after a warm-up, when the error becomes
non-finite, or when a catastrophic ratio (`> 2.0`) appears. A speed guard skips
the ratio rule when the experiment is far behind the baseline's epoch progress.

When a run stops because its relative error is too large, retrain it
automatically **at most once**: a fresh process with a shifted `NEMS_SEED` and
the same (or a `retrain_config`-overridden) config. Preserve attempt 1 artifacts
under `experiments/<exp_id>/attempts/`. Record the retrain outcome in
`result.json` under `retrain`. Set `max_retrains=0` to disable this behavior.

### 6. Report (Phase 5)

Call `agent.result_analyzer.analyze` to compare the experiment, baseline, and
history at equal wall-clock, then ask the LLM for likely causes. Write the
per-experiment `REPORT.md` with `agent.report_generator`. Report the numbers
that were measured; do not claim an improvement that the aligned comparison does
not support.

### 7. Incremental summary (Phase 6)

After the user edits the model, hash the files and re-read only the changed ones.
Write `.nems/project_summary_diff.md` and label untouched parts
`unchanged (see <exp_id>)`. Do not re-scan the whole project.

### 8. Micro-tune (Phase 7)

Call `agent.param_tuner.run_tuning`. Adjust at most 3 small parameters, at most 2
candidate values each, with short runs against the baseline. Mark each result
`建议保留` (keep) or `建议回退` (revert). Do not tune large architecture changes
through this path; demonstrate those through an explicit source edit.

### 9. Maintain history (Phase 8)

Update `.nems/history.json`, `.nems/INDEX.md`, and
`.nems/figures/metrics_evolution.png` with
`agent.history_manager` and `agent.report_generator`.

## Artifact contract

All durable state is written under `.nems/` at the managed project root. JSON
stores data; Markdown stores reports.

```text
project_root/
├── <adapter module>.py               # generated parameter index (name-adaptive)
├── .nems/
│   ├── project_summary.json          # LLM functional summary
│   ├── project_summary_diff.md       # incremental diff after a model update
│   ├── model_structure.json          # AST skeleton (+ optional torch.fx graph)
│   ├── model_structure.md            # human-readable model structure
│   ├── stream_adapter.json           # loss-stream source spec (default jsonl)
│   ├── hyperparameters.json          # LLM-identified candidate hyperparameters
│   ├── hyperparameters_confirmed.json# user-confirmed hyperparameters
│   ├── hyperparameter_history.json   # structured injection history
│   ├── hyperparameter_history.md     # human-readable injection history
│   ├── data_flow_confirmed.json      # user-confirmed data-flow graph
│   ├── source_hashes.json            # file hash manifest for incremental reads
│   ├── ENVIRONMENT_SETUP.md          # environment guidance
│   ├── PROJECT_OVERVIEW.md           # human-readable project overview
│   ├── history.json                  # experiment history
│   ├── INDEX.md                      # experiment index
│   ├── current_config.json           # current experiment config (link/copy)
│   ├── code_snapshots/<stamp>/       # original sources + diff.md before edits
│   ├── baseline/                     # baseline loss-time + metrics + report
│   ├── experiments/<exp_id>/         # one directory per experiment
│   └── figures/                      # metrics_evolution.png and other figures
```

The managed program must read `NEMS_OUTPUT_DIR` and write `loss_time.jsonl` and
`metrics.json`; it should honor `NEMS_MAX_WALL_CLOCK`, `NEMS_SEED`, and
`NEMS_RETRAIN_ATTEMPT`. When it cannot write `loss_time.jsonl`, a `loss_stream`
adapter reads its output instead. The full schema is in `README.md`.

## Approval rules

- Show the config and plan before every experiment; get user approval first.
- Source edits require approval; always back up before editing.
- Compute the baseline once and reuse it.
- Keep default runs small and CPU-friendly (a single run under 5 minutes).
- The early-stop threshold defaults to `0.20` (current error / baseline error).
- A too-large relative error triggers at most one automatic retrain
  (`max_retrains=1`).
- Small-parameter tuning defaults to at most 3 parameters and at most 2 values
  each.

## Bundled resources

- `agent/code_reader.py`: collect `.py` files (main-first priority) and hashes.
- `agent/model_structure.py`: AST model skeleton plus optional `torch.fx`
  augmentation; writes `.nems/model_structure.json`/`.md` and structural diffs.
- `agent/loss_stream.py`: pluggable loss sources (jsonl / stdout_regex / sidecar /
  completion_only / custom) that normalize a program's output to loss points.
- `agent/llm_analyzer.py`: structure, diff, and report JSON (remote or offline).
- `agent/hyperparameter_injector.py`: back up source, rewrite selected params to
  `get_param(...)`, generate the adapter module.
- `agent/experiment_runner.py`: run an experiment subprocess, read the loss stream
  through a `LossSource`, record wall-clock loss, drive early stop, retrain once.
- `agent/early_stopper.py`: per-second alignment, `0.20` relative-error decision,
  `trigger_kind` classification.
- `agent/result_analyzer.py`: compare experiment vs baseline vs history at equal
  wall-clock.
- `agent/param_tuner.py`: choose 1-3 small parameters, mark keep/revert.
- `agent/report_generator.py`: write `REPORT.md`, `INDEX.md`,
  `BASELINE_REPORT.md`, overview/environment docs, and figures.
- `agent/history_manager.py`: maintain `.nems/history.json` and the index.
- `scripts/check_environment.py`: dependency-free JSON probe of required and
  optional packages.
- `scripts/analyze_project.py`: dependency-free, offline JSON project scan
  (structure + hyperparameters), the small command-line entry point.
- `prompts/`: LLM prompt templates the agent fills.
- `references/early_stop_strategies.md`: the stopping and retrain rules.
- `references/hyperparameter_patterns.md`: hyperparameter-scanning patterns.
- `references/model_structure_methods.md`: the AST/torch.fx snapshot methods.
- `references/loss_stream_adapters.md`: loss-source selection and extension.
- `examples/demo_mlp_sin/`: self-contained CPU demo of the full lifecycle.
- `examples/demo_no_stream/`: CPU demo of a project with no `loss_time.jsonl`
  (reads printed output via the `stdout_regex` source).
- `examples/demo_poisson/`: 2D Poisson problem with a PINN (`main.py`), a
  weak-form ParticleWNN (`particle_wnn.py`), and a traditional 5-point
  finite-difference solver (`traditional_solver.py`); see its `PROBLEM.md`.
- `tests/`: `unittest` suite (no pytest required).
