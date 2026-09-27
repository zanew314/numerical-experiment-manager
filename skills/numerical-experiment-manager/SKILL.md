---
name: numerical-experiment-manager
description: >-
  Guidance-first skill for managing the lifecycle of a Python numerical/ML
  experiment project with a coding agent. Keeps a versioned structure JSON of the
  project and explains performance changes from structural diffs; on request it
  runs the program, traces its data flow with torch.fx, draws a precise structure
  diagram, produces an experiment report, early-stops a run whose relative error
  stays above 0.10 (a local-optimum signature), and micro-tunes small parameters.
  Use when a numerical, scientific-computing, or machine-learning project needs
  structure/version management, reproducible wall-clock experiments, early
  stopping against a baseline, or small-parameter tuning driven by a coding agent
  (opencode, Codex, Claude Code). Do not use for Julia or MATLAB projects, for
  pure documentation tasks, or as a standalone command-line pipeline.
---

# AI Numerical Experiment Manager

This skill tells a coding agent how to run a Python numerical/ML experiment
project as a managed, reproducible campaign. The agent does the thinking and the
interactive steps; a handful of small, dependency-free scripts under `scripts/`
do the deterministic bookkeeping. This document is the source of truth — prefer
it over any script's own help text.

Two workflows are in scope:

- **Structure versioning** — read a project once, keep a versioned structure JSON,
  and whenever the author edits the numerical-experiment model, assign a new
  version, diff it, and have the LLM explain what the change should do to
  performance.
- **Automated testing** — on the user's request, run the program, optionally trace
  its data flow with `torch.fx`, run experiments against a fixed baseline,
  early-stop a run that looks trapped in a local optimum, write a report, and
  micro-tune a few small parameters.

Everything durable is written under `.nems/` at the managed project root. Never
commit `.nems/` outputs; they are evidence, not source.

## When to use, when not to

Use it when the project is Python and can emit, at minimum, a per-step metric
(see *Instrumentation*), or can at least print parseable progress.

Do **not** use it for Julia or MATLAB projects, for pure documentation or
packaging tasks with no experiment to run, or when the program produces no
usable per-run output at all.

Treat every recorded measurement as evidence: preserve the raw loss series,
compare **best-so-far** values at equal wall-clock, and never replace a
measurement with a plausible-looking summary.

## Workflow

The agent decides which workflow to enter from the user's intent. Both workflows
share the `.nems/` layout in `references/artifacts.md`.

### A. Structure versioning (project management)

Trigger when the project is first seen, and again whenever the author changes the
model or the numerical-experiment file (they are trying to improve performance).

1. **Scan.** Run:

   ```bash
   python scripts/scan_structure.py <project_root>
   ```

   It parses the project's Python files with the standard-library `ast`, writes a
   body-free structural snapshot, and manages versions for you:
   - `.nems/structure/index.json` — the version ledger;
   - `.nems/structure/vNNNN.json` and `vNNNN.md` — one snapshot per version.

   A new version is created only when the content hash of the tracked files
   changes. Re-running on an unchanged project is a no-op. Use `--force` only
   when you deliberately want to re-snapshot.

2. **Summarize.** Read `prompts/structure_scan.md`, fill it with the snapshot, and
   ask the LLM (the coding agent itself) for a short JSON summary: what the model
   is, its data flow, its parameters, and the likely knobs for performance. Store
   it in the version's JSON under `summary` (edit the file) or in
   `.nems/reports/vNNNN_summary.md`.

3. **Attribute a change.** When a run finishes for a new version, produce the
   structural diff and ask for the cause:

   ```bash
   python scripts/diff_structure.py <project_root> --from v0001 --to v0002
   ```

   Then read `prompts/structure_diff.md`, fill it with that diff plus the metrics
   delta (baseline vs this run), and write
   `.nems/reports/v0001_to_v0002_change.md`. State plainly whether the change is
   expected to help, hurt, or is inconclusive, and why. Never claim an improvement
   the aligned metrics do not support.

### B. Automated testing and tuning

Trigger when the user says something like "帮我测试" / "run this". Confirm the
plan first (see *Approval rules*).

1. **Confirm the loss source.** Prefer the default contract in
   *Instrumentation*. If the program does not write the standard loss file, decide
   with the user how to read its output and pass that to `watch_experiment.py`
   (`jsonl`, `stdout_regex`, `sidecar`, or `completion_only`). See
   `references/early_stop_and_tuning.md`.

2. **Baseline.** Run the unmodified program once, store the wall-clock loss-time
   curve under `.nems/baseline/`. Compute it once and reuse it; do not silently
   recompute it per experiment.

3. **Data flow (optional, precise).** When PyTorch is available and the model can
   be instantiated from a small input, run:

   ```bash
   python scripts/fx_dataflow.py <model_file> --entry <ModelClass> --save .nems/figures
   ```

   It uses `torch.fx` to trace the forward pass and writes a Mermaid/DOT diagram
   plus operator and shape tables. This is best-effort: on any tracing failure,
   keep the AST snapshot from workflow A and move on.

4. **Run and monitor.** Run the experiment with `watch_experiment.py`:

   ```bash
   python scripts/watch_experiment.py <project_root> --run-id run_001 \
       --cmd "python main.py" --baseline .nems/baseline/loss_time.jsonl \
       --tolerance 0.10 --sustain 3
   ```

   It launches the program, records wall-clock loss, and stops it early (see
   *Early stop*) while writing `.nems/runs/<run_id>/`.

5. **Report.** Read `prompts/experiment_report.md`, fill it with the metrics,
   early-stop record, and the structural diff, and write
   `.nems/runs/<run_id>/REPORT.md`. Include the data-flow diagram when one exists.

6. **Micro-tune (optional).** If the user wants to "炼丹", adjust at most 3 small
   parameters with at most 2 candidate values each, using short runs against the
   baseline, and mark each result `keep` or `revert`. Do not push architecture
   changes through this path; demonstrate those as an explicit source edit plus a
   new structure version.

## Early stop

The stopper answers "at equal wall-clock, is this run clearly not worth the
compute?". It compares the experiment's best-so-far error with the baseline's at
each integer second.

- Default tolerance `0.10`: a run is a suspected local optimum when its relative
  error stays above `1 + 0.10` for `--sustain` consecutive seconds (default `3`)
  after a short warm-up. The user tunes this; `0.10` is the project default.
- Stop immediately on a non-finite error (`nan` / `inf`) or a catastrophic ratio
  (`> 2.0`).
- A speed guard skips the ratio rule when the run's epoch progress lags far behind
  the baseline, so machine jitter is not mistaken for a bad configuration.
- On a too-large relative error the runner may retrain **at most once** (fresh
  process, shifted seed) before recording the run as early-stopped.

Full rules and the `trigger_kind` table are in
`references/early_stop_and_tuning.md`.

## Instrumentation

The managed program reads `NEMS_OUTPUT_DIR` and should write:

- `loss_time.jsonl` — one JSON object per step:
  `{"wall_clock": <sec>, "loss": <float>, "epoch": <int>}`;
- `metrics.json` — `{"final_loss": ..., "final_error": ...,
  "total_wall_clock": ..., "epochs": ...}`.

It should also honor `NEMS_MAX_WALL_CLOCK`, `NEMS_SEED` (default `0`), and
`NEMS_RETRAIN_ATTEMPT` (`0` first run, `1` on the retrain). The stopper converts
each `loss` to an RMSE-style error with `sqrt`, so the program must log a
non-negative loss. When the program cannot write `loss_time.jsonl`, choose a loss
source instead.

## Approval rules

- Show the plan and config before every experiment; get approval first.
- Source edits require approval; back up the file before editing.
- Compute the baseline once and reuse it.
- Keep default runs small and CPU-friendly (one run under ~5 minutes).
- Early stop defaults to `tolerance = 0.10`, `sustain = 3` seconds.
- Retrain at most once.
- Micro-tune at most 3 parameters, at most 2 candidate values each.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/scan_structure.py` | AST scan of a project → versioned `.nems/structure/` JSON + Markdown. |
| `scripts/diff_structure.py` | Structural diff between two versions → JSON for the LLM. |
| `scripts/fx_dataflow.py` | Optional `torch.fx` trace → Mermaid/DOT data-flow diagram + tables. |
| `scripts/watch_experiment.py` | Run a program, record wall-clock loss, early-stop, write run artifacts. |

## References

- `references/artifacts.md` — the `.nems/` contract and versioning scheme.
- `references/early_stop_and_tuning.md` — stop rules, retrain, and tuning.
- `references/torch_fx_dataflow.md` — when and how to trace data flow.
- `prompts/` — fill-in prompts for scan, diff, and report.
