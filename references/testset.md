# Equation Bank and Case Contract

Workflow C probes a managed model on a bank of test equations. The bank is a set
of Markdown cards under `references/test_equations/` (one file per family, see
`references/test_equations/INDEX.md`). The agent adapts each card to the model's
own problem-description interface and materializes it as a case under
`.nems/testset/`. This file defines the mechanics; the diagnosis prose lives in
`prompts/model_diagnosis.md`.

## Interface detection

Before generating anything, read the structure snapshot (`workflow A`) and the
model's source, and answer:

- What is the problem-description object? A `Problem` class, a config dict, a
  dataset loader, a `forward` signature, or CLI arguments?
- What does the model consume — scalar parameters, a time grid, a field, a
  coefficient field, initial/boundary data?
- What does it output, and in what shape?

Match that answer against each card's `input_format` and the *Input encoding*
line in its `## Expected Modeling Signals`. Only materialize cards whose format
fits. Record skipped cards as coverage gaps; do not force a card into a shape the
model cannot accept.

## `case.json` contract

Each case is a JSON object written to `.nems/testset/<case_id>/case.json`:

```json
{
  "case_id": "poisson-2d-smooth",
  "card": "poisson-2d",
  "group": "general",
  "difficulty": ["elliptic", "boundary_value"],
  "equation": "-delta u = f, u = 0 on the boundary",
  "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
  "initial_condition": null,
  "boundary_condition": "u = 0",
  "coefficients": {"f": "2*pi^2*sin(pi*x)*sin(pi*y)"},
  "reference": {"kind": "exact", "expression": "sin(pi*x)*sin(pi*y)", "path": null},
  "input": {"encoding": "grid coordinates + source field", "points": "N x 2 plus N"},
  "run": {"command": "python main.py --case .nems/testset/poisson-2d-smooth/case.json",
           "max_wall_clock": 60, "iterations": 200},
  "tolerance": 0.10
}
```

Required fields: `case_id`, `card`, `group`. `group` is `general` or `pde`.
`reference.kind` is `exact` or `numeric`; for `numeric`, store or reference the
precomputed reference next to the case and never substitute a plausible summary.
`run.max_wall_clock` must be positive and bounded (default 60 s).

Use `python scripts/testset.py scaffold <card_id> --out <case.json>` to start a
case from a card, then fill the placeholders from the model interface.

## Running cases

Run each case with `watch_experiment.py`, exactly as in workflow B, but with a
short budget:

```bash
python scripts/watch_experiment.py <project_root> --run-id <case_id> \
    --cmd "<the command derived from the model interface>" \
    --baseline <baseline for this case or the default> \
    --tolerance 0.10 --sustain 3 --max-wall-clock 60
```

Write the per-case artifacts under `.nems/testset/<case_id>/` (set
`NEMS_OUTPUT_DIR` there). Run at most 6 cases by default, and keep every case
tiny CPU-friendly. If the model honors a `NEMS_TESTCASE_FILE` environment
variable, pass the case path through it; otherwise pass the path through the
model's existing input mechanism. Source edits to add that hook require approval.

## Aggregation

After the runs, reduce the artifacts with:

```bash
python scripts/testset.py aggregate <project_root> --testset .nems/testset \
    --output .nems/testset/summary.json
```

`aggregate` reads each `<case_id>/result.json` (or `metrics.json`) plus the raw
`loss_time.jsonl`, computes the **best-so-far RMSE** per case, and groups the
results by `group` and by each `difficulty` axis. `best_error` is the best
(best-so-far) value; a low `best_error` means the model eventually fit that case
under the budget, not that it is accurate at equal wall-clock.

`.nems/testset/index.json` (optional) lists the selected `cases`; `aggregate`
reports any listed case that has no run directory under `missing_cases`.

## From diagnosis to improvement

The diagnosis report (from `prompts/model_diagnosis.md`) names the weak equation
classes and the structural hypotheses that explain them. Improvements are
**advisory only** in workflow C: do not edit the model source. If the user wants
a change applied, make it an explicit source edit under workflow A (creating a
new structure version), then re-run the affected cases under workflow B/C to see
whether the weakness actually moved.
