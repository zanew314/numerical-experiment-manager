# Artifacts and Versioning

All durable state lives under `.nems/` at the managed project root. JSON stores
data; Markdown stores human-readable reports. `.nems/` is evidence, never source:
add it to `.gitignore`.

```text
project_root/
└── .nems/
    ├── structure/
    │   ├── index.json              # version ledger
    │   ├── v0001.json              # AST snapshot + summary + hash
    │   ├── v0001.md                # human-readable snapshot
    │   └── ...
    ├── baseline/
    │   ├── loss_time.jsonl         # fixed baseline wall-clock curve
    │   └── metrics.json
    ├── runs/
    │   └── <run_id>/
    │       ├── loss_time.jsonl     # this run's curve (raw evidence)
    │       ├── metrics.json
    │       ├── early_stop.json     # decision + trigger_kind, when stopped
    │       ├── run.log             # captured stdout/stderr
    │       └── REPORT.md
    ├── testset/
    │   ├── index.json              # selected cards/cases for workflow C
    │   ├── summary.json            # grouped aggregation of the cases
    │   └── <case_id>/
    │       ├── case.json           # materialized equation instance
    │       ├── loss_time.jsonl     # short per-case run curve
    │       ├── metrics.json
    │       └── result.json
    ├── reports/
    │   ├── v0001_summary.md
    │   ├── v0001_to_v0002_change.md
    │   └── diagnosis_<tag>.md      # workflow C model diagnosis
    └── figures/
        ├── dataflow_v0002.md       # Mermaid
        └── dataflow_v0002.dot      # Graphviz
```

## Version numbering

A **version** identifies the model/experiment structure, not a run. `scan_structure.py`
computes a SHA-256 over the normalized AST snapshots of the tracked files:

- the first scan creates `v0001`;
- re-scanning with the same hash does nothing;
- any change to the parsed structure creates `vNNNN+1` and records
  `created_at`, `parent`, `hash`, `files`, and the `summary` field in
  `structure/index.json`.

Format versions as zero-padded `v0001`, `v0002`, … so lexical sort matches
chronological order.

## `structure/index.json`

```json
{
  "schema_version": 1,
  "current": "v0002",
  "versions": [
    {
      "version": "v0001",
      "created_at": "2026-09-27T12:00:00Z",
      "parent": null,
      "hash": "<sha256>",
      "files": ["main.py", "model.py"],
      "summary": "short LLM summary or null"
    }
  ]
}
```

## `structure/vNNNN.json`

```json
{
  "schema_version": 1,
  "version": "v0002",
  "created_at": "...",
  "hash": "<sha256>",
  "files": ["main.py"],
  "modules": [
    {
      "path": "model.py",
      "imports": ["torch", "torch.nn"],
      "classes": [
        {
          "name": "MLP",
          "bases": ["nn.Module"],
          "methods": ["__init__", "forward"],
          "attributes": ["fc1", "fc2"]
        }
      ],
      "functions": [
        {"name": "train", "args": ["epochs", "lr"]}
      ],
      "constants": {"HIDDEN": 64}
    }
  ],
  "summary": null
}
```

Function bodies are never copied. The snapshot stores signatures, class bases,
imports, module-level constants, and `self.<attr> = ...` assignments.

## `runs/<run_id>/`

- `loss_time.jsonl` is the raw measurement and must not be rewritten.
- `metrics.json` holds the final numbers actually measured.
- `early_stop.json` exists only when the stopper fired; it records
  `{"stopped": true, "trigger_kind": ..., "at_second": ..., "ratio": ...}`.
- `REPORT.md` is generated from the fill-in prompt, never fabricated numbers.

Keep the run id short and sortable, e.g. `run_001`, `run_002`.

## `testset/` (workflow C)

- `testset/index.json` (optional) lists the selected cards and `cases` so
  `aggregate` can report any case that was planned but never ran.
- `testset/<case_id>/case.json` is the materialized equation instance; the field
  contract is in `references/testset.md`.
- `testset/<case_id>/loss_time.jsonl` and `metrics.json` are per-case evidence,
  identical in shape to a `runs/<run_id>/` run.
- `testset/summary.json` is the grouped reduction (by `group` and by each
  `difficulty` axis) written by `python scripts/testset.py aggregate`.
- `reports/diagnosis_<tag>.md` is generated from `prompts/model_diagnosis.md`.

`.nems/testset/` is generated evidence, never source, and is gitignored with the
rest of `.nems/`.
