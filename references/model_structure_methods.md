# Model-Structure Snapshot Methods

The manager records what a model *is* without copying its function bodies. Two
layers are available; the AST layer always runs, the dynamic layer is opt-in.

## Layer 1 — AST skeleton (standard library only)

`agent.model_structure.extract_model_structure` parses every collected `.py` with
`ast` and keeps only structural facts:

- imports;
- module-level constants (`NAME = <literal>`);
- classes: bases, class attributes, method **signatures**, and
  `self.<name> = SomeModule(...)` layer assignments;
- top-level function signatures plus the dotted calls they make.

Function bodies, comments, and string bodies are **not** copied. A `SyntaxError`
in one file is recorded in `parse_errors` and does not abort the scan.

Why this shape:

- no third-party dependency, so it works in any environment;
- no execution, so it works on partially generated or injected sources;
- it is a card an LLM or a human can read quickly, and it is small on disk.

### Detected layer modules

`LAYER_CALL_HINTS` covers the common architecture blocks: `Linear`, `Conv*`,
`BatchNorm*`, `LayerNorm`, `GroupNorm`, `Dropout`, `Embedding`, `LSTM`, `GRU`,
`RNN`, `MultiheadAttention`, `TransformerEncoderLayer`, `Sequential`,
`ModuleList`, `ModuleDict`, `ParameterList`, `ParameterDict`.

A class is treated as a model class when it inherits a module-like base, defines
`forward`, or contains layer assignments.

## Layer 2 — dynamic graph (torch.fx, opt-in)

`agent.model_structure.augment_with_torch_fx(structure, model, example_inputs)`
traces a real model instance and merges an operator graph:

- `framework`, `total_parameters`;
- `nodes`: `{name, op, target, args, meta.shape}`;
- tensor shapes when `torch.fx.passes.shape_prop.ShapeProp` succeeds.

Tracing is **best-effort**: any failure is stored under `dynamic_graph_error`
and the AST skeleton remains valid. Use it only when PyTorch is importable, the
model can be instantiated, and an example input is available. Data-dependent
control flow or unsupported ops may prevent tracing; that is not a project
error.

## Artifacts

- `.nems/model_structure.json` — full machine-readable snapshot;
- `.nems/model_structure.md` — the same structure as a readable tree.

## Incremental updates

`diff_model_structures(old, new)` compares two snapshots and returns
`changes` (layer-count changes, added/removed model classes, parameter-count
changes) plus `changed_files` from the source-hash manifest. Use it after a
model edit so only the changed structure is re-described, keeping the pattern
"untouched parts unchanged (see `<exp_id>`)".

## Scope limits

- The AST layer reads static structure; layers built in data-dependent loops or
  via `getattr`/`setattr` may not appear.
- Shapes are only present in the dynamic layer.
- The snapshot is descriptive evidence, not proof that the code runs.
