# torch.fx Data Flow

The AST snapshot from `scan_structure.py` says what the model *is* (classes,
layers, signatures) without executing it. When you need to see what the model
*does* — the actual operator sequence and tensor shapes for one forward pass — use
`torch.fx`.

## When to trace

Trace only when all of these hold:

- PyTorch is installed (`import torch` succeeds);
- the model class can be imported from a file without side effects;
- a small example input of the right shape can be constructed.

Otherwise skip tracing. The AST snapshot remains the authoritative structure.

## How

```bash
python scripts/fx_dataflow.py <model_file> \
    --entry <ModelClassOrFunction> \
    --input-shape 1,3,32,32 \
    --save .nems/figures
```

`--input-shape` is a comma-separated shape; a batch dimension is included as
written. For models that take several arguments, use `--input-shape` repeated (one
per argument). Outputs:

- `.nems/figures/dataflow_<name>.dot` — Graphviz;
- `.nems/figures/dataflow_<name>.md` — a Mermaid graph plus an operator table;
- stdout — a JSON summary `{"nodes": [...], "edges": [...], "ops": {...}}` that
  the agent can quote in the report.

If the model uses control flow, data-dependent shapes, or non-traceable ops,
`torch.fx.symbolic_trace` will fail; that is expected. Do not force it. Record the
failure in the report and fall back to the AST snapshot.

## Reading the result

- The Mermaid diagram is the "precise structure diagram": one node per traced
  operator, edges following the forward data flow.
- The operator table lists `op`, `target`, input shapes, and output shapes, which
  is what makes a shape-driven performance regression visible.
- Compare diagrams across versions in the change report; a new large operator or a
  changed tensor shape often explains a wall-clock shift.

## Limits

- Tracing runs the model symbolically; it does **not** train. Keep the example
  input tiny.
- Results are best-effort. A missing diagram is not a failure of the experiment,
  only of the visualization.
