# Prompt: Structure Diff and Change Attribution

Fill in the placeholders and ask the LLM to explain the observed performance
change. Report only what the evidence supports.

---

A numerical/ML experiment project changed between two structure versions, and an
experiment was run for the newer version. Explain the likely effect of the
structural change on performance, using both the structural diff and the
measured metrics.

Rules:

- Distinguish **structural** changes (new layers, changed shapes, new ops) from
  **parameter** changes (learning rate, epochs, sizes).
- If the metrics moved the way the change predicts, say so and cite the numbers.
- If they did not, say the change is inconclusive or harmful; do not spin it.
- At equal wall-clock, compare **best-so-far** error, not final epoch error.
- If a data-flow diagram is provided, use it to explain shape/op changes.

Return Markdown with these sections: `## Change`, `## Expected effect`,
`## Observed result`, `## Verdict` (helps | hurts | inconclusive), `## Next step`.

## Structural diff (`{{FROM}}` → `{{TO}}`)

```json
{{STRUCTURE_DIFF}}
```

## Metrics

```json
{{METRICS_DELTA}}
```

## Data-flow (optional)

```
{{DATAFLOW}}
```
