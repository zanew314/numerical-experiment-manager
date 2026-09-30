# Prompt: Model Diagnosis

Fill in the placeholders and ask the LLM to diagnose where the model is weak
across the test-equation bank. Report only what the evidence supports; proposed
improvements are advisory and must not be applied here.

---

Write `.nems/reports/diagnosis_<tag>.md` for one workflow C campaign. Use the
aggregated per-case numbers exactly as given. Where a case has no result, say so
instead of guessing.

Return Markdown with these sections:

- `## Detected interface` — what problem-description interface the model exposes
  and which equation `input_format`s it can accept.
- `## Coverage` — which cards were instantiated, which were skipped, and why.
- `## Per-family results` — one row per case: group, difficulty axes, best-so-far
  RMSE, stopped/trigger, and whether it beat the reference tolerance.
- `## Weak spots` — the equation families or difficulty axes where the model
  clearly underperforms its peers, with the aligned numbers that show it.
- `## Hypotheses` — the structural reasons the weak spots are expected, tied to
  the structure snapshot (`{{VERSION}}`), not to speculation.
- `## Recommended structural changes` — concrete, prioritised suggestions. Mark
  each `advisory`; do not claim an improvement the aligned metrics do not show.
- `## Next step` — one concrete follow-up (usually: apply one change as a new
  structure version and re-run the affected cases).

State plainly that accuracy is compared at **equal wall-clock** using best-so-far
error, and that a low error under a short budget is not a proof of accuracy.

## Inputs

Operator tag: `{{TAG}}`

Detected interface:

```json
{{INTERFACE}}
```

Aggregated testset summary:

```json
{{TESTSET_SUMMARY}}
```

Structure version: `{{VERSION}}`

Structure snapshot (body-free):

```json
{{STRUCTURE_JSON}}
```

Early-stop records (per case):

```json
{{EARLY_STOPS}}
```
