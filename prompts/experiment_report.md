# Prompt: Experiment Report

Fill in the placeholders and ask the LLM to write a concise, evidence-based
report. Never invent numbers; use only the provided data.

---

Write `.nems/runs/<run_id>/REPORT.md` for one experiment run. Use the measured
numbers exactly as given. State uncertainty where it exists.

Return Markdown with these sections:

- `## Summary` — one paragraph: what was run, and the outcome.
- `## Configuration` — the parameters that were varied.
- `## Result` — final and best-so-far metrics; the wall-clock at which the
  baseline was matched or beaten.
- `## Early stop` — whether and why the run stopped (`trigger_kind`, second,
  ratio); omit if it ran to completion.
- `## Structural context` — the structure version, and the change that this run
  tests.
- `## Verdict` — `improved` | `no change` | `worse` | `inconclusive`, with the
  aligned comparison that supports it.
- `## Next step` — one concrete suggestion.

Be explicit that accuracy is compared at **equal wall-clock** using best-so-far
error. Do not claim an improvement the aligned numbers do not support.

## Inputs

Run id: `{{RUN_ID}}`

```json
{{RUN_METRICS}}
```

```json
{{BASELINE_METRICS}}
```

```json
{{EARLY_STOP}}
```

Structure version: `{{VERSION}}`

Structure diff:

```json
{{STRUCTURE_DIFF}}
```
