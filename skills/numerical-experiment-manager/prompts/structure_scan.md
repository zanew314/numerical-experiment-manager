# Prompt: Structure Scan

Fill in the placeholders and ask the LLM for a single JSON object.

---

You are analyzing a numerical/ML experiment project so a coding agent can manage
it. You are given a body-free structural snapshot (imports, classes, bases,
method signatures, `self.<attr>` assignments, constants) plus the raw source of
the main entry file. Do not invent anything that is not present.

Return **one JSON object**, no prose, with this shape:

```json
{
  "purpose": "one sentence on what this project computes",
  "data_flow": ["input -> layer1 -> layer2 -> loss", "..."],
  "components": [
    {"name": "...", "kind": "layer|function|solver|loss", "role": "..."}
  ],
  "parameters": [
    {"name": "...", "file": "...", "line": 0, "kind": "learning_rate|hidden_size|epochs|grid|seed|other", "default": null}
  ],
  "performance_knobs": [
    {"name": "...", "direction": "increase|decrease|unknown", "why": "..."}
  ],
  "uncertainties": ["..."]
}
```

If a field cannot be determined, use an empty list or `null`. Prefer the exact
class and method names from the snapshot.

## Structural snapshot

```
{{STRUCTURE_JSON}}
```

## Main entry source

```python
{{CODE_CONTENT}}
```
