# Loss-Stream Adapters

The manager needs the experiment's error over wall-clock time to do align-and-
compare early stopping. The default contract asks the program to write
`loss_time.jsonl`. When a program does not, a **loss-stream adapter** reads
whatever it *does* produce at run time and normalizes it to the same point shape.

## Point shape

Every adapter yields the same points:

```json
{"wall_clock": <seconds>, "loss": <float>, "epoch": <int or null>}
```

The early stopper converts `loss` to an RMSE-style error with `sqrt`, so a
recorded loss must be non-negative.

## Selecting an adapter

The agent writes a spec to `.nems/stream_adapter.json` (or passes `stream_spec`
to `ExperimentRunner.run`):

```json
{"type": "<name>", "config": { ... }}
```

## Built-in adapters

| type | reads | live | notes |
| --- | --- | --- | --- |
| `jsonl` (default) | `loss_time.jsonl` | yes | the instrumentation contract |
| `stdout_regex` | the captured stdout / `logs/run.log` | yes | parses printed per-step lines |
| `sidecar` | a CSV / JSON / JSONL file the program already writes | yes | column mapping |
| `completion_only` | final `metrics.json` only | no | no curve; final comparison only |

### `stdout_regex`

```json
{"type": "stdout_regex",
 "config": {"pattern": "epoch=(?P<epoch>\\d+) loss=(?P<loss>[-+0-9.eEnanif]+) elapsed=(?P<wall_clock>[0-9.]+)"}}
```

- `loss` is required. `wall_clock` and `epoch` are optional named groups.
- When the pattern has **no** `wall_clock` group the axis is synthesized as
  `index * time_step` (`time_step` defaults to `1.0`) and the source sets
  `synthetic_time = true`. Reports and `result.json` record this so the reader
  knows the wall-clock axis is approximate.
- Unmatched lines are skipped; a malformed numeric value is skipped too.
- The runner sets `PYTHONUNBUFFERED=1` so printed lines arrive for live tailing.

### `sidecar`

```json
{"type": "sidecar",
 "config": {"path": "history*.csv", "format": "csv",
            "columns": {"wall_clock": "t", "loss": "mse", "epoch": "step"}}}
```

- `path` may be a glob. `format` is `csv`, `json`, or `jsonl`.
- `columns` maps the logical fields to the file's column names; the defaults are
  `wall_clock`, `loss`, `epoch`.

### `completion_only`

```json
{"type": "completion_only", "config": {}}
```

No live points; `read_final` returns `metrics.json`. Early stopping and the
speed guard are inactive, but the final-error comparison against the baseline
still works.

## Extension

Register an adapter in the agent session:

```python
from agent.loss_stream import LossSource, register_source

class MySource(LossSource):
    name = "my_source"
    live = False
    def read_points(self, exp_dir, log_path=None):
        ...
    def read_final(self, exp_dir, log_text=""):
        ...

register_source("my_source", MySource)
```

Or point at one on disk (for a project-local adapter):

```json
{"type": "custom", "module": "my_project_adapters", "class": "MySource",
 "config": {"path": "out/metrics.log"}}
```

`available_sources()` lists the registered names. Choosing an unknown type is an
error that names the available sources, so a typo never silently degrades.

## Honesty rules

- Report `synthetic_time` when the wall-clock axis was synthesized.
- Never fabricate points; a source that finds nothing yields an empty curve, and
  the run is compared on final metrics only.
- Keep the raw program output (the run log) so a parsed curve can be audited.
