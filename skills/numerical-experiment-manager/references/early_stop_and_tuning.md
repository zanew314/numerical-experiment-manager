# Early Stop, Retrain, and Tuning

The manager compares every experiment with a fixed baseline on a **wall-clock**
axis, not an epoch axis. The default criterion is a 10% relative-error
tolerance sustained for a few seconds.

## Wall-clock alignment

- Both curves are `loss_time.jsonl` lines
  `{"wall_clock": <sec>, "loss": <float>, "epoch": <int>}`.
- Time is quantized to whole seconds; a run shorter than one second still
  occupies second `1` (ceil rule).
- To compare at second `s`, reduce each curve to the **best (lowest) error at or
  before `s`** — the best-so-far envelope. It avoids false alarms from transient
  loss spikes. If a curve has no sample at or before `s`, skip that second.

## Relative error

```
error(series, s) = min over t <= s of sqrt(loss(t))      # best-so-far RMSE
ratio(s)         = error_experiment(s) / error_baseline(s)
```

If the baseline has stopped while the experiment continues, reuse the baseline's
last available point (hold-last extrapolation).

## Stop rule

```
stop if ratio(s) > 2.0                                   # catastrophic, any second
stop if error is nan/inf                                 # non-finite
stop if s >= warm_up
        and ratio(s) > 1 + tolerance
        for `sustain` consecutive seconds                # local-optimum suspicion
```

Defaults: `warm_up = 2`, `tolerance = 0.10`, `sustain = 3`, catastrophic ratio
`2.0`. The user may change `tolerance` and `sustain` per run.

`trigger_kind` values:

| kind | meaning |
| --- | --- |
| `non_finite` | experiment error became `nan` / `inf` (divergence) |
| `catastrophic` | `ratio(s) > 2.0` at any second |
| `threshold` | `ratio(s) > 1 + tolerance` held for `sustain` seconds |
| `max_wall_clock` | the run hit its own soft time budget |

## Retrain once

A bad wall-clock result at a fixed configuration is often an unlucky
initialization, not a bad configuration. When the stopper fires with a too-large
relative error, retrain **at most once** in a fresh process with a shifted seed
(`NEMS_SEED`), then record the run as early-stopped if it is still bad.

Attempt 1 artifacts are preserved under `.nems/runs/<run_id>/attempts/attempt_1/`.
Set the retrain count to `0` to disable.

## Speed guard

Compare epoch progress at the same wall-clock second. If the experiment has
completed fewer than half the epochs the baseline had, skip the ratio rule for
that second — the run is behind because of machine speed, not a bad
configuration. Use the recorded `epoch` field in both curves.

## Micro-tuning

When the user wants to "炼丹", tune **small** parameters only:

- at most 3 parameters, at most 2 candidate values each;
- short runs against the fixed baseline;
- one parameter change at a time, so the effect is attributable.

Mark each result `keep` (better or equal at equal wall-clock) or `revert`
(worse). Never route architecture changes through this path; make those an
explicit source edit that creates a new structure version (workflow A), then test
it with workflow B.
