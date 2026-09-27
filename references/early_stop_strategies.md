# Early-Stop Strategies

The manager compares every experiment against a fixed baseline on a **wall-clock** axis, not an
epoch axis. The default criterion is a 20% relative-error threshold.

## Wall-clock alignment

- Both the baseline and the current experiment store `loss_time.jsonl` with lines
  `{"wall_clock": <sec>, "loss": <float>, "epoch": <int>}`.
- `wall_clock` is measured with `time.perf_counter() - t0` inside the training process.
- Time is quantized to whole seconds. A run shorter than one second still occupies second `1`
  (ceil rule: "不足一秒按一秒计算").
- To compare at integer second `s`, each curve is reduced to the **best (lowest) error achieved
  at or before `s`**. The quantity of interest is "at equal wall-clock, what accuracy have you
  reached so far?", and the best-so-far envelope avoids false alarms from transient loss spikes.
  If a curve has no sample at or before `s`, the comparison is skipped.

## Relative error

```
error(series, s) = min over t <= s of sqrt(loss(t))      # best-so-far RMSE
ratio(s)         = error_experiment(s) / error_baseline(s)
```

If the baseline is still running while the experiment continues, the baseline's last available
time point is reused for later seconds (hold-last extrapolation).

## Stop rule

```
stop if ratio(s) > CATASTROPHIC_RATIO                      # any second
stop if s >= MIN_SECONDS and ratio(s) > 1 + TOLERANCE      # after warm-up
```

Defaults: `MIN_SECONDS = 2`, `TOLERANCE = 0.20`, `CATASTROPHIC_RATIO = 2.0`.

Every trigger carries a `trigger_kind`:

| kind | meaning |
| --- | --- |
| `non_finite` | experiment error became `nan` / `inf` (divergence) |
| `catastrophic` | `ratio(s) > 2.0` at any second |
| `threshold` | `ratio(s) > 1 + TOLERANCE` after the warm-up |

## Retrain once on a too-large relative error

A bad wall-clock result at a fixed configuration is often caused by an unlucky
initialization, not by the configuration itself. So when the early stopper fires
with a relative error that is too large, the runner does **not** abandon the run
immediately: it retrains the experiment automatically, **at most once**
(`max_retrains`, default `1`).

```
attempt 1 -> early stop (trigger_kind in {non_finite, catastrophic, threshold})
          -> retrain (fresh process, shifted NEMS_SEED, same/overridden config)
          -> attempt 2 still too bad? -> record early_stopped and stop
```

- The retrain gets a fresh subprocess and `NEMS_SEED = attempt - 1`, so projects
  that honor `NEMS_SEED` (see the instrumentation contract) can recover from an
  unlucky seed. `NEMS_RETRAIN_ATTEMPT` is `0` for the first run, `1` for the retrain.
- `retrain_config` (a dict merged over the base config) can override
  hyperparameters for the retrain, e.g. a smaller learning rate; it defaults to the
  same config.
- Attempt 1 artifacts are preserved under
  `experiments/<exp_id>/attempts/attempt_1/`; the root `loss_time.jsonl`,
  `metrics.json` and `early_stop.json` always describe the final attempt.
- `result.json` carries a `retrain` block:
  `{"attempted": true, "count": 1, "max_retrains": 1, "attempts": [...]}`.
- Set `max_retrains=0` to restore the original stop-on-first-failure behavior.

Additional guards:

- A non-finite experiment error (`nan` / `inf`, typical of an exploding learning rate) triggers an
  immediate stop.
- The warm-up window (2 seconds) skips first-second process-startup jitter, so a faster-converging
  configuration is not stopped merely because its first second was slow to schedule. Clearly broken
  configurations still stop immediately through the catastrophic ratio or the non-finite guard.
- A speed guard compares epoch progress at the same wall-clock time. If the experiment has completed
  fewer than half the epochs the baseline had at that second, the ratio rule is skipped for that
  point: the experiment is behind because of machine speed, not because the configuration is bad.
  Both the recorded `epoch` field and the baseline's are used.
- Live monitoring polls `loss_time.jsonl` sub-second and kills the subprocess when a rule fires.
  Because very fast runs can finish before the first poll, the full recorded series is re-evaluated
  once at the end so the decision is still recorded in `early_stop.json`.

## Rationale

Wall-clock alignment answers the practical question "at equal compute budget, which method is
more accurate?" A 20% tolerance filters noise while still catching divergence and clearly bad
hyperparameters quickly.
