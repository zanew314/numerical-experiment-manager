"""Wall-clock aligned early stopping.

The baseline and the running experiment are aligned to whole seconds. A run
shorter than one second still occupies second ``1`` ("不足一秒按一秒计算"). At each
second the reported accuracy is the best (lowest) RMSE achieved by that
wall-clock time, so transient loss spikes do not create false regressions. The
default stop rule fires when
``best_error_experiment / best_error_baseline > 1 + 0.20`` for any second past
the warm-up (default 2s, which skips first-second startup jitter). A run is also
stopped immediately when its error becomes non-finite or when the ratio exceeds
``CATASTROPHIC_RATIO`` (2.0), so clearly broken configurations do not need to
survive the warm-up.

A speed guard compares epoch progress at the same wall-clock time: if the
experiment is less than half as far along as the baseline (for example because
its process got less CPU), ratio-based stopping is skipped for that point, so
machine-speed jitter does not masquerade as a bad configuration.

Every trigger is classified by ``trigger_kind`` (``non_finite``, ``catastrophic``
or ``threshold``). The runner uses this classification to decide whether a run
whose relative error is too large gets one automatic retraining attempt before
being abandoned ("重新训练，至多一次").
"""

from __future__ import annotations

import math

from . import save_json

DEFAULT_TOLERANCE = 0.20
DEFAULT_MIN_SECONDS = 2
CATASTROPHIC_RATIO = 2.0

# Trigger kinds that count as "relative error too large" and are therefore
# eligible for a single automatic retraining attempt.
RETRAINABLE_TRIGGER_KINDS = ("non_finite", "catastrophic", "threshold")


def loss_to_error(loss) -> float:
    """Convert a recorded loss into an RMSE-style error."""
    try:
        value = float(loss)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(value) or math.isinf(value):
        return value
    if value < 0:
        return value
    return math.sqrt(value)


def _second_of(wall_clock) -> int:
    try:
        wc = float(wall_clock)
    except (TypeError, ValueError):
        wc = 0.0
    if wc <= 0:
        return 1
    return max(1, int(math.ceil(wc)))


def _better(candidate, current):
    """Return the smaller finite error, ignoring nan and keeping finite values."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    if isinstance(candidate, float) and math.isnan(candidate):
        return current
    if isinstance(current, float) and math.isnan(current):
        return candidate
    return candidate if candidate < current else current


def align_series(series) -> dict:
    """Return ``{second: best-so-far error}`` with ceil-to-second buckets.

    At each integer second the value is the best (lowest) RMSE achieved by that
    wall-clock time. Loss spikes therefore do not create false regressions, and a
    run shorter than one second still occupies second ``1``.
    """
    points = [p for p in series if p is not None]
    if not points:
        return {}
    points = sorted(points, key=lambda p: float(p.get("wall_clock", 0.0) or 0.0))
    per_second = {}
    for point in points:
        sec = _second_of(point.get("wall_clock", 0.0))
        per_second[sec] = loss_to_error(point.get("loss", float("nan")))
    best = loss_to_error(points[0].get("loss", float("nan")))
    max_sec = max(per_second)
    aligned = {}
    for sec in range(1, max_sec + 1):
        if sec in per_second:
            best = _better(per_second[sec], best)
        aligned[sec] = best
    return aligned


def relative_ratio(current_error, baseline_error):
    """Return ``current_error / baseline_error`` handling zero and non-finite."""
    if current_error is None or baseline_error is None:
        return None
    if isinstance(current_error, float) and (math.isnan(current_error) or math.isinf(current_error)):
        return float("inf")
    if baseline_error == 0:
        return 0.0 if current_error <= 0 else float("inf")
    return current_error / baseline_error


def progress_series(series, key="epoch"):
    """Return ``{second: last value of key reached by that second}``."""
    points = [p for p in series if p is not None and p.get(key) is not None]
    if not points:
        return {}
    points = sorted(points, key=lambda p: float(p.get("wall_clock", 0.0) or 0.0))
    per_second = {}
    for point in points:
        per_second[_second_of(point.get("wall_clock", 0.0))] = point.get(key)
    max_sec = max(per_second)
    aligned = {}
    current = per_second[min(per_second)]
    for sec in range(1, max_sec + 1):
        if sec in per_second:
            current = per_second[sec]
        aligned[sec] = current
    return aligned


def aligned_comparison(experiment_series, baseline_series, max_seconds=None):
    """Compare two series on the shared integer-second grid."""
    exp = align_series(experiment_series)
    base = align_series(baseline_series)
    if not exp or not base:
        return []
    last_base_sec = max(base)
    rows = []
    limit = min(max(exp), max_seconds) if max_seconds else max(exp)
    for sec in range(1, limit + 1):
        base_err = base.get(sec, base[last_base_sec])
        exp_err = exp.get(sec)
        rows.append({
            "wall_clock_sec": sec,
            "baseline_error": base_err,
            "experiment_error": exp_err,
            "ratio": relative_ratio(exp_err, base_err),
        })
    return rows


class EarlyStopper:
    """Tracks a running experiment against a fixed baseline."""

    def __init__(self, baseline_series=None, tolerance=DEFAULT_TOLERANCE, min_seconds=DEFAULT_MIN_SECONDS):
        self.tolerance = tolerance
        self.min_seconds = min_seconds
        self.baseline = align_series(baseline_series or [])
        self.baseline_last_second = max(self.baseline) if self.baseline else 0
        raw_samples = sorted(
            [(float(p.get("wall_clock", 0.0) or 0.0), loss_to_error(p.get("loss")), p.get("epoch"))
             for p in (baseline_series or [])],
            key=lambda item: item[0],
        )
        # running best-so-far per sample time for a fair live comparison
        self.baseline_samples = []
        best = None
        for time_point, error, _epoch in raw_samples:
            best = _better(error, best)
            self.baseline_samples.append((time_point, best))
        self.baseline_epoch_series = progress_series(baseline_series or [], "epoch")
        self.speed_guard = 0.5
        self.observations = []
        self.best_error = None
        self.triggered = False
        self.trigger_second = None
        self.trigger_ratio = None
        self.trigger_reason = None
        self.trigger_kind = None

    def _baseline_at(self, second):
        if not self.baseline:
            return None
        if second in self.baseline:
            return self.baseline[second]
        return self.baseline[self.baseline_last_second]

    def _baseline_at_time(self, wall_clock):
        """Baseline value at the same elapsed time (fair live comparison)."""
        if not self.baseline_samples:
            return self._baseline_at(_second_of(wall_clock))
        value = self.baseline_samples[0][1]
        for time_point, error in self.baseline_samples:
            if time_point <= wall_clock:
                value = error
            else:
                break
        return value

    def _baseline_epoch_at_time(self, wall_clock):
        if not self.baseline_epoch_series:
            return None
        second = _second_of(wall_clock)
        if second in self.baseline_epoch_series:
            return self.baseline_epoch_series[second]
        return self.baseline_epoch_series[max(self.baseline_epoch_series)]

    def _speed_guarded(self, epoch, baseline_epoch):
        """True when the experiment is far behind, so a time ratio is unfair."""
        if epoch is None or baseline_epoch in (None, 0):
            return False
        return epoch < baseline_epoch * self.speed_guard

    def observe(self, wall_clock, loss, epoch=None):
        """Record one point; return True if the stop rule now fires."""
        error = loss_to_error(loss)
        self.best_error = _better(error, self.best_error)
        second = _second_of(wall_clock)
        base_err = self._baseline_at_time(wall_clock)
        guarded = self._speed_guarded(epoch, self._baseline_epoch_at_time(wall_clock))
        ratio = relative_ratio(self.best_error, base_err)
        self.observations.append({
            "wall_clock": float(wall_clock),
            "second": second,
            "loss": loss,
            "error": error,
            "best_error": self.best_error,
            "baseline_error": base_err,
            "ratio": ratio,
            "speed_guarded": guarded,
        })
        if self.triggered:
            return True
        if base_err is None:
            return False
        if isinstance(error, float) and (math.isnan(error) or math.isinf(error)):
            self._mark(second, ratio, "实验误差出现非有限值（发散）", "non_finite")
            return True
        if guarded:
            return False
        if ratio is not None and ratio > CATASTROPHIC_RATIO:
            self._mark(second, ratio, "相对误差倍率出现灾难性劣化", "catastrophic")
            return True
        if second >= self.min_seconds and ratio is not None and ratio > (1.0 + self.tolerance):
            self._mark(second, ratio, "相对误差倍率超过阈值", "threshold")
            return True
        return False

    def _mark(self, second, ratio, reason, kind="threshold"):
        self.triggered = True
        self.trigger_second = second
        self.trigger_ratio = ratio
        self.trigger_reason = reason
        self.trigger_kind = kind

    def is_retrainable(self) -> bool:
        """True when the trigger means the relative error is too large."""
        return self.triggered and self.trigger_kind in RETRAINABLE_TRIGGER_KINDS

    def evaluate_series(self, series) -> dict:
        """Re-evaluate the full recorded series (catches runs faster than 1s)."""
        for point in series or []:
            error = loss_to_error(point.get("loss"))
            if isinstance(error, float) and (math.isnan(error) or math.isinf(error)):
                self._mark(_second_of(point.get("wall_clock", 0.0)), float("inf"),
                           "实验误差出现非有限值（发散）", "non_finite")
                break
        rows = aligned_comparison(series, [
            {"wall_clock": s, "loss": (e ** 2)} for s, e in self.baseline.items()
        ]) if self.baseline else []
        exp_epochs = progress_series(series, "epoch")
        for row in rows:
            if self.triggered:
                break
            sec = row["wall_clock_sec"]
            guarded = self._speed_guarded(exp_epochs.get(sec), self.baseline_epoch_series.get(sec))
            row["speed_guarded"] = guarded
            if guarded:
                continue
            ratio = row["ratio"]
            if ratio is not None and ratio > CATASTROPHIC_RATIO:
                self._mark(sec, ratio, "相对误差倍率出现灾难性劣化", "catastrophic")
                break
            if sec >= self.min_seconds and ratio is not None and ratio > (1.0 + self.tolerance):
                self._mark(sec, ratio, "相对误差倍率超过阈值", "threshold")
                break
        return self.report(rows)

    def report(self, rows=None) -> dict:
        if rows is None:
            rows = aligned_comparison(self.observations_to_series(), [
                {"wall_clock": s, "loss": (e ** 2)} for s, e in self.baseline.items()
            ]) if self.baseline else []
        return {
            "triggered": self.triggered,
            "trigger_second": self.trigger_second,
            "trigger_ratio": self.trigger_ratio,
            "trigger_reason": self.trigger_reason,
            "trigger_kind": self.trigger_kind,
            "retrainable": self.is_retrainable(),
            "tolerance": self.tolerance,
            "threshold_ratio": 1.0 + self.tolerance,
            "min_seconds": self.min_seconds,
            "baseline_last_second": self.baseline_last_second,
            "aligned": rows,
        }

    def observations_to_series(self):
        return [{"wall_clock": o["wall_clock"], "loss": o["loss"]} for o in self.observations]

    def write(self, path) -> str:
        return save_json(path, self.report())
