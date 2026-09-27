import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.early_stopper import (  # noqa: E402
    EarlyStopper,
    align_series,
    aligned_comparison,
    relative_ratio,
)


class TestAlignment(unittest.TestCase):
    def test_ceil_to_second(self):
        aligned = align_series([
            {"wall_clock": 0.2, "loss": 0.25},
            {"wall_clock": 1.4, "loss": 0.01},
        ])
        self.assertAlmostEqual(aligned[1], 0.5)
        self.assertAlmostEqual(aligned[2], 0.1)

    def test_hold_last_value(self):
        aligned = align_series([
            {"wall_clock": 1.0, "loss": 0.04},
            {"wall_clock": 3.0, "loss": 0.01},
        ])
        self.assertAlmostEqual(aligned[1], 0.2)
        self.assertAlmostEqual(aligned[2], 0.2)
        self.assertAlmostEqual(aligned[3], 0.1)

    def test_relative_ratio(self):
        self.assertAlmostEqual(relative_ratio(2.0, 1.0), 2.0)
        self.assertEqual(relative_ratio(1.0, 0.0), float("inf"))
        self.assertEqual(relative_ratio(0.0, 0.0), 0.0)
        self.assertEqual(relative_ratio(float("nan"), 1.0), float("inf"))


class TestEarlyStopper(unittest.TestCase):
    def _baseline(self):
        return [
            {"wall_clock": 1.0, "loss": 1.0},
            {"wall_clock": 2.0, "loss": 0.1},
            {"wall_clock": 3.0, "loss": 0.01},
        ]

    def test_triggers_on_catastrophic_divergence(self):
        stopper = EarlyStopper(baseline_series=self._baseline(), tolerance=0.20)
        stopper.observe(1.0, 9.0)  # error 3.0 vs baseline 1.0 -> ratio 3.0 > 2.0
        self.assertTrue(stopper.triggered)
        self.assertEqual(stopper.trigger_second, 1)

    def test_no_trigger_when_improving(self):
        stopper = EarlyStopper(baseline_series=self._baseline(), tolerance=0.20)
        stopper.observe(1.0, 0.25)  # error 0.5 vs baseline 1.0
        stopper.observe(2.0, 0.01)
        self.assertFalse(stopper.triggered)

    def test_transient_startup_lag_survives(self):
        # first sample is 1.7x worse, then converges much better: no stop
        stopper = EarlyStopper(baseline_series=self._baseline(), tolerance=0.20)
        stopper.observe(0.6, 2.89)   # error 1.7 vs baseline 1.0 -> ratio 1.7
        stopper.observe(1.1, 0.0001)
        self.assertFalse(stopper.triggered)

    def test_nonfinite_triggers(self):
        stopper = EarlyStopper(baseline_series=self._baseline())
        stopper.observe(1.0, float("inf"))
        self.assertTrue(stopper.triggered)

    def test_evaluate_full_series(self):
        stopper = EarlyStopper(baseline_series=self._baseline(), tolerance=0.20)
        report = stopper.evaluate_series([
            {"wall_clock": 0.5, "loss": 0.9},
            {"wall_clock": 1.2, "loss": 9.0},
        ])
        self.assertTrue(report["triggered"])
        self.assertEqual(report["threshold_ratio"], 1.2)

    def test_speed_guard_skips_unfair_comparison(self):
        baseline = [
            {"wall_clock": 1.0, "loss": 1.0, "epoch": 100},
            {"wall_clock": 2.0, "loss": 0.1, "epoch": 200},
            {"wall_clock": 3.0, "loss": 0.01, "epoch": 300},
        ]
        stopper = EarlyStopper(baseline_series=baseline, tolerance=0.20)
        # hugely worse, but only 10 epochs done vs baseline's 100 -> guarded
        stopper.observe(1.0, 9.0, epoch=10)
        self.assertFalse(stopper.triggered)
        # same ratio but comparable progress now -> stop
        stopper.observe(2.0, 9.0, epoch=210)
        self.assertTrue(stopper.triggered)

    def test_aligned_comparison_holds_baseline_tail(self):
        rows = aligned_comparison(
            [{"wall_clock": 1.0, "loss": 0.01}, {"wall_clock": 3.0, "loss": 0.01}],
            [{"wall_clock": 1.0, "loss": 0.01}],
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["baseline_error"], rows[0]["baseline_error"])


if __name__ == "__main__":
    unittest.main()
