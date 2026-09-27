import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import result_analyzer  # noqa: E402


class DummyAnalyzer:
    def analyze_report(self, experiment, baseline, history, aligned):
        return {"summary": "stub", "possible_reasons": [], "recommendations": []}


class TestResultAnalyzer(unittest.TestCase):
    def _baseline(self):
        return {
            "exp_id": "baseline",
            "metrics": {"final_error": 0.1, "total_wall_clock": 1.0},
            "series": [
                {"wall_clock": 0.5, "loss": 0.25},
                {"wall_clock": 1.0, "loss": 0.01},
            ],
        }

    def test_improved(self):
        record = {
            "exp_id": "exp_001",
            "early_stopped": False,
            "metrics": {"final_error": 0.05},
            "series": [
                {"wall_clock": 0.5, "loss": 0.0025},
                {"wall_clock": 1.0, "loss": 0.0025},
            ],
        }
        analysis = result_analyzer.analyze(record, self._baseline(), analyzer=DummyAnalyzer())
        self.assertEqual(analysis["verdict"], "improved")
        self.assertAlmostEqual(analysis["final_error_ratio"], 0.5)

    def test_worse(self):
        record = {
            "exp_id": "exp_bad",
            "early_stopped": False,
            "metrics": {"final_error": 0.2},
            "series": [
                {"wall_clock": 0.5, "loss": 0.4},
                {"wall_clock": 1.0, "loss": 0.04},
            ],
        }
        analysis = result_analyzer.analyze(record, self._baseline(), analyzer=DummyAnalyzer())
        self.assertEqual(analysis["verdict"], "worse")

    def test_failed_when_early_stopped(self):
        record = {
            "exp_id": "exp_diverge",
            "early_stopped": True,
            "early_stop": {"triggered": True, "trigger_second": 1},
            "metrics": {"final_error": 3.0},
            "series": [{"wall_clock": 1.0, "loss": 9.0}],
        }
        analyzer = DummyAnalyzer()
        analyzer.analyze_report = lambda *a, **k: {"verdict": "improved", "summary": "s"}
        analysis = result_analyzer.analyze(record, self._baseline(), analyzer=analyzer)
        self.assertEqual(analysis["verdict"], "failed")

    def test_time_to_target(self):
        series = [
            {"wall_clock": 0.5, "loss": 0.25},
            {"wall_clock": 1.2, "loss": 0.04},
            {"wall_clock": 2.1, "loss": 0.0004},
        ]
        self.assertEqual(result_analyzer.time_to_target(series, 0.1), 3)
        self.assertIsNone(result_analyzer.time_to_target(series, 1e-9))


if __name__ == "__main__":
    unittest.main()
