"""Tests for the retrain-once early-stop mechanism in the experiment runner."""

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.experiment_runner import ExperimentRunner  # noqa: E402

FAKE_MAIN = textwrap.dedent(
    """
    import json
    import os

    out = os.environ["NEMS_OUTPUT_DIR"]
    seed = int(os.environ.get("NEMS_SEED", "0"))
    attempt = int(os.environ.get("NEMS_RETRAIN_ATTEMPT", "0"))

    params = {}
    try:
        with open(os.path.join(out, "current_config.json"), encoding="utf-8") as fh:
            params = json.load(fh).get("params", {})
    except OSError:
        params = {}

    quality = params.get("quality", "good")
    good = quality == "good" or (quality == "bad" and seed >= 1)
    loss = 0.0001 if good else 1.0

    with open(os.path.join(out, "loss_time.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"wall_clock": 1.0, "loss": loss, "epoch": 10}) + "\\n")
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"final_loss": loss, "final_error": loss ** 0.5, "epochs": 10,
                   "seed": seed, "retrain_attempt": attempt}, fh)
    """
)

BASELINE = [
    {"wall_clock": 1.0, "loss": 0.01, "epoch": 10},
    {"wall_clock": 2.0, "loss": 0.0001, "epoch": 20},
]


class TestExperimentRunnerRetrain(unittest.TestCase):
    def _make_project(self, tmp):
        root = Path(tmp)
        (root / "main.py").write_text(FAKE_MAIN, encoding="utf-8")
        return root

    def _runner(self, root):
        return ExperimentRunner(root, entry="main.py", timeout=30, monitor_interval=0.05)

    def test_retrain_recovers_a_bad_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_project(tmp)
            runner = self._runner(root)
            record = runner.run(
                "exp_bad", {"quality": "bad"}, baseline_series=BASELINE,
                early_stop=True, max_retrains=1,
            )
            self.assertTrue(record["retrain"]["attempted"])
            self.assertEqual(record["retrain"]["count"], 1)
            self.assertEqual(len(record["retrain"]["attempts"]), 2)
            # first attempt was too bad, retrain succeeded -> final run completed
            self.assertEqual(record["retrain"]["attempts"][0]["status"], "early_stopped")
            self.assertFalse(record["early_stopped"])
            self.assertEqual(record["status"], "completed")
            archived = root / ".nems" / "experiments" / "exp_bad" / "attempts" / "attempt_1"
            self.assertTrue((archived / "loss_time.jsonl").exists())

    def test_retrain_at_most_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_project(tmp)
            runner = self._runner(root)
            record = runner.run(
                "exp_always_bad", {"quality": "bad_always"}, baseline_series=BASELINE,
                early_stop=True, max_retrains=1,
            )
            self.assertEqual(record["retrain"]["count"], 1)
            self.assertEqual(len(record["retrain"]["attempts"]), 2)
            self.assertTrue(record["early_stopped"])
            self.assertEqual(record["status"], "early_stopped")

    def test_no_retrain_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_project(tmp)
            runner = self._runner(root)
            record = runner.run(
                "exp_disabled", {"quality": "bad"}, baseline_series=BASELINE,
                early_stop=True, max_retrains=0,
            )
            self.assertFalse(record["retrain"]["attempted"])
            self.assertEqual(len(record["retrain"]["attempts"]), 1)
            self.assertTrue(record["early_stopped"])

    def test_retrain_config_override_is_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_project(tmp)
            runner = self._runner(root)
            record = runner.run(
                "exp_override", {"quality": "bad"}, baseline_series=BASELINE,
                early_stop=True, max_retrains=1, retrain_config={"quality": "good"},
            )
            self.assertEqual(record["final_config"]["quality"], "good")
            self.assertFalse(record["early_stopped"])


if __name__ == "__main__":
    unittest.main()
