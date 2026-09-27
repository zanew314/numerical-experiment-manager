"""End-to-end runner test for a project with no loss_time.jsonl stream.

The fake program only prints progress lines; the runner must read them through a
``stdout_regex`` source and drive early stopping. This runs without torch.
"""

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.experiment_runner import ExperimentRunner  # noqa: E402

FAKE_PRINT_MAIN = textwrap.dedent(
    """
    import json
    import os

    out = os.environ["NEMS_OUTPUT_DIR"]
    params = {}
    try:
        with open(os.path.join(out, "current_config.json"), encoding="utf-8") as fh:
            params = json.load(fh).get("params", {})
    except OSError:
        params = {}

    quality = params.get("quality", "good")
    loss = 0.0001 if quality == "good" else 4.0
    for epoch in range(5):
        print(f"epoch={epoch} loss={loss} elapsed={epoch * 0.5}", flush=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"final_loss": loss, "final_error": loss ** 0.5, "epochs": 5}, fh)
    """
)

STDOUT_SPEC = {
    "type": "stdout_regex",
    "config": {
        "pattern": r"epoch=(?P<epoch>\d+) loss=(?P<loss>[-+0-9.eEnanif]+) elapsed=(?P<wall_clock>[0-9.]+)"
    },
}

BASELINE = [
    {"wall_clock": 1.0, "loss": 0.01},
    {"wall_clock": 2.0, "loss": 0.0001},
]


class TestNoStreamRunner(unittest.TestCase):
    def _make_project(self, tmp):
        root = Path(tmp)
        (root / "main.py").write_text(FAKE_PRINT_MAIN, encoding="utf-8")
        return root

    def _runner(self, root):
        return ExperimentRunner(root, entry="main.py", timeout=30, monitor_interval=0.05)

    def test_stdout_source_feeds_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_project(tmp)
            record = self._runner(root).run(
                "exp_good", {"quality": "good"}, baseline_series=BASELINE,
                early_stop=True, stream_spec=STDOUT_SPEC,
            )
            self.assertEqual(record["stream"]["type"], "stdout_regex")
            self.assertGreaterEqual(record["num_points"], 1)
            self.assertFalse(record["early_stopped"])
            self.assertEqual(record["status"], "completed")
            self.assertAlmostEqual(record["metrics"]["final_error"], 0.01)

    def test_stdout_source_triggers_early_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_project(tmp)
            record = self._runner(root).run(
                "exp_bad", {"quality": "bad"}, baseline_series=BASELINE,
                early_stop=True, stream_spec=STDOUT_SPEC,
            )
            self.assertTrue(record["early_stopped"])
            self.assertEqual(record["status"], "early_stopped")

    def test_stream_spec_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_project(tmp)
            from agent import loss_stream

            loss_stream.save_stream_spec(root, STDOUT_SPEC)
            record = self._runner(root).run(
                "exp_file", {"quality": "good"}, baseline_series=BASELINE, early_stop=True,
            )
            self.assertEqual(record["stream"]["type"], "stdout_regex")
            self.assertGreaterEqual(record["num_points"], 1)


if __name__ == "__main__":
    unittest.main()
