"""End-to-end integration test for the no-loss-stream demo.

Runs ``examples/demo_no_stream/run_demo.py`` (CPU, under a minute), so it is
skipped unless ``NEMS_RUN_INTEGRATION=1`` is set. It also requires PyTorch.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(os.environ.get("NEMS_RUN_INTEGRATION") == "1", "set NEMS_RUN_INTEGRATION=1 to run")
@unittest.skipUnless(HAS_TORCH, "the no-stream demo requires torch")
class TestDemoNoStream(unittest.TestCase):
    def test_full_no_stream_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            result = subprocess.run(
                [sys.executable, str(SKILL_ROOT / "examples" / "demo_no_stream" / "run_demo.py"),
                 "--workspace", str(workspace)],
                cwd=str(SKILL_ROOT),
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("DEMO COMPLETE", result.stdout)
            nems = workspace / ".nems"
            for rel in [
                "model_structure.json",
                "model_structure.md",
                "stream_adapter.json",
                "baseline/baseline.json",
                "experiments/exp_good/REPORT.md",
                "experiments/exp_bad/REPORT.md",
                "REPORT.md",
                "INDEX.md",
                "figures/metrics_evolution.png",
            ]:
                self.assertTrue((nems / rel).exists(), msg=f"missing {rel}")


if __name__ == "__main__":
    unittest.main()
