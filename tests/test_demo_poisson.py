"""End-to-end integration test for the Poisson demo (PINN + finite difference).

Runs ``examples/demo_poisson/run_demo.py`` (CPU, a few minutes at most), so it is
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
@unittest.skipUnless(HAS_TORCH, "the Poisson demo requires torch")
class TestDemoPoisson(unittest.TestCase):
    def test_full_poisson_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            result = subprocess.run(
                [sys.executable, str(SKILL_ROOT / "examples" / "demo_poisson" / "run_demo.py"),
                 "--workspace", str(workspace)],
                cwd=str(SKILL_ROOT),
                capture_output=True,
                text=True,
                timeout=900,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("DEMO COMPLETE", result.stdout)
            nems = workspace / ".nems"
            for rel in [
                "model_structure.json",
                "traditional/poisson_fd.json",
                "traditional/TRADITIONAL_REPORT.md",
                "particlewnn/metrics.json",
                "particlewnn/loss_time.jsonl",
                "baseline/baseline.json",
                "experiments/exp_lr1e-2/REPORT.md",
                "experiments/exp_lr1.0/early_stop.json",
                "REPORT.md",
                "INDEX.md",
                "figures/metrics_evolution.png",
            ]:
                self.assertTrue((nems / rel).exists(), msg=f"missing {rel}")


if __name__ == "__main__":
    unittest.main()
