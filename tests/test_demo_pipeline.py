"""End-to-end integration test for the demo lifecycle.

This runs the full ``run_demo.py`` (CPU, a few minutes at most), so it is
skipped unless ``NEMS_RUN_INTEGRATION=1`` is set. Run it explicitly with:

    NEMS_RUN_INTEGRATION=1 python -m unittest tests.test_demo_pipeline
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("NEMS_RUN_INTEGRATION") == "1", "set NEMS_RUN_INTEGRATION=1 to run")
class TestDemoPipeline(unittest.TestCase):
    def test_full_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            result = subprocess.run(
                [sys.executable, str(SKILL_ROOT / "examples" / "demo_mlp_sin" / "run_demo.py"),
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
                "project_summary.json",
                "model_structure.json",
                "model_structure.md",
                "hyperparameters_confirmed.json",
                "baseline/baseline.json",
                "experiments/exp_001/REPORT.md",
                "experiments/exp_002_earlystop/early_stop.json",
                "project_summary_diff.md",
                "REPORT.md",
                "INDEX.md",
                "figures/metrics_evolution.png",
            ]:
                self.assertTrue((nems / rel).exists(), msg=f"missing {rel}")


if __name__ == "__main__":
    unittest.main()
