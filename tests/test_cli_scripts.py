"""Dependency-free tests for the command-line entry points.

These run without torch or matplotlib, so they can serve as the platform's
zero-skip smoke tests for the package.
"""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_script(script, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class TestCheckEnvironment(unittest.TestCase):
    def test_emits_one_json_document(self):
        result = run_script("check_environment.py")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("", result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["schema_version"], "nems-environment-v1")
        self.assertIn("numpy", report["packages"])
        self.assertEqual(report["ready"], report["packages"]["numpy"]["available"])


class TestAnalyzeProject(unittest.TestCase):
    def test_scans_a_tiny_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text(
                textwrap.dedent(
                    """
                    import torch.nn as nn

                    class Config:
                        LR = 0.01

                    class MLP(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.fc1 = nn.Linear(1, 8)

                        def forward(self, x):
                            return x
                    """
                ),
                encoding="utf-8",
            )
            result = run_script("analyze_project.py", str(root))
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["schema_version"], "nems-project-analysis-v1")
            self.assertEqual(report["file_count"], 1)
            self.assertEqual(report["summary"]["entry_point"], "main.py")
            names = [h["name"] for h in report["summary"]["hyperparameters"]]
            self.assertIn("LR", names)
            self.assertIn("main.py", report["structure"]["modules"])

    def test_utf8_bom_source_still_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("X = 1\n", encoding="utf-8-sig")
            result = run_script("analyze_project.py", str(root))
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(result.stdout)
            self.assertIn("main.py", report["structure"]["modules"])

    def test_bad_input_is_json_error_on_stderr(self):
        result = run_script("analyze_project.py", str(ROOT / "does-not-exist"))
        self.assertEqual(2, result.returncode)
        self.assertEqual("", result.stdout)
        error = json.loads(result.stderr)
        self.assertEqual(error["error"]["kind"], "NotADirectoryError")


if __name__ == "__main__":
    unittest.main()
