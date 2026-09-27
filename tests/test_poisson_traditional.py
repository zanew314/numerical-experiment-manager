"""Tests for the traditional finite-difference Poisson solver."""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SOLVER_PATH = SKILL_ROOT / "examples" / "demo_poisson" / "traditional_solver.py"

try:
    _spec = importlib.util.spec_from_file_location("nems_traditional_solver", SOLVER_PATH)
    solver = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(solver)
    IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - numpy may be missing
    solver = None
    IMPORT_ERROR = exc


@unittest.skipIf(solver is None, f"solver import failed: {IMPORT_ERROR}")
class TestTraditionalPoisson(unittest.TestCase):
    def test_solves_repository_problem_to_cg_tolerance(self):
        report = solver.solve_poisson_2d(n=63)
        self.assertTrue(report["converged"])
        self.assertLess(report["residual_inf"], 1e-8)
        self.assertGreater(report["l2_rel_error"], 0.0)
        self.assertLess(report["l2_rel_error"], 5e-2)
        self.assertEqual(report["schema_version"], solver.SCHEMA_VERSION)

    def test_second_order_convergence(self):
        e31 = solver.solve_poisson_2d(n=31)["l2_rel_error"]
        e63 = solver.solve_poisson_2d(n=63)["l2_rel_error"]
        e127 = solver.solve_poisson_2d(n=127)["l2_rel_error"]
        self.assertLess(e127, e63)
        self.assertLess(e63, e31)
        self.assertAlmostEqual(solver.observed_order(e63, e31, 2), 2.0, delta=0.3)
        self.assertAlmostEqual(solver.observed_order(e127, e63, 2), 2.0, delta=0.3)

    def test_exact_solution_is_zero_on_the_boundary(self):
        import numpy as np

        freq = 4.0 * np.pi
        corners = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
        values = solver.exact_sine(corners[:, 0:1], corners[:, 1:2], freq)
        self.assertLess(float(np.max(np.abs(values))), 1e-12)

    def test_invalid_n_is_rejected(self):
        with self.assertRaises(ValueError):
            solver.solve_poisson_2d(n=4)

    def test_cli_emits_one_json_document(self):
        result = subprocess.run(
            [sys.executable, str(SOLVER_PATH), "--n", "31"],
            cwd=str(SKILL_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["n"], 31)
        self.assertEqual(report["case"], "sine")


if __name__ == "__main__":
    unittest.main()
