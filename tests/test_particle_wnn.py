"""Tests for the ParticleWNN weak-form Poisson solver."""

import importlib.util
import math
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = SKILL_ROOT / "examples" / "demo_poisson" / "particle_wnn.py"

try:
    import torch  # noqa: F401

    _spec = importlib.util.spec_from_file_location("nems_particle_wnn", MODULE_PATH)
    pw = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(pw)
    IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - torch may be missing
    pw = None
    IMPORT_ERROR = exc


@unittest.skipIf(pw is None, f"module import failed: {IMPORT_ERROR}")
class TestParticleWNN(unittest.TestCase):
    def test_weak_form_holds_on_the_exact_solution(self):
        freq = math.pi
        n_center = 64
        radius = torch.full((n_center, 1, 1), 1e-3)
        grid = pw.integral_grid(9)
        v, dv = pw.wendland_test_function(grid)
        n_grid = grid.shape[0]
        centers = (torch.rand(n_center, 1, 2) * 2 - 1) * 0.99

        x = (centers + grid[None, :, :] * radius).reshape(-1, 2)
        x.requires_grad_(True)
        u = pw.exact_u(x, freq)
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=False)[0]
        v_flat = v[None, :, :].expand(n_center, n_grid, 1).reshape(-1, 1)
        dv_flat = (dv[None, :, :] / radius).reshape(-1, 2)
        flux = (grad_u * dv_flat).sum(-1).reshape(n_center, n_grid).mean(-1)
        source = (pw.source_f(x, freq) * v_flat).reshape(n_center, n_grid).mean(-1)

        mismatch = float(torch.norm(source - flux) / torch.norm(source))
        self.assertLess(mismatch, 0.05)

    def test_training_reduces_the_error(self):
        config = {
            "FREQ": math.pi, "HIDDEN": 32, "DEPTH": 3, "ACTIVATION": "Tanh_Sin",
            "LR": 1e-3, "EPOCHS": 120, "N_CENTER": 96, "RADIUS": 1e-3,
            "N_MESH": 9, "N_TEST": 25, "WEIGHT_DECAY": 1e-4,
            "STEP_SIZE": 200, "GAMMA": 0.5, "W_PDE": 5.0,
        }
        metrics = pw.train(config)
        self.assertTrue(metrics["final_l2_error"] < 0.05, metrics)
        self.assertGreater(metrics["final_loss"], 0.0)
        self.assertAlmostEqual(metrics["final_error"], math.sqrt(metrics["final_loss"]), places=6)

    def test_unknown_activation_is_rejected(self):
        with self.assertRaises(ValueError):
            pw.make_activation("not_an_activation")

    def test_integral_grid_stays_in_unit_disk(self):
        grid = pw.integral_grid(9)
        self.assertGreater(grid.shape[0], 0)
        self.assertLess(float(torch.linalg.norm(grid, dim=1).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
