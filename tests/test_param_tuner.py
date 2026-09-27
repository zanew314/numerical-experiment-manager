import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.param_tuner import select_params  # noqa: E402


class TestParamTuner(unittest.TestCase):
    def setUp(self):
        self.confirmed = [
            {"name": "LR", "value": 0.01, "type": "float"},
            {"name": "NUM_LAYERS", "value": 4, "type": "int"},
            {"name": "HIDDEN_DIM", "value": 64, "type": "int"},
            {"name": "EPOCHS", "value": 1000, "type": "int"},
        ]

    def test_selects_small_params_only(self):
        selected = select_params(self.confirmed, max_params=3, values_per_param=2)
        names = [item["name"] for item in selected]
        self.assertNotIn("EPOCHS", names)
        self.assertEqual(names[0], "LR")
        self.assertLessEqual(len(selected), 3)

    def test_candidate_values(self):
        selected = {item["name"]: item for item in select_params(self.confirmed)}
        self.assertEqual(len(selected["LR"]["values"]), 2)
        self.assertIn(round(0.01 * 3, 8), selected["LR"]["values"])
        self.assertTrue(all(v != selected["HIDDEN_DIM"]["base"] for v in selected["HIDDEN_DIM"]["values"]))

    def test_layer_values_are_additive(self):
        selected = {item["name"]: item for item in select_params(self.confirmed)}
        self.assertEqual(sorted(selected["NUM_LAYERS"]["values"]), [2, 6])


if __name__ == "__main__":
    unittest.main()
