"""Tests for the body-free model-structure snapshot (AST + optional torch.fx)."""

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import model_structure  # noqa: E402

SAMPLE = textwrap.dedent(
    '''
    import torch
    import torch.nn as nn

    DEFAULT_HIDDEN = 32

    class Config:
        LR = 0.001
        HIDDEN_DIM = 64

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.fc1 = nn.Linear(1, config.HIDDEN_DIM)
            self.fc2 = nn.Linear(config.HIDDEN_DIM, 1)
            assert "BODY_ONLY_TOKEN_9f3" == "BODY_ONLY_TOKEN_9f3"

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            return torch.tanh(self.fc2(x))

    def make_data(n=8):
        return torch.linspace(0.0, 1.0, n)
    '''
)


class TestModelStructure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "main.py").write_text(SAMPLE, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_extracts_classes_layers_and_constants(self):
        structure = model_structure.extract_model_structure(self.root)
        self.assertEqual(structure["schema_version"], model_structure.SCHEMA_VERSION)
        module = structure["modules"]["main.py"]
        class_names = [cls["name"] for cls in module["classes"]]
        self.assertIn("Config", class_names)
        self.assertIn("MLP", class_names)

        models = {cls["name"]: cls for cls in structure["model_classes"]}
        self.assertIn("MLP", models)
        self.assertEqual(models["MLP"]["layer_count"], 2)
        self.assertEqual(
            [layer["attr"] for layer in models["MLP"]["layers"]],
            ["fc1", "fc2"],
        )

        config = next(cls for cls in module["classes"] if cls["name"] == "Config")
        self.assertEqual(
            {attr["name"]: attr["value"] for attr in config["attributes"]},
            {"LR": 0.001, "HIDDEN_DIM": 64},
        )
        constants = {c["name"]: c["value"] for c in module["module_constants"]}
        self.assertEqual(constants.get("DEFAULT_HIDDEN"), 32)

    def test_function_bodies_are_not_copied(self):
        structure = model_structure.extract_model_structure(self.root)
        dumped = json.dumps(structure)
        self.assertNotIn("BODY_ONLY_TOKEN_9f3", dumped)
        self.assertNotIn("assert ", dumped)

    def test_write_model_structure_outputs_json_and_markdown(self):
        result = model_structure.write_model_structure(self.root)
        self.assertTrue(Path(result["json"]).exists())
        self.assertTrue(Path(result["markdown"]).exists())
        markdown = Path(result["markdown"]).read_text(encoding="utf-8")
        self.assertIn("class MLP", markdown)
        self.assertNotIn("BODY_ONLY_TOKEN_9f3", markdown)

    def test_syntax_error_is_recorded_not_fatal(self):
        (self.root / "broken.py").write_text("def oops(:\n", encoding="utf-8")
        structure = model_structure.extract_model_structure(self.root)
        self.assertTrue(structure["parse_errors"])
        self.assertIn("main.py", structure["modules"])

    def test_diff_reports_layer_count_change(self):
        old = model_structure.extract_model_structure(self.root)
        changed = SAMPLE.replace(
            "        self.fc2 = nn.Linear(config.HIDDEN_DIM, 1)\n",
            "        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)\n"
            "        self.fc3 = nn.Linear(config.HIDDEN_DIM, 1)\n",
        )
        new = model_structure.extract_model_structure(self.root, files={"main.py": changed})
        diff = model_structure.diff_model_structures(old, new)
        fields = {change["field"] for change in diff["changes"]}
        self.assertIn("layer_count", fields)

    def test_torch_fx_augmentation_is_best_effort(self):
        try:
            import torch
            import torch.nn as nn
        except Exception:  # noqa: BLE001 - torch is optional
            self.skipTest("torch is not installed")

        structure = model_structure.extract_model_structure(self.root)
        model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
        model_structure.augment_with_torch_fx(structure, model, (torch.zeros(1, 2),))
        self.assertIsNotNone(structure["dynamic_graph"])
        self.assertEqual(structure["dynamic_graph"]["total_parameters"], 4 * 2 + 4 + 4 + 1)


if __name__ == "__main__":
    unittest.main()
