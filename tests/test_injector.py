import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from agent import hyperparameter_injector  # noqa: E402
from agent.llm_analyzer import extract_hyperparameters  # noqa: E402

DEMO_MAIN = (SKILL_ROOT / "examples" / "demo_mlp_sin" / "main.py").read_text(encoding="utf-8")


class TestInjector(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copyfile(SKILL_ROOT / "examples" / "demo_mlp_sin" / "main.py", self.root / "main.py")
        self.params = extract_hyperparameters({"main.py": DEMO_MAIN})
        self.keep = [p for p in self.params if p["name"] in ("LR", "NUM_LAYERS", "HIDDEN_DIM")]

    def tearDown(self):
        self.tmp.cleanup()

    def test_inject_selected_only(self):
        report = hyperparameter_injector.inject_hyperparameters(self.root, self.keep)
        text = (self.root / "main.py").read_text(encoding="utf-8")
        self.assertIn('get_param("LR", default=0.001)', text)
        self.assertIn('get_param("NUM_LAYERS", default=4)', text)
        self.assertIn('get_param("HIDDEN_DIM", default=64)', text)
        self.assertNotIn('get_param("EPOCHS"', text)
        self.assertIn("EPOCHS = 1000", text)
        self.assertIn("from nems_config import get_param", text)
        self.assertEqual(len(report["injected"]), 3)
        self.assertEqual(report["config_module"], "nems_config.py")

    def test_backup_and_diff(self):
        report = hyperparameter_injector.inject_hyperparameters(self.root, self.keep)
        snapshot = Path(report["snapshot_dir"])
        self.assertTrue((snapshot / "main.py").exists())
        self.assertTrue((snapshot / "manifest.json").exists())
        self.assertTrue((snapshot / "diff.md").exists())
        self.assertIn("get_param", (snapshot / "diff.md").read_text(encoding="utf-8"))

    def test_config_module_resolves_from_current_config(self):
        import json
        report = hyperparameter_injector.inject_hyperparameters(self.root, self.keep)
        module_path = Path(report["config_module_path"])
        self.assertTrue(module_path.exists())
        nems = self.root / ".nems"
        nems.mkdir(exist_ok=True)
        (nems / "current_config.json").write_text(
            json.dumps({"params": {"LR": 0.02}}), encoding="utf-8"
        )
        sys.path.insert(0, str(self.root))
        try:
            module = __import__("nems_config")
            self.assertEqual(module.get_param("LR", default=0.001), 0.02)
            self.assertEqual(module.get_param("MISSING", default=7), 7)
        finally:
            sys.path.remove(str(self.root))
            sys.modules.pop("nems_config", None)

    def test_name_collision_fallback(self):
        (self.root / "nems_config.py").write_text("existing\n", encoding="utf-8")
        self.assertEqual(
            hyperparameter_injector.resolve_config_module_name(self.root),
            "nems_param_loader.py",
        )

    def test_double_inject_is_idempotent(self):
        hyperparameter_injector.inject_hyperparameters(self.root, self.keep)
        report2 = hyperparameter_injector.inject_hyperparameters(self.root, self.keep)
        self.assertEqual(report2["injected"], [])
        self.assertEqual(len(report2["skipped"]), 3)


if __name__ == "__main__":
    unittest.main()
