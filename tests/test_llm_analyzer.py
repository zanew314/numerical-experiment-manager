import os
import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from agent.llm_analyzer import (  # noqa: E402
    LLMAnalyzer,
    extract_hyperparameters,
    parse_json,
    parse_literal,
)

DEMO_MAIN = (SKILL_ROOT / "examples" / "demo_mlp_sin" / "main.py").read_text(encoding="utf-8")


class TestLiteral(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(parse_literal("0.001"), (0.001, "float"))
        self.assertEqual(parse_literal("64"), (64, "int"))
        self.assertEqual(parse_literal("1e-3"), (0.001, "float"))

    def test_bool_str_none(self):
        self.assertEqual(parse_literal("True"), (True, "bool"))
        self.assertEqual(parse_literal('"adam"'), ("adam", "str"))
        self.assertEqual(parse_literal("None"), (None, "null"))

    def test_get_param_default(self):
        self.assertEqual(parse_literal('get_param("LR", default=0.001)'), (0.001, "float"))

    def test_invalid(self):
        self.assertIsNone(parse_literal("config.LR"))
        self.assertIsNone(parse_literal("epoch + 1"))

    def test_parse_json_fenced(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json('noise {"b": 2} tail'), {"b": 2})
        self.assertEqual(parse_json(""), {})


class TestStructureAnalysis(unittest.TestCase):
    def setUp(self):
        for key in ("NEMS_LLM_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(key, None)

    def test_offline_structure(self):
        analyzer = LLMAnalyzer(prompts_dir=SKILL_ROOT / "prompts")
        summary = analyzer.analyze_structure({"main.py": DEMO_MAIN})
        self.assertEqual(summary["_source"], "offline_fallback")
        self.assertEqual(summary["model_type"], "MLP")
        self.assertEqual(summary["architecture"]["layers"], 4)
        self.assertEqual(summary["architecture"]["activation"], "tanh")
        self.assertEqual(summary["architecture"]["neurons_per_layer"], [64, 64, 64, 1])
        self.assertEqual(summary["optimizer"]["type"], "Adam")
        self.assertEqual(summary["optimizer"]["lr"], 0.001)
        self.assertEqual(summary["test_function"], "sin(x)")
        self.assertEqual(summary["entry_point"], "main.py")
        self.assertTrue(any(t["name"] == "mse" for t in summary["loss"]["terms"]))
        names = {h["name"] for h in summary["hyperparameters"]}
        self.assertEqual(names, {"LR", "NUM_LAYERS", "HIDDEN_DIM", "EPOCHS"})
        self.assertEqual(len(summary["data_flow"]), 6)  # x, y, 3 hidden links, y_pred

    def test_hyperparameter_locations(self):
        params = {h["name"]: h for h in extract_hyperparameters({"main.py": DEMO_MAIN})}
        self.assertEqual(params["LR"]["confidence"], "high")
        self.assertIn("Config.LR", params["LR"]["location"])
        self.assertEqual(params["EPOCHS"]["value"], 1000)

    def test_prompt_template_used(self):
        analyzer = LLMAnalyzer(prompts_dir=SKILL_ROOT / "prompts")
        prompt = analyzer.build_structure_prompt({"main.py": DEMO_MAIN})
        self.assertIn("model_type", prompt)
        self.assertIn("class Config", prompt)
        self.assertNotIn("{{CODE_CONTENT}}", prompt)


if __name__ == "__main__":
    unittest.main()
