from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

TOKEN = "TO" + "DO"


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
BANK = ROOT / "references" / "test_equations"
TESTSET_SCRIPT = SCRIPTS / "testset.py"
SKILL = ROOT / "SKILL.md"
REFERENCE = ROOT / "references" / "testset.md"
PROMPT = ROOT / "prompts" / "model_diagnosis.md"


def load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TESTSET_MODULE = load_script_module("nems_testset_release_gate", TESTSET_SCRIPT)

EXPECTED_CARDS = {
    "linear-ode": "general",
    "stiff-ode-robertson": "general",
    "harmonic-oscillator": "general",
    "poisson-2d": "general",
    "nonlinear-system": "general",
    "advection-diffusion": "general",
    "heat-equation": "pde",
    "viscous-burgers": "pde",
    "allen-cahn": "pde",
    "darcy-flow": "pde",
    "wave-equation": "pde",
    "navier-stokes-2d": "pde",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_case(case_id: str, card: str, group: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "card": card,
        "group": group,
        "difficulty": ["elliptic"],
        "equation": "u'' = 0",
        "reference": {"kind": "exact", "expression": "u = x"},
        "run": {"command": "python main.py", "max_wall_clock": 60},
        "tolerance": 0.10,
    }


class TestsetSkillTests(unittest.TestCase):
    def run_json(self, *args: str) -> tuple[subprocess.CompletedProcess[str], dict]:
        result = subprocess.run(
            [sys.executable, str(TESTSET_SCRIPT), *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stream = result.stdout if result.stdout.strip() else result.stderr
        return result, json.loads(stream)

    def test_skill_resources_exist_without_placeholders(self) -> None:
        expected = [
            SKILL,
            REFERENCE,
            PROMPT,
            TESTSET_SCRIPT,
            BANK / "INDEX.md",
        ]
        expected += [BANK / f"{card}.md" for card in EXPECTED_CARDS]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)
                self.assertNotIn(TOKEN, path.read_text(encoding="utf-8"))

    def test_bank_has_twelve_cards_grouped_six_and_six(self) -> None:
        cards = TESTSET_MODULE.iter_cards(BANK)
        observed = {card["card_id"]: card["group"] for card in cards}
        self.assertEqual(EXPECTED_CARDS, observed)
        self.assertEqual(
            6, sum(1 for group in observed.values() if group == "general")
        )
        self.assertEqual(6, sum(1 for group in observed.values() if group == "pde"))
        for card in cards:
            with self.subTest(card=card["card_id"]):
                self.assertTrue(card["has_prompt_body"])
                self.assertTrue(card["has_expected_signals"])
                self.assertTrue(card["difficulty"])
                self.assertIn(card["group"], TESTSET_MODULE.ALLOWED_GROUPS)

    def test_validate_case_normalizes_and_checks(self) -> None:
        case = TESTSET_MODULE.validate_case(
            valid_case("poisson-2d-001", "poisson-2d", "general")
        )
        self.assertEqual("poisson-2d-001", case["case_id"])
        self.assertTrue(case["structural_checks"]["known_group"])
        self.assertTrue(case["structural_checks"]["has_run_command"])
        self.assertTrue(case["structural_checks"]["bounded_wall_clock"])

    def test_validate_case_rejects_missing_and_unknown_group(self) -> None:
        with self.assertRaises(TESTSET_MODULE.InputError):
            TESTSET_MODULE.validate_case({"case_id": "x"})
        with self.assertRaises(TESTSET_MODULE.InputError):
            TESTSET_MODULE.validate_case(
                valid_case("x", "linear-ode", "nonsense")
            )

    def test_validate_case_rejects_unbounded_wall_clock(self) -> None:
        case = valid_case("x", "linear-ode", "general")
        case["run"]["max_wall_clock"] = 10000
        with self.assertRaises(TESTSET_MODULE.InputError):
            TESTSET_MODULE.validate_case(case)

    def test_load_case_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(
                '{"case_id":"a","case_id":"b","card":"x","group":"general"}',
                encoding="utf-8",
            )
            with self.assertRaises(TESTSET_MODULE.InputError):
                TESTSET_MODULE.load_case(path)

    def test_load_card_requires_front_matter_and_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_group = Path(directory) / "broken.md"
            missing_group.write_text(
                "---\ntitle: X\ncard_id: x\n---\n\n# X\n", encoding="utf-8"
            )
            with self.assertRaises(TESTSET_MODULE.InputError):
                TESTSET_MODULE.load_card(missing_group)

            no_front_matter = Path(directory) / "nofm.md"
            no_front_matter.write_text("# X\n", encoding="utf-8")
            with self.assertRaises(TESTSET_MODULE.InputError):
                TESTSET_MODULE.load_card(no_front_matter)

    def test_scaffold_case_is_valid_and_uses_card(self) -> None:
        cards = {card["card_id"]: card for card in TESTSET_MODULE.iter_cards(BANK)}
        case = TESTSET_MODULE.scaffold_case(cards["darcy-flow"])
        self.assertEqual("darcy-flow", case["card"])
        self.assertEqual("pde", case["group"])
        TESTSET_MODULE.validate_case(case)

    def test_best_error_uses_best_so_far_rmse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loss = Path(directory) / "loss_time.jsonl"
            loss.write_text(
                "\n".join(
                    [
                        json.dumps({"wall_clock": 1.0, "loss": 4.0}),
                        json.dumps({"wall_clock": 2.0, "loss": 1.0}),
                        json.dumps({"wall_clock": 3.0, "loss": 9.0}),
                        "not json",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(1.0, TESTSET_MODULE._best_error(loss))

    def test_aggregate_groups_results_and_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            testset = project / ".nems" / "testset"
            write_json(testset / "index.json", {"cases": ["case-a", "case-missing"]})

            write_json(
                testset / "case-a" / "case.json",
                valid_case("case-a", "poisson-2d", "general"),
            )
            (testset / "case-a" / "loss_time.jsonl").write_text(
                json.dumps({"wall_clock": 1.0, "loss": 4.0}) + "\n",
                encoding="utf-8",
            )
            write_json(
                testset / "case-a" / "result.json",
                {"final": {"stopped": False, "metrics": {"final_loss": 4.0}}},
            )

            write_json(
                testset / "case-b" / "case.json",
                valid_case("case-b", "viscous-burgers", "pde"),
            )
            (testset / "case-b" / "loss_time.jsonl").write_text(
                json.dumps({"wall_clock": 1.0, "loss": 9.0}) + "\n",
                encoding="utf-8",
            )
            write_json(
                testset / "case-b" / "metrics.json",
                {"final_loss": 9.0, "final_error": 3.0},
            )

            report = TESTSET_MODULE.aggregate(project, testset)
            self.assertEqual(2, report["totals"]["cases"])
            self.assertEqual(["case-missing"], report["missing_cases"])
            self.assertEqual(2.0, report["groups"]["general"]["best_error"])
            self.assertEqual(3.0, report["groups"]["pde"]["best_error"])
            self.assertEqual(2.0, report["difficulty"]["elliptic"]["best_error"])
            self.assertEqual(
                sorted(["case-a", "case-b"]),
                sorted(record["case_id"] for record in report["cases"]),
            )

    def test_cli_list_emits_one_json_document(self) -> None:
        result, report = self.run_json("list")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("", result.stderr)
        self.assertEqual("nems-testset-v1", report["schema_version"])
        self.assertEqual(12, report["count"])
        self.assertEqual(6, len(report["groups"]["general"]))
        self.assertEqual(6, len(report["groups"]["pde"]))

    def test_cli_scaffold_and_validate_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case_path = Path(directory) / "case.json"
            result, report = self.run_json(
                "scaffold", "poisson-2d", "--out", str(case_path)
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(case_path.is_file())

            result, report = self.run_json("validate", str(case_path))
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(report["valid"])

    def test_cli_unknown_card_is_input_error(self) -> None:
        result, report = self.run_json("scaffold", "does-not-exist")
        self.assertEqual(2, result.returncode)
        self.assertEqual("", result.stdout)
        self.assertEqual("input", report["error"]["kind"])

    def test_cli_aggregate_writes_atomic_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            testset = project / ".nems" / "testset"
            write_json(testset / "case-a" / "case.json",
                       valid_case("case-a", "linear-ode", "general"))
            (testset / "case-a" / "loss_time.jsonl").write_text(
                json.dumps({"wall_clock": 1.0, "loss": 1.0}) + "\n",
                encoding="utf-8",
            )
            output = testset / "summary.json"
            result, report = self.run_json(
                "aggregate", str(project), "--output", str(output)
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(report, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual([], list(testset.glob(".summary.json.*.tmp")))

    def test_aggregate_missing_directory_is_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            with self.assertRaises(TESTSET_MODULE.InputError):
                TESTSET_MODULE.aggregate(project, project / ".nems" / "testset")


if __name__ == "__main__":
    unittest.main()
