"""Tests for the pluggable loss-stream adapters."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import loss_stream  # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_builtin_sources_are_registered(self):
        names = loss_stream.available_sources()
        for expected in ("jsonl", "stdout_regex", "sidecar", "completion_only"):
            self.assertIn(expected, names)

    def test_build_default_is_jsonl(self):
        source = loss_stream.build_source(None)
        self.assertEqual(source.name, "jsonl")

    def test_unknown_type_names_available_sources(self):
        with self.assertRaises(ValueError) as ctx:
            loss_stream.build_source({"type": "nope"})
        self.assertIn("jsonl", str(ctx.exception))

    def test_save_and_load_spec_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            loss_stream.save_stream_spec(tmp, {"type": "completion_only", "config": {}})
            spec = loss_stream.load_stream_spec(tmp)
            self.assertEqual(spec["type"], "completion_only")

    def test_normalize_point_drops_malformed(self):
        self.assertIsNone(loss_stream.normalize_point({"loss": "x"}))
        self.assertIsNone(loss_stream.normalize_point({"epoch": 1}))
        point = loss_stream.normalize_point({"wall_clock": "2", "loss": "0.25", "epoch": "3"})
        self.assertEqual(point, {"wall_clock": 2.0, "loss": 0.25, "epoch": 3})


class TestStdoutRegexSource(unittest.TestCase):
    def _log(self, tmp, lines):
        exp = Path(tmp)
        (exp / "logs").mkdir(parents=True, exist_ok=True)
        (exp / "logs" / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return exp

    def test_parses_wall_clock_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = self._log(tmp, [
                "epoch=0 loss=1.0 elapsed=0.10",
                "noise line",
                "epoch=1 loss=0.25 elapsed=1.20",
            ])
            source = loss_stream.build_source({
                "type": "stdout_regex",
                "config": {"pattern": r"epoch=(?P<epoch>\d+) loss=(?P<loss>[\d.]+) elapsed=(?P<wall_clock>[\d.]+)"},
            })
            points = source.read_points(exp)
            self.assertEqual(len(points), 2)
            self.assertEqual(points[1]["epoch"], 1)
            self.assertAlmostEqual(points[1]["wall_clock"], 1.2)
            self.assertFalse(source.synthetic_time)

    def test_synthesizes_time_when_group_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = self._log(tmp, ["step loss=0.5", "step loss=0.4"])
            source = loss_stream.build_source({
                "type": "stdout_regex",
                "config": {"pattern": r"step loss=(?P<loss>[\d.]+)", "time_step": 2.0},
            })
            self.assertTrue(source.synthetic_time)
            points = source.read_points(exp)
            self.assertAlmostEqual(points[0]["wall_clock"], 0.0)
            self.assertAlmostEqual(points[1]["wall_clock"], 2.0)


class TestSidecarSource(unittest.TestCase):
    def test_reads_csv_with_column_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            (exp / "history.csv").write_text(
                "t,mse,step\n0.5,0.36,1\n1.5,0.09,2\n", encoding="utf-8"
            )
            source = loss_stream.build_source({
                "type": "sidecar",
                "config": {"path": "history.csv", "format": "csv",
                           "columns": {"wall_clock": "t", "loss": "mse", "epoch": "step"}},
            })
            points = source.read_points(exp)
            self.assertEqual(len(points), 2)
            self.assertAlmostEqual(points[1]["loss"], 0.09)
            self.assertEqual(points[1]["epoch"], 2)

    def test_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            (exp / "points.jsonl").write_text(
                json.dumps({"wall_clock": 1, "loss": 0.04}) + "\n", encoding="utf-8"
            )
            source = loss_stream.build_source({"type": "sidecar", "config": {"path": "points.jsonl", "format": "jsonl"}})
            points = source.read_points(exp)
            self.assertEqual(points[0]["loss"], 0.04)


class TestCompletionOnlySource(unittest.TestCase):
    def test_reads_final_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            (exp / "metrics.json").write_text(json.dumps({"final_loss": 0.01}), encoding="utf-8")
            source = loss_stream.build_source({"type": "completion_only"})
            self.assertFalse(source.live)
            self.assertEqual(source.read_points(exp), [])
            self.assertEqual(source.read_final(exp)["final_loss"], 0.01)


class TestCustomSourceRegistration(unittest.TestCase):
    def test_register_and_build_custom_class(self):
        class AlwaysSource(loss_stream.LossSource):
            name = "always_test"

            def read_points(self, exp_dir, log_path=None):
                return [{"wall_clock": 1.0, "loss": 0.25, "epoch": 0}]

        loss_stream.register_source("always_test", AlwaysSource)
        source = loss_stream.build_source({"type": "always_test"})
        self.assertEqual(source.read_points(".")[0]["loss"], 0.25)

    def test_register_rejects_non_source(self):
        with self.assertRaises(ValueError):
            loss_stream.register_source("bad", object)


if __name__ == "__main__":
    unittest.main()
