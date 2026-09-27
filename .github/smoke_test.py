#!/usr/bin/env python3
"""CI smoke test for the structure scripts (standard library only).

Versions a tiny in-memory project, checks the v0001 -> unchanged -> v0002 path,
and diffs the two versions. Run from the repository root.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def run(script: str, *args: str) -> dict:
    output = subprocess.check_output(
        [sys.executable, str(SCRIPTS / script), *args], text=True)
    return json.loads(output)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp)
        (project / "model.py").write_text(
            "HIDDEN = 64\n\n\nclass M:\n    def f(self):\n        return HIDDEN\n",
            encoding="utf-8")
        (project / "main.py").write_text(
            "LR = 0.01\n\n\ndef train(lr=LR):\n    return lr\n", encoding="utf-8")

        first = run("scan_structure.py", str(project))
        assert first["status"] == "created" and first["version"] == "v0001", first

        second = run("scan_structure.py", str(project))
        assert second["status"] == "unchanged", second

        (project / "model.py").write_text(
            "HIDDEN = 128\n\n\nclass M:\n    def f(self):\n        return HIDDEN\n"
            "\n    def g(self):\n        return 0\n",
            encoding="utf-8")
        third = run("scan_structure.py", str(project))
        assert third["version"] == "v0002", third

        delta = run("diff_structure.py", str(project), "--from", "v0001", "--to", "v0002")
        assert delta["from"] == "v0001" and delta["to"] == "v0002", delta
        assert delta["summary"]["structural"] is True, delta

    print("smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
