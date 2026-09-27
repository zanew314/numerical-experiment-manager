#!/usr/bin/env python3
"""Scan a Python project and emit a deterministic structure summary as JSON.

Uses the offline analyzer (no network, no LLM key) so the output is reproducible
and dependency-light. This is the small command-line entry point; the full
workflows live in ``agent/`` and the runnable demonstrations in ``examples/``.

    python scripts/analyze_project.py path/to/project
    python scripts/analyze_project.py path/to/project --max-files 40

On success one JSON document is written to stdout. On a bad input one JSON error
document is written to stderr and the exit code is ``2``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import model_structure  # noqa: E402
from agent.code_reader import CodeReader  # noqa: E402
from agent.llm_analyzer import LLMAnalyzer  # noqa: E402

SCHEMA_VERSION = "nems-project-analysis-v1"


def analyze_project(project_root, max_files=None) -> dict:
    root = Path(project_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    reader = CodeReader(root)
    files = reader.read_map(max_files=max_files)

    analyzer = LLMAnalyzer()
    analyzer.api_key = None  # force the deterministic offline analyzer
    summary = analyzer.analyze_structure(files)

    structure = model_structure.extract_model_structure(root, files=files)
    return {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(root),
        "file_count": len(files),
        "files": list(files),
        "structure": model_structure.summarize_model_structure(structure),
        "summary": summary,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", help="directory of the Python project to scan")
    parser.add_argument("--max-files", type=int, default=None, help="cap the number of files read")
    parser.add_argument("--json", action="store_true",
                        help="accepted for symmetry; the report is always JSON")
    args = parser.parse_args(argv)
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be a positive integer")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        report = analyze_project(args.project_root, max_files=args.max_files)
    except (OSError, ValueError) as exc:
        error = {"schema_version": SCHEMA_VERSION,
                 "error": {"kind": type(exc).__name__, "message": str(exc)}}
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
