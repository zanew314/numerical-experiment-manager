#!/usr/bin/env python3
"""Probe the Python environment for the numerical-experiment-manager skill.

Required for the agent tools: ``numpy``. Optional (only needed by the bundled
PyTorch demos): ``torch`` and ``matplotlib``.

Prints exactly one JSON document to stdout. The exit code is ``0`` when the
required stack is importable and ``2`` otherwise, so the probe can be used as a
non-mutating gate.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import sys

SCHEMA_VERSION = "nems-environment-v1"
REQUIRED = ("numpy",)
OPTIONAL = ("torch", "matplotlib")
INSTALL_HINT = "python -m pip install numpy"


def probe_package(name: str) -> dict:
    """Return ``{available, version}`` for a package without importing side effects."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return {"available": False, "version": None}
    version = None
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
    except Exception as exc:  # noqa: BLE001 - report, never crash the probe
        return {"available": True, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": version}


def build_report() -> dict:
    packages = {name: probe_package(name) for name in REQUIRED + OPTIONAL}
    ready = all(packages[name]["available"] for name in REQUIRED)
    return {
        "schema_version": SCHEMA_VERSION,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": packages,
        "required": list(REQUIRED),
        "optional": list(OPTIONAL),
        "ready": ready,
        "install_hint": None if ready else INSTALL_HINT,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="accepted for symmetry; the report is always JSON")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    parse_args(argv)
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
