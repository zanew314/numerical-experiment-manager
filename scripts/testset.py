#!/usr/bin/env python3
"""List the skill's equation cards, validate cases, and aggregate run results.

The equation bank lives under ``references/test_equations`` as one Markdown card
per equation family, each with YAML front matter. A managed project materializes
those cards into ``.nems/testset/<case_id>/case.json`` and runs each case with
``watch_experiment.py``. This script does the deterministic bookkeeping only:
parse cards, check a case against the contract, and reduce the per-case run
artifacts into one grouped summary. It never runs the managed program and never
edits the project's source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "nems-testset-v1"
SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BANK = SKILL_ROOT / "references" / "test_equations"
DEFAULT_TESTSET = Path(".nems") / "testset"
INDEX_FILENAME = "INDEX.md"
ALLOWED_GROUPS = ("general", "pde")
MAX_CASE_WALL_CLOCK = 300.0
CARD_REQUIRED_FIELDS = ("title", "card_id", "group")
CASE_REQUIRED_FIELDS = ("case_id", "card", "group")
RESULT_NAMES = ("result.json", "metrics.json")
LOSS_FILENAME = "loss_time.jsonl"

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_KEY_VALUE_RE = re.compile(r"^([A-Za-z0-9_]+)\s*:\s*(.*)$")


class InputError(ValueError):
    """Raised when a card or case file is not a valid runner input."""


def _strip_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_strip_quotes(item) for item in inner.split(",")]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return _strip_quotes(text)


def _parse_front_matter(text: str, label: str) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        raise InputError(f"{label} is missing YAML front matter")
    metadata: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        entry = _KEY_VALUE_RE.match(line)
        if entry is None:
            raise InputError(
                f"{label} has an unparsable front-matter line: {line!r}"
            )
        metadata[entry.group(1)] = _coerce_scalar(entry.group(2))
    return metadata, text[match.end():]


def load_card(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read card {path}: {exc}") from exc

    metadata, body = _parse_front_matter(text, f"card {path.name}")
    missing = [field for field in CARD_REQUIRED_FIELDS if not metadata.get(field)]
    if missing:
        raise InputError(
            f"card {path.name} is missing front-matter field(s): "
            f"{', '.join(missing)}"
        )
    group = str(metadata["group"])
    if group not in ALLOWED_GROUPS:
        raise InputError(
            f"card {path.name} group must be one of {', '.join(ALLOWED_GROUPS)}"
        )

    card = dict(metadata)
    card["card_id"] = str(metadata["card_id"])
    card["group"] = group
    card["path"] = path.name
    card["has_prompt_body"] = "## Prompt Body" in body
    card["has_expected_signals"] = "## Expected Modeling Signals" in body
    return card


def iter_cards(bank: Path) -> list[dict[str, Any]]:
    if not bank.is_dir():
        raise InputError(f"test-equation bank not found: {bank}")
    cards = [
        load_card(path)
        for path in sorted(bank.glob("*.md"))
        if path.name != INDEX_FILENAME
    ]
    if not cards:
        raise InputError(f"no equation cards found under {bank}")
    return cards


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}") from exc


def _string_list(raw: Any, label: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InputError(f"{label} must be a JSON array of strings")
    return [str(item) for item in raw]


def validate_case(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InputError("the case root must be a JSON object")
    missing = [field for field in CASE_REQUIRED_FIELDS if not raw.get(field)]
    if missing:
        raise InputError(
            f"case is missing required field(s): {', '.join(missing)}"
        )

    group = str(raw["group"])
    if group not in ALLOWED_GROUPS:
        raise InputError(f"case group must be one of {', '.join(ALLOWED_GROUPS)}")

    difficulty = _string_list(raw.get("difficulty"), "difficulty")
    reference = raw.get("reference") or {}
    if not isinstance(reference, dict):
        raise InputError("reference must be a JSON object")
    run = raw.get("run") or {}
    if not isinstance(run, dict):
        raise InputError("run must be a JSON object")

    max_wall_clock = run.get("max_wall_clock", 60)
    if isinstance(max_wall_clock, bool) or not isinstance(
        max_wall_clock, (int, float)
    ):
        raise InputError("run.max_wall_clock must be numeric")
    if not math.isfinite(float(max_wall_clock)) or max_wall_clock <= 0:
        raise InputError("run.max_wall_clock must be finite and positive")
    if max_wall_clock > MAX_CASE_WALL_CLOCK:
        raise InputError(
            f"run.max_wall_clock {max_wall_clock} exceeds the case limit "
            f"{MAX_CASE_WALL_CLOCK:g}; workflow C cases must stay short"
        )

    return {
        "case_id": str(raw["case_id"]),
        "card": str(raw["card"]),
        "group": group,
        "difficulty": difficulty,
        "equation": str(raw.get("equation", "")),
        "domain": raw.get("domain"),
        "initial_condition": raw.get("initial_condition"),
        "boundary_condition": raw.get("boundary_condition"),
        "coefficients": raw.get("coefficients") or {},
        "reference": {
            "kind": str(reference.get("kind", "unknown")),
            "expression": reference.get("expression"),
            "path": reference.get("path"),
        },
        "input": raw.get("input") or {},
        "run": {
            "command": str(run.get("command", "")),
            "max_wall_clock": float(max_wall_clock),
            "iterations": run.get("iterations"),
        },
        "tolerance": float(raw.get("tolerance", 0.10)),
        "structural_checks": {
            "known_group": True,
            "has_difficulty_axis": bool(difficulty),
            "has_reference_kind": (
                str(reference.get("kind", "unknown")) != "unknown"
            ),
            "has_run_command": bool(run.get("command")),
            "bounded_wall_clock": max_wall_clock <= MAX_CASE_WALL_CLOCK,
        },
    }


def load_case(path: Path) -> dict[str, Any]:
    return validate_case(_read_json(path))


def scaffold_case(card: dict[str, Any]) -> dict[str, Any]:
    card_id = str(card["card_id"])
    return {
        "case_id": f"{card_id}-001",
        "card": card_id,
        "group": card["group"],
        "difficulty": _string_list(card.get("difficulty"), "difficulty"),
        "equation": "<fill from the card Prompt Body>",
        "domain": {"<axis>": ["<lower>", "<upper>"]},
        "initial_condition": "<fill or null>",
        "boundary_condition": "<fill or null>",
        "coefficients": {"<name>": "<value>"},
        "reference": {
            "kind": str(card.get("reference_kind", "numeric")),
            "expression": "<fill when an exact solution exists>",
            "path": None,
        },
        "input": {
            "encoding": "<how the model receives the equation, from the card>",
            "points": "<shape the model consumes>",
        },
        "run": {
            "command": "<the concrete command derived from the model interface>",
            "max_wall_clock": 60,
            "iterations": "<small iteration budget>",
        },
        "tolerance": 0.10,
    }


def _best_error(loss_path: Path) -> float | None:
    if not loss_path.is_file():
        return None
    best = math.inf
    try:
        with open(loss_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    point = json.loads(line)
                except json.JSONDecodeError:
                    continue
                loss = point.get("loss")
                if isinstance(loss, (int, float)) and loss >= 0:
                    best = min(best, math.sqrt(float(loss)))
    except OSError:
        return None
    return None if best == math.inf else best


def _case_metadata(case_dir: Path) -> dict[str, Any]:
    case_path = case_dir / "case.json"
    if not case_path.is_file():
        return {}
    try:
        return validate_case(_read_json(case_path))
    except InputError:
        return {}


def _case_record(case_dir: Path) -> dict[str, Any]:
    metadata = _case_metadata(case_dir)
    record: dict[str, Any] = {
        "case_id": case_dir.name,
        "card": metadata.get("card"),
        "group": metadata.get("group", "unknown"),
        "difficulty": metadata.get("difficulty", []),
        "best_error": _best_error(case_dir / LOSS_FILENAME),
        "final_error": None,
        "stopped": None,
        "trigger": None,
        "has_result": False,
    }

    payload: dict[str, Any] = {}
    for name in RESULT_NAMES:
        path = case_dir / name
        if path.is_file():
            try:
                loaded = _read_json(path)
            except InputError:
                continue
            if isinstance(loaded, dict):
                payload = loaded
            break

    if payload:
        record["has_result"] = True
        raw_final = payload.get("final")
        final = raw_final if isinstance(raw_final, dict) else payload
        record["stopped"] = final.get("stopped")
        record["trigger"] = final.get("trigger") or final.get("trigger_kind")
        metrics = (
            final.get("metrics")
            if isinstance(final.get("metrics"), dict)
            else final
        )
        if isinstance(metrics, dict):
            if isinstance(metrics.get("final_error"), (int, float)):
                record["final_error"] = float(metrics["final_error"])
            elif (
                isinstance(metrics.get("final_loss"), (int, float))
                and metrics["final_loss"] >= 0
            ):
                record["final_error"] = math.sqrt(float(metrics["final_loss"]))
    return record


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        record["best_error"]
        for record in records
        if isinstance(record["best_error"], (int, float))
    ]
    if not errors:
        return {
            "cases": len(records),
            "scored": 0,
            "best_error": None,
            "worst_error": None,
            "mean_error": None,
        }
    return {
        "cases": len(records),
        "scored": len(errors),
        "best_error": min(errors),
        "worst_error": max(errors),
        "mean_error": sum(errors) / len(errors),
    }


def aggregate(project_root: Path, testset_dir: Path) -> dict[str, Any]:
    if not testset_dir.is_dir():
        raise InputError(f"testset directory not found: {testset_dir}")

    index_cases: list[str] = []
    index_path = testset_dir / "index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        if isinstance(index, dict):
            index_cases = _string_list(index.get("cases"), "index.cases")

    records = [
        _case_record(child)
        for child in sorted(testset_dir.iterdir())
        if child.is_dir()
    ]

    groups: dict[str, list[dict[str, Any]]] = {}
    axes: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["group"]), []).append(record)
        for axis in record["difficulty"]:
            axes.setdefault(str(axis), []).append(record)

    present = {record["case_id"] for record in records}
    missing = [case_id for case_id in index_cases if case_id not in present]
    scored = [record for record in records if record["has_result"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(project_root),
        "testset": str(testset_dir),
        "totals": {
            "cases": len(records),
            "with_result": len(scored),
            "missing": len(missing),
        },
        "groups": {name: _summarize(items) for name, items in sorted(groups.items())},
        "difficulty": {name: _summarize(items) for name, items in sorted(axes.items())},
        "cases": sorted(records, key=lambda record: (
            str(record["group"]), str(record["case_id"]))),
        "missing_cases": missing,
        "note": (
            "best_error is best-so-far RMSE from loss_time.jsonl; compare cases "
            "grouped by group and difficulty axis before drawing conclusions."
        ),
    }


def _write_report(
    report: dict[str, Any],
    output: Path | None,
    indent: int,
) -> None:
    rendered = json.dumps(report, indent=indent, sort_keys=True, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, output)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    print(rendered)


def _error_report(kind: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "error": {"kind": kind, "message": message},
    }


def _command_list(args: argparse.Namespace) -> int:
    cards = iter_cards(args.bank)
    report = {
        "schema_version": SCHEMA_VERSION,
        "bank": str(args.bank),
        "count": len(cards),
        "groups": {
            group: [card["card_id"] for card in cards if card["group"] == group]
            for group in ALLOWED_GROUPS
        },
        "cards": cards,
    }
    _write_report(report, args.output, args.indent)
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    case = load_case(args.case)
    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "case": case,
    }
    _write_report(report, args.output, args.indent)
    return 0


def _command_scaffold(args: argparse.Namespace) -> int:
    cards = {card["card_id"]: card for card in iter_cards(args.bank)}
    card = cards.get(args.card_id)
    if card is None:
        raise InputError(
            f"unknown card {args.card_id!r}; known cards: "
            f"{', '.join(sorted(cards))}"
        )
    _write_report(scaffold_case(card), args.out, args.indent)
    return 0


def _command_aggregate(args: argparse.Namespace) -> int:
    project_root = args.project_root.expanduser().resolve(strict=False)
    testset_dir = (project_root / args.testset).resolve(strict=False)
    report = aggregate(project_root, testset_dir)
    _write_report(report, args.output, args.indent)
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="list the equation cards")
    listing.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    listing.add_argument("--output", type=Path, default=None)
    listing.set_defaults(handler=_command_list)

    validate = subparsers.add_parser("validate", help="validate one case file")
    validate.add_argument("case", type=Path)
    validate.add_argument("--output", type=Path, default=None)
    validate.set_defaults(handler=_command_validate)

    scaffold = subparsers.add_parser(
        "scaffold", help="write a case template for one card"
    )
    scaffold.add_argument("card_id")
    scaffold.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    scaffold.add_argument("--out", type=Path, default=None)
    scaffold.set_defaults(handler=_command_scaffold)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="summarize per-case run artifacts"
    )
    aggregate_parser.add_argument(
        "project_root", nargs="?", type=Path, default=Path(".")
    )
    aggregate_parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    aggregate_parser.add_argument("--output", type=Path, default=None)
    aggregate_parser.set_defaults(handler=_command_aggregate)

    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be nonnegative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.handler(args)
    except InputError as exc:
        print(
            json.dumps(_error_report("input", str(exc)), indent=args.indent),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
