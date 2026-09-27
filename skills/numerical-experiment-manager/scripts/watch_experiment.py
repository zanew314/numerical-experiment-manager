#!/usr/bin/env python3
"""Run a numerical experiment, record wall-clock loss, and stop it early.

The stopper compares the run's best-so-far error with a fixed baseline at each
integer wall-clock second and stops when the relative error stays above the
tolerance for several seconds (a local-optimum signature), becomes non-finite, or
turns catastrophic. A run stopped that way is retrained at most once.

Usage:
    python watch_experiment.py <project_root> --run-id run_001 --cmd "python main.py" \
        --baseline .nems/baseline/loss_time.jsonl --tolerance 0.10 --sustain 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

DEFAULT_LOSS_RE = re.compile(
    r"(?:loss|error)\s*[=:]\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)", re.I)


class JsonlTail:
    """Incremental reader for loss_time.jsonl-style files."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.points = []

    def poll(self):
        if not self.path.is_file():
            return
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.offset)
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t, loss = obj.get("wall_clock"), obj.get("loss")
                    if isinstance(t, (int, float)) and isinstance(loss, (int, float)):
                        self.points.append((float(t), float(loss), obj.get("epoch")))
                self.offset = handle.tell()
        except OSError:
            return self.points


def best_so_far(points, second):
    """Return (best RMSE at or before `second`, non_finite, best epoch)."""
    best = math.inf
    non_finite = False
    epoch = None
    for t, loss, ep in points:
        if t > second:
            continue
        if loss is None or loss != loss or loss < 0:
            non_finite = True
        else:
            value = math.sqrt(loss)
            if value < best:
                best = value
        if ep is not None and (epoch is None or ep > epoch):
            epoch = ep
    return (None if best == math.inf else best), non_finite, epoch


def _copy_retry(src: Path, dst: Path, tries: int = 15):
    for attempt in range(tries):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError:
            time.sleep(0.2)
    shutil.copy2(src, dst)


def _remove_retry(path: Path, tries: int = 15):
    for attempt in range(tries):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.2)


def load_series(path: Path):
    tail = JsonlTail(path)
    tail.poll()
    return tail.points


def run_attempt(project_root: Path, run_dir: Path, args, attempt: int, seed: int) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NEMS_OUTPUT_DIR"] = str(run_dir)
    env["NEMS_SEED"] = str(seed)
    env["NEMS_RETRAIN_ATTEMPT"] = str(attempt)
    if args.max_wall_clock is not None:
        env["NEMS_MAX_WALL_CLOCK"] = str(args.max_wall_clock)

    log_path = run_dir / "run.log"
    log = open(log_path, "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        args.cmd, cwd=str(project_root), shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )

    start = time.perf_counter()
    stdout_points = []
    lock = threading.Lock()
    pattern = re.compile(args.loss_regex) if args.loss_regex else DEFAULT_LOSS_RE

    def reader():
        try:
            for line in proc.stdout:
                try:
                    log.write(line)
                    log.flush()
                except (ValueError, OSError):
                    break
                if args.loss_source == "stdout_regex":
                    match = pattern.search(line)
                    if match:
                        with lock:
                            stdout_points.append(
                                (time.perf_counter() - start, float(match.group(1)), None))
        except (ValueError, OSError):
            pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    baseline = load_series(args.baseline) if args.baseline else []
    tail = None
    if args.loss_source == "jsonl":
        tail = JsonlTail(run_dir / "loss_time.jsonl")
    elif args.loss_source == "sidecar" and args.loss_file:
        tail = JsonlTail(Path(args.loss_file))

    def current_points():
        if tail is not None:
            tail.poll()
            return list(tail.points)
        with lock:
            return list(stdout_points)

    trigger, at_second, ratio = None, None, None
    consecutive = 0
    last_second = 0
    wall_clock_guard = args.max_wall_clock

    def evaluate(points, second, stopped_already):
        nonlocal consecutive, trigger, at_second, ratio
        err_exp, non_finite, exp_ep = best_so_far(points, second)
        if non_finite:
            trigger, at_second = "non_finite", second
            return True
        if stopped_already or not baseline or err_exp is None:
            return stopped_already
        err_base, _, base_ep = best_so_far(baseline, second)
        if err_base is None or err_base <= 0:
            return stopped_already
        current_ratio = err_exp / err_base
        if current_ratio > args.catastrophic:
            trigger, at_second, ratio = "catastrophic", second, current_ratio
            return True
        if second < args.warm_up:
            return stopped_already
        if base_ep and exp_ep is not None and exp_ep < 0.5 * base_ep:
            return stopped_already  # speed guard
        if current_ratio > 1.0 + args.tolerance:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= args.sustain:
            trigger, at_second, ratio = "threshold", second, current_ratio
            return True
        return stopped_already

    stopped = False
    while proc.poll() is None:
        time.sleep(0.2)
        elapsed = time.perf_counter() - start
        second = max(1, int(math.ceil(elapsed)))
        points = current_points()
        for s in range(last_second + 1, second + 1):
            if evaluate(points, s, stopped):
                stopped = True
                break
        last_second = max(last_second, second)
        if stopped:
            proc.terminate()
            break
        if wall_clock_guard is not None and elapsed > wall_clock_guard + 5:
            trigger, at_second = "max_wall_clock", second
            proc.terminate()
            stopped = True
            break

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    if proc.stdout is not None:
        try:
            proc.stdout.close()
        except OSError:
            pass
    thread.join(timeout=5)
    log.close()

    points = current_points()
    if not stopped:
        elapsed = time.perf_counter() - start
        evaluate(points, max(1, int(math.ceil(elapsed))), False)

    if args.loss_source == "stdout_regex" and points:
        loss_path = run_dir / "loss_time.jsonl"
        with open(loss_path, "w", encoding="utf-8") as handle:
            for index, (t, loss, ep) in enumerate(points, start=1):
                handle.write(json.dumps(
                    {"wall_clock": t, "loss": loss, "epoch": ep if ep is not None else index}) + "\n")

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
    else:
        metrics = {}
    if points and "final_loss" not in metrics:
        last = points[-1][1]
        metrics.update({
            "final_loss": last,
            "final_error": math.sqrt(last) if last >= 0 else None,
            "total_wall_clock": points[-1][0],
            "epochs": len(points),
        })
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if stopped and trigger:
        (run_dir / "early_stop.json").write_text(json.dumps({
            "stopped": True, "trigger_kind": trigger,
            "at_second": at_second, "ratio": ratio,
        }, indent=2) + "\n", encoding="utf-8")

    return {"attempt": attempt, "stopped": stopped, "trigger": trigger,
            "at_second": at_second, "ratio": ratio, "metrics": metrics}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run and monitor an experiment.")
    parser.add_argument("project_root", nargs="?", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cmd", required=True, help="command to run the experiment")
    parser.add_argument("--baseline", default=None, help="baseline loss_time.jsonl")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--sustain", type=int, default=3)
    parser.add_argument("--warm-up", type=int, default=2)
    parser.add_argument("--catastrophic", type=float, default=2.0)
    parser.add_argument("--max-retrains", type=int, default=1)
    parser.add_argument("--loss-source", choices=["jsonl", "stdout_regex", "sidecar", "completion_only"],
                        default="jsonl")
    parser.add_argument("--loss-file", default=None, help="sidecar loss file")
    parser.add_argument("--loss-regex", default=None, help="override the stdout loss regex")
    parser.add_argument("--max-wall-clock", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    run_dir = project_root / ".nems" / "runs" / args.run_id
    attempts = []
    max_attempt = args.max_retrains if args.loss_source != "completion_only" else 0

    result = run_attempt(project_root, run_dir, args, attempt=0, seed=args.seed)
    attempts.append(result)

    retrain_triggers = {"non_finite", "catastrophic", "threshold"}
    if result["stopped"] and result["trigger"] in retrain_triggers and max_attempt >= 1:
        preserve = run_dir / "attempts" / "attempt_1"
        preserve.mkdir(parents=True, exist_ok=True)
        for name in ("loss_time.jsonl", "metrics.json", "early_stop.json", "run.log"):
            src = run_dir / name
            if src.is_file():
                _copy_retry(src, preserve / name)
                _remove_retry(src)
        result = run_attempt(project_root, run_dir, args, attempt=1, seed=args.seed + 1)
        attempts.append(result)

    (run_dir / "result.json").write_text(json.dumps({
        "run_id": args.run_id,
        "loss_source": args.loss_source,
        "attempts": attempts,
        "final": attempts[-1],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "stopped": attempts[-1]["stopped"],
        "trigger": attempts[-1]["trigger"],
        "at_second": attempts[-1]["at_second"],
        "ratio": attempts[-1]["ratio"],
        "retrained": len(attempts) > 1,
        "metrics": attempts[-1]["metrics"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
