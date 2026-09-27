"""Run an experiment and record its wall-clock loss against a baseline.

The runner owns one experiment directory, points ``.nems/current_config.json`` at
that experiment's config, launches the project entry point as a subprocess, reads
the loss stream through a ``LossSource`` (default: the ``loss_time.jsonl``
instrumentation file; see ``loss_stream``), and drives the early stopper. Very
fast runs are re-evaluated from the full recorded series after the process exits.

When the early stopper fires because the relative error is too large
(``non_finite`` / ``catastrophic`` / ``threshold``), the run is retrained
automatically -- at most once (``max_retrains``, default 1). A bad wall-clock
result is often caused by an unlucky initialization, so the retrain gets a fresh
process and a shifted seed (``NEMS_SEED``) while keeping the same (or a supplied
``retrain_config``) hyperparameters. If the retrain is still too bad, the run is
recorded as early-stopped and abandoned. Attempt 1 artifacts are preserved under
``experiments/<exp_id>/attempts/``.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import ensure_dir, load_json, nems_dir, save_json
from .early_stopper import EarlyStopper, loss_to_error
from .loss_stream import build_source, load_stream_spec

DEFAULT_MAX_RETRAINS = 1


class ExperimentRunner:
    """Execute experiments with wall-clock loss tracking and early stopping."""

    def __init__(self, project_root, entry="main.py", python=None, timeout=300, monitor_interval=1.0, stream_spec=None):
        self.project_root = Path(project_root).resolve()
        self.entry = entry
        self.python = python or sys.executable
        self.timeout = timeout
        self.monitor_interval = monitor_interval
        self.stream_spec = stream_spec

    # ------------------------------------------------------------------ setup
    def experiments_dir(self) -> Path:
        return ensure_dir(nems_dir(self.project_root) / "experiments")

    def prepare_experiment(self, exp_id, config, metadata=None):
        exp_dir = ensure_dir(self.experiments_dir() / exp_id)
        config_path = exp_dir / "current_config.json"
        save_json(config_path, {"params": config, "exp_id": exp_id})
        save_json(exp_dir / "run_meta.json", {
            "exp_id": exp_id,
            "entry": self.entry,
            "config": config,
            "metadata": metadata or {},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        # fresh loss log
        (exp_dir / "loss_time.jsonl").write_text("", encoding="utf-8")
        ensure_dir(exp_dir / "logs")
        self.point_current_config(config_path)
        return exp_dir

    def point_current_config(self, config_path) -> str:
        """Point ``.nems/current_config.json`` at the experiment config."""
        current = nems_dir(self.project_root) / "current_config.json"
        current.parent.mkdir(parents=True, exist_ok=True)
        if current.exists() or current.is_symlink():
            try:
                current.unlink()
            except OSError:
                pass
        try:
            os.symlink(str(config_path), str(current))
            return "symlink"
        except (OSError, NotImplementedError):
            import shutil
            shutil.copyfile(str(config_path), str(current))
            return "copy"

    # ------------------------------------------------------------- retraining
    def _archive_attempt(self, exp_dir, attempt_no) -> Path:
        """Preserve the artifacts of a finished attempt before retraining."""
        dst = ensure_dir(exp_dir / "attempts" / f"attempt_{attempt_no}")
        for name in ("loss_time.jsonl", "early_stop.json", "metrics.json"):
            src = exp_dir / name
            if src.exists():
                shutil.copyfile(str(src), str(dst / name))
        log_src = exp_dir / "logs" / "run.log"
        if log_src.exists():
            shutil.copyfile(str(log_src), str(dst / "run.log"))
        # clear attempt-scoped outputs so a failed retrain cannot reuse stale data
        for name in ("metrics.json", "early_stop.json"):
            stale = exp_dir / name
            if stale.exists():
                try:
                    stale.unlink()
                except OSError:
                    pass
        return dst

    # -------------------------------------------------------------------- run
    def run(
        self,
        exp_id,
        config,
        baseline_series=None,
        early_stop=True,
        max_wall_clock=None,
        metadata=None,
        timeout=None,
        max_retrains=DEFAULT_MAX_RETRAINS,
        retrain_config=None,
        stream_spec=None,
    ) -> dict:
        exp_dir = self.prepare_experiment(exp_id, config, metadata=metadata)
        loss_path = exp_dir / "loss_time.jsonl"
        log_path = exp_dir / "logs" / "run.log"

        spec = stream_spec if stream_spec is not None else (self.stream_spec or load_stream_spec(self.project_root))
        source = build_source(spec)
        source.prepare(exp_dir)

        max_retrains = max(0, int(max_retrains))
        attempts = []
        retrain_used = 0
        attempt_no = 0
        final = None

        while True:
            attempt_no += 1
            attempt_config = config if attempt_no == 1 else {**config, **(retrain_config or {})}

            if attempt_no > 1:
                self._archive_attempt(exp_dir, attempt_no - 1)
                config_path = exp_dir / "current_config.json"
                save_json(config_path, {"params": attempt_config, "exp_id": exp_id})
                self.point_current_config(config_path)
                loss_path.write_text("", encoding="utf-8")

            env = dict(os.environ)
            env["NEMS_OUTPUT_DIR"] = str(exp_dir)
            env["NEMS_PROJECT_ROOT"] = str(self.project_root)
            env["NEMS_EXP_ID"] = exp_id
            env["NEMS_RETRAIN_ATTEMPT"] = str(attempt_no - 1)
            # keep printed progress unbuffered so a stdout-based source can tail it
            env["PYTHONUNBUFFERED"] = "1"
            if attempt_no > 1:
                # a retrain varies the initialization so an unlucky seed can recover
                env["NEMS_SEED"] = str(attempt_no - 1)
            if max_wall_clock:
                env["NEMS_MAX_WALL_CLOCK"] = str(max_wall_clock)
            else:
                env.pop("NEMS_MAX_WALL_CLOCK", None)

            stopper = None
            if early_stop and baseline_series:
                stopper = EarlyStopper(baseline_series=baseline_series)

            status, returncode, stdout_text, duration = self._run_attempt(
                exp_dir, loss_path, log_path, env, stopper, timeout, source
            )
            series = source.read_points(exp_dir, log_path)
            metrics = load_json(exp_dir / "metrics.json", default={}) or {}
            if not metrics:
                metrics = source.read_final(exp_dir, stdout_text) or {}

            early_report = None
            if stopper is not None:
                early_report = stopper.evaluate_series(series)
                if stopper.triggered and status == "completed":
                    status = "early_stopped"
                save_json(exp_dir / "early_stop.json", early_report)

            if not metrics and series:
                final_loss = series[-1].get("loss")
                metrics = {
                    "final_loss": final_loss,
                    "final_error": loss_to_error(final_loss),
                    "total_wall_clock": series[-1].get("wall_clock"),
                    "epochs": series[-1].get("epoch"),
                }
            metrics.setdefault("exp_id", exp_id)
            metrics.setdefault("attempt", attempt_no)

            attempt = {
                "attempt": attempt_no,
                "config": attempt_config,
                "status": status,
                "returncode": returncode,
                "duration_sec": duration,
                "metrics": metrics,
                "early_stop": early_report,
                "early_stopped": bool(early_report and early_report.get("triggered")),
                "num_points": len(series),
                "log_tail": "\n".join(stdout_text.splitlines()[-20:]),
                "series": series,
            }
            attempts.append(attempt)
            final = attempt

            if attempt["early_stopped"] and retrain_used < max_retrains:
                retrain_used += 1
                continue
            break

        total_duration = sum(a["duration_sec"] for a in attempts)
        record = {
            "exp_id": exp_id,
            "config": config,
            "final_config": final["config"],
            "status": final["status"],
            "returncode": final["returncode"],
            "duration_sec": total_duration,
            "metrics": final["metrics"],
            "early_stop": final["early_stop"],
            "early_stopped": final["early_stopped"],
            "num_points": final["num_points"],
            "stream": source.describe(),
            "retrain": {
                "attempted": retrain_used > 0,
                "count": retrain_used,
                "max_retrains": max_retrains,
                "retrain_config": retrain_config or {},
                "attempts": [
                    {
                        "attempt": a["attempt"],
                        "status": a["status"],
                        "early_stopped": a["early_stopped"],
                        "trigger_kind": (a["early_stop"] or {}).get("trigger_kind"),
                        "trigger_second": (a["early_stop"] or {}).get("trigger_second"),
                        "final_error": (a["metrics"] or {}).get("final_error"),
                        "duration_sec": a["duration_sec"],
                    }
                    for a in attempts
                ],
            },
            "paths": {
                "dir": str(exp_dir),
                "loss_time": str(loss_path),
                "metrics": str(exp_dir / "metrics.json"),
                "log": str(log_path),
                "early_stop": str(exp_dir / "early_stop.json"),
                "attempts": str(exp_dir / "attempts"),
            },
            "series": final["series"],
            "log_tail": final["log_tail"],
        }
        save_json(exp_dir / "result.json", {k: v for k, v in record.items() if k != "series"})
        return record

    def _run_attempt(self, exp_dir, loss_path, log_path, env, stopper, timeout, source):
        """Launch the entry point once and monitor it. Returns attempt stats."""
        started = time.perf_counter()
        status = "completed"
        returncode = None
        stdout_text = ""

        with open(log_path, "w", encoding="utf-8", errors="replace") as log_fh:
            log_fh.write(f"$ {self.python} {self.entry}\n")
            log_fh.flush()
            try:
                process = subprocess.Popen(
                    [self.python, self.entry],
                    cwd=str(self.project_root),
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                status = "failed"
                log_fh.write(f"failed to launch: {exc}\n")
                process = None

            if process is not None:
                seen = 0
                deadline = started + (timeout or self.timeout)
                while process.poll() is None:
                    time.sleep(self.monitor_interval)
                    points = source.read_points(exp_dir, log_path)
                    if stopper is not None:
                        for point in points[seen:]:
                            if stopper.observe(point.get("wall_clock", 0.0), point.get("loss"), point.get("epoch")):
                                status = "early_stopped"
                                self._terminate(process)
                                break
                        seen = len(points)
                    if stopper is not None and stopper.triggered:
                        break
                    if time.perf_counter() > deadline:
                        status = "timeout"
                        self._terminate(process)
                        break
                returncode = process.poll()
                if status == "completed" and returncode not in (0, None):
                    status = "failed"

        duration = time.perf_counter() - started
        try:
            stdout_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stdout_text = ""
        return status, returncode, stdout_text, duration

    @staticmethod
    def _terminate(process) -> None:
        try:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        except OSError:
            pass
