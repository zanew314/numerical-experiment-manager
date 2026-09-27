#!/usr/bin/env python3
"""Classical finite-difference solver for the 2D Poisson problem.

Solve ``-Delta u = f`` on Omega = [-1, 1]^2 with ``u = 0`` on the boundary using
the second-order 5-point stencil on a uniform grid and a matrix-free
conjugate-gradient solver (numpy only, no scipy). This is the traditional
numerical reference for the PINN in ``main.py``.

Target problem (adapted from the Poisson example of the repository
"Physics-Driven-Deep-Learning-for-PDEs", user ``yaohua32``; re-implemented, no
upstream code copied):

    freq  = 4*pi
    u*(x) = sin(freq*x1) * sin(freq*x2)
    f(x)  = 2 * freq^2 * u*(x),      so -Delta u* = f exactly.

On a uniform grid the sampled sine solution is a discrete Laplacian
eigenvector, so the discrete system is solved to the CG tolerance; the remaining
error against the exact continuum solution is the amplitude discretization error
and converges as O(h^2). The solver reports the CG residual (linear-solve
accuracy) separately from the L2/Linf error (discretization accuracy).

Run:

    python traditional_solver.py --n 64
    python traditional_solver.py --n 127 --output fd.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "nems-traditional-poisson-v1"


def exact_sine(x, y, freq):
    return np.sin(freq * x) * np.sin(freq * y)


def source_sine(x, y, freq):
    return 2.0 * freq ** 2 * exact_sine(x, y, freq)


def _apply_negative_laplacian(u, h):
    """Discrete ``-Delta u`` with zero Dirichlet boundary, 5-point stencil."""
    out = 4.0 * u
    out[1:, :] -= u[:-1, :]
    out[:-1, :] -= u[1:, :]
    out[:, 1:] -= u[:, :-1]
    out[:, :-1] -= u[:, 1:]
    return out / (h * h)


def conjugate_gradient(apply_a, rhs, tol=1e-12, max_iter=None):
    """Matrix-free CG for a symmetric positive-definite operator."""
    n = rhs.size
    max_iter = max_iter or max(1000, 20 * int(math.isqrt(n)) + 1000)
    x = np.zeros_like(rhs)
    r = rhs.copy()
    p = r.copy()
    rs_old = float(np.vdot(r, r))
    rhs_norm = math.sqrt(float(np.vdot(rhs, rhs))) or 1.0
    target = tol * rhs_norm
    iterations = 0
    rs_new = rs_old
    for iterations in range(1, max_iter + 1):
        ap = apply_a(p)
        denom = float(np.vdot(p, ap))
        if denom == 0.0:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = float(np.vdot(r, r))
        if math.sqrt(rs_new) <= target:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, iterations, math.sqrt(rs_new)


def solve_poisson_2d(n=64, freq=4.0 * math.pi, tol=1e-12, max_iter=None):
    """Solve ``-Delta u = f`` and compare against the exact sine solution."""
    if n < 8:
        raise ValueError("n must be at least 8")
    h = 2.0 / (n + 1)
    interior = -1.0 + h * np.arange(1, n + 1)
    xx, yy = np.meshgrid(interior, interior, indexing="ij")

    rhs = source_sine(xx, yy, freq)
    rhs_vector = rhs.reshape(-1)

    def apply_a(vector):
        return _apply_negative_laplacian(vector.reshape(n, n), h).reshape(-1)

    started = time.perf_counter()
    u, iterations, residual = conjugate_gradient(apply_a, rhs_vector, tol=tol, max_iter=max_iter)
    solve_wall_clock = time.perf_counter() - started
    u = u.reshape(n, n)

    exact_values = exact_sine(xx, yy, freq)
    linf_error = float(np.max(np.abs(u - exact_values)))
    denominator = float(np.linalg.norm(exact_values))
    l2_rel_error = float(np.linalg.norm(u - exact_values) / denominator) if denominator else float("nan")
    residual_inf = float(np.max(np.abs(_apply_negative_laplacian(u, h) - rhs)))

    return {
        "schema_version": SCHEMA_VERSION,
        "case": "sine",
        "n": n,
        "h": h,
        "freq": freq,
        "iterations": iterations,
        "converged": residual <= tol * max(1.0, float(np.linalg.norm(rhs_vector))),
        "cg_residual": residual,
        "residual_inf": residual_inf,
        "l2_rel_error": l2_rel_error,
        "linf_error": linf_error,
        "solve_wall_clock": solve_wall_clock,
    }


def observed_order(fine, coarse, ratio=2):
    """Estimate the convergence order from two errors at grid ratio ``ratio``."""
    if fine <= 0.0 or coarse <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64, help="interior nodes per axis")
    parser.add_argument("--freq", type=float, default=4.0 * math.pi, help="sine frequency")
    parser.add_argument("--tol", type=float, default=1e-12, help="CG relative tolerance")
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.n < 8:
        raise SystemExit("--n must be at least 8")
    report = solve_poisson_2d(n=args.n, freq=args.freq, tol=args.tol)
    rendered = json.dumps(report, indent=args.indent, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
