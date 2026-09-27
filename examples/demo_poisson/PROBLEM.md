# 2D Poisson: PINN and a traditional finite-difference solver

## Problem

Find `u` on `Omega = [-1, 1]^2` such that

```
-Delta u = f        in Omega
         u = 0      on the boundary
```

with the manufactured exact solution

```
u*(x1, x2) = sin(freq*x1) * sin(freq*x2)
f(x1, x2)  = 2 * freq^2 * u*(x1, x2)
```

so that `-Delta u* = f` exactly and `u* = 0` on `∂Omega` (because
`sin(±freq) = 0` for integer multiples of `pi`).

Source of the problem setup (re-implemented here; no upstream code copied):
`yaohua32/Physics-Driven-Deep-Learning-for-PDEs`,
`Examples/01_Poisson` and `Problems/Poisson/Poisson_2d.py`. The repository
notebooks use `freq = 4*pi`; the `Problem` class takes `freq` as a parameter and
its own self-test uses `freq = pi`.

### Frequency choice in this demo

This example runs `freq = pi`. The repository's `4*pi` target is much harder for
a small CPU network: with the same `[2, 32, 32, 32, 1]` tanh MLP and a short
budget, training at `4*pi` barely moves (observed L2 error stays near 1),
whereas `pi` converges to a small L2 error. Reproducing `4*pi` needs a wider or
deeper network and/or a longer schedule; `freq` is exposed as a hyperparameter so
that experiment is a natural extension.

## Solvers of the same problem

### `main.py` — physics-informed neural network (PINN)

- Network: fully connected `tanh` MLP, default `2 -> 32 -> 32 -> 32 -> 1`.
- Boundary handling: the output is multiplied by the mollifier
  `sin(pi*x1) sin(pi*x2)`, so `u = 0` on the boundary exactly.
- Loss: mean square of the PDE residual `-Delta u_theta - f`, with the Laplacian
  evaluated by autograd (second derivatives).
- The manager runs this program and reads `loss_time.jsonl`; `metrics.json` also
  records the final L2 relative error against `u*`.

### `traditional_solver.py` — classical finite difference

- Uniform grid with `n` interior nodes per axis, spacing `h = 2/(n+1)`.
- Second-order 5-point stencil for `-Delta u`, zero Dirichlet boundary.
- Matrix-free conjugate gradient (numpy only; no scipy) to a tight tolerance.
- Reports the CG residual (linear-solve accuracy) separately from the L2 / Linf
  error against `u*` (discretization accuracy).

The sampled sine solution is a discrete Laplacian eigenvector, so the discrete
system is solved to machine precision by CG and the remaining error is the
second-order amplitude error. Observed order is 2.0, e.g. L2 relative error
`2.0e-4` at `n = 127`.

### `particle_wnn.py` — ParticleWNN (weak form)

Reproduces `Examples/01_Poisson/01_Poisson_ParticleWNN.ipynb` (re-implemented;
no upstream code copied):

- Network: fully connected `2 -> 40 -> 40 -> 40 -> 1` with the repository's
  `Tanh_Sin` activation `a(x) = tanh(sin(pi*(x+1))) + x`.
- Weak form: for random centers `xc` with radius `R` and a compactly supported
  Wendland test function `v` (gradient `dv`) on a local grid, enforce
  `<grad u, grad v> = <f, v>` per center.
- Boundary: the mollifier `sin(pi*x1) sin(pi*x2)` makes `u = 0` exact.
- Loss: `5 * ||left - right||_2 / sqrt(n_center)` (`lp_abs`), Adam with
  `weight_decay = 1e-4` and a `StepLR` schedule.
- The manager runs this program and reads `loss_time.jsonl`; `metrics.json` also
  records the final L2 relative error against `u*`.

## Results on the same PDE

| solver | setting | L2 rel. error vs `u*` | wall-clock |
| --- | --- | --- | --- |
| finite difference | `n=127`, freq=4π | `3.2e-3` | < 1 ms |
| ParticleWNN | freq=π, 256 centers, 300 epochs | `1.5e-3` | ~7.5 s |
| ParticleWNN | freq=4π, 512 centers, 1200 epochs | `5.2e-2` | ~52 s |
| PINN (small) | freq=π, 600 epochs | `4.8e-4` | ~2 s |
| PINN (small) | freq=4π, 600 epochs | `~1.0` (fails) | ~2 s |

At the repository's `4π` target, the weak-form ParticleWNN is far more capable
than the small strong-form PINN, at the cost of a larger budget.

## Why both

The finite-difference solver is a deterministic numerical reference for the same
PDE. It lets the manager's PINN experiments be judged against a known, verifiable
accuracy at a fixed budget, and it demonstrates that the same project can hold
both a learned solver and a classical one (the model-structure snapshot records
both).

## Run

```bash
python examples/demo_poisson/run_demo.py            # full managed lifecycle
python examples/demo_poisson/traditional_solver.py --n 127
python examples/demo_poisson/main.py                 # PINN standalone (no NEMS_OUTPUT_DIR)
python examples/demo_poisson/particle_wnn.py         # ParticleWNN standalone
```
