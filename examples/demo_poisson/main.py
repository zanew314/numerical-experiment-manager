#!/usr/bin/env python3
"""Physics-informed neural network for the 2D Poisson problem.

Problem (adapted from the Poisson example of the repository
"Physics-Driven-Deep-Learning-for-PDEs", user ``yaohua32``; re-implemented here,
no upstream code copied):

    -Delta u = f                on Omega = [-1, 1]^2
             u = 0              on dOmega
    freq  = 4*pi
    u*(x) = sin(freq*x1) * sin(freq*x2)
    f(x)  = 2 * freq^2 * u*(x)          (so -Delta u* = f exactly)

The network output is multiplied by the mollifier ``sin(pi*x1) sin(pi*x2)``, so
the zero Dirichlet boundary condition holds exactly. The loss is the mean square
of the PDE residual ``-Delta u_theta - f``, where the Laplacian is evaluated with
autograd.

The experiment manager runs this program with ``NEMS_OUTPUT_DIR`` set. The
program writes ``loss_time.jsonl`` (one JSON object per recorded epoch) and
``metrics.json``, and reads hyperparameters from ``current_config.json`` in
``NEMS_OUTPUT_DIR`` when present. It is CPU-friendly and small on purpose.
"""

import json
import math
import os
import time

import torch
import torch.nn as nn

PI = math.pi


class Config:
    FREQ = 4.0 * PI
    HIDDEN = 32
    DEPTH = 3
    LR = 1e-3
    EPOCHS = 600
    N_COLLOCATION = 600
    N_TEST = 31


def load_params(out_dir):
    """Read the experiment parameters the manager wrote, if any."""
    if not out_dir:
        return {}
    try:
        with open(os.path.join(out_dir, "current_config.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload.get("params", payload) or {}
    except (OSError, ValueError):
        pass
    return {}


def exact_u(x, freq):
    """Exact solution u*(x), x shaped [N, 2]."""
    return torch.sin(freq * x[:, 0:1]) * torch.sin(freq * x[:, 1:2])


def source_f(x, freq):
    """Right-hand side f(x) = 2 freq^2 u*(x)."""
    return 2.0 * freq ** 2 * exact_u(x, freq)


def mollifier(x):
    """Boundary mollifier sin(pi x1) sin(pi x2), zero on dOmega."""
    return torch.sin(PI * x[:, 0:1]) * torch.sin(PI * x[:, 1:2])


class MLP(nn.Module):
    """Fully connected tanh network: 2 -> [hidden]*depth -> 1."""

    def __init__(self, hidden=32, depth=3):
        super().__init__()
        sizes = [2] + [hidden] * depth + [1]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)


def laplacian(u, x):
    """Delta u with autograd (needs second derivatives)."""
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_xx + u_yy


def relative_l2(pred, ref):
    denom = float(torch.linalg.vector_norm(ref))
    if denom == 0.0:
        return float("nan")
    return float(torch.linalg.vector_norm(pred - ref) / denom)


def make_test_grid(n):
    axis = torch.linspace(-1.0, 1.0, n)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def main() -> dict:
    out_dir = os.environ.get("NEMS_OUTPUT_DIR")
    params = load_params(out_dir)
    freq = float(params.get("FREQ", Config.FREQ))
    hidden = int(params.get("HIDDEN", Config.HIDDEN))
    depth = int(params.get("DEPTH", Config.DEPTH))
    lr = float(params.get("LR", Config.LR))
    total_epochs = int(params.get("EPOCHS", Config.EPOCHS))
    n_collocation = int(params.get("N_COLLOCATION", Config.N_COLLOCATION))
    n_test = int(params.get("N_TEST", Config.N_TEST))

    seed = int(os.environ.get("NEMS_SEED", "0"))
    retrain_attempt = int(os.environ.get("NEMS_RETRAIN_ATTEMPT", "0"))
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("NEMS_TORCH_THREADS", "1")))
    device = torch.device("cpu")

    model = MLP(hidden=hidden, depth=depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_test = make_test_grid(n_test).to(device)
    u_test = exact_u(x_test, freq)

    max_wall = float(os.environ.get("NEMS_MAX_WALL_CLOCK", "0") or 0.0)
    record_every = max(1, total_epochs // 100)

    out_loss_path = os.path.join(out_dir, "loss_time.jsonl") if out_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    start = None
    final_loss = float("nan")
    epochs_done = 0
    for epoch in range(total_epochs):
        x = (torch.rand(n_collocation, 2, device=device) * 2.0 - 1.0).requires_grad_(True)
        optimizer.zero_grad()
        u = model(x) * mollifier(x)
        residual = -laplacian(u, x) - source_f(x, freq)
        loss = (residual ** 2).mean()
        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())
        epochs_done = epoch + 1
        if start is None:
            start = time.perf_counter()
            wall = 0.0
        else:
            wall = time.perf_counter() - start
        if out_loss_path and (epoch % record_every == 0 or epoch == total_epochs - 1):
            with open(out_loss_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "wall_clock": wall,
                    "loss": final_loss,
                    "epoch": epoch,
                }) + "\n")
        if max_wall and wall >= max_wall:
            break

    total = time.perf_counter() - start
    with torch.no_grad():
        u_pred = model(x_test) * mollifier(x_test)
        l2_error = relative_l2(u_pred, u_test)
    rmse = math.sqrt(final_loss) if final_loss == final_loss and final_loss >= 0 else final_loss

    metrics = {
        "final_loss": final_loss,
        "final_error": rmse,
        "total_wall_clock": total,
        "epochs": epochs_done,
        "seed": seed,
        "retrain_attempt": retrain_attempt,
        "final_l2_error": l2_error,
        "freq": freq,
        "hidden": hidden,
        "depth": depth,
    }
    if out_dir:
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
    print(f"epochs={epochs_done} pde_loss={final_loss:.6e} l2_err={l2_error:.6e} wall={total:.3f}s")
    return metrics


if __name__ == "__main__":
    main()
