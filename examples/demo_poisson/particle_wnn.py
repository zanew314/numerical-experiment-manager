#!/usr/bin/env python3
"""ParticleWNN (weak-form) solver for the 2D Poisson problem.

Reproduces the method of ``Examples/01_Poisson/01_Poisson_ParticleWNN.ipynb``
from the repository "Physics-Driven-Deep-Learning-for-PDEs" (user ``yaohua32``;
re-implemented here, no upstream code copied).

Same PDE as ``main.py``:

    -Delta u = f                on Omega = [-1, 1]^2
             u = 0              on dOmega
    freq  = 4*pi
    u*(x) = sin(freq*x1) * sin(freq*x2)
    f(x)  = 2 * freq^2 * u*(x)

Method (weak form / "particle" test functions):

- network: fully connected net with the repository's ``Tanh_Sin`` activation,
  ``a(x) = tanh(sin(pi*(x+1))) + x``;
- for random centers ``xc`` with radius ``R`` and a compactly supported Wendland
  test function ``v`` (with gradient ``dv``) on a local grid, enforce the local
  weak form ``<grad u, grad v> = <f, v>``;
- the zero Dirichlet boundary condition is enforced by the mollifier
  ``sin(pi*x1) sin(pi*x2)``;
- the loss is the ``lp_abs`` norm of the per-center weak residual
  (``5 * ||left - right||_2 / sqrt(n_center)``).

The program reads hyperparameters from ``current_config.json`` in
``NEMS_OUTPUT_DIR`` when present, writes ``loss_time.jsonl`` and ``metrics.json``,
and is CPU-friendly.
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
    HIDDEN = 40
    DEPTH = 3
    ACTIVATION = "Tanh_Sin"
    LR = 1e-3
    EPOCHS = 300
    N_CENTER = 256
    RADIUS = 1e-3
    N_MESH = 9
    N_TEST = 31
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 200
    GAMMA = 0.5
    W_PDE = 5.0


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


# ------------------------------------------------------------- test functions
def wendland_test_function(grid):
    """Wendland test function and its gradient on a local grid (dim=2).

    Matches the repository's compactly supported Wendland kernel for order
    ``l = floor(dim/2) + 3``. Returns ``(v, dv)`` with shapes ``(m, 1)`` and
    ``(m, 2)``; both are detached (test functions carry no parameters).
    """
    r = torch.linalg.norm(grid, dim=-1, keepdim=True)
    l = 4
    r2 = r * r
    v = (1 - r) ** (l + 2) * ((l * l + 4 * l + 3) * r2 + (3 * l + 6) * r + 3) / 3.0
    dv_dr_divide_by_r = (
        (1 - r) ** (l + 1) * (-(l ** 3 + 8 * l ** 2 + 19 * l + 12) * r - (l * l + 7 * l + 12)) / 3.0
    )
    dv = dv_dr_divide_by_r * grid
    return v.detach(), dv.detach()


def integral_grid(n_mesh):
    """Mesh grid on the unit disk (in [-1,1]^2) for the local integrals."""
    axis = torch.linspace(-1.0, 1.0, n_mesh)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    return grid[torch.linalg.norm(grid, dim=1) < 1.0]


class TanhSin(nn.Module):
    """The repository's ``Tanh_Sin`` activation: tanh(sin(pi*(x+1))) + x."""

    def forward(self, x):
        return torch.tanh(torch.sin(PI * (x + 1.0))) + x


def make_activation(name):
    mapping = {
        "Tanh_Sin": TanhSin,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "ReLU": nn.ReLU,
        "SiLU": nn.SiLU,
    }
    if name not in mapping:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(mapping)}")
    return mapping[name]()


class FCNet(nn.Module):
    """Fully connected net: 2 -> [hidden]*depth -> 1 with the given activation."""

    def __init__(self, hidden=40, depth=3, activation="Tanh_Sin"):
        super().__init__()
        sizes = [2] + [hidden] * depth + [1]
        self.net = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])])
        self.activation = make_activation(activation)

    def forward(self, x):
        for layer in self.net[:-1]:
            x = self.activation(layer(x))
        return self.net[-1](x)


def mollifier(x):
    """Boundary mollifier sin(pi x1) sin(pi x2), zero on dOmega."""
    return torch.sin(PI * x[:, 0:1]) * torch.sin(PI * x[:, 1:2])


def exact_u(x, freq):
    return torch.sin(freq * x[:, 0:1]) * torch.sin(freq * x[:, 1:2])


def source_f(x, freq):
    return 2.0 * freq ** 2 * exact_u(x, freq)


def make_test_grid(n):
    axis = torch.linspace(-1.0, 1.0, n)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


# ------------------------------------------------------------------ weak form
def weak_residual(model, centers, radius, grid, v, dv, freq):
    """Per-center weak residual ``<grad u, grad v> - <f, v>`` (vector, size nc)."""
    n_center = centers.shape[0]
    n_grid = grid.shape[0]
    x = (centers + grid[None, :, :] * radius).reshape(-1, 2)
    x.requires_grad_(True)
    u = model(x) * mollifier(x)
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_flat = v[None, :, :].expand(n_center, n_grid, 1).reshape(-1, 1)
    dv_flat = (dv[None, :, :] / radius).reshape(-1, 2)
    flux_term = (grad_u * dv_flat).sum(dim=-1).reshape(n_center, n_grid).mean(dim=-1)
    source_term = (source_f(x, freq) * v_flat).reshape(n_center, n_grid).mean(dim=-1)
    return source_term - flux_term


def relative_l2(pred, ref):
    denom = float(torch.linalg.norm(ref))
    if denom == 0.0:
        return float("nan")
    return float(torch.linalg.norm(pred - ref) / denom)


def train(config, device=None):
    """Train the ParticleWNN and return the metrics dict."""
    device = device or torch.device("cpu")
    seed = int(os.environ.get("NEMS_SEED", "0"))
    retrain_attempt = int(os.environ.get("NEMS_RETRAIN_ATTEMPT", "0"))
    torch.manual_seed(seed)

    out_dir = os.environ.get("NEMS_OUTPUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    grid = integral_grid(config["N_MESH"]).to(device)
    v, dv = wendland_test_function(grid)
    n_grid = grid.shape[0]

    model = FCNet(hidden=config["HIDDEN"], depth=config["DEPTH"],
                  activation=config["ACTIVATION"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"],
                                 weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["STEP_SIZE"],
                                                gamma=config["GAMMA"])

    x_test = make_test_grid(config["N_TEST"]).to(device)
    u_test = exact_u(x_test, config["FREQ"])

    radius = torch.full((config["N_CENTER"], 1, 1), float(config["RADIUS"]), device=device)
    max_wall = float(os.environ.get("NEMS_MAX_WALL_CLOCK", "0") or 0.0)
    record_every = max(1, config["EPOCHS"] // 100)
    loss_path = os.path.join(out_dir, "loss_time.jsonl") if out_dir else None

    start = None
    final_loss = float("nan")
    epochs_done = 0
    for epoch in range(config["EPOCHS"]):
        centers = (torch.rand(config["N_CENTER"], 1, 2, device=device) * 2.0 - 1.0)
        centers = centers * (1.0 - float(config["RADIUS"]))
        residual = weak_residual(model, centers, radius, grid, v, dv, config["FREQ"])
        loss = torch.norm(residual) / (config["N_CENTER"] ** 0.5) * config["W_PDE"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss = float(loss.item())
        epochs_done = epoch + 1
        if start is None:
            start = time.perf_counter()
            wall = 0.0
        else:
            wall = time.perf_counter() - start
        if loss_path and (epoch % record_every == 0 or epoch == config["EPOCHS"] - 1):
            with open(loss_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"wall_clock": wall, "loss": final_loss, "epoch": epoch}) + "\n")
        if max_wall and wall >= max_wall:
            break

    total = time.perf_counter() - start
    with torch.no_grad():
        pred = model(x_test) * mollifier(x_test)
        l2_error = relative_l2(pred, u_test)
    rmse = math.sqrt(final_loss) if final_loss == final_loss and final_loss >= 0 else final_loss

    metrics = {
        "final_loss": final_loss,
        "final_error": rmse,
        "total_wall_clock": total,
        "epochs": epochs_done,
        "seed": seed,
        "retrain_attempt": retrain_attempt,
        "final_l2_error": l2_error,
        "freq": config["FREQ"],
        "hidden": config["HIDDEN"],
        "depth": config["DEPTH"],
        "activation": config["ACTIVATION"],
        "n_center": config["N_CENTER"],
        "n_grid": n_grid,
        "radius": config["RADIUS"],
    }
    if out_dir:
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
    print(f"epochs={epochs_done} weak_loss={final_loss:.6e} l2_err={l2_error:.6e} wall={total:.3f}s")
    return metrics


def load_config(params):
    config = {
        "FREQ": float(params.get("FREQ", Config.FREQ)),
        "HIDDEN": int(params.get("HIDDEN", Config.HIDDEN)),
        "DEPTH": int(params.get("DEPTH", Config.DEPTH)),
        "ACTIVATION": str(params.get("ACTIVATION", Config.ACTIVATION)),
        "LR": float(params.get("LR", Config.LR)),
        "EPOCHS": int(params.get("EPOCHS", Config.EPOCHS)),
        "N_CENTER": int(params.get("N_CENTER", Config.N_CENTER)),
        "RADIUS": float(params.get("RADIUS", Config.RADIUS)),
        "N_MESH": int(params.get("N_MESH", Config.N_MESH)),
        "N_TEST": int(params.get("N_TEST", Config.N_TEST)),
        "WEIGHT_DECAY": float(params.get("WEIGHT_DECAY", Config.WEIGHT_DECAY)),
        "STEP_SIZE": int(params.get("STEP_SIZE", Config.STEP_SIZE)),
        "GAMMA": float(params.get("GAMMA", Config.GAMMA)),
        "W_PDE": float(params.get("W_PDE", Config.W_PDE)),
    }
    torch.set_num_threads(int(os.environ.get("NEMS_TORCH_THREADS", "1")))
    return config


def main() -> dict:
    return train(load_config(load_params(os.environ.get("NEMS_OUTPUT_DIR"))))


if __name__ == "__main__":
    main()
