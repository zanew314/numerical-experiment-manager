#!/usr/bin/env python3
"""Demo project WITHOUT a loss_time.jsonl stream.

It fits y = sin(x) on [0, 1] with an explicit-layer MLP, but it never writes the
``loss_time.jsonl`` instrumentation file. Instead it prints one progress line per
recorded epoch:

    epoch=<i> loss=<v> elapsed=<seconds>

The experiment manager reads those printed lines with the ``stdout_regex``
loss-stream source (see ``..\\..\\references\\loss_stream_adapters.md``). The
program still writes ``metrics.json`` for the final metric, and it reads its
hyperparameters from ``current_config.json`` in ``NEMS_OUTPUT_DIR`` when present.

This is intentionally small and CPU-friendly so the no-stream lifecycle can be
demonstrated in well under a minute.
"""

import json
import math
import os
import time

import torch
import torch.nn as nn


def load_params(out_dir):
    """Read the experiment parameters the manager wrote, if any."""
    if not out_dir:
        return {}
    path = os.path.join(out_dir, "current_config.json")
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload.get("params", payload) or {}
    except (OSError, ValueError):
        pass
    return {}


class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.tanh(self.fc3(x))


def make_data(n=256):
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    return x, torch.sin(x)


def main() -> dict:
    out_dir = os.environ.get("NEMS_OUTPUT_DIR")
    params = load_params(out_dir)
    lr = float(params.get("LR", 0.01))
    hidden_dim = int(params.get("HIDDEN_DIM", 64))
    total_epochs = int(params.get("EPOCHS", 400))

    seed = int(os.environ.get("NEMS_SEED", "0"))
    retrain_attempt = int(os.environ.get("NEMS_RETRAIN_ATTEMPT", "0"))
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("NEMS_TORCH_THREADS", "1")))

    x, y = make_data()
    model = MLP(hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    max_wall = float(os.environ.get("NEMS_MAX_WALL_CLOCK", "0") or 0.0)
    record_every = max(1, total_epochs // 100)

    start = None
    final_loss = float("nan")
    epochs_done = 0
    for epoch in range(total_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())
        epochs_done = epoch + 1
        if start is None:
            start = time.perf_counter()
            wall = 0.0
        else:
            wall = time.perf_counter() - start
        if epoch % record_every == 0 or epoch == total_epochs - 1:
            print(f"epoch={epoch} loss={final_loss!r} elapsed={wall:.4f}", flush=True)
        if max_wall and wall >= max_wall:
            break

    total = time.perf_counter() - start
    rmse = math.sqrt(final_loss) if final_loss == final_loss and final_loss >= 0 else final_loss

    metrics = {
        "final_loss": final_loss,
        "final_error": rmse,
        "total_wall_clock": total,
        "epochs": epochs_done,
        "seed": seed,
        "retrain_attempt": retrain_attempt,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    main()
