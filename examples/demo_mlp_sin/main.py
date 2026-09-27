#!/usr/bin/env python3
"""Demo project: fit y = sin(x) on [0, 1] with an explicit-layer MLP.

This is intentionally small and CPU-friendly so the full experiment-manager
lifecycle can be demonstrated in under three minutes. The training loop records
wall-clock time together with the loss and writes them to the directory named by
``NEMS_OUTPUT_DIR`` when the experiment manager runs it.
"""

import json
import math
import os
import time

import torch
import torch.nn as nn


class Config:
    LR = 0.001
    NUM_LAYERS = 4
    HIDDEN_DIM = 64
    EPOCHS = 1000


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(1, config.HIDDEN_DIM)
        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        self.fc3 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        self.fc4 = nn.Linear(config.HIDDEN_DIM, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


def make_data(n=256):
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    y = torch.sin(x)
    return x, y


def main() -> dict:
    config = Config()
    seed = int(os.environ.get("NEMS_SEED", "0"))
    retrain_attempt = int(os.environ.get("NEMS_RETRAIN_ATTEMPT", "0"))
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("NEMS_TORCH_THREADS", "1")))
    device = torch.device("cpu")

    x, y = make_data()
    x, y = x.to(device), y.to(device)

    model = MLP(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_fn = nn.MSELoss()

    out_dir = os.environ.get("NEMS_OUTPUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    loss_path = os.path.join(out_dir, "loss_time.jsonl") if out_dir else None
    max_wall = float(os.environ.get("NEMS_MAX_WALL_CLOCK", "0") or 0.0)
    total_epochs = int(config.EPOCHS)
    record_every = max(1, total_epochs // 200)

    start = None
    final_loss = float("nan")
    epochs_done = 0
    for epoch in range(total_epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())
        epochs_done = epoch + 1
        if start is None:
            # start the training clock after one-time initialization so that
            # wall-clock comparisons are not dominated by the first-step cost
            start = time.perf_counter()
            wall = 0.0
        else:
            wall = time.perf_counter() - start
        if loss_path and (epoch % record_every == 0 or epoch == total_epochs - 1):
            with open(loss_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "wall_clock": wall,
                    "loss": final_loss,
                    "epoch": epoch,
                }) + "\n")
        if max_wall and wall >= max_wall:
            break

    total = time.perf_counter() - start
    if final_loss == final_loss and final_loss >= 0:
        rmse = math.sqrt(final_loss)
    else:
        rmse = final_loss

    metrics = {
        "final_loss": final_loss,
        "final_error": rmse,
        "total_wall_clock": total,
        "epochs": epochs_done,
        "seed": seed,
        "retrain_attempt": retrain_attempt,
    }
    if out_dir:
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
    print(f"epochs={epochs_done} final_loss={final_loss:.6e} rmse={rmse:.6e} wall={total:.3f}s")
    return metrics


if __name__ == "__main__":
    main()
