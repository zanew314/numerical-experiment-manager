---
name: nems
description: Route numerical/ML experiment-management tasks to the bundled skill package in skills/numerical-experiment-manager.
---

# Numerical Experiment Manager

Use this repository as a routing layer.

## Packages

- `skills/numerical-experiment-manager/`: version a Python project's structure
  and, on request, run wall-clock experiments against a fixed baseline with
  torch.fx data-flow tracing, early stopping, and small-parameter tuning. Read its
  `SKILL.md`, `README.md`, `references/`, and `scripts/` before use.

Prefer the package-local instructions over this router when running a concrete
workflow.
