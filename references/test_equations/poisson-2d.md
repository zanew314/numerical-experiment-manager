---
title: 2D Dirichlet Poisson Problem
card_id: poisson-2d
group: general
problem_family: partial_differential_equation
equation_class: elliptic_boundary_value_problem
input_format: grid_and_source_term
difficulty: [elliptic, boundary_value]
reference_kind: exact
run_level: tiny_cpu_possible
approval_required: true
---

# 2D Dirichlet Poisson Problem

## Prompt Body

Use $numerical-experiment-manager.

Consider the Poisson problem on the unit square with homogeneous Dirichlet data

```text
-delta u = f     in (0, 1)^2,
       u = 0     on the boundary,
```

and the manufactured solution

```text
u(x, y) = sin(pi * x) * sin(pi * y),
f(x, y) = 2 * pi^2 * sin(pi * x) * sin(pi * y).
```

Instantiate on a modest uniform grid; the small size keeps the run bounded.

## Expected Modeling Signals

- Predicted quantity: the scalar field `u(x, y)` on the grid.
- Input encoding: the source field `f` and the grid coordinates (or an
  interior-point set); the model must accept a 2D structured or scattered input.
- Difficulty axes: `elliptic`, `boundary_value`.
- Reference solution: exact; also check that the boundary residual is near zero.
- Risk checks: whether boundary values are enforced or only penalized, and the
  grid resolution the model expects.
