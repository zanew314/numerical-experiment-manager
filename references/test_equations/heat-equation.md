---
title: 1D Heat Equation
card_id: heat-equation
group: pde
problem_family: partial_differential_equation
equation_class: parabolic_initial_boundary_value_problem
input_format: initial_condition_field_and_time
difficulty: [parabolic, smoothing]
reference_kind: exact
run_level: tiny_cpu_possible
approval_required: true
---

# 1D Heat Equation

## Prompt Body

Use $numerical-experiment-manager.

Consider the one-dimensional heat equation with homogeneous Dirichlet data

```text
u_t = alpha * u_xx     x in (0, 1), t in (0, T],
u(x, 0) = sin(pi * x),    u(0, t) = u(1, t) = 0,
```

with the exact solution `u(x, t) = exp(-alpha * pi^2 * t) * sin(pi * x)`.

For an operator-learning model, instantiate a bundle of initial conditions and
learn the map `u(x, 0) -> u(x, T)`; for a residual/PINN model, solve a single
initial condition over the space-time domain.

## Expected Modeling Signals

- Predicted quantity: the field `u(x, t)`, or the terminal field `u(x, T)`.
- Input encoding: the initial-condition field plus the space-time coordinates.
- Difficulty axes: `parabolic`, `smoothing`.
- Reference solution: exact; the high-frequency content of the initial
  condition decays fast, so a correct model should smooth over time.
- Risk checks: whether the model enforces the two boundary ends and the initial
  time slice, and whether it overfits the initial condition without decaying.
