---
title: First-Order Linear ODE
card_id: linear-ode
group: general
problem_family: ordinary_differential_equation
equation_class: linear_initial_value_problem
input_format: time_grid_and_scalar_parameters
difficulty: [baseline, linear]
reference_kind: exact
run_level: tiny_cpu_possible
approval_required: true
---

# First-Order Linear ODE

## Prompt Body

Use $numerical-experiment-manager.

Consider the scalar first-order linear initial-value problem

```text
dy/dt = a * y + b,    t in [t0, t1],    y(t0) = y0,
```

with the closed-form solution

```text
y(t) = (y0 + b / a) * exp(a * (t - t0)) - b / a,    a != 0.
```

This is the baseline sanity case: a model that cannot reduce the error on a
linear ODE will not be trusted on anything harder. Instantiate a few coefficient
sets (stable `a < 0`, growing `a > 0`, and `b != 0`) and keep the time grid small
and fixed across them.

## Expected Modeling Signals

- Predicted quantity: the solution trajectory `y(t)` on the supplied time grid.
- Input encoding: `(t0, t1, y0, a, b)` plus the query grid; the model must accept
  the coefficients and evaluation times in the format its own interface exposes.
- Difficulty axes: `baseline`, `linear`.
- Reference solution: exact; compare against `y(t)` above.
- Risk checks: sign convention for `a`, treatment of the `b` term, and whether
  the model evaluates on a grid or at scattered points.
