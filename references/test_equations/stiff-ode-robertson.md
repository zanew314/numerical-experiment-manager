---
title: Robertson Stiff System
card_id: stiff-ode-robertson
group: general
problem_family: ordinary_differential_equation
equation_class: stiff_initial_value_problem
input_format: time_interval_and_parameters
difficulty: [stiff, multi_scale]
reference_kind: numeric
run_level: tiny_cpu_possible
approval_required: true
---

# Robertson Stiff System

## Prompt Body

Use $numerical-experiment-manager.

Consider the classic Robertson stiff system

```text
y1' = -0.04 * y1 + 1e4 * y2 * y3
y2' =  0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2^2
y3' =                         3e7 * y2^2
y(0) = (1, 0, 0),    t in [0, 1e5],
```

whose components evolve on timescales separated by many orders of magnitude.
Instantiate the interval on a log-spaced time grid and keep the horizon bounded.

## Expected Modeling Signals

- Predicted quantity: the three-component state `(y1, y2, y3)(t)`.
- Input encoding: initial state and the requested times; the model must accept a
  vector state and a large, unevenly spaced time interval.
- Difficulty axes: `stiff`, `multi_scale`.
- Reference solution: numeric — precompute a high-accuracy reference (for
  example an implicit/BDF solve) once and store it next to the case; do not
  compare against an explicit-Euler trajectory.
- Risk checks: a model relying on fixed explicit steps will diverge; watch for
  `nan`/`inf`, mass conservation `sum(y) = 1`, and negative concentrations.
- This is the canonical early-stop trigger case: a stiffness-blind model should
  be caught by the `non_finite` or `catastrophic` rule rather than run to budget.
