---
title: Harmonic Oscillator
card_id: harmonic-oscillator
group: general
problem_family: ordinary_differential_equation
equation_class: second_order_oscillatory_problem
input_format: time_grid_and_frequency
difficulty: [oscillatory, long_horizon]
reference_kind: exact
run_level: tiny_cpu_possible
approval_required: true
---

# Harmonic Oscillator

## Prompt Body

Use $numerical-experiment-manager.

Consider the undamped harmonic oscillator written as a first-order system

```text
x'  =  v
v'  = -omega^2 * x,    t in [0, T],    x(0) = 1, v(0) = 0,
```

with the exact solution `x(t) = cos(omega * t)` and conserved energy
`E = 0.5 * (v^2 + omega^2 * x^2) = 0.5 * omega^2`.

Instantiate at least one small and one large horizon to expose phase drift.

## Expected Modeling Signals

- Predicted quantity: position `x(t)` (and optionally velocity `v(t)`).
- Input encoding: frequency `omega`, initial state, and the time grid.
- Difficulty axes: `oscillatory`, `long_horizon`.
- Reference solution: exact; also check energy conservation over the horizon.
- Risk checks: amplitude decay (numerical damping) and phase error growing with
  horizon length are the two failure modes to look for.
