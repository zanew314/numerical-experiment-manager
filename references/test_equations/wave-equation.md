---
title: 1D Wave Equation
card_id: wave-equation
group: pde
problem_family: partial_differential_equation
equation_class: hyperbolic_initial_boundary_value_problem
input_format: initial_condition_field_and_speed
difficulty: [hyperbolic, oscillatory]
reference_kind: exact
run_level: tiny_cpu_possible
approval_required: true
---

# 1D Wave Equation

## Prompt Body

Use $numerical-experiment-manager.

Consider the one-dimensional wave equation with fixed ends

```text
u_tt = c^2 * u_xx     x in (0, 1), t in (0, T],
u(x, 0) = sin(pi * x),    u_t(x, 0) = 0,    u(0, t) = u(1, t) = 0,
```

with the exact solution `u(x, t) = cos(c * pi * t) * sin(pi * x)`.

Instantiate a moderate horizon and a long horizon to expose dispersion error.

## Expected Modeling Signals

- Predicted quantity: the field `u(x, t)`.
- Input encoding: the initial displacement and velocity fields and the wave
  speed `c`; this is a second-order-in-time problem.
- Difficulty axes: `hyperbolic`, `oscillatory`.
- Reference solution: exact; also check energy conservation of the wave.
- Risk checks: numerical dispersion (phase error growing with `t`) and amplitude
  drift are the expected signatures; a first-order-in-time reduction that drops
  `u_t` is a common modeling error.
