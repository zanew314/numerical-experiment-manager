---
title: Viscous Burgers Equation
card_id: viscous-burgers
group: pde
problem_family: partial_differential_equation
equation_class: nonlinear_convection_diffusion
input_format: initial_condition_field_and_viscosity
difficulty: [nonlinear, shock]
reference_kind: numeric
run_level: tiny_cpu_possible
approval_required: true
---

# Viscous Burgers Equation

## Prompt Body

Use $numerical-experiment-manager.

Consider the viscous Burgers equation on a periodic interval

```text
u_t + u * u_x = nu * u_xx     x in (-1, 1), t in (0, T],
u(x, 0) = -sin(pi * x),
```

with periodic boundary conditions. At small `nu` the solution steepens into a
moving near-shock; at moderate `nu` it stays smooth.

Instantiate a moderate `nu` and a small `nu` to separate smooth-regime accuracy
from shock-capturing ability.

## Expected Modeling Signals

- Predicted quantity: the field `u(x, t)`.
- Input encoding: the initial field, the viscosity `nu`, and the periodic
  space-time domain.
- Difficulty axes: `nonlinear`, `shock`.
- Reference solution: numeric — precompute a high-resolution reference per `nu`.
- Risk checks: models with too little spatial resolution will smear or oscillate
  near the front; report error separately in the smooth and steep regions.
