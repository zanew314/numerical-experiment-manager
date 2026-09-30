---
title: 2D Incompressible Navier-Stokes (Taylor-Green)
card_id: navier-stokes-2d
group: pde
problem_family: partial_differential_equation
equation_class: incompressible_navier_stokes
input_format: initial_velocity_field_and_reynolds_number
difficulty: [nonlinear, incompressible, chaotic_sensitive]
reference_kind: numeric
run_level: tiny_cpu_possible
approval_required: true
---

# 2D Incompressible Navier-Stokes (Taylor-Green)

## Prompt Body

Use $numerical-experiment-manager.

Consider the two-dimensional incompressible Navier-Stokes equations on a
periodic domain

```text
u_t + (u . grad) u = -grad p + nu * lap(u),
div u = 0,
```

initialized with the Taylor-Green vortex

```text
u(x, y, 0) = ( sin(x) cos(y), -cos(x) sin(y) ),    (x, y) in [0, 2*pi)^2.
```

Instantiate a modest Reynolds number so the flow stays resolved; keep the
horizon short and record when the solution first becomes under-resolved.

## Expected Modeling Signals

- Predicted quantity: the velocity field `(u, v)(x, y, t)` (and pressure).
- Input encoding: the initial velocity field and viscosity `nu`; the model must
  accept a two-component field and a divergence-free constraint.
- Difficulty axes: `nonlinear`, `incompressible`, `chaotic_sensitive`.
- Reference solution: numeric — precompute a spectral reference at the same
  resolution.
- Risk checks: divergence `div u` drift, energy dissipation rate, and whether
  the model enforces periodicity; treat divergence blow-up as a structural
  failure separate from accuracy.
