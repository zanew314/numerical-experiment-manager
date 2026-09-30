---
title: Steady Advection-Diffusion
card_id: advection-diffusion
group: general
problem_family: partial_differential_equation
equation_class: convection_diffusion_boundary_value_problem
input_format: grid_and_peclet_number
difficulty: [boundary_layer, convection_dominated]
reference_kind: exact
run_level: tiny_cpu_possible
approval_required: true
---

# Steady Advection-Diffusion

## Prompt Body

Use $numerical-experiment-manager.

Consider the steady advection-diffusion problem on the unit interval

```text
c * u_x = nu * u_xx     in (0, 1),
u(0) = 0,    u(1) = 1,
```

whose exact solution is the boundary-layer profile

```text
u(x) = (exp(c * x / nu) - 1) / (exp(c / nu) - 1).
```

Instantiate at moderate and small `nu` (large cell Peclet number) to expose a
thin boundary layer near `x = 1`.

## Expected Modeling Signals

- Predicted quantity: the scalar profile `u(x)`.
- Input encoding: `c`, `nu`, and the grid; the grid must be dense enough that
  the layer is resolvable, or the case measures resolution handling.
- Difficulty axes: `boundary_layer`, `convection_dominated`.
- Reference solution: exact; a central-difference-like model will oscillate when
  `c / nu` is large.
- Risk checks: non-monotone / oscillatory profiles and boundary-condition error
  are the expected failure signatures.
