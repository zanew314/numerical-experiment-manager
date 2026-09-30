---
title: Allen-Cahn Phase Field
card_id: allen-cahn
group: pde
problem_family: partial_differential_equation
equation_class: reaction_diffusion
input_format: initial_condition_field_and_interface_width
difficulty: [reaction_diffusion, sharp_interface]
reference_kind: numeric
run_level: tiny_cpu_possible
approval_required: true
---

# Allen-Cahn Phase Field

## Prompt Body

Use $numerical-experiment-manager.

Consider the Allen-Cahn equation with double-well reaction

```text
u_t = eps^2 * u_xx + u - u^3     x in (-1, 1), t in (0, T],
u(x, 0) = <smooth step or noisy phase field>,    periodic boundary,
```

which coarsens into flat `+1`/`-1` regions separated by interfaces of width
`O(eps)`. Small `eps` produces thin interfaces that stress resolution.

Instantiate a moderate `eps` and a small `eps`, and record interface locations.

## Expected Modeling Signals

- Predicted quantity: the field `u(x, t)` in `[-1, 1]`.
- Input encoding: the initial phase field, the interface-width parameter
  `eps`, and the periodic space-time domain.
- Difficulty axes: `reaction_diffusion`, `sharp_interface`.
- Reference solution: numeric — precompute a fine reference per `eps`.
- Risk checks: interface smearing or spurious nucleation; check that `u` stays
  within `[-1, 1]` and that interfaces move at the correct speed.
