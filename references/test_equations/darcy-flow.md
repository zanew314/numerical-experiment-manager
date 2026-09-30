---
title: Darcy Flow with Random Coefficient
card_id: darcy-flow
group: pde
problem_family: partial_differential_equation
equation_class: elliptic_operator_learning
input_format: coefficient_field_to_solution_field
difficulty: [elliptic, random_coefficient, operator_learning]
reference_kind: numeric
run_level: tiny_cpu_possible
approval_required: true
---

# Darcy Flow with Random Coefficient

## Prompt Body

Use $numerical-experiment-manager.

Consider the Darcy flow problem on the unit square

```text
-div( a(x) * grad u ) = f     in (0, 1)^2,
u = 0     on the boundary,
```

where the coefficient `a(x)` is a fixed log-normal random field and `f` is a
fixed source. This is the canonical operator-learning benchmark: learn the map
`a(x) -> u(x)`.

Instantiate a small number of coefficient samples with a fixed random seed.

## Expected Modeling Signals

- Predicted quantity: the pressure field `u(x)` for each coefficient field.
- Input encoding: the coefficient field `a` on a grid (or a low-dimensional
  parameterization); the model must accept a field, not a scalar.
- Difficulty axes: `elliptic`, `random_coefficient`, `operator_learning`.
- Reference solution: numeric — precompute a finite-element / sparse-solve
  reference for each sampled coefficient field.
- Risk checks: the coefficient field must be the *only* varying input; fix the
  seed so samples are reproducible, and check relative `L2` error on held-out
  samples (in-distribution vs out-of-distribution length scales).
