---
title: 1D Bratu Reaction
card_id: nonlinear-system
group: general
problem_family: nonlinear_boundary_value_problem
equation_class: nonlinear_reaction_diffusion
input_format: grid_and_parameter_lambda
difficulty: [nonlinear, branch_sensitive]
reference_kind: numeric
run_level: tiny_cpu_possible
approval_required: true
---

# 1D Bratu Reaction

## Prompt Body

Use $numerical-experiment-manager.

Consider the one-dimensional Bratu problem

```text
u'' + lambda * exp(u) = 0     in (0, 1),
u(0) = u(1) = 0,
```

which has two solution branches below the critical value
`lambda_c ~= 3.5138`. Instantiate `lambda` below and near `lambda_c` and record
which branch a solver returns.

## Expected Modeling Signals

- Predicted quantity: the scalar field `u(x)`.
- Input encoding: the parameter `lambda` and the grid; the model must accept a
  scalar control parameter.
- Difficulty axes: `nonlinear`, `branch_sensitive`.
- Reference solution: numeric — precompute a converged reference per `lambda`
  (for example by Newton or continuation) and store branch labels.
- Risk checks: a model may converge to the wrong branch near `lambda_c`; treat a
  high residual near the fold as a branch failure, not a precision failure.
