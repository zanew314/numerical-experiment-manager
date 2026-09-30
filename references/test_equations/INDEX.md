# Test-Equation Bank Index

One Markdown card per equation family. Each card carries machine-readable YAML
front matter (`card_id`, `group`, `problem_family`, `equation_class`,
`input_format`, `difficulty`, `reference_kind`) plus a human `## Prompt Body` and
review-only `## Expected Modeling Signals`.

The bank is **not** run directly. During workflow C the agent reads the model's
problem-description interface, picks the cards whose `input_format` fits, and
materializes each card into a concrete case under `.nems/testset/<case_id>/`.
Use `python scripts/testset.py list` to print the parsed catalog as JSON.

Two groups are available:

- `general` — small, often analytically solvable problems that probe the model
  as a general numerical solver/regressor.
- `pde` — space-time PDE and operator-learning problems that probe a PINN /
  DeepONet / FNO-style model.

## General numerical cards

- `linear-ode.md` — first-order linear initial-value problem (baseline sanity).
- `stiff-ode-robertson.md` — Robertson stiff system (stiffness, multi-scale).
- `harmonic-oscillator.md` — undamped oscillator (oscillation, long horizon).
- `poisson-2d.md` — 2D Dirichlet Poisson problem (elliptic, boundary value).
- `nonlinear-system.md` — 1D Bratu reaction (nonlinear, branch-sensitive).
- `advection-diffusion.md` — steady advection-diffusion (boundary layers).

## PDE / operator cards

- `heat-equation.md` — 1D heat equation (parabolic, smoothing).
- `viscous-burgers.md` — viscous Burgers (nonlinear, shock formation).
- `allen-cahn.md` — Allen-Cahn phase field (reaction-diffusion, sharp interface).
- `darcy-flow.md` — Darcy flow with random coefficient (operator learning).
- `wave-equation.md` — 1D wave equation (hyperbolic, oscillatory).
- `navier-stokes-2d.md` — 2D incompressible flow (nonlinear, incompressible).

## Adding a card

Copy any existing card, change `card_id`/`group`/`difficulty`, and keep the two
body sections. `scripts/testset.py list` picks up new `*.md` cards automatically;
`INDEX.md` is the only file it skips.
