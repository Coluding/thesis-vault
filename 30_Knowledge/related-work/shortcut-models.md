---
type: paper
status: living
last_updated: 2026-05-15
title: "Shortcut Models"
authors: []
venue:
year:
url:
local_pdf: docs/paper/shortcut_models.pdf
relevance: theory, framework
deliverable: D3
---

# Shortcut Models

> Source of the step-size-conditioned consistency formulation behind D3.
> The closest direct precedent for the thesis's shortcut adapter.

## Status of this note

**Stub.** Vendored PDF at `docs/paper/shortcut_models.pdf`. Title,
authors, venue, year, URL — _needs verification from the PDF_. The
existing in-repo write-up at `docs/shortcut_action_summary.md` is the
best second source — extract its claims into this note when fleshing the
stub out.

## Why it matters for the thesis

- Shortcut Models is the direct precedent for the **D3 contribution** —
  training a generative model with a step-size argument `d` and a
  multi-step self-consistency objective so it can roll out in few steps.
- The key relationship the thesis re-uses:
  ```
  s(x_t, t, 2d, a_t) ≈ ½ s(x_t, t, d, a_t) + ½ s(x_{t+d}, t+d, d, a_t)
  ```
- Implemented in the codebase as:
  - `losses/consistency.py` — `shortcut_direction`, `local_consistency`,
    `multistep_self_consistency` losses.
  - `adapters/output/shortcut_direction.py` —
    `ShortcutDirectionOutputAdapter` with `include_step_size=True`.
  - `TrainingConfig.shortcut_target_method` ∈ {`linear`, `two_step`}.
  - Configs `flow_output_shortcut.yaml`,
    `flow_hyper_shortcut_stepwise.yaml`,
    `diffusion_output_shortcut_noise.yaml`,
    `diffusion_output_shortcut_velocity.yaml`,
    `diffusion_output_dynamicrafter_shortcut_test.yaml`.

## How the thesis differs from Shortcut Models

The thesis's D3 contribution is **not** to introduce the shortcut idea
(that's this paper). The contribution is:

1. **Adapter-only realisation.** Shortcut Models train the full model
   with the shortcut objective. We freeze the base and put the
   step-size conditioning + consistency loss into the adapter alone.
2. **Action conditioning + step-size conditioning together.** The
   adapter takes both `a_t` and `d`. Shortcut Models do not consider
   action conditioning.
3. **Adapter-family ablations.** We run the shortcut training across
   output / hidden-state / hypernetwork adapters and report the
   trade-off. Shortcut Models do not have this design surface.

These three deltas are the precise reason D3 is its own chapter rather
than a one-paragraph extension.

## Open questions for the chapter

- Exact loss equation in the Shortcut Models paper — _needs verification
  from the PDF, then derive in `30_Knowledge/theory/shortcut-loss-derivation.md`_.
- Whether Shortcut Models use a `g(d)` gain like our composition rule
  `f_base + g(d) · Δ_φ`. _needs verification_.
- Reported few-step regime (1, 2, 4, 8 steps?) and how the thesis's
  numbers should align for an apples-to-apples comparison.
- Relationship to [[consistency-models]] — is Shortcut Models a strict
  generalisation, a different objective, or an orthogonal cut?

## Related

- [[_MOC]]
- [[consistency-models]] · [[self-distillation]] · [[dpm-solver]] — the few-step-sampling cluster
- [[../../10_now/positioning]] — D3 deliverable
- [[../../10_now/architecture]] — see Losses
- `docs/shortcut_action_summary.md` — existing in-repo notes
- `src/generative_flow_adapters/losses/consistency.py`
- `src/generative_flow_adapters/adapters/output/shortcut_direction.py`
