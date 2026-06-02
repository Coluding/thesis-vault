---
type: figure-design
status: needs-revision
last_updated: 2026-05-28
target: advisor-presentation
source: figures/shortcut-explainer.html
deliverable: D3
---

# Explainer — Shortcut training (from-scratch vs. adapter)

> **2026-05-28 — needs revision.** Section 3 of the explainer
> (currently the "adapter approach") describes the contribution as
> "the base supplies the local tangent, the adapter learns the
> step-size-dependent correction." That framing is still correct for
> the adapter architecture, but the *supervision* it gestures at
> (Heun-corrected base target) has been deprecated as a shortcut
> training mode; see
> [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]].
> The supervision story has to be rewritten: the adapter is trained
> by **paper-faithful self-consistency** (`distillation`) — the
> adapted model bootstraps its own chord-velocity targets across the
> dyadic recurrence, anchored by the standard diffusion/flow loss at
> the smallest step size. The base contributes a tangent input to the
> adapter at *inference* time, but is no longer the supervisory
> teacher.
>
> Until the revision lands, **do not use this explainer in advisor
> slides** — the headline pedagogy in Section 3 misrepresents the
> current training scheme.

> Multi-section pedagogical HTML for the advisor meeting. Distinct
> from [[figure-shortcut-training]] (the training data-flow figure):
> this one motivates *why* the shortcut idea matters, sketches the
> from-scratch formulation, and then frames our contribution as an
> adapter that consumes the frozen base's local prediction.

## What this doc teaches

Three concepts in order:

1. **Generative models as ODE trajectories.** Visual: probability-flow
   curve between two vertical axes (noise / data), with eight small
   Euler-step chords approximating it. The narrative point: each chord
   costs one base-model call, so step count is the cost lever.
2. **Shortcut models — predict the full step.** Visual: the same
   trajectory traversed in two big shortcut steps. Self-consistency
   formula written out:
   `s(x_t, t, 2d) ≈ ½·s(x_t, t, d) + ½·s(x_{t+d}, t+d, d)`. Caveat:
   this normally requires training from scratch.
3. **Our adapter approach.** Visual: zoomed-in view of a single point
   on the trajectory, showing the base's tangent (`v_base`) versus the
   chord to `x_{t+d}` (`v_shortcut`), with an inset block showing the
   adapter's inputs and outputs. The framing: the base contributes a
   useful starting direction; the adapter learns the
   step-size-dependent correction.

## What this doc deliberately omits

- The full self-consistency derivation (only the headline formula is
  shown).
- The classifier-free dropout path on conditioning.
- The contrast with consistency models / score distillation — out of
  scope for the advisor walkthrough; mentioned in
  [[../related-work/consistency-models]] and
  [[../related-work/self-distillation]] if needed.

## Anchors used (each claim sourced)

- Shortcut adapter takes `cond["base_direction"]`:
  `adapters/output/shortcut_direction.py:20, 72–80`. (Still current —
  the inference-time tangent input survives the deprecation.)
- ~~Two-step base-derived target:~~ **Deprecated 2026-05-28.** The
  active supervision is now paper-faithful self-consistency at
  `training/shortcut_targets.py:54-78` and `training/trainer.py:385-414`
  (the `distillation` branch of `_maybe_prepare_shortcut`). See
  [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]] and
  [[../tech/shortcut-training-modes]] for the catalogue.
- Shortcut models paper formulation:
  [[../related-work/shortcut-models]] (citing
  `docs/paper/shortcut_models.pdf` in the impl repo).

## Decisions (2026-05-19)

- **Hand-drawn SVG**, not Mermaid: the figures need smooth Bezier
  curves and free-positioned tangent/chord vectors, which Mermaid can't
  express. Coordinates are baked into the SVG (no JS layout step).
- **Three SVGs in one HTML doc**, not three separate files: the doc is
  read top-to-bottom as one narrative, so they belong together.
- **Section 3 framing**: emphasise that the adapter receives
  `v_base` *and* the step size, and learns to map the tangent to the
  chord. This is the user's framing of the contribution.

## Open questions

- The `d → 0 ⇒ adapter ≈ v_base` claim in Section 4 — is the
  architecture actually able to learn the trivial pass-through when
  `d` is small? The features are concatenated and passed through MLP
  layers, so it can in principle, but worth verifying empirically. Not
  blocking the advisor narrative, but a footnote worth tracking.

## Related

- [[figure-shortcut-training]] — companion figure (training data flow,
  not pedagogy); also pending redraw under the same decision
- [[../theory/shortcut-training]] §4 — the up-to-date conceptual
  framing of the contribution (distillation, not base-as-teacher)
- [[../theory/heun-smoothness-regularizer]] — the Heun construction,
  now repurposed as a regularizer; if the explainer wants to keep the
  Heun visualisation, it belongs in a side-explainer for this term
- [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]] —
  the decision that motivated this revision
- [[../related-work/shortcut-models]] — paper note
- [[../related-work/consistency-models]] · [[../related-work/self-distillation]]
- Code: `src/generative_flow_adapters/adapters/output/shortcut_direction.py`
- Code: `src/generative_flow_adapters/training/shortcut_targets.py`
