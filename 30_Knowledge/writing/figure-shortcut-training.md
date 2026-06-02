---
type: figure-design
status: needs-redraw
last_updated: 2026-05-28
target: advisor-presentation
source: figures/shortcut-training.html
deliverable: D3
---

# Figure — Shortcut training

> **2026-05-28 — needs redraw.** The current SVG depicts the
> `two_step` target construction (two frozen-base passes, Heun
> average). That mode was deprecated on 2026-05-28; see
> [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]].
> The figure must be redrawn to depict the `distillation` target
> instead (two no-grad calls of the **adapted** model at
> `step_level = s/2`, chained across one DDIM micro-step, averaged
> and supervising the adapter at `step_level = s`). The
> base-anchored Heun construction can either be omitted entirely or
> moved to a separate small figure for the `heun_smoothness`
> regularizer ([[../theory/heun-smoothness-regularizer]]).
>
> Until the redraw lands, **do not use this figure in slides or the
> thesis** — it depicts a code path that no longer exists.

> One-slide figure showing the shortcut training loop as implemented in
> this repo. Focus: how the training target for the step-size-conditioned
> adapter is constructed from the frozen base, and what loss the adapter
> is optimised against. **This is a training-time data-flow figure**, not
> a forward-time architecture figure (the adapter family — output / LoRA
> / hypernet — is orthogonal and is figured separately).

## What this figure asserts

Each claim is traceable to code.

1. **The adapter is step-size-conditioned.** It takes `(x_t, t, c, d)`
   where `d` is the step size; this is what distinguishes a shortcut
   adapter from a vanilla one. Source:
   `adapters/output/shortcut_direction.py:14–47` —
   `include_step_size=True` adds `d` as input; the standalone shortcut
   adapter is `ShortcutDirectionOutputAdapter`. Other adapter families
   (output / hypernet) receive `d` via
   `cond["step_size"]` / `cond["step_level"]`.
2. **The training target is computed from the frozen base, not from the
   adapter.** Targets are detached and the base is run under
   `torch.no_grad()`. Source:
   `training/shortcut_targets.py:83, 111` (`with torch.no_grad():`) and
   `:65–66` (`shortcut_target.detach()`).
3. **The figure shows the `two_step` target method (paper-faithful).**
   The base is run *twice* to produce a synthetic shortcut target:
   - `v_b1 = f_base(x_t, t, c)`
   - `x_t' = x_t + d · v_b1`
   - `v_b2 = f_base(x_t', t, c)`
   - `target = ½(v_b1 + v_b2)`
   Source: `_compute_two_step_shortcut_target`, lines 94–125.
4. **The alternative `linear` method uses one base pass.**
   `target = d · base(x_t, t, c)` (with optional normalisation). One
   forward pass; cheaper but less accurate. Mentioned in caption only.
   Source: `_compute_linear_shortcut_target`, lines 70–91.
5. **The loss is MSE against the (detached) base-derived target.**
   `loss = ‖adapter(x_t, t, c, d) − target‖²`. Source:
   `losses/consistency.py:11–12` (`shortcut_direction_loss`) and
   trainer wiring at `training/trainer.py:144–149`.
6. **Three loss heads share the same target tensor in this codebase.**
   `shortcut_direction`, `local_consistency`, and
   `multistep_self_consistency` are all MSE losses; in the current
   trainer `shortcut_target` and `self_consistency_target` are *set to
   the same detached tensor* (`shortcut_targets.py:65–66`). The figure
   shows them as one loss head with a caption note.

## Known divergence from paper — superseded by the redraw

> **2026-05-28 update.** The earlier `t` vs `t + d` divergence
> (originally tracked in
> [[../../20_Tickets/bug-training-shortcut-target-timestep]]) was
> structurally resolved when the trainer switched to
> `ddim_micro_step_v` (the second base call now correctly uses
> `prev_t`, not `t`). That ticket is closed. Subsequently the entire
> `two_step` mode was deprecated, so the asserted "figure shows the
> paper; code will be fixed" claim no longer applies. The redraw will
> depict `distillation` instead, where the paper-faithful
> self-consistency target is already what the code does.

## What this figure deliberately omits

- The standard diffusion/flow loss term (the trainer adds the shortcut
  loss *on top of* the base task loss). Mentioned in caption.
- The `local_consistency` and `multistep_self_consistency` loss heads,
  since they currently use the same target tensor as
  `shortcut_direction`. Mentioned in caption.
- The internals of the adapter — we show it as a single trainable
  block; the architecture inside (output / hypernet / etc.) is the
  subject of the other figures.
- Classifier-free guidance dropout on `c` (handled by the structural
  encoder).

## Layout intent (original — refers to deprecated `two_step`)

> Preserved here for reference; the redraw will mirror this layout
> with the **adapted model** in both teacher rows instead of the
> frozen base, and a DDIM-micro-step compute node in place of the
> naive Euler `x_t' = x_t + d · v_b1`.

Three parallel rows showing the three forward passes that happen each
training step, mirroring the "frozen base run twice" visual pattern
established in `figure-hyperalign`:

- **Top row (trainable):** `[x_t, t, c, d]` → adapter → `prediction`.
  The `d` input is highlighted as the distinguishing feature.
- **Middle row (frozen, pass A):** `[x_t, t, c]` → frozen base
  (`no_grad`) → `v_b1`. Below this row, a small compute node forms
  `x_t' = x_t + d · v_b1`.
- **Bottom row (frozen, pass B, shared weights with middle):**
  `[x_t', t, c]` → frozen base → `v_b2`. A small "same weights" dashed
  link to the middle row makes the weight sharing visual.
- **Right side:** average node `target = ½(v_b1 + v_b2)` with a
  stop-grad annotation, then loss node `‖prediction − target‖²`.
- Loss connects back to the adapter via a dashed "grad" arrow,
  emphasising that *only* the adapter receives gradient.

## Redraw plan (2026-05-28)

Same three-row layout, but the two "teacher" rows are now passes of
the **adapted** model in `no_grad` / `eval` mode at half the step
size. Concretely:

- **Top row (trainable):** `[x_t, t, c, step_level=s]` → adapter →
  `prediction`. The `step_level = s` input is highlighted as the
  distinguishing feature.
- **Middle row (`no_grad`, pass A — adapted, not base):**
  `[x_t, t, c, step_level=s/2]` → `f_θ = base + adapter` → `v_1`.
  Below this row, a small compute node forms
  `x_mid = ddim_micro_step_v(x_t, v_1, t, t - jump)` where
  `jump = round((s/2) · T)`.
- **Bottom row (`no_grad`, pass B — adapted, shared weights with
  middle):** `[x_mid, t - jump, c, step_level=s/2]` → `f_θ` → `v_2`.
  A "same weights" dashed link to the middle row.
- **Right side:** average node `target = ½(v_1 + v_2)` with a
  stop-grad annotation, then loss node `‖prediction − target‖²`.
- Loss connects back to the adapter via a dashed "grad" arrow,
  emphasising that only the adapter receives gradient on the
  trainable row.

Also add an "anchor branch" annotation in the caption: with probability
`shortcut_anchor_prob` the figure's training step is replaced by the
standard diffusion/flow loss at `step_level = schedule.smallest()`,
not the self-consistency target.

## Decisions

- **2026-05-28 — Target method shown:** `distillation` (paper-faithful,
  the only mode after deprecation). The Heun construction migrates to a
  separate optional figure for `heun_smoothness_regularizer` if needed
  in D2 talks.
- **2026-05-19 — Adapter family:** drawn as a generic "trainable
  adapter" block, agnostic of family. This figure is the *training
  scheme*, not the adapter architecture.
- **2026-05-19 — Loss heads:** one loss head shown; siblings mentioned
  in caption.

## Open questions for the advisor

- Does the redrawn `distillation` figure need to also show the
  anchor branch as a parallel "case B" mini-panel, or is a caption
  note sufficient? Probably caption-only, but worth confirming.
- Should we make a companion figure for `heun_smoothness` (one
  trainable row + one no-grad-self-distillation row)? Decision
  parked until we run the first regularised experiment.

## Related

- [[../related-work/shortcut-models]] — paper-side note
- [[../related-work/consistency-models]] · [[../related-work/self-distillation]] — the few-step-sampling cluster
- [[figure-hyperalign]] · [[figure-avid]] · [[figure-structural-encoder]]
- Code: `src/generative_flow_adapters/losses/consistency.py`
- Code: `src/generative_flow_adapters/training/shortcut_targets.py`
- Code (trainer wiring): `src/generative_flow_adapters/training/trainer.py:137–156`
- Code: `src/generative_flow_adapters/adapters/output/shortcut_direction.py`
- Repo doc: `docs/shortcut_action_summary.md`
