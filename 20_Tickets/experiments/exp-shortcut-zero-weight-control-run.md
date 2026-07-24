---
type: exp
scope: shortcut
status: open
priority: medium
created: 2026-07-09
updated: 2026-07-14
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]", "[[../../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[exp-adapter-adaln-gatelow-metaworld-run]]"]
---

# exp: shortcut_direction_weight = 0 control run

## Hypothesis

On the 20260907 run, the **only visibly-learning loss component is the coarse
shortcut rung (N001)** — a self-referential consistency objective the adapter can
satisfy *without using actions*. Suspicion: the shortcut term is capturing the
gradient budget while the action-conditioned `base_loss` stays flat.

See [[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]].

## Procedure

Re-run the 20260907 config on the **same new data** with
`shortcut_direction_weight = 0` (and `local_consistency_weight` /
`multistep_consistency_weight = 0`) — i.e. **pure action-conditioned flow
matching, no shortcut**. Everything else identical.

## Decision rule

- **`base_loss` now descends** ⇒ the shortcut/consistency terms were interfering
  with (or masking) action-conditioned learning → revisit shortcut weighting /
  scheduling ([[../feat-shortcut-per-stepsize-loss-reweighting]]) before combining.
- **`base_loss` still flat** ⇒ the problem is upstream of the shortcut term
  (conditioning path / data / adapter) — the shortcut term is exonerated.

## Notes

Complements [[exp-conditioning-action-shuffle-ablation]]: shuffle isolates
*actions*, this run isolates the *shortcut term*. Together they localise the
failure. Deliverable-wise this is the clean D2 (action-only) baseline the D4
combined run should have had anyway.

## Sharper mechanism (2026-07-14)

[[../../30_Knowledge/tech/why-adapter-underlearns-diagnosis]] found a more specific
version of this hypothesis worth checking alongside the weight=0 control:
`trainer.py`'s base flow-matching loss fires **unconditionally** every step
against the point-velocity target, even on non-anchor/coarser `step_level`
steps — so on those steps the same prediction is regressed toward *two*
different targets simultaneously (the base target and the shortcut
self-consistency target). This is a real design characteristic, not yet
established as harmful vs. a standard combined-loss pattern (many consistency-
model formulations do exactly this on purpose) — **do this run first**; if
`base_loss` still doesn't descend with `shortcut_direction_weight=0`, the
target-overlap mechanism is moot and something further upstream is the cause.
Run this **after** the composition/gate fix
([[../bug-adapter-gate-saturation-mask-mix]]) so it isn't confounded by the
gradient throttle — see do-now order in the diagnosis note.

## Config bug found (2026-07-15) — fixed via a new sibling, not a patch

The original `*_noshortcut.yaml` config meant to run this control doesn't set
`composition`/`gate_bias` at all (silently defaults to `output_composition="add"`,
not the `mask_mix` being compared against AVID). On closer look it's also
drifted in other ways unrelated to this ticket — wrong script
(`train_wan22_i2v_metaworld.py`, not `_external.py`), `temporal_length: 121`
vs. the current family's `41`, 256px resolution its own comment calls
"washed." Multiple mismatches, not just composition — patching it in place
wouldn't produce a clean sibling of the current gate-fixed baseline.

**Fixed by forking instead:** new config
`configs/diffusion_wan22_avid_gatelow_noshortcut_metaworld.yaml` — an exact
copy of `diffusion_wan22_avid_gatelow_metaworld.yaml` (the 2026-07-15
gate_bias=0.0 + grad_accum + warmup baseline —
[[exp-adapter-adaln-gatelow-metaworld-run]]) with **only**
`shortcut_anchor_prob: 0.6 → 1.0` changed (100% anchor steps → shortcut term
never fires, matching the original config's own disabling mechanism). This is
now the config to run for this ticket — genuinely matched to its shortcut-ON
sibling on everything except the shortcut term itself.
