---
type: experiment
date: 2026-06-27
config: configs/diffusion_wan22_avid_i2v_metaworld_noshortcut.yaml
commit:
wandb_run_id:              # wandb project: wan22-avid-i2v-metaworld-noshortcut
ckpt_path:
status: planned            # NOT YET a result — see the note below
deliverable: D3
metrics:
notes: >
  The no-shortcut control that separates a D2 failure (action conditioning
  collapses to base-parity on Wan) from a D3 failure (the consistency
  objective is at fault). Setup is fully specified; no observed outputs
  recorded yet.
---

# Wan2.2 AVID adapter, action-conditioned, NO shortcut consistency

> ⚠ **This note documents a *planned* run, not an observed result.** It has
> no logged outputs, no wandb run id, and no metrics. Per CLAUDE.md hard
> rule 6, planned runs belong in
> `20_Tickets/experiments/exp-{scope}-{slug}.md`, not in
> `30_Knowledge/experiments/`. **Either promote it (fill in run id + metrics
> once it lands) or move it to a ticket** — flagged 2026-07-26, left in place
> pending that call. Do not cite it as evidence in the meantime.

## Why this run matters to the storyline

This is the **control that protects the D3 chapter**. The narrative arrives at
Wan via the shortcut argument (curvature bias → flow matching → Wan), so a
reader will read the base-parity collapse on Wan as *the shortcut objective
failing*. It is not — it is a **D2** failure, measured on action-conditioning
runs. This run proves that, by removing the consistency loss entirely and
showing whether the collapse persists.

See [[../writing/thesis-storyline]] §6 and [[../writing/ablation-axes]]
(hypothesis row: "the shortcut objective itself").

## Goal

Does the AVID-style `backbone: wan` output-delta, action-conditioned on a
frozen Wan2.2-TI2V-5B base, learn a useful world model from **plain flow
matching alone**, with no shortcut self-consistency supervision?

## Setup

- Config: `configs/diffusion_wan22_avid_i2v_metaworld_noshortcut.yaml`
  (clone of `diffusion_wan22_avid_i2v_metaworld.yaml`).
- Job: `jobs/submit_train_wan22_avid_noshortcut.sh`
  (script `scripts/train_wan22_i2v_metaworld.py`, ckpt `ckpts/Wan2.2-TI2V-5B`).
- Only change vs the shortcut run: `shortcut_anchor_prob: 1.0`.

## Why `anchor_prob=1.0` = no consistency loss

In `Trainer._maybe_prepare_shortcut`, the flow-distillation branch computes
`do_anchor = anchor_prob >= 1.0 or ...`. With `anchor_prob=1.0` every step is
an anchor step — it returns a **None** shortcut target (and clamps
`step_level` to the smallest rung). `training_step` only adds the
shortcut/consistency terms when `"shortcut_target" in batch`, so with a None
target the sole supervision is the standard flow-matching loss.
`local_consistency_weight` / `multistep_consistency_weight` are 0 too.

`step_level` conditioning stays wired (constant smallest rung) so the adapter
input contract is unchanged — the only difference from the shortcut run is the
absence of the consistency target/loss. The multi-N eval grid is a pure
diagnostic and is unaffected.

## Compare against

The shortcut run `wan22-avid-i2v-metaworld` (same adapter/base, shortcut ON).
wandb project: `wan22-avid-i2v-metaworld-noshortcut`.

## Results

_Not yet run / not yet recorded._

## Related

- [[../writing/thesis-storyline]] — §6, the D2-vs-D3 separation this run serves
- [[../writing/ablation-axes]] — the hypothesis table
- [[20260724-metaworld-cap-shift-triangle-base-parity]] — the collapse this controls for
