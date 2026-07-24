---
type: feat
scope: data
status: open
priority: medium
created: 2026-07-24
updated: 2026-07-24
resolution:
resolution_note:
closed_at:
related: ["[[chore-data-action-frame-alignment-audit]]", "[[bug-data-acwm-decord-dataloader-fork-deadlock]]", "[[../50_Decisions/open/multimodal-adapter-broadening]]"]
---

# feat: add BridgeData V2 as a real-world action-conditioned dataset (action-only)

## What

Add BridgeData V2 (Walke, Black, … Levine — arXiv:2308.12952, CoRL 2023) as a
second dataset behind the WAN-2.2 i2v pipeline, alongside MetaWorld. It becomes
the first **real-world** action-conditioned source: ~53.9k teleop trajectories,
24 scenes, WidowX 250 6-DOF arm, 5 Hz, over-the-shoulder RGB (native 640×480).

## Why (fit under the WAN-2.2 base)

- WAN-2.2's frozen prior is a **real-world** video model, so BridgeData scenes
  are plausibly **in-distribution** for the frozen base (analysed estimate — WAN
  is a general real-video model vs. synthetic MetaWorld renders; not yet
  measured). That makes the base-vs-adapted delta test honest, which is exactly
  what the MetaWorld xattn/replace diagnoses have been fighting to get.
- Actions are **7-DoF continuous end-effector deltas** — a richer, physically
  grounded action signal than MetaWorld's, i.e. a better testbed for "does the
  adapter follow actions vs. clone the base." The shuffle/zero-action probe
  transfers directly.

## Scope — action-only (explicit decision, this ticket)

- **Adapter carries the 7-DoF action, and only the action.**
- **Ignore BridgeData's language instructions.** At train/eval time feed the
  base a **generic / null-ish prompt** (constant across trajectories) so the
  text-context path carries no task information. Rationale (user, 2026-07-24):
  we want the adapter to *actually contribute* the dynamics signal — if the base
  gets a per-trajectory language instruction it can solve the task from text and
  the adapter is again left with nothing to learn. General prompts force the
  action to be the only route to the future frames.
- **i2v first-frame conditioning stays** (that's how the base is anchored) —
  it's the language channel we're deliberately starving, not the image channel.
- Over-the-shoulder RGB view only. Defer depth / wrist / randomized-pose cams.

## Open sub-decisions (resolve during implementation)

1. **Source format / resolution.** Raw 640×480 (JPEG/PNG/pkl) vs. the
   pre-processed 256×256 RLDS/TFDS release. WAN runs at higher native res, so
   raw 640×480 may suit the base better — weigh against the disk + conversion
   cost of the raw pipeline. _needs decision_
2. **Action representation.** BridgeData actions are EE deltas + gripper; confirm
   the exact 7-dim layout and whether/how to normalise to match the adapter's
   expected `[B, T, A]` per-frame `action_seq` (see the per-frame vs. aggregated
   landmine below). _needs verification against the raw data_

## Implementation notes

- New translator `data/translators/bridgedata_v2.py` implementing the
  `Translator` contract (`list_episodes` → `EpisodeRef`, `load_clip(ref, start,
  length, stride) -> dict`) in the sibling repo, register it in the dataset
  registry, mirror the MetaWorld/ACWM translators.
- **Fork-safety is mandatory** (see [[bug-data-acwm-decord-dataloader-fork-deadlock]]):
  drop any open file/video handles in `__getstate__`, lazily re-open per worker,
  `close()` after the geometry probe, spawn context when `num_workers > 0`.
- Feed the adapter **per-frame `action_seq` [B,T,A]**, not the aggregated `[B,A]`
  vector — the aggregation path is the documented WAN action-signal landmine; the
  action↔latent-frame alignment must be audited exactly as in
  [[chore-data-action-frame-alignment-audit]] (5 Hz control → WAN latent frame
  rate mapping).
- Run through the **`wan2.2_external` script** (`train_wan22_i2v_metaworld_external.py`
  / a bridgedata variant) — it's the only one wired for `action_per_frame` /
  `action_seq_len` and loads the real pretrained WAN. Do **not** use the plain
  `wan2.2` provider (unverified/random prior).

## Acceptance

- [ ] Translator lists episodes and streams clips (RGB + 7-DoF action_seq) with
      `num_workers > 0`, no fork deadlock.
- [ ] Latent precompute (`precompute_latents.py`) runs end-to-end on a subset.
- [ ] Generic-prompt path confirmed: base receives a constant null-ish text
      context (verified in the batch, not just config).
- [ ] One smoke training run on a scene subset with base-vs-adapted eval metrics
      + action shuffle/zero probe logged.

## Links

- Dataset: https://bridgedata-v2.github.io/ · https://rail-berkeley.github.io/bridgedata/
- Code/loader ref: https://github.com/rail-berkeley/bridge_data_v2
- Paper: arXiv:2308.12952
