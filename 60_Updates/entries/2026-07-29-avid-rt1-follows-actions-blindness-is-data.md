---
date: 2026-07-29
category: finding
deliverable: D2
meeting:
sources: ["[[20260729-avid-rt1-follows-actions-control]]", "[[20260728-acwm-robotarm-matrix-action-blind]]", "[[avid-vs-ours-action-conditioning]]"]
---

# The action-blindness is our data, not the recipe — AVID follows actions on RT-1

> **⚠ SUPERSEDED 2026-07-30 by
> [[2026-07-30-avid-follows-actions-on-our-data-its-our-implementation]].**
> The RT-1 measurement below is sound; the conclusion drawn from it is not. The
> only synthetic-side AVID datapoint available at the time was the 64-clip
> memorization-confounded Push Cube smoke. Running the same recipe on **full-data
> ACWM Robot Arm** shows it follows actions there too (effect_rel 0.029475, null
> 0) — so the fault is **our implementation**, not the data. Do not present the
> "it's a data problem" framing at the meeting.

## What

Probed the **original AVID recipe** trained on its **own in-distribution real
data (RT-1, full, 15k steps)** with our action-sensitivity metric. It **follows
actions**: `action_effect_rel = 0.0495` (shuffle), null control exactly 0, and
**~66% of the adapter's contribution is action-driven**. Our three ACWM adapter
runs were action-*blind* (effect_rel 0.001–0.006, ~5% action-driven).

## Why it matters

This is the decisive control for the whole D2 base-parity story. The same family
of approach (frozen base + trained action-conditioned correction) follows actions
on in-distribution real video but collapses to action-independence on our OOD
synthetic ACWM domains. **So the blindness is a data/OOD problem — not the recipe,
not the probe.** It reframes D2 from "adapters can't follow actions" to "adapters
follow actions when the data is in-distribution for the frozen base." (It also
corrects the earlier confounded read where a 64-clip AVID smoke looked blind —
that was tiny-data memorization.)

## Evidence / sources

- AVID RT-1 `93qrvr5v`, ckpt `epoch=14-step=15000`, 120 samples: effect_rel
  0.0495/0.0443 (shuffle/zero), cos 0.997, base_null 0.0, adapter_rel_contribution
  0.0755, gate 0.905. Probe: `probe_action_sensitivity.py --config avid_11M.yaml`.
- Contrast: Wan `ncztxyyo` 0.0056, DC `c3pcewxk` 0.0034, SkyReels `8zjjn7wl` 0.0013
  ([[20260728-acwm-robotarm-matrix-action-blind]]).
- Full note: [[20260729-avid-rt1-follows-actions-control]].

## Next

- **Run our Wan/DC/SkyReels adapters on RT-1** (translator + `--dataset rt1`
  built) — the direct test of whether OUR lightweight adapter follows actions
  in-distribution where ACWM failed. Add per-dim action std-normalization first.
- Caveat for the slide: AVID's is a full separate action-UNet, not our output
  adapter, and the effect is modest (0.05) — real and null-clean, not dramatic.
