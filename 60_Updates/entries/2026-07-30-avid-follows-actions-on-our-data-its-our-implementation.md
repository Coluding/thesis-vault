---
date: 2026-07-30
category: finding
deliverable: D2
meeting:
sources: ["[[20260730-avid-robotarm-follows-actions-recipe-not-data]]", "[[20260729-avid-rt1-follows-actions-control]]", "[[20260728-acwm-robotarm-matrix-action-blind]]", "[[../../20_Tickets/experiments/exp-adapter-our-framework-avid-replication-robotarm]]"]
---

# The action-blindness is ours, not the data — AVID follows actions on the very dataset we failed on

## What

Ran the **unmodified AVID recipe on ACWM Robot Arm** — the exact synthetic
dataset where our Wan/DC/SkyReels adapters were action-blind — and probed it with
our action-sensitivity metric. It **follows actions**: `action_effect_rel =
0.029475` (shuffle), null control exactly 0, **~42% of the adapter's contribution
action-driven**. Ours on the same data: 0.0056 / 0.0034 / 0.0013, ~5%
action-driven.

Same frozen base weights, same data, same probe. Only the implementation differs,
and the metric moves by an order of magnitude.

## Why it matters

**This reverses last week's conclusion.** The 2026-07-29 entry
([[2026-07-29-avid-rt1-follows-actions-blindness-is-data]]) argued the blindness
was a **data/OOD** problem, because AVID followed actions on real in-distribution
RT-1 while everything on synthetic ACWM went blind. But the only synthetic-side
AVID datapoint available then was a **64-clip memorization smoke** (`423pjv8y`,
effect_rel 0.0015) that the note itself flagged as unusable. With the clean
full-data synthetic cell now filled, the recipe follows actions on synthetic OOD
data too.

So the data is **not** the blocker. D2's problem is in **our implementation or our
adapter design** — which is a far more tractable place to be, because we now have
a reference that provably follows actions on the same substrate, making the gap
bisectable instead of speculative.

It also retires a five-week dead end: since 2026-07-12 the working hypothesis has
cycled through injection mechanism, gate saturation, optimization traps and
finally data/OOD. This is the first result that localises the fault.

## Evidence / sources

- **AVID · Robot Arm** `rqp4s3gp`, ckpt `epoch=4-step=5000`, 120 samples
  (8 batches × 5 timesteps × 3 draws): effect_rel 0.029475 shuffle / 0.027991
  zero, cos 0.9990, `base_null_violation` 0.000, `adapter_rel_contribution`
  0.069838, gate 0.8759. Training healthy: loss 0.279→0.015, `mask_mean`
  0.501→0.88, FID 60.2, FVD 216.5.
- **Ours, same data:** Wan `ncztxyyo` 0.0056, DC `c3pcewxk` 0.0034 (step-~801
  snapshot; **0.004238** at the run's endpoint), SkyReels `8zjjn7wl` 0.0013
  ([[20260728-acwm-robotarm-matrix-action-blind]]).
- **The sharper contrast — ours isn't cloning, it's contributing the wrong
  thing.** Our DC adapter does **4.6× more work** than AVID's
  (`adapter_rel_contribution` 0.319 vs 0.070; gate 0.597 ≈40% adapter weight vs
  0.876 ≈12%) yet only **1.5%** of it is action-driven, against AVID's **42%**.
  Composition form is identical, so this is not the base-cloning failure mode —
  it is a large, purely action-independent correction, which is exactly what the
  concat-vs-add mechanism predicts.
- **Prior real-data reference:** AVID · RT-1 `93qrvr5v` 0.0495, ratio 0.66.
- Full note: [[20260730-avid-robotarm-follows-actions-recipe-not-data]].

## Caveat for the slide

**Not training-matched.** The probe is at step 5000; our comparison runs were at
~1172 / ~801 / ~897. AVID has had 4–6× more training. Step 5000 is the earliest
checkpoint that exists (saved every 5 epochs), so the match has to come from our
side: push the Wan/DC Robot Arm runs to ~5000 and re-probe. Until then the
headline is provisional — this is the same class of confound that already
invalidated the Push Cube reference, so it is not a formality.

Also worth stating plainly at the meeting: **training health and good-looking
videos did not predict this.** Push Cube trained cleanly and probed blind; our
2026-07-21 run beat the base on PSNR at every eval while shuffled actions moved
the loss <1e-5. Only the probe separates them.

## Next

- Step-matched probe (~1000 steps) — gates everything above.
- Re-probe at 15000 for parity with the RT-1 reference.
- **Our framework vs the reference on the same substrate:**
  [[../../20_Tickets/experiments/exp-adapter-our-framework-avid-replication-robotarm]].
