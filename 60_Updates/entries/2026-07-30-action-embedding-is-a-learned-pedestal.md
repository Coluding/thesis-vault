---
date: 2026-07-30
category: finding
deliverable: D2
meeting:
sources: ["[[20260730-dc-parity-arms-null-action-embedding-pedestal]]", "[[20260730-avid-robotarm-follows-actions-recipe-not-data]]", "[[../../20_Tickets/experiments/exp-conditioning-decouple-encoder-bias]]"]
---

# Our action encoder learns to emit a constant — the blindness has a mechanism

## What

Ran six parity arms against the AVID reference on ACWM Robot Arm. **All four
treatments came back null** against a measured noise floor. But the diagnostic
probes built to read them found the actual mechanism:

**Our action embedding is 14× larger than the time embedding and 99.7% of it is a
constant.** Only ~0.3% varies with the action. The AVID reference on the same
data sits at ~24% — a **48× difference** on the same measurement.

And a trajectory probe shows **the constant is learned, not architectural**: at
initialisation our encoder is healthy (output *smaller* than the time embedding,
2.4% varying). By step 600 its magnitude has grown **106×** while the varying
fraction collapsed 7×.

## Why it matters

D2 moves from "our adapter is action-blind" to a specific, measured mechanism
with a reference value. The encoder is being used as a **bias generator** — a
large constant into every ResBlock helps fit the denoising objective without
using actions, which is exactly the "action-independent domain adjustment"
verdict from 2026-07-21, now with a cause.

It also **kills two of our own hypotheses**, which is the point of measuring:

- The two catalogued config divergences (`action_time_combine`, `frame_stride`)
  are not the cause — tested directly, both null.
- The encoder-architecture story (6 layers vs AVID's 2, 792k params vs 9k) is not
  the natural explanation either, since the architecture behaves fine at init.
  The planned narrow/shallow encoder arms are **on hold**.

## Evidence / sources

- Six arms, all `base_null_violation` 0: control `n3dbgq4q` 0.003288, seed
  control `l2jcz9nx` 0.003533 (**floor 0.000245**), concat `hbuu4lwx` 0.003540,
  stride-1 `1e0fe9ei` 0.003663, both `2us8hugq` 0.004310, full parity `t62nhyfu`
  0.004121. Target ≥0.02; AVID 0.029475.
- Mechanism, same probe both sides: realised/RMS per element **0.0050** (ours)
  vs **0.238** (AVID); embedding÷time 14.45× vs 0.83×.
- Trajectory: init 0.0276 RMS / 2.4% varying → step 600 2.933 RMS / 0.28%.
- Full note: [[20260730-dc-parity-arms-null-action-embedding-pedestal]].

## Caveat for the slide

Blindness is uniform across the noise schedule (0.0033–0.0034 at t=100…900), and
running our probe on AVID's timestep grid reproduces our own number — so this is
not a measurement artifact, and it retroactively validates the earlier
our-vs-AVID comparisons.

The backward-direction (gradient) measurement is weaker and should not be led
with: AVID is 3–7× higher, straddling the pre-registered bands, with 2.75× seed
spread on our side. The forward measurement carries the finding.

## Next

Intervention: prevent the encoder from supplying a constant — centre/normalise
`cond_emb` before it enters `emb`.
→ [[../../20_Tickets/experiments/exp-conditioning-decouple-encoder-bias]]
