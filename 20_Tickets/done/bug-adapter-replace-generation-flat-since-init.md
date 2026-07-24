---
type: bug
scope: adapter
status: done
priority: high
created: 2026-07-18
updated: 2026-07-21
resolution: shipped
resolution_note: >
  Root cause found 2026-07-20 by seam instrumentation, NOT the σ=1 /
  1-step-eval story below: every generation-eval site built adapted_cond
  without the per-frame `action_seq` the xattn adapter trained on, so
  Wan21OutputAdapter silently fell back to one aggregated OOD action token
  and the adapter output collapsed (cos vs base 0.997 -> 0.634 measured).
  Under replace, that collapsed output IS the velocity -> noise rollouts,
  flat since init. The 2026-07-19 RETRACTION of exactly this hypothesis was
  itself wrong — `action_per_frame: false` only gates the encoder input;
  the preprocessor emits `action_seq` unconditionally and wan.py reads it
  directly. Fixed in trainer.py (both eval paths thread action_seq),
  compare script, and a fail-loud raise in wan.py. Validated end-to-end on
  wandb y1jrgxqp (adapted FID 518->58 by step 600, ≈ base). Consequence:
  generation metrics of all earlier action_seq-xattn runs are invalid.
  Full numbers: 30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe.md
closed_at: 2026-07-21
related: ["[[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]]", "[[../experiments/exp-adapter-wan-replace-metaworld-run]]", "[[../experiments/exp-adapter-xattn-gatelow-metaworld-run]]", "[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]"]
---

# bug: replace-composition eval video is flat at the noise floor since step 0, despite descending training loss

## RESOLVED (2026-07-21) — read this first; the sections below are the investigation trail

The cause was the **train/generation conditioning mismatch on the
cross-attention token path** — the hypothesis retracted mid-ticket on
2026-07-19. The retraction was wrong: `action_per_frame: false` governs only
the *encoder* input (`cond["action"]`), not the xattn token path
(`cond["action_seq"]`, emitted unconditionally by the preprocessor and read
directly in `adapters/output/wan.py`). Training cross-attended over per-frame
tokens; every generation-eval site passed only the aggregated `action`, and
the adapter's silent fallback collapsed its output (measured cos vs base
0.997 → 0.634 on the step-1500 checkpoint). The "1-step quality metrics"
reframe above was a red herring — after the fix, the same eval pipeline
produces coherent rollouts at FID ≈ base (wandb `y1jrgxqp`, step 300+).

Fix + validation + the follow-up σ-sweep/action-probe measurements:
[[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]].
Remaining real problem (base-cloning, action-blindness) tracked in
[[../experiments/exp-adapter-replace-nobase-overfit]] and
[[../bug-adapter-gate-saturation-mask-mix]].

## RESOLUTION-GRADE REFRAME (2026-07-19, late) — the quality metrics are 1-STEP generations

Code-read finding that supersedes everything below: **every quality-metric
call site — PSNR/SSIM/LPIPS/MSE *and* FID/FVD — passes
`num_steps = quality_eval_num_steps or inference_num_steps`
(`training/trainer.py:575,632,704`), and this config sets
`quality_eval_num_steps: 1`.** All the "generation is noise" numbers are
single-Euler-step generations `x̂0 = x1 − v(x1, σ=1)` — there is no 25-50-step
rollout in them, so every compounding/exposure-bias framing below rested on a
false premise. The metric exclusively probes the velocity field at σ=1:

- σ=1 is the one input with **zero ground-truth leakage** (for σ<1 the
  training input hands the model `(1−σ)·z0` of the answer);
- flat-uniform training σ almost never supervises it (the shift-schedule
  sampler that would is the dead code in
  [[bug-losses-flow-boundary-sampling-unused]]);
- the pretrained base is strong exactly there (1-step FID 66).

With this, all observations are consistent with **H2 alone** (σ≈1 never
learned; uniform-σ parity is a leakage-inflated weak certificate): replace's
1-step FID flat at ~520 = "adapter's v(noise, σ=1) is useless at every
checkpoint"; gatelow's 81 = "the mask keeps the σ=1-competent base in the
output"; training loss and eval delta at parity = "measured where leakage
dominates." No compounding story required; no seam bug required (both
inspected candidates were retracted).

**Secondary confound to measure while testing:** eval generation runs real
CFG at `inference_guide_scale: 5.0` with prompts, so under `replace` the
CFG formula `neg + 5·(pos − neg)` runs on *adapter* outputs — any spurious
pos/neg-branch difference is amplified 5x.

**Decisive confirmation test (no generation pipeline involved):** paired
denoise delta at forced σ ∈ {0.5, 0.9, 0.99, 1.0} on a trained replace
adapter. Prediction under H2: delta ≈ 0 at σ=0.5, collapsing toward the
untrained value as σ→1. Needs trained weights → short local re-run
(~600-900 steps on the 3090, `training.output_dir` set so a checkpoint is
finally saved, new gate/grad instrumentation live). Caveat to carry: the
step-size *grid* videos in wandb include 25/50-step rows — if those are
also pure noise, that's consistent with H2 too (a σ≈1-poisoned start), but
worth confirming visually which rows fail.

## Observed (sourced: wandb `5cxstyh4` replace vs `bcipghvw` gatelow, pulled via `wandb.Api()` 2026-07-18)

| run | step | `eval/adapted/fid` | `eval/adapted/psnr` | `eval/adapted/ssim` |
|---|---|---|---|---|
| replace | 0 | 530.18 | 11.36 | 0.329 |
| replace | 900 | 521.44 | 10.98 | 0.299 |
| replace | 1500 | 524.35 | 11.16 | 0.318 |
| replace | 2400 | 520.88 | 11.32 | 0.333 |
| gatelow | 0 | 482.99 | 13.17 | 0.433 |
| gatelow | 300 | 81.34 | 16.12 | 0.797 |
| gatelow | 600 | 96.82 | 16.50 | 0.827 |

`replace`'s decoded-video quality metrics at step 2400 are statistically
indistinguishable from step 0, over the whole run — despite `train/loss`
visibly descending 1.63 → 0.109 over that same window
([[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]]).
Sibling `gatelow` (same adapter — `hidden_dim: 256`, zero-init final layer,
same backbone, same dataset) moves off the same noise-level FID (483) to
near-base quality (81) within 300 steps, 8x fewer steps than replace's still-flat
2400. This is the user's direct observation ("loss went down but the video is
just noise... trained for 3k steps, so maybe there is a deeper issue").

## CURRENT UNDERSTANDING (2026-07-19) — supersedes the hypothesis sections below

Reset of the analysis after the earlier layers (capacity → exposure bias →
sigma-region → probe-as-input-class) proved muddled. The sections below are
kept for history; this section is the operative one.

**The two load-bearing facts:**

1. **The replace rollout's endpoint is statistically identical to the
   untrained (zero-init) endpoint** — FID 530→521, PSNR 11.4→11.3 from step 0
   to step 3600. At step 0 the zero-init head means v̂=0 exactly, so the
   rollout literally ends at its noise starting point. An identical endpoint
   after 3600 steps of training means the adapter's effective velocity
   *inside `generate()`* is still ≈0 (or unstructured of similar effect).
   This kills every "modest error compounds" story: a modestly-wrong model
   moves x *somewhere*; here x doesn't move. The right question is not "why
   does error compound" but **"why does the adapter output ≈nothing inside
   the sampler while reaching base parity on training batches."**
2. **Every good metric is computed at the training seam
   (`AdaptedModel.forward` on preprocessor batches); every bad metric at the
   generation seam (`generate()` → `_ComposedDiT` → `compose_fn`).** The
   contradiction maps exactly onto that code split.

**Corrected inference error:** gatelow's decent videos never certified the
adapter's generation path. Gatelow adapted FID 81 vs base 66 — *slightly
worse than base* — is exactly what the mask_mix composition produces when the
adapter contributes ≈0/garbage at generation and the base carries the video.
**Nothing in any run so far has ever verified that the adapter produces a
meaningful prediction inside `generate()`.** Replace is simply the first
config with no base to hide behind.

**~~Concrete verified mismatch found at the seam~~ RETRACTED (2026-07-19,
same day):** the claimed train/inference `action_seq`-vs-`action` mismatch
does **not** apply to these runs. Both
`diffusion_wan22_avid_xattn_replace_metaworld.yaml:119` and
`..._gatelow_metaworld.yaml:118` set `action_per_frame: false` (despite the
gatelow config's own header comment describing `true` as the intent — a
separate config-vs-comment inconsistency worth noting), so the preprocessor
never emitted `action_seq` during training either. Training and generation
both feed the aggregated `action` vector through the same
`Wan21OutputAdapter` fallback — the forms match; no mismatch on this run.
The fallback + in-code TODO ("if we trained on action_seq then we shouldn't
be able to pass aggregated actions during inference") remains a real footgun
for future `action_per_frame: true` runs, but cannot explain `5cxstyh4`.
With this and the state-clearing asymmetry both ruled out by code-read, no
specific seam bug has been identified by inspection — the seam-diff test
below is the way to find (or exonerate) one empirically.

**Surviving second hypothesis (independent of the bug):** training-seam
parity is a weak certificate because for σ<1, `x_t = (1−σ)z0 + σ·noise`
leaks `(1−σ)` worth of ground truth into the input. Uniform-σ batch averages
are dominated by inputs where much of the answer is given. Generation starts
at σ≈1 (nothing given; `shift=5.0` concentrates solver steps there) — the
base handles σ≈1 from pretraining, the adapter saw σ>0.98 in ~2% of samples.
Both this and the seam bug can be true simultaneously. The frozen-probe
"input-class" story is retracted as evidence (n=1, unknown σ).

**Revised experiment order (supersedes the numbered list below):**

1. **Seam-diff test — no checkpoint needed.** Identical `(x_t, t, cond)`
   through `AdaptedModel.forward` vs the `_ComposedDiT`/`compose_fn` path;
   diff adapter inputs and outputs. Also log the actual `cond`/`t` the
   eval-video path passes vs a training batch's. Surfaces the `action_seq`
   mismatch mechanically plus any t-form/encoder-shape divergence. Works
   with untrained weights; minutes.
2. **Instrument `generate()`:** per solver step, log ‖adapter velocity‖ and
   ‖Δx‖ — distinguishes "outputs ≈0 in the sampler" from "outputs something
   large but wrong."
3. **Fix the conditioning mismatch** (thread per-frame `action_seq` through
   the eval-video path) and re-run short — with `training.output_dir` set so
   checkpoints exist this time, and the new gate/grad-norm logging live.
4. **σ-sliced eval delta** (forced σ ∈ {0.9, 0.98, 1.0}) on that checkpoint —
   tests the leakage hypothesis separately.

---

## Why this needs investigation, not just an interpretation

A zero-init adapter head explains step 0 (composed output = literally zero
velocity under `replace`, so the rollout never leaves the initial noise). It
does **not** explain 2400 steps of clearly-improving single-step denoising
loss producing **zero** measurable improvement in full 25-50-step generation.
Single-step training loss and full-rollout generation have never been
directly compared on the *same* checkpoint in this project — that comparison
is the missing piece.

## Leading hypothesis (analysed estimate — reasoning shown, not confirmed)

**Revised 2026-07-18** (the original "capacity gap" framing below doesn't
survive a direct challenge: if the adapter's single-step loss is ≈ the
base's, why would generation be pure noise rather than just "somewhat worse
than base"? A capacity shortfall alone predicts graceful degradation, not
collapse to noise.)

**Exposure bias / off-manifold drift, unconstrained by the training
objective.** `train/loss` is measured only on ground-truth-anchored inputs
(`x_t` built directly from a real encoded clip via the flat-uniform-σ recipe
in `wan22_batch_preprocessor.py`). Matching the base's loss there only means
the adapter predicts an accurate velocity *when shown an `x_t` a real clip
actually produced* — it says nothing about its behavior off that manifold.
`generate()` integrates the adapter's **own** predictions for 25-50 steps
from pure noise; after step 1, every subsequent `x_t` is something the
adapter itself produced, not a ground-truth-anchored sample — exactly the
regime the training loss never constrains. A small, systematic (not
per-sample-random) bias in the field — fully compatible with a low averaged
L2 loss, since MSE against a noisy per-sample target doesn't pin down the
field everywhere — compounds across dozens of sequential steps and can push
`x_t` far outside anything the network or the VAE decoder has ever seen.

This also explains the gatelow-vs-replace contrast better than capacity did:
gatelow's composed output (`base·gate + adapter·(1-gate)`) keeps the
(well-behaved, massively-pretrained) base present at **every** step, so even
a biased adapter contribution gets pulled back toward the base's own
trajectory each step. `replace` has no such anchor — nothing corrects drift
once it starts.

Capacity (`hidden_dim: 256` vs. the ~4-5 order-of-magnitude larger frozen
Wan2.2 TI2V-5B, `self.dit` in `models/base/wan_ti2v.py`) may still be a
secondary contributing factor — a larger net might generalize its
self-correcting behavior better off-manifold — but it's not the primary
explanation on its own.

**Not ruled out:** a wiring/parameterization bug specific to the `replace`
path through `generate()` → `_ComposedDiT` (`models/base/wan_ti2v.py:54-99`)
that only manifests under full multistep rollout, not under the single-step
training forward (`AdaptedModel.forward`, `models/adapted_model.py:90-112`).
These two code paths are structurally different (`_ComposedDiT.__call__`
memoizes/replaces per-step during `generate()`; training calls
`_compose_with_adapter` once per batch) and have never been cross-checked
against each other on the same weights. Experiment #1 below is designed to
separate this from exposure bias directly.

## The concrete discrepancy (2026-07-19) — not a hypothesis, measured

Full `eval_denoise_adapter_delta` (batch-averaged, fresh randomly-resampled
held-out batch each eval) vs. `eval_probe_denoise_delta` (one frozen
`(x_t, t, noise, target)` sample, fixed forever) history for `5cxstyh4`:

| step | `eval_denoise_adapter_delta` | `eval_probe_denoise_delta` |
|---|---|---|
| 900 | −0.00125 | −0.11193 |
| 1800 | +0.00734 | −0.10101 |
| 2400 | +0.00148 | −0.07364 |
| 3000 | +0.01892 | −0.07225 |
| 3600 | +0.00894 | −0.06723 |

The batch-averaged metric reaches parity with base by ~step 900 and is
**net-positive at several later checkpoints** (adapter beats base on
average). The frozen probe shows a **persistent, plateaued** gap — flat at
−0.07 to −0.11 from step ~900 onward (not continuing to close; corrects the
"still closing" framing given earlier in
[[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]],
which was based on an incomplete read of this run's history). On that one
fixed sample, `probe_denoise_adapted` (~0.16-0.29) is 2-3x `probe_denoise_base`
(constant 0.088) — a substantial, single-step, non-rollout failure, not a
small bias.

**This means the fit is not uniform across input space.** On "typical"
randomly-sampled inputs (dominating the batch-averaged metric), the adapter
is genuinely at or above base parity. On whatever specific input the frozen
probe is, it's substantially and persistently worse — invisible in the
batch-averaged number because it's a minority-of-input-space failure diluted
into an average dominated by easy cases. This raises the floor of confidence
under the front-loaded/sigma-region hypothesis below considerably — it's no
longer speculation that *some* region of input space is badly undertrained,
it's measured. What's still open is *which* characteristic of the probe
sample (its σ/timestep pattern, specific clip, specific action) is
responsible — decoding it directly (experiment #1) answers this without
needing to probe sigma-space blind.

## The magnitude doesn't add up (2026-07-19, discussion follow-up)

The frozen-probe gap (`probe_denoise_adapted` 0.16-0.29 vs. `probe_denoise_base`
0.088, i.e. 2-3x base's own error) is a **modest** single-step degradation.
Integrated smoothly over a 25-50 step rollout, an error of that magnitude
should produce blurrier/lower-fidelity video, not something statistically
indistinguishable from random noise (`eval/adapted/fid` ~520, matching the
untrained zero-init baseline). The measured delta is too small to explain the
observed collapse on its own. Two live implications, not mutually exclusive:

1. **The frozen probe's one fixed sigma isn't the worst case.** It only
   samples one point in sigma-space; there could be a much worse region
   (plausibly near pure noise, where generation starts) with a far larger gap
   that this metric has never measured. Motivates measuring error across the
   *full* sigma range the sampler actually visits, not just the one frozen
   point — promoted to a required experiment, not optional (see #2 below,
   revised).
2. **Or the ~0.1 probe delta is largely a red herring** and something not
   captured by this training-objective metric at all is broken in the
   generation-time composition path specifically — "modest measured error,
   catastrophic output" is more consistent with something concretely wrong
   (scale mismatch, sign error, NaN/inf, a prediction-type mismatch in how
   `_ComposedDiT` feeds the adapter's output into the UniPC solver) than with
   smooth quality degradation. Raises the priority of experiment #5
   (`_ComposedDiT` vs. training-seam cross-check) relative to the
   sigma-undertraining story.

## Sharpest current guess (2026-07-18, discussion follow-up)

The *complete flatness* of eval quality (not gradual improvement, not even
slow) across 2400+ steps of visibly-improving single-step loss argues against
plain gradual exposure bias on its own — a 15x-improving field should buy
something under compounding unless the failure is front-loaded. Point
estimate, ranked:

1. **Front-loaded failure at the high-sigma start of the rollout.**
   `generate()` starts from pure noise (high sigma) and `sample_solver="unipc"`
   with `shift=5.0` concentrates early steps there by design. Training sigma
   is flat-uniform (`wan22_batch_preprocessor.py`) — untested whether it
   under-serves that specific high-sigma band relative to what the warped
   inference schedule needs (this is the live, not-yet-tested part of
   [[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]] /
   `bug-losses-flow-boundary-sampling-unused` — note the *original*
   "v-pred degenerates near t→0" justification for that ticket was already
   disproven; this is a narrower, still-open claim). If the first 1-2 steps
   are already badly wrong, the trajectory is doomed before gradual
   compounding ever gets a chance to be gradual — while the aggregate loss
   keeps improving because it's averaged over the whole (mostly easier,
   mid-range) sigma distribution. Reconciles "loss improving" with "video
   never moves" better than plain exposure bias alone.
2. ~~Wiring bug: `forward()` clears adapter state before composing,
   `_compose_with_adapter` (what `generate()`'s `compose_fn` calls) doesn't.~~
   **Ruled out 2026-07-19** — checked `adapters/output/wan.py`:
   `Wan21OutputAdapter` implements neither `clear_captured_base_features` nor
   `clear_dynamic_parameters` (only `hypernetworks/`, `low_rank/common.py`,
   and `hidden_states/unicon.py` do). The `hasattr()` guards in `forward()`
   are no-ops for this adapter in both the training and generation paths —
   this specific candidate is dead, not a live asymmetry.
3. Gradual exposure bias (the original framing) — still plausible, ranked
   third given the flatness argument above.

Experiment #1 (single-step decode) is the fastest way to weaken/rule out #2.
Experiment #2 (per-step drift, specifically: is step 1 already bad, or does
it degrade gradually?) is the fastest way to distinguish #1 from #3.

## Experiments to run (prioritized — cheapest/most decisive first)

1. **Single-step probe-batch denoise decode (the decisive discriminator).**
   Load the current replace checkpoint, take the frozen probe batch's
   `(x_t, t)` at a couple of representative sigma values (ground-truth-anchored,
   built exactly like training does), run `AdaptedModel.forward()` once (the
   *training* seam, not `generate()`), decode the resulting velocity/x0
   through the VAE, and look at it directly. **If it looks reasonable**, the
   on-manifold fit is genuinely fine and the failure is specific to the
   iterative `generate()` rollout — confirms exposure bias/off-manifold drift
   (or, separately, a `_ComposedDiT` wiring bug — see #5). **If it already
   looks incoherent**, the loss-vs-video paradox has a different explanation
   than either hypothesis above and needs a fresh look (e.g. the decode step
   itself, or the loss metric not meaning what we think). Cheapest, no new
   training required.
2. **Trajectory-drift tracking during an actual `generate()` rollout, PLUS
   per-sigma-bucket eval loss across the full range (not just the one frozen
   point).** At each sampler step, decode (or otherwise compare) the current
   `x_t` against what the *base's own* rollout looks like at the matched
   step, or against real encoded-clip latent statistics — distinguishes
   "drift starts small and compounds gradually" from "diverges immediately at
   step 1." Separately, bucket held-out eval loss by sigma decile (cheap
   addition to the existing eval loop) to check whether some region has a
   gap far larger than the frozen probe's 2-3x — needed because the probe's
   modest ~0.1 delta is, by itself, too small to explain noise-level output
   (see "The magnitude doesn't add up" above).
3. **Sampler step-count sweep on the existing checkpoint.** Regenerate the
   same probe conditioning at `sampling_steps` = 4, 8, 16, 25, 50. Exposure
   bias predicts quality degrading as step count grows (more steps = more
   compounding); a hard wiring bug predicts bad output regardless of step
   count.
4. **Latent-statistics sanity check.** Compare per-channel mean/std of the
   adapter's raw predicted output (a few sigma values) against real encoded-clip
   latent statistics — rules a scale/parameterization mismatch in or out as a
   confound.
5. **`_ComposedDiT` vs. training-seam cross-check.** Directly compare
   `AdaptedModel.forward()`'s output against `_ComposedDiT.__call__`'s output
   for the *identical* `(x_t, t)` and checkpoint — the two code paths have
   never been checked against each other on the same weights, and are
   structurally different enough (memoization, per-step vs per-batch
   composition) that a divergence here would be a concrete, fixable bug
   rather than an inherent exposure-bias limitation.
6. **Extended-duration replace run** (only after #1-5 narrow the cause) — run
   well past 2400 steps to see whether it *ever* escapes the noise floor,
   informative on "needs more steps to become robust off-manifold" vs
   "structurally can't without an anchor." Most expensive, sequence last.

## Does this unify with the mask_mix "looks like a copy" symptom? (2026-07-19)

No — checked and rejected. If `replace`'s failure were the same "adapter
converged to clone base" fact that explains mask_mix
([[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]]),
`replace` (which outputs `adapter_output` directly, nothing else) should
produce roughly **base-quality** video — cloning and outputting the clone
looks like a worse copy of base, not random noise. It doesn't: FID stays at
literal noise level (520+) for the whole run. So mask_mix's symptom is fully
explained at the training seam alone (no generation-pipeline defect needed —
mixing two near-identical predictions at any gate ratio gives ≈base,
directly from the measured `denoise_adapter_delta` collapse); `replace`'s
symptom requires something additional, located specifically between the
training seam and the `generate()` rollout — which is what this ticket's
remaining branches (front-loaded high-sigma undertraining vs. a genuine
`_ComposedDiT` pipeline defect) are for. The two composition modes share the
same underlying training-level weakness, but it's the *different composition
math* (base-anchored vs. unanchored), not a shared generation-pipeline bug,
that explains why it manifests differently.

## Guardrails

- Don't conflate this with the separate weak-action-signal question
  ([[../experiments/exp-conditioning-action-shuffle-ablation]],
  [[../experiments/exp-shortcut-action-free-isolation]]) — this ticket is about whether
  `replace`-mode generation reflects *any* of what the adapter has learned at
  all, prior to and independent of whether what it learns uses the action.
- `replace` is already flagged diagnostic-only, not a D2/D4 evidence path
  ([[../experiments/exp-adapter-wan-replace-metaworld-run]] guardrails) — this investigation
  is about understanding the failure mode cleanly, not about rescuing
  `replace` as a shipped composition mode.
