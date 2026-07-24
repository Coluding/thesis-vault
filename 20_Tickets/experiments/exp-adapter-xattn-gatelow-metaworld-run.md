---
type: exp
scope: adapter
status: open
priority: high
created: 2026-07-15
updated: 2026-07-16
resolution:
resolution_note:
closed_at:
related: ["[[../../50_Decisions/open/action-conditioning-injection-mechanism]]", "[[../feat-adapter-wan-action-cross-attention]]", "[[../../30_Knowledge/experiments/20260712-wan-xattn-action-no-improvement]]", "[[exp-adapter-adaln-gatelow-metaworld-run]]", "[[../../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]]"]
---

# exp: cross-attention Wan adapter with gate_bias fixed — isolates the gate effect on the xattn arm

**Config:** `configs/diffusion_wan22_avid_xattn_gatelow_metaworld.yaml`.
**As of 2026-07-15 the config tests gate_bias in isolation, NOT the fully-fixed
arm** — see "What this config actually tests" below, corrected after
re-reading the live file (it's been hand-edited since first written).

## Why

The original cross-attention run (`xb76ptw2`) is **retracted** — it violated
its own design's non-negotiable precondition (unbinned, unmasked action
tokens — see
[[../../30_Knowledge/experiments/20260712-wan-xattn-action-no-improvement]] and
[[../feat-adapter-wan-action-cross-attention]]) *and* ran at the saturated
`gate_bias: 4.0`. Two confounds, neither controlled.

## What this config actually tests (current state, re-verified 2026-07-15)

- `gate_bias: 4.0 → 0.0` ✅ fixed — matches the AVID-validated balanced init.
- `action_per_frame: false` — **the binning fix is currently OFF.**
  `action_seq_len` is unset, so `cond["action_seq"]` is still the unbinned
  raw-pixel-frame sequence (~41 tokens) against the 11-latent-frame grid —
  the same shape as the original `xb76ptw2` bug. This run therefore isolates
  **"does gate_bias alone recover health, even with the binning bug still
  present"** — a distinct, valid question, not the doubly-fixed arm the
  original version of this ticket described.
- `shortcut_anchor_prob: 1.0` — shortcut self-consistency is effectively
  **disabled** (100% anchor steps, no shortcut-target steps ever sampled).
  Pure single-step flow-matching loss — removes the "entangled shortcut loss"
  confound (finding #5 in the AVID-vs-ours comparison) as a variable here,
  and moves this run closer to AVID's own training regime (no shortcut
  analogue at all).
- `grad_accum_steps` / `linear_warmup_steps` — **not set** (still effective
  batch = physical batch only, no warmup). Not matched to the AdaLN sibling
  run yet.

## Procedure

Run **alongside** [[exp-adapter-adaln-gatelow-metaworld-run]]. For a genuinely
matched comparison, either add `grad_accum_steps`/`linear_warmup_steps` here
to match the AdaLN run, or accept this as a looser comparison and note the
mismatch when reading results.

**Open question for whoever runs this next:** is a doubly-fixed sibling
(gate_bias=0.0 **and** `action_per_frame: true`, the fully-controlled
injection-mechanism test) still wanted as a follow-up? That's the version
needed to cleanly resolve
[[../../50_Decisions/open/action-conditioning-injection-mechanism]] — this
config alone only tells you about the gate, not about binned cross-attention
vs. AdaLN.

## Decision rule

Compare against the AdaLN sibling run ([[exp-adapter-adaln-gatelow-metaworld-run]])
on `train/loss` trajectory shape and `eval_base_loss`. **Because binning is
still off here, this comparison can only speak to the gate, not to injection
mechanism:**

- **This run is healthy (comparable shape to the AdaLN sibling / the AVID
  reference)** ⇒ gate_bias was the dominant confound on the xattn arm too,
  independent of binning — good news, and motivates actually building the
  doubly-fixed sibling to get the real injection-mechanism answer.
- **Still flat/noisy despite gate_bias=0.0** ⇒ either the binning bug alone is
  enough to break the xattn arm regardless of gate, or unmasked global
  attention itself (present with or without binning) is the deeper problem —
  can't distinguish these two without the doubly-fixed sibling run.

**Resolving [[../../50_Decisions/open/action-conditioning-injection-mechanism]]
still requires the doubly-fixed sibling** (gate_bias=0.0 *and*
`action_per_frame: true`) — this run alone is informative but not sufficient
for that decision.

## Guardrails

Both this and the AdaLN sibling run must be genuinely matched (data, steps,
gate_bias, accumulation, warmup) or the comparison is meaningless — check
configs side-by-side before launching, don't rely on memory.

## Result (2026-07-16, wandb `bcipghvw`, crashed @ 624 steps)

Ran. `train/denoise_adapter_delta` collapses from −0.46 (first-20-step mean)
to +0.00125 (last-20-step mean) within ~70 steps and stays pinned there —
composed output converges to ≈ the frozen base's own prediction, not a new
optimum below it. Eval quality metrics: adapted is a **wash** vs base
(marginally better MSE/PSNR/SSIM, worse FID/FVD/LPIPS) — no clear win either
way. Full numbers and cross-run comparison (incl. the overfit and replace
runs, which show the same clone-base convergence independent of gate_bias) in
[[../../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]].

**Decision rule outcome:** "healthy, comparable shape to AdaLN sibling" —
partially. The pathological gate saturation is gone (delta no longer stuck at
a large negative value), but the run does **not** show the adapter learning
a genuinely useful action-conditioned correction — it shows the adapter
converging to redundancy with the base. This reframes the open question: the
doubly-fixed sibling (gate_bias=0.0 **and** `action_per_frame: true`) is
still worth running for
[[../../50_Decisions/open/action-conditioning-injection-mechanism]], but expect
it to face the same clone-base dynamic unless the weak-action-signal problem
([[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]])
is addressed first. Priority should shift toward
[[exp-conditioning-action-shuffle-ablation]] and
[[exp-shortcut-action-free-isolation]], which can actually distinguish
"ignores the action" from "found a degenerate optimum regardless of action."
