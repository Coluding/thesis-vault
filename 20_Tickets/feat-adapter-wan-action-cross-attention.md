---
type: feat
scope: adapter
status: open
priority: high
created: 2026-07-11
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../50_Decisions/open/action-conditioning-injection-mechanism]]", "[[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]", "[[experiments/exp-conditioning-action-shuffle-ablation]]", "[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]"]
---

# feat: wire the action into the WAN adapter's cross-attention (per-frame tokens)

Implements option B (+ C) of
[[../50_Decisions/open/action-conditioning-injection-mechanism]]: move the action
out of the AdaLN broadcast and into the WAN output adapter's **existing but unused
`t2v_cross_attn` context path**, using **per-frame action tokens**.

## Context (why this is cheap)

`backbones/wan/modules/action_model.py` already builds `WanAttentionBlock`s with a
`t2v_cross_attn` + a `text_embedding`→`context` path, currently fed a null/zero
token (`use_text_context=False`). So the cross-attention slot exists; this ticket
fills it with action tokens rather than adding a new mechanism.

## Build

1. **Preprocessor** (`data/wan_batch_preprocessor.py`): add an aggregation mode
   that bins the per-step deltas onto the latent temporal grid →
   `cond["action"]` shape `[B, L, 4]` (`L = (frame_num−1)/4 + 1`), summing within
   each bin. Keep the existing `[B,4]` sum mode for the AdaLN arm.
2. **`ActionWanModel`** (`backbones/wan/modules/action_model.py`): add an
   `action_embedding` mirroring `text_embedding`:
   `Linear(4→dim) → GELU → Linear(dim→dim)` + temporal position embedding →
   context tokens `[B, L, dim]`. Zero-init the last Linear.
3. **Wiring:** feed those tokens as `ctx` to the blocks (reuse the `t2v_cross_attn`
   path); keep `timestep` + `step_level` in the AdaLN `e` embedding.
4. **Flag:** `action_injection: adaln | cross_attention | both` (through
   `adapter.extra`), so the ablation is one config switch. `adaln` = current
   behaviour (regression-safe default).
5. **Configs:** `*_wan22_*_xattn_metaworld.yaml` for the cross-attn + `both` arms.

## Validate

- Smoke-run end-to-end on the real WAN base with random weights (like
  `examples/wan22_generate_cond_frames.py`) — shapes + cross-attn path fire, base
  untouched at init (Δ≈0 from the zero-init final proj).
- Then the ablation run: `adaln` vs `cross_attention` vs `both`, same base/data,
  compared on base-vs-adapted delta, the shuffle counterfactual
  ([[experiments/exp-conditioning-action-shuffle-ablation]]), and the NFE-row grid.

## Guardrails

- **Temporal position embedding is required** (cross-attn is permutation-invariant
  over KV).
- Don't touch the frozen WAN base forward — this lives entirely in the adapter.
- Quantify, don't eyeball (hard rule 8).

## Outcome (2026-07-12, run `xb76ptw2`) — negative, adapted worse than base

Built + run (`coluding/Wan2.2-avid-xattn-i2v-metaworld/xb76ptw2`, commit `1c77db61`,
killed @ step 2661). The cross-attn adapter is **worse than the frozen WAN base on
every eval metric** (PSNR −0.055, SSIM −0.021, MSE +0.00017, LPIPS +0.124,
FVD +126, FID +243 ≈ 2.5×). Crucially it **loses the reconstruction gain** the AdaLN
button-press run had (which beat base on PSNR/SSIM/MSE). Full write-up + confounds
(early kill @ 2661 steps, different MetaWorld subset):
[[../30_Knowledge/experiments/20260712-wan-xattn-action-no-improvement]].

## ⚠️ Correction (2026-07-14) — the run violated its own design's precondition; do not treat as conclusive

Re-investigated as part of [[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]].
Confirmed by direct grep: `configs/diffusion_wan22_avid_xattn_i2v_metaworld.yaml`
**never sets `action_seq_len`**, and the preprocessor default is `None` (raw,
unbinned passthrough). So the run fed **unbinned raw-pixel-frame action tokens**
(not binned to the 11-latent-frame grid) into **globally unmasked** cross-attention —
exactly the failure mode this decision's own "Non-negotiable coupling" section
warned about *before the run* (decision doc written 2026-07-11, one day before
`xb76ptw2` ran on 2026-07-12): *"a lone KV token attended by all queries collapses
back to a global bias → cross-attn ≈ AdaLN and the ablation is uninformative...
the cross-attn arm MUST use a per-frame action-token sequence."* That requirement
was not met.

**The observed eval numbers above still stand** (they're real measurements). **The
causal conclusion below them does not** — this was not a clean test of localized
cross-attention injection vs. AdaLN. It should be **re-run with `action_seq_len`
pinned to the latent frame count** (11, for the current 41-frame/stride-4 config)
before drawing any conclusion about injection mechanism vs. capacity.

~~Implication: moving the action into the cross-attn context path is **not** the
lever... Evidence for the **capacity** hypothesis over the **injection-mechanism**
hypothesis...~~ — **retracted pending a clean re-run.**

**Next lever, still valid regardless:** capacity —
[[feat-adapter-dynamicrafter-output-on-wan-base]] (UNet-scale `Δ_φ` on the frozen
WAN base) remains worth pursuing on its own merits; it just isn't validated *by
this run* as the answer to a resolved "capacity vs. mechanism" question, since
mechanism was never cleanly tested.

## Cleanup 2026-08-01 — **OBSOLETE**

Cross-attention injection is built and *works* (transports 44-56% action-driven signal). Not the bottleneck.

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
