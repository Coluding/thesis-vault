---
type: writing
status: living
last_updated: 2026-08-07
sources:
  - "[[conditioning-injection-site-literature]]"
  - "[[../writing/rubric/01-originality]]"
  - "[[../experiments/20260802-avid-wan-cleanroom-perframe-causal]]"
  - "[[../experiments/20260731-wan-action-trace-value-pathway-drowns]]"
---

# The pathway claim: what is prior art, and what survives

> **Verdict from the conditioning scan, 2026-08-07: the empirical result is
> not novel; a version of the frozen-backbone result is nearly claimed; the
> mechanism appears to be ours.** The headline must be restated. This note
> is the record of what can and cannot be claimed.

## 🛑 Not novel: the ranking

Normalised-multiplicative conditioning beating additive or token-based
conditioning for low-dimensional signals is one of the better-replicated
results in the field, **including on action conditioning**:

| Paper | Setting | Numbers |
|---|---|---|
| ADM, arXiv:2105.05233, Table 3 | class+timestep diffusion | AdaGN **13.06** vs Addition+GroupNorm **15.08** FID |
| DiT, arXiv:2212.09748, Fig. 5 | class-conditional DiT | adaLN-Zero **19.47** vs cross-attention **26.14** FID |
| **IOI**, arXiv:2606.23296, Table 5 | **action-conditioned video** | Add 87.31 → Concat 67.89 → CrossAttn 62.56 → **AdaLN 56.47** FVD |
| **Nano World Models**, arXiv:2605.23993, Table 3 | **action-conditioned, RT-1** | CrossAttn **20.82** PSNR / 51.12 FID vs FiLM **23.20** / 40.62 |

⚠ **Do not write "per-frame AdaLN beats cross-attention for action
conditioning" as a finding.** It reads as a re-run of DiT Figure 5, and two
2026 preprints already report it on our exact task.

## ⚠ Nearly claimed: the frozen-backbone version

**Decoupled Action Expert** (arXiv:2511.12101v2 §IV-E, venue unverified)
ran seven conditioning mechanisms on one backbone under normal versus
**frozen** training, on MimicGen:

| Mechanism | Normal | Frozen |
|---|---|---|
| Cross-Attention | 61.3 % | **19.8 %** |
| Prefix Tuning | 56.3 % | **19.5 %** |
| Additive Injection | 57.0 % | 45.3 % |
| FiLM | 59.5 % | 51.8 % |
| AdaLN | 62.0 % | **62.3 %** |
| AdaLN-Zero | 64.0 % | **64.5 %** |

Verbatim: *"modulation-based methods that keep conditioning external to the
backbone are robust to freezing, while token-based methods that route
conditions through backbone projections collapse."*

**Cite this in the related-work chapter and do not let a reviewer find it
first.** Their conclusion is ours.

**Their explanation is not ours, and their own data favours ours.** They
attribute the collapse to *projection staleness*: conditioning routed
through frozen `W_q/W_k/W_v` stuck at the stage-1 input distribution. But
their **additive injection** row bypasses backbone projections entirely and
still degrades (57.0 → 45.3). Staleness does not explain that; a magnitude
account does. **This is the single strongest argument that our mechanism is
different from theirs**, and it costs nothing to make.

## 🛑 The paper that refutes the claim as loosely stated

**Motif-Video 2B** (arXiv:2604.16503 §6.2) performs *our exact
measurement*: Frobenius norm of the cross-attention contribution relative to
the self-attention residual, per block and denoising step.

**7.6 % mean, 21.7 % maximum, 5.2 % weakest block**, cosine to the
self-attention output ≈ −0.008. Conclusion: *"no block is dormant"*;
cross-attention is a live, complementary pathway.

Our ratio (~0.01 against a stream of 1.8–3.0, i.e. **~0.3–0.6 %**) is an
order of magnitude below their weakest block.

⚠ **So "cross-attention output is negligible against the residual stream" is
false in general and citably so.** The defensible statement is conditional
on our setting:

> In a **frozen** backbone with a **from-scratch** adapter carrying a
> **low-dimensional** action signal, the cross-attention write lands roughly
> an order of magnitude below the level at which trained-from-scratch models
> operate their cross-attention pathway.

Motif's shared-K/V design, reusing the backbone's own projections so the
write is grounded in the existing manifold, is arguably an instance of the
same principle and worth reading as such.

**Meltdown** (arXiv:2602.11130) supplies the complementary caution: a small
cross-attention write can still be causally decisive. **The RMS ratio alone
proves nothing.** What carries our claim is the *intervention* evidence, the
matched-contribution A/B and the temporal control, not the magnitude number.

## ✅ What survives, and it is enough

The mechanism is unclaimed. Every explanation on offer is something else:

| Explanation | Source |
|---|---|
| spatial granularity | GenTron, the incumbent |
| initialisation / identity-at-init | DiT; the OpenReview adaLN-Zero analysis |
| similarity detection via multiplicative interaction | the Distill survey, which explicitly does *not* discuss magnitude |
| projection staleness under freezing | Decoupled Action Expert |
| task-dependence, unexplained | Nano World Models |

The two halves of our account exist separately and **have never been
joined**. Residual-stream norm growth attenuating later writes is stated in
general form by NAG (arXiv:2606.16112): *"the norm of the residual stream
can grow rapidly with depth… contributions systematically suppressed by
residual norm expansion"*. And the entire zero-init lineage (ControlNet,
Flamingo, LLaMA-Adapter, ReZero, DeepNet) controls write magnitude into a
residual stream. **But every one of those is about the write being too large
at initialisation. None argues the converse: that a conditioning write ends
up too small to matter.**

Applying the norm-attenuation argument to a **conditioning branch** rather
than a **layer branch** appears unclaimed.

## The claim to make

Three parts, in this order:

1. **The measured magnitude ratio in a frozen-backbone adapter setting**,
   scoped as above and stated against Motif's numbers rather than in
   isolation.
2. **The localisation.** The signal *survives the attention layers* and dies
   *at the residual addition*. Our 23-depth trace shows action-driven share
   at 44–56 % across all ten blocks, then a fall to 0.0085 in one addition.
   **This is a sharper localisation than anything in the scan**, and it is
   what separates a mechanism from a ranking.
3. **The resulting design rule**, scoped explicitly to low-dimensional,
   spatially-global conditioning signals.

**Scope it or be refuted.** GenTron (text, spatially varying) and DexAC-WM
(high-DoF dexterous actions need the token pathway) are ready-made
counterexamples for an unscoped version.

## ➜ Two manual verifications before relying on them

- **CogVideoX** (arXiv:2408.06072): the expert-AdaLN passage reportedly
  invokes *numerical scale mismatch between modalities* as the architectural
  driver. Only reachable through a search summary. If it verifies, it is the
  closest thing to a scale-based argument from a production system.
- **OpenReview `E4roJSM9RM`**, "Unveiling the Secret of AdaLN-Zero":
  reportedly isolates zero-init as the cause of adaLN-Zero's advantage.
  **The likeliest place for a direct scoop.** Automated fetch was blocked
  twice; check manually.
