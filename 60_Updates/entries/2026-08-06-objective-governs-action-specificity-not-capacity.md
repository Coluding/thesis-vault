---
date: 2026-08-06
category: finding
deliverable: D2
meeting:
sources:
  - "[[../../30_Knowledge/experiments/20260806-objective-governs-action-specificity-not-adapter-capacity]]"
  - "[[../../30_Knowledge/experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]"
  - "[[../../30_Knowledge/writing/ablation-axes]]"
  - "[[../../10_now/compute-spend-ledger]]"
---

# The objective governs action-specificity; the backbone governs adaptability

## What

Integrating **EasyAnimate** as a frozen base produced two results at once.
First, the same output-adapter family that barely moved Wan reshapes
EasyAnimate substantially — so the Wan ceiling was **base-specific, not
intrinsic to output adapters**. Second, comparing EasyAnimate **V5
(diffusion)** against **V5.1 (flow)** at matched video backbone: the two
objectives adapt **equally well** but differ in how much of that adaptation
is **action-conditioned**.

## Why it matters

We were heading toward the conclusion that output adapters simply cannot
move a frozen video base. That would have been **wrong**, and it would have
been the thesis's central claim.

The corrected picture separates two variables that the Wan campaign had
conflated:

- **The backbone decides how much an adapter can reshape at all.**
- **The objective decides what fraction of that reshaping is
  action-conditioned.**

This **generalises the 0.45 % loss-share result** already in hand: both say
the objective *allocates gradient*, and that capacity is not the binding
constraint. Together they give the thesis one governing claim rather than
several separate findings.

## Evidence

Same adapter family (34.9M, `composition: add`, cross-attention,
`condition_on_base_outputs: false`), ACWM Robot Arm:

| run | base loss | adapted | Δ | `adapter_base_cosine` | `rel_contrib` |
|---|---|---|---|---|---|
| Wan token-norm + add (`25192313`) | 0.13362 | 0.12917 | −3.3 % | 0.9989 | 0.047 |
| **EA V5 diffusion** (`25240257`) | 0.24290 | **0.06098** | **−74.9 %** | 0.868 | **0.521** |
| **EA V5.1 flow** (`25241732`) | 0.44620 | **0.11759** | **−73.6 %** | 0.886 | **0.484** |

Wan's `adapter_base_cosine` of 0.9989 means the adapted prediction was
*numerically almost the frozen base* — the adapter was cosmetic there.

Objective contrast (V5 vs V5.1): loss reduction tied (−74.9 % vs −73.6 %);
`action_effect_rel` **+36 %** and `action_effect_vs_adapter` **+26 %** for
diffusion. Across independent families, both diffusion backbones sit above
both flow backbones — **n=2 per objective**, not a single pair.

⚠ **Interim.** Both arms still running, steps not matched (9000 vs 8200),
n=1 per arm; the direction held across eight consecutive hourly evals.
EA-vs-Wan differs in backbone, VAE, objective *and* text conditioning; V5 vs
V5.1 differ in text backbone (BERT+T5 vs Qwen2VL). It is **"flow is
weaker", not "flow fails"** — EA-flow is ~3× Wan-flow.

⚠ **Only 6–8 % of the adapter's large contribution is action-driven**
(`effect_vs_adapter` 0.076 / 0.060). "The adapter is powerful" is
established; "powerful *at action conditioning*" is not.

**Provenance.** These numbers exist only because the base was fixed first:
before 2026-08-05 the EasyAnimate base rendered **noise** while every
shape/finiteness/file-existence check passed — caught by looking at one
frame. All `effect_rel` logged before that fix are **void**
([[../../10_now/compute-spend-ledger]]).

## Next

- Matched-step end-of-run numbers with spread across eval draws.
- Recorded as hypothesis **H12** in
  [[../../30_Knowledge/writing/ablation-axes]], with a new axis 14
  (training objective at matched video backbone).
- Storyline claim 3 ("the backbone is not the limit") reworded — the
  backbone *is* a limit on reshaping, not on addressability.
