---
type: writing
status: living
last_updated: 2026-08-03
sources:
  - "[[thesis-storyline]]"
  - "[[rubric/03-experimental-evaluation]]"
  - "[[../experiments/_index]]"
  - "[[../experiments/20260802-avid-wan-cleanroom-perframe-causal]]"
  - "[[../experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]"
  - "[[../experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]]"
  - "[[../experiments/20260731-why-wan-copies-the-base-decomposed]]"
  - "[[../experiments/20260731-wan-action-signal-is-a-global-bag]]"
  - "[[../../20_Tickets/experiments/exp-adapter-lora-vs-output-comparison]]"
---

# Ablation design — hypotheses, the axes that discriminate them, verdicts

> **⚠ REWRITTEN 2026-08-03.** The previous version of this note (dated
> 2026-07-24) described a *search* for a configuration that escapes
> base-parity, listed five axes, and stated that LoRA "is not run". It
> predates the entire 07-30 → 08-02 campaign, which **answered** the
> question. This version is organised the way the chapter must be written:
> **hypothesis-first, verdict-bearing**. Old version in Git history.

This is the **experimental-design layer**. [[thesis-storyline]] is the
narrative layer; [[../../10_now/positioning]] the contribution layer. This
note is the direct source for **Ch4 §4.5** (ablation design) and supplies
the ordering for **Ch5**.

---

## Why hypothesis-first is not a stylistic choice

Thirteen axes were exercised. Presented axis-first — *"we varied backbone,
dataset, adapter size, injection, composition…"* — that reads as flailing,
and it invites the rubric's failing row ("errors are made in the process").
Presented as *"the failure has N candidate explanations; here is the design
that discriminates each"*, the same runs read as a discriminating study
([[rubric/03-experimental-evaluation]]).

**Every row below ends in a mechanism claim, not a result.** That is the
difference between a contribution and a postmortem.

---

## The question

> An adapter on a frozen video base learns to improve prediction. **Does it
> learn to use the actions** — and if not, why not?

Sharpened by the campaign into two separable questions, because the answer
turned out to differ between them:

- **Q-a (sensitivity/structure):** does the action change the prediction, in
  a *structured* way — directional, temporally aligned, spatially localised?
- **Q-b (control):** does that structure convert into rollout-level
  control — does swapping the action change where the trajectory goes?

Conflating these is the trap the campaign spent weeks in. `effect_rel`
answers neither cleanly on its own; it is monotone in action-pathway gain.

---

## The hypotheses and their verdicts

| # | Candidate explanation | Discriminating axis | Verdict |
|---|---|---|---|
| H1 | **Our data does not reward actions** | 1 Dataset | **Killed.** The unmodified AVID recipe follows actions on *our* ACWM Robot Arm (effect_rel 0.0295, null 0) where our three adapters were blind (0.0013–0.0056) — same frozen weights, same data, same probe |
| H2 | **The base is too strong; the adapter clones it** | 2 Backbone · 7 Base-pred injection | **Refined, not confirmed.** ~87% of the pred–base cosine is shared-target convergence, present with **no** base input — high cosine ≠ copying. The removable part is *oracle-reading* |
| H3 | **Optimisation traps (gate)** | 6 Composition · 8 Gate bias/cap · 9 Gate pretraining | **Real but insufficient.** The traps exist and are fixable; fixing them did not unlock action use |
| H4 | **Insufficient capacity** | 4 Adapter size | **Killed.** A structurally clean 7.5M adapter settles *below* the DiT-clone arms (~0.0025 vs 0.008–0.011); the DiT inductive bias is worth 3–4× on this data |
| H5 | **Wrong adapter family** | 3 Family (output vs LoRA) | **Open — run in flight** ([[../../20_Tickets/experiments/exp-adapter-lora-vs-output-comparison]]) |
| H6 | **The signal arrives mis-scaled** | 10 Scale calibration | **Confirmed — two opposite failures at the same interface.** DC: a *learned pedestal* (embedding ×106, 99.7% constant, 14× the time embedding). Wan: the mirror image, faithful but ~250× too quiet |
| H7 | **Action tokens are temporally unaligned** | 11 Token binning | **A real defect, not the cause.** px→latent correspondence at chance; enforcing binning buys nothing on its own |
| H8 | **Actions only carry signal at high noise** | 12 σ-shift | **Killed.** effect_rel flat across the σ sweep — the mismatch hypothesis is dead |
| H9 | **The injection pathway is wrong** | 5 Action injection | **✅ THE ANSWER.** Per-frame modulation of normalised activations is *causal* at matched adapter contribution and matched mask; cross-attention fails at every scale and alignment tested |
| H10 | **The objective does not pay for actions** | analysis, not an axis | **Confirmed as the bound.** Actions explain ~0.45% of the teacher-forced denoising loss; appearance correction always outbids them |
| H11 | **The consistency loss is at fault** (D3 contamination) | 13 + `anchor_prob: 1.0` control | **Killed.** The D2 failure is measured on runs carrying no consistency term |

**H9 + H10 are the thesis.** H9 is the mechanism that is fixable; H10 is the
bound that is not. Everything else is the elimination that makes those two
credible — and the eliminations must be *reported*, not compressed away
([[rubric/05-reflection]]).

---

## The axis inventory

Grouped by what they vary. **This is a toolbox that was pulled adaptively,
not a factorial** — say so in the chapter, and report the axis count and
comparison count honestly.

### Reporting axes (varied deliberately, every cell)

| # | Axis | Values exercised |
|---|---|---|
| 1 | **Dataset** | MetaWorld `five_task` (action-**redundant** control) · ACWM {Push Cube, Robot Arm, Reacher} (informative) · RT-1 (diverse real) · OpenVid |
| 2 | **Frozen backbone** | Wan2.2-TI2V-5B (flow, DiT, 4× temporal compression) · SkyReels-1.3B · DynamiCrafter (diffusion, UNet) |

### Architecture axes

| # | Axis | Values | Note |
|---|---|---|---|
| 3 | **Adapter family** | output adapter · **LoRA** | the LoRA arm is in flight; it reopens the D1 quality comparison |
| 4 | **Adapter size** | 7.5M simple transformer · 34.97M · 47M | 34.97M of 5.03B = 0.69% trainable |
| 5 | **Action injection** | cross-attention · **per-frame AdaLN** | **the decisive axis** |
| 6 | **Composition** | `mask_mix` · `replace` | replace cannot stand alone at 34M; mask_mix gate-saturates |
| 7 | **Base-prediction injection** | `condition_on_base_outputs` on/off | a **D1** design axis with a measured cost — see below |

### Training / calibration axes

| # | Axis | Values |
|---|---|---|
| 8 | **Gate bias / cap** | `gate_bias` · `gate_cap` {off, 0.9} |
| 9 | **Gate pretraining** | AVID warmup `pretrain_steps` {0, N} |
| 10 | **Action scale calibration** | `action_token_norm` (Wan) · `condition_center` (DC) |
| 11 | **Action token binning** | `action_seq_len`, enforced px→latent temporal binning |
| 12 | **Noise schedule** | σ-shift {off, 5.0} |

### D3 axis

| # | Axis | Values |
|---|---|---|
| 13 | **Shortcut target × base geometry** | `v_average` · `endpoint_inversion` × {flow (κ=0), diffusion (curved)} |

---

## Two results that are D1, not D2

Worth separating explicitly, because they belong in a different chapter
section and are easy to lose inside the D2 story:

1. **The composition interface is a real design axis** (axis 7). A
   single-flag change trades quality against action-conditioning: with the
   oracle on, every quality metric is better (FVD +63.7% vs +45.1%) and the
   prediction sits closer to the base (cosine 0.914 vs 0.851), while
   following actions **25% less** (0.0062 vs 0.0077). Whether `Δ_φ` sees
   `f_base(x_t,t)` is a framework decision with a measurable cost, not an
   implementation detail. ⚠ n=1 per arm, slightly different steps, different
   bases — quote % improvement over each arm's *own* base.
2. **Base-swapping is a configuration change** (axis 2), demonstrated
   across three backbone families plus a port of AVID's own recipe into
   AVID's own repository on a new base family.

---

## The per-cell metric set

Identical everywhere, so any outcome is a result.

1. **Action sensitivity** — `effect_rel` (cheap screen; **monotone in
   action-pathway gain**, so never load-bearing alone) and the **Action
   Error Ratio** (AVID §4.2) as the external, AVID-comparable readout.
2. **Action structure** — the triad, each against its chance level:
   steering cosine (direction), temporal alignment (px→latent
   correspondence), spatial concentration (localisation).
3. **Control** — rollout-action-swap: does the true-action rollout track
   ground truth better than a wrong-action or zero-action one?
4. **Prediction accuracy** — masked MSE / PSNR / SSIM.
5. **Generation quality** — FID / FVD / LPIPS.
6. **Base parity** — pred-vs-base cosine.
7. **Nulls** — `base_null_violation` (frozen-base invariance), verified
   before every run, and the clip-null for genericity.
8. **Cost** — trainable params, FLOPs/step, GPU-hours.

**Metrics 1–3 form a ladder** — sensitivity, then structure, then control —
and cells dissociate along it. That dissociation *is* the headline result
(see below), so the ladder must be presented before the results, in Ch4.

---

## The dissociation the design uncovered

The reason the metric ladder matters, stated as the chapter should state it:

| | Wan × ACWM | DC arm E |
|---|---|---|
| Quality vs frozen base | **beats it 6/6** (FVD 1118 → 406, −64%) | **unknown** — no DC run logs quality metrics (all 18 checked) |
| Sensitivity (`effect_rel`) | 0.0062 | 0.115 |
| Structure (triad) | **all three at chance** | **all three above chance** (steering +0.117 vs 0.000; temporal 1.000 vs 0.313; spatial 0.470 vs 0.100) |

**The same adapter design is a strong domain adapter on Wan and a working
action conditioner on DC — never both on either.** A 2.75× FVD improvement
carrying no action information is a domain correction.

Two scope conditions the claim needs:
- The Wan quality win is **ACWM-only**. On RT-1 the same adapter is a net
  *perceptual* regression (FID 143.1 vs base 120.8; LPIPS 0.427 vs 0.376),
  better only on pixel metrics — the mean-regression signature.
- It is a **Wan** failure, not an adapter-family failure. DC processes
  actions with the same family. "The adapter cannot process actions" is
  false in general.

---

## What is still open

- **H5 — adapter family.** LoRA vs output, in flight. Pre-register the
  decision rule before it lands.
- **Control (Q-b) on the DC cell.** DC has structure above chance; whether
  that converts to rollout-level control is **untested there**. The
  rollout-swap null is a **Wan** result. Do not write "control is
  demonstrated nowhere" — write what is true per cell.
- **The structure triad on the binned RT-1 checkpoint** (`0fqjrqjl`) has
  never been run, and it is the intervention most likely to change the Wan
  verdict.
- **Quality metrics on any DC run** — none exist, so the DC cell's
  domain-adaptation performance is unknown.
- **Seeds.** Every headline is n=1.

---

## Related

- [[evidence-map]] — the per-cell verdicts mapped to sections and rubric items
- [[thesis-storyline]] — the narrative these axes serve
- [[rubric/03-experimental-evaluation]] — how this table is graded
- [[../experiments/_index]] — the results ledger (every axis traces to runs)
- [[../../70_Thesis/outline]] §4.5 — the chapter this feeds
