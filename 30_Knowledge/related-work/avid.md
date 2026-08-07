---
type: paper
status: living
last_updated: 2026-07-26
title: "AVID: Adapting Video Diffusion Models to World Models"
authors: ["Marc Rigter", "Tarun Gupta", "Agrin Hilmkil", "Chao Ma"]
venue: "RLC 2025 / Reinforcement Learning Journal 2025, paper 64 (also ICLR 2025 workshop poster). arXiv:2410.12822"
year: 2024
url: https://arxiv.org/abs/2410.12822
local_pdf: docs/paper/avid.pdf
relevance: framework, baseline
deliverable: D1, D2
---

# AVID

> **Action-conditioned** world model built by training an adapter on a
> *frozen* pretrained video diffusion model, using only its noise
> predictions (no weight access). The direct precedent for the thesis's
> "additive correction on a frozen base" architecture **and for the
> action-conditioning result itself**; the thesis generalises across
> adapter families and adds step-size conditioning.

## Status of this note

**Verified against the PDF 2026-07-26** (title, authors, venue, mechanism,
evaluation, planning status). Vendored at `docs/paper/avid.pdf`. Remaining
gaps flagged inline.

## Correction (2026-07-26) — AVID *is* action-conditioned

This note previously stated "AVID does not condition on actions or
step-size. The D2 and D3 contributions of the thesis are exactly the two
axes AVID does not cover." **The first half is wrong**, and the error is
mirrored in [[../../10_now/positioning]] ("output-level and not action /
step-size-aware").

From the abstract: *"we propose to adapt pretrained video diffusion models
to action-conditioned world models, without access to the parameters of the
pretrained model. Our approach, AVID, trains an adapter on a small
domain-specific dataset of action-labelled videos."* Action conditioning on
a frozen video base **is AVID's contribution**, not ours.

**Step-size conditioning (D3) remains genuinely uncovered by AVID** — no
few-step / consistency / shortcut content in the paper.

## Why it matters for the thesis

- The thesis's framework (D1) explicitly mirrors AVID's
  "frozen-base + trainable residual" shape, then generalises it across
  four adapter families. The thesis must cite AVID as the direct
  architectural precedent for the output-adapter family.
- The codebase contains an explicit AVID-style starting point:
  - `configs/diffusion_output_avid_training_test.yaml` — replicates the
    AVID setup on top of DynamiCrafter.
  - `src/external_deps/avid_utils/` — vendored AVID evaluation utilities.

## How it maps to our adapter taxonomy

- AVID's adapter is **output-level** in our terminology: it adds a
  trainable correction on top of the frozen base's output (the noise
  prediction or velocity).
- In our factory it lives under `adapters/output/`, with the AVID-flavoured
  variant most closely matched by `dynamicrafter` (the
  `DynamicCrafterOutputAdapter`, see `adapters/output/dynamicrafter.py`).
- **Composition: mask-mix, confirmed.** Abstract: *"AVID uses a learned mask
  to modify the intermediate outputs of the pretrained model."* This is our
  `composition: mask_mix`, and it is why `init_mask_bias: 0.0` (σ=0.5) is
  the AVID-native init — see
  [[../../50_Decisions/decided/avid-adapter-init]] and
  [[../experiments/20260715-avid-metaworld-native-gate-healthy]].

## Baselines AVID compares against (§4.1, "no access to pretrained params")

Our D2 comparison should slot into this same table rather than inventing one:

- **Action-Conditioned Diffusion** — trained from scratch, param-matched to AVID.
- **Classifier Guidance** — action classifier `f_ϕ(a|x_i)` steers sampling.
- **Product of Experts** — `λ_p·ϵ_adapt + (1−λ_p)·ϵ_pre`.
- **Action Classifier-Free Guidance** — `ϵ_pre + λ_a(ϵ_adapt(a) − ϵ_adapt(∅))`,
  action dropout `p=0.2` (cf. [[../../50_Decisions/decided/condition-dropout-cfg-for-world-models]]).
- ControlNet / ControlNet-Small are also reported, but require weight access.

Headline finding (§5): AVID is *similar or slightly better overall* than
ControlNet/ControlNet-Small on Coinrun500k and RT1, **without needing base
weights** — the framing is parity-at-lower-access-cost, not a large win.

## Evaluation protocol (§4.2) — 1024 held-out videos

Metrics: **Action Error Ratio**, FVD, FID, SSIM, PSNR.

**Action Error Ratio is the action-sensitivity metric to adopt.** A model is
trained to predict actions from *real* videos; the ratio is its error on
generated videos ÷ its error on real videos. This is AVID-comparable and
stronger than our shuffled/zeroed-action loss gap — worth adding to the
metric list in [[../writing/ablation-axes]] so the D2 table can carry an
AVID-replica row.

## What we do that AVID doesn't

- **Step-size conditioning + consistency training (D3)** — absent from AVID entirely.
- Multiple adapter families behind one composition rule (D1).
- Coverage of flow matching, not just diffusion (D1).
- **Planning** — AVID names this as *future work*, not a result (below).
- _Not_ action conditioning — that is AVID's own contribution. See the
  Correction section above.

## Planning: AVID's stated future work, not its result

AVID evaluates **video-prediction quality only** — there is no planning,
MPC, reward model, or policy-evaluation experiment in the paper. The
conclusion (§7) states the results *"demonstrate the considerable potential
of adapting these models to world models that are suitable for planning"*
and that they *"aim to explore the use of synthetic data generated by AVID
adapters for planning tasks"*.

**Consequence for the thesis:** a planning demonstration on top of the
DynamiCrafter + AVID setup is a citable, explicitly-invited extension of
this paper — the strongest available framing for that step of the
storyline. It is a *demonstration*, not a claimed contribution (see
[[../../10_now/positioning]] anti-positioning: "Not a control / RL paper").

## Open questions for the chapter

- Exact composition equation as written in the paper (mask-mix confirmed at
  the abstract level; the per-term equation still to be transcribed from §4).
- Coinrun500k / RT1 dataset details, if we want a domain-comparability
  sentence.

## Related

- [[_MOC]]
- [[../../10_now/architecture]] — see Adapter families, output column
- [[hyperalign]] · [[unicon]] · [[cafm]] — the other adapter-side neighbours
- `src/external_deps/avid_utils/`
- `configs/diffusion_output_avid_training_test.yaml`
