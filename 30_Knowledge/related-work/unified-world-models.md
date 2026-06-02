---
type: paper
status: living
last_updated: 2026-05-25
title: "Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets"
authors: [Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, Abhishek Gupta]
venue: "arXiv preprint (2504.02792v3)"
year: 2025
url: https://weirdlabuw.github.io/uwm/
local_pdf: docs/paper/unified_wms.pdf
relevance: framework, theory, baseline
deliverable: D2, D-multimodal
---

# UWM — Unified World Models

> One diffusion transformer over `(o, a, o')` (current obs, action, next
> obs) with **independent diffusion timesteps per modality** under a
> **single shared denoising objective**. The "noising = partial masking"
> identity then turns one trained model into policy / forward-dynamics /
> inverse-dynamics / video-prediction just by choosing each modality's
> inference timestep. This is the principled answer to the timestep
> question for the thesis's multimodal extension — see
> [[../../50_Decisions/open/multimodal-adapter-broadening]].

## Status of this note

**Read 2026-05-25** (pp. 1–10, the full method + experiments + related
work + limitations). PDF at `docs/paper/unified_wms.pdf`. Author list,
venue, arXiv id, URL taken from the title page; venue is an arXiv preprint
(v3, 23 May 2025) — _no peer-reviewed venue confirmed_.

## The one idea

UWM trains a coupled score model `s_θ(o, a_{t_a}, o'_{t_o'}, t_a, t_o')`
that predicts the noise on **both** the action and the next observation,
each corrupted at **its own** diffusion timestep:

- Training (Eq. 1): sample `t_a, t_o' ~ U(0, T)` **independently**, corrupt
  each modality at its own timestep, minimise a weighted sum of the two
  denoising losses:
  `ℓ(θ) = E[ w_a‖ε̂_a − ε_a‖² + w_o'‖ε̂_o' − ε_o'‖² ]`.
- The objective is **shared** (one loss, summed over modalities) but the
  **noise schedules are decoupled** (one timestep per modality). This is
  the exact phrasing the user flagged as the inspiration for our timestep
  choice.

### Why decoupled timesteps are the whole point: noising ↔ masking

UWM's key insight is a connection between the forward diffusion process and
**partial masking**: a modality fully noised (`t = T`) is indistinguishable
from being **masked out / marginalised**; a modality left clean (`t = 0`)
is **conditioned on**. So by setting inference timesteps you select which
conditional/marginal of `p(o, a, o')` you sample:

| Inference mode | Set | Denoise |
|---|---|---|
| Policy `p(a\|o)` | `t_o' = T` (marginalise obs) | action |
| Video prediction `p(o'\|o)` | `t_a = T` (marginalise action) | next obs |
| Forward dynamics `p(o'\|o, a)` | `t_a = 0` (condition on clean action) | next obs |
| Inverse dynamics `p(a\|o, o')` | `t_o' = 0` (condition on clean obs) | action |

One trained model, four inferences, chosen at test time by timestep alone.

### Co-training on incomplete-modality data

The same identity lets UWM train on **action-free video**: fix `t_a = T`
and impute the missing action with random noise `ε_a ~ N(0,1)`, then
optimise the same Eq. 1 loss. The model still gets the video-prediction
gradient. This is how UWM absorbs large action-free video corpora during
pretraining.

## Architecture (for reference)

- Diffusion transformer; per-modality sinusoidal timestep embeddings
  concatenated with features and injected via **AdaLN**.
- Images: ResNet-18 per-frame encoder for the *conditioning* obs; for the
  *diffused* next-obs they use a latent-diffusion SDXL VAE (224²→(28,28,4)),
  patchified `(4,4,2)`.
- Actions: shallow per-timestep MLP on action chunks.
- **Register tokens** (randomly initialised, discarded after) added to give
  the modalities an intermediary to exchange info — they found this helps,
  hypothesising the meaningful noise-prediction outputs leave "no room" for
  cross-modal communication otherwise.

## How it maps to the thesis

- **Not adapter-first.** UWM is a single from-scratch monolithic DiT trained
  on the joint distribution. The thesis is the opposite shape: a **frozen**
  pretrained base + trainable adapters
  (`f(x_t,t,a,d) = f_base + g(d)·Δ_φ`). So we **adopt the training
  scheme** (independent per-modality timesteps + shared denoising
  objective), not the architecture.
- **It subsumes the AVID learned mask.** The AVID/`mask_mix` gate
  (`adapted_model.py:88-94`, [[../tech/mask-mix-gate]]) is a learned per-pixel
  blend between base and adapter. UWM's timestep-as-mask is the *principled*
  version of "how much does each stream contribute" — and it's free at
  inference rather than a trained parameter.
- **It is the answer to the timestep sub-decision.** The thesis's
  multimodal extension (`(x^video, x^prop, …) = f(x_t, t, a, d)`, vault
  CLAUDE.md optional extension) should give **each output modality its own
  independent timestep**, sampled `~U(0,T)` in training, summed under one
  weighted denoising loss. The "denoising optional" (purple) boxes in
  `docs/composite (2).png` are exactly UWM's `t = T` marginalisation.
- **Cross-modal sharing happens differently for us.** UWM shares features
  through joint self-attention + register tokens in one transformer. In the
  adapter setting the sharing routes through (a) the shared conditioning
  embedding ([[../tech/structural-encoder]]) and (b) the compositional mask
  coupling each modality's *video adjustment* back into the shared latent
  (`docs/composite (2).png`).

## What UWM does that we (would) do differently

- **PAD is the cautionary baseline.** UWM explicitly positions against PAD
  [20], which uses a **shared** diffusion timestep across modalities and
  conditions by concatenating clean latents on the channel dim. UWM argues
  (and shows, Table I) the shared timestep gives a "sub-optimal shared
  representation that lacks a causal understanding." **Our channel-stack
  baseline is PAD-shaped** — so UWM predicts it should be the weakest of our
  three variants, which is a useful prior for
  [[../../50_Decisions/open/multimodal-adapter-broadening]].

## Results (brief)

- Pretrain on DROID (2000 trajs) + co-train on 2000 action-free trajs;
  finetune on 5 real Franka tasks. UWM beats DP, PAD, GR1 across all 5 (ID
  and OOD), by up to ~20% (Table I, Fig. 6). PAD is worst across the board.
- LIBERO sim (Table II): UWM avg 0.79 vs DP 0.71 / GR1 0.58 / PAD 0.57.
- Co-training on action-free video helps OOD robustness (Table IV).

## Open questions for the chapter / for us

- **Weighting `w_a, w_o'`.** UWM trades off the per-modality losses with
  fixed scalar weights. For n>2 modalities at very different scales
  (proprio vs depth vs tactile) this weighting is non-trivial — open
  sub-decision in the design note.
- **Does the timestep-as-mask trick survive on a frozen base?** UWM's
  marginalisation relies on the *joint* model having seen all timestep
  combinations. With a frozen video base that only knows the video
  schedule, the "set `t_video = T` to marginalise video" direction may not
  behave — _needs thought / experiment_.
- **Register tokens** — do we need an analogue for cross-modal sharing in
  the compositional-adapter variant, or does the mask + shared embedding
  suffice? _open._

## Related

- [[_MOC]]
- [[../../50_Decisions/open/multimodal-adapter-broadening]] — the design this
  paper anchors (timestep scheme).
- [[avid]] — the learned-mask precedent UWM's timestep-masking subsumes.
- [[../tech/mask-mix-gate]] — our current learned gate.
- [[../tech/structural-encoder]] — the input-side multimodal path that
  already exists.
- `docs/paper/unified_wms.pdf` — Eq. 1 (objective), Eqs. 2–5 (the four
  inference modes), Fig. 1–3 (architecture).
