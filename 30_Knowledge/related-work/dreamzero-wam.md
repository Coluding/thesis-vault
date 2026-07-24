---
type: paper
status: living
last_updated: 2026-06-27
title: "World Action Models are Zero-shot Policies (DreamZero)"
authors: ["Seonghyeon Ye", "Yunhao Ge", "Kaiyuan Zheng", "Shenyuan Gao", "...", "Joel Jang", "Jim Fan", "Yuke Zhu"]
venue: arXiv:2602.15922
year: 2026
url: https://arxiv.org/abs/2602.15922
local_pdf:
relevance: D2 / theory — joint video+action prediction as an alternative to action-as-conditioning
---

# DreamZero — World Action Models (WAMs)

A 14B robot foundation model built on a **frozen-initialised, fully-finetuned**
Wan2.1-I2V-14B video-diffusion backbone. The core move relevant to this thesis:
**action is predicted jointly with video, not used as a conditioning input.**
Treats video as a dense world-state representation; action learning becomes
inverse dynamics aligned to a predicted visual future.

## The one equation that matters for us (Eq. 1)

Joint video+action prediction **factorises** into video prediction × an
inverse-dynamics model (IDM), and is trained end-to-end as a single model:

```
π(o_{l:l+H}, a_{l:l+H} | o_{0:l}, c, q_l)
   = π(o_{l:l+H} | o_{0:l}, c, q_l)        ·  π(a_{l:l+H} | o_{0:l+H}, q_l)
       video prediction (autoregressive)       inverse dynamics (IDM)
```

- `o` = video observations, `a` = actions, `c` = language, `q_l` = proprioceptive state.
- They train **one** model for the joint, not separate video + IDM models (Pai
  et al., Li et al. do the separate version). Claim: end-to-end joint training
  gives tighter video↔action alignment.
- **Note action is not pure-output:** they still *condition* on proprio `q_l`,
  past observations `o_{0:l}`, and language. Only *future action* and *future
  video* move to the predicted side. The real variable is which streams are
  clamped-clean (conditioning) vs. noised-and-predicted.

## Training objective — joint flow matching over `[video; action]`

Standard flow matching (Lipman/Liu), velocity target on the **concatenation** of
video latent and normalised action:

```
z^k_{t_k} = t_k·z^k_1 + (1−t_k)·z^k_0 ,   a^k_{t_k} = t_k·a^k_1 + (1−t_k)·a^k_0
v^k := [z^k_1, a^k_1] − [z^k_0, a^k_0]
L = E[ (1/K) Σ_k w(t_k) · ‖ u_θ([z^k_{t_k}, a^k_{t_k}]; C_k, c, q_k, t_k) − v^k ‖² ]
```

Autoregressive over **chunks**: each chunk K latent frames = action horizon;
teacher forcing on clean previous chunks; chunks get **independent** timesteps,
but **within a chunk video and action share `t_k`** (the DreamZero default).

## Two design axes that map onto OUR open decisions

1. **Shared vs. independent video/action timestep.** DreamZero **shares** the
   timestep across video+action "for faster convergence at the beginning of
   training" — an explicit *counterpoint* to the UWM independent-per-modality
   scheme we adopted ([[unified-world-models]],
   [[../../50_Decisions/open/multimodal-adapter-broadening]] sub-decision 1).
2. **DreamZero-Flash — decoupled schedule for few-step inference.** Bias video
   toward high noise via `t^video = 1 − η, η~Beta(7,1)` (E[t]=0.125, mostly
   noisy) while keeping action timestep **uniform**. Trains the model to predict
   **clean actions from still-noisy video context** — matching the 1-step
   inference regime. Cuts diffusion steps 4→1 (~350ms→~150ms). This is directly
   relevant to our **shortcut line**: "predict clean action from few-step-noisy
   video" is a shortcut-flavoured idea on the action stream.

## Other findings (sourced, from the paper)

- **>2× generalization over SOTA VLAs** (GR00T N1.6, π0.5) to unseen
  tasks/environments in real-robot evals; world-modelling objective lets them
  learn from **diverse, non-repetitive** data (no repeated demos needed).
- **Cross-embodiment transfer:** video-only human/other-robot data (10–20 min)
  → +42% relative on unseen tasks; **30 min** of play data adapts to a new robot.
- **Real-time control at 7Hz** for a 14B model via 38× inference speedup
  (CFG parallelism, DiT velocity-caching, NVFP4 quant, async closed-loop with
  KV-cache, Flash). Async exec: replace predicted frames with ground-truth obs
  in the KV cache after each chunk → kills compounding error (a WAM-only trick).
- **LoRA underperformed** full-DiT updates ("suboptimal results") — they update
  all DiT blocks + state/action encoders/decoders, freeze text/image encoders + VAE.
- **Policy quality ∝ video-generation quality** (larger backbone → better video →
  better actions). Echoes our multimodal "works" criterion (2): video improving
  is the lever.

## Difference from our setting (why we can't just copy it)

- **They finetune the whole 14B DiT; we keep the base frozen and train only a
  small adapter.** Their video↔action coupling lives in joint self-attention
  across an unfrozen DiT. We must route all coupling through the adapter
  (`ModalityEncoder` / `VideoReadout`, the 2026-06-19 design), because our video
  base is frozen and video-only. So the frozen-base coupling question
  (decision sub-decision 3) is the load-bearing risk for porting this.
- **They're autoregressive (KV-cache, chunked);** our DynamiCrafter base is a
  fixed-horizon bidirectional diffusion. Their AR-vs-bidirectional argument
  (modality alignment, native FPS) is a separate axis we haven't taken on.

## Related

- [[unified-world-models]] — the independent-timestep marginalisation scheme;
  DreamZero is the *shared-timestep* counter-example.
- [[avid]] — our frozen-base output-adapter starting point.
- [[../../50_Decisions/open/multimodal-adapter-broadening]] — where the
  "action-as-modality" option is captured.
- [[surfing-uncertainty]] — action-oriented predictive processing: action as
  part of the same forward-prediction machinery (conceptual sibling to Eq. 1).
</content>
</invoke>
