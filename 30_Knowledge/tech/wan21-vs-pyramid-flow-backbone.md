---
type: tech-note
status: living
last_updated: 2026-06-18
sources:
  - "code: external_repos/Wan2.1/wan/text2video.py"
  - "code: external_repos/Wan2.1/wan/utils/fm_solvers_unipc.py"
  - "code: external_repos/Wan2.1/wan/utils/fm_solvers.py"
  - "code: external_repos/Wan2.1/wan/configs/"
  - "code: external_repos/Pyramid-Flow/diffusion_schedulers/scheduling_flow_matching.py"
  - "code: external_repos/Pyramid-Flow/pyramid_dit/pyramid_dit_for_video_gen_pipeline.py"
  - "doc: docs/open-video-base-models.md"
relevance: D1 / D2  # choice of frozen flow-matching backbone for the adapter framework
---

# Wan2.1 vs Pyramid Flow as the flow-matching backbone

> **DECISION (2026-06-18): Wan2.1-T2V-1.3B is the primary flow-matching base.
> Pyramid Flow is *not* the primary base — it is reserved as a robustness /
> stress-test backbone.** The reason is architectural, not quality: Pyramid
> Flow breaks the single uniform `(x_t, t) → Δ` contract the whole adapter
> framework is built on.

Both are flow-matching, velocity-prediction video DiTs with **linear** paths.
They are *both* legitimate flow-matching examples. The split is in **how the
flow trajectory is structured**, and that structure is exactly what our
composition rule `f_base(x_t,t) + g(d)·Δ_φ(x_t,t,a_t,d)` depends on.

## Wan2.1 — single-stage rectified flow (clean contract)

- One global σ ∈ [0,1], one resolution, one velocity head.
- Path: `x_t = (1−σ)·x_0 + σ·noise`; model predicts `v`; recovery
  `x_0 = x_t − σ·v` (`fm_solvers_unipc.py:323`, `_sigma_to_alpha_sigma_t`
  returns `(1−σ, σ)`).
- `prediction_type="flow_prediction"`, `num_train_timesteps=1000`,
  solvers `FlowUniPCMultistepScheduler` / `FlowDPMSolverMultistepScheduler`,
  `shift` reweights σ. Authors state it outright (README.md:624: "Flow
  Matching framework within … Diffusion Transformers").
- Maps **directly** onto `BaseGenerativeModel.forward(x_t, t, cond)` and the
  additive Δ rule. `t` is a single scalar; latent shape is fixed.
- 1.3B (dim 1536 / 30 layers / 12 heads) and 14B variants; Wan-VAE
  `vae_stride=(4,8,8)`, patch `(1,2,2)`, 16-ch latent, umt5-xxl text encoder.
  Apache-2.0.

## Pyramid Flow — pyramidal, multi-stage + autoregressive (broken contract)

`PyramidFlowMatchEulerDiscreteScheduler`
(`scheduling_flow_matching.py`) is fundamentally piecewise:

- `stages: int = 3` — flow split into **per-stage σ ranges**
  (`sigmas_per_stage`, `start_sigmas`, `end_sigmas`), each stage at a
  **different spatial resolution** (the pyramid: coarse → fine).
- **Renoising jumps at stage boundaries** — `gamma` correction re-injects
  noise when crossing resolutions
  (`corrected_sigma = (1/(sqrt(1+1/γ)·(1−σ)+σ))·σ`, lines ~113-117). The
  trajectory is *not* a single straight line in one space.
- **Autoregressive in time** — later video chunks condition on earlier
  generated ones (README: "Autoregressive Video Generation").

## Why this matters for the adapter framework (D1)

| | Wan2.1 | Pyramid Flow |
|---|---|---|
| `forward(x_t,t)` contract | global scalar t | t means different things per stage |
| Latent shape in sampling | fixed | **changes across stages** |
| `f_base + Δ_φ` composition | trivial | Δ must be stage/resolution-aware |
| Ecosystem / weights | Apache-2.0, Diffusers, VACE family | smaller, less standardized |
| Published param count | explicit 1.3B | not published (inferred <7B) |

Designing adapters against Pyramid Flow means fighting a moving target
(varying resolution + renoising boundaries + AR carry-over) instead of
isolating the **adapter mechanism**, which is the thesis contribution. That
is a confound, not a feature.

## How Pyramid Flow stays useful

Its multi-stage structure is the right **robustness experiment**: once an
adapter works on Wan's clean single-stage flow, ask *"does Δ_φ survive a
multi-stage / multi-resolution / autoregressive flow?"* So keep it on the
bench as a stress test, not the primary backbone.

For a *second clean* single-stage flow base (cross-backbone generalization),
prefer **Open-Sora 1.2/1.3** over Pyramid Flow — same simple contract, ~1B.

See [[../../docs/open-video-base-models]] (the April survey already ranked
Wan #1 and Pyramid Flow as the "alternative flow-matching" base). Next step:
wrap Wan2.1 as a `BaseGenerativeModel` mirroring the DynamiCrafter
integration — see [[concat-condition]] for the existing wrapper pattern.
