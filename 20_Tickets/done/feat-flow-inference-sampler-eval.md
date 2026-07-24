---
type: feat
scope: training
status: done
priority: high
created: 2026-06-22
updated: 2026-06-22
resolution: completed
resolution_note: "NEW inference/flow.py FlowInferenceSampler (FlowUniPC, CFG, velocity, autocast); trainer auto-selects it for model_type==flow via _build_inference_sampler (both adapted + base-only). Timestep fed as scheduler_t/timestep_scale (default 1000) to match our [0,1] training. Diffusion path unchanged. Tests: test_trainer_selects_flow_inference_sampler (CPU), test_flow_inference_sampler_eval_rollout (GPU, real base, decodes+stores mp4). The deeper train-vs-native [0,1] vs [0,1000] convention issue remains open (see body) — own ticket."
closed_at: 2026-06-22
related:
  - "[[../30_Knowledge/tech/wan21-model-architecture.md]]"
  - "[[../30_Knowledge/tech/wan21-vs-pyramid-flow-backbone.md]]"
  - "[[feat-wan21-backbone-integration]]"
  - "[[fix-wan-flow-eval-video-grid-never-fired]]"
---

> **FOLLOW-UP 2026-06-23** — this ticket landed the sampler, but eval video
> still never fired for WAN end to end (guards, unbuilt logger, velocity-as-GT
> bug, orphaned logger). Fixed in
> [[fix-wan-flow-eval-video-grid-never-fired]].

# Flow inference sampler in the trainer eval loop

The trainer's eval/rollout path uses `DiffusionInferenceSampler`
(`training/trainer.py:58,70`), which is **DDIM/diffusion** — wrong for a
flow-matching model like Wan. So today there is no eval-time rollout/video for
flow models: a flow run either gets a semantically-incorrect DDIM rollout or
none. The correct flow sampling currently lives only in the standalone
`examples/wan_generate_video.py`. Wire a flow-native sampler into the trainer.

## Why
D2/D4 eval needs flow rollouts during training (loss alone is weak signal;
the video panels + base-vs-adapter comparison are the real read). Wan is our
flow backbone, so the eval loop must denoise with the rectified-flow ODE.

## Design
- NEW `inference/flow.py` `FlowInferenceSampler` mirroring the
  `DiffusionInferenceSampler` interface (`sample`, `sample_from_batch`) so it
  is a drop-in. Uses the vendored `FlowUniPCMultistepScheduler`
  (`backbones/wan/utils/fm_solvers_unipc.py`), CFG-capable, velocity output.
- EDIT `training/trainer.py`: select sampler by `model_type` — flow/flow_matching
  → `FlowInferenceSampler`, else `DiffusionInferenceSampler`; same for the
  base-only `base_inference_sampler`. Pass the trainer's amp dtype so the Wan
  DiT's fp32 time-embedding reconciles with bf16 weights (same autocast gotcha
  as the shortcut target).

## Timestep-convention caveat (important)
Our flow training samples `t ∈ [0,1]` (`FlowMatchingTrainingObjective.
sample_timesteps`, logit-normal sigmoid), and `WanBatchPreprocessor` feeds that
to the model. But the FlowUniPC scheduler's `timesteps` run `[0,1000]`, and the
**pretrained Wan base was trained at `[0,1000]`** (the proven
`wan_generate_video.py` corgi run feeds raw `[0,1000]`).
=> The sampler must divide the scheduler timestep by `num_train_timesteps`
(default 1000) before feeding the model, so eval matches *our* training
convention. Exposed as `timestep_scale` (set 1.0 for native `[0,1000]`).

Deeper related issue (separate ticket-worthy): our adapter training feeds the
frozen pretrained base `t∈[0,1]`, which is **off-distribution** for a base
pretrained at `[0,1000]`. The adapter compensates, but ideally training should
feed the base its native `[0,1000]` so the frozen velocity is in-distribution.
Flagged here; not fixed by this ticket.

## Acceptance
- Flow model eval produces a coherent rollout tensor of the right shape.
- `sample_from_batch` works from a `{target, cond}` batch (shape inference).
- Trainer auto-selects flow sampler for `model_type=="flow"`; diffusion path
  unchanged.
- Test: a flow rollout runs and is finite; CPU-tiny + (guarded) GPU-real.
