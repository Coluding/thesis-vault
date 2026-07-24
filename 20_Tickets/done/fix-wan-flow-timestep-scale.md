---
type: bug
scope: training
status: done
priority: high
created: 2026-06-22
updated: 2026-06-22
resolution: fixed
resolution_note: "Native [0,1000] convention everywhere via single knob flow_timestep_scale (default 1000): model fed t=sigma*scale. Edits: wan_batch_preprocessor (sample sigma, output t=sigma*scale), shortcut_targets.compute_self_consistency_target_v_flow (t_next=t-d*scale, x_mid uses sigma-step d), trainer (_flow_timestep_scale threaded to shortcut + FlowInferenceSampler), inference/flow.py (t=sigma*scale), train script passes scale from training.extra. STRONG VALIDATION: flow loss dropped 2.30->1.48 (old, base off-distribution) to 0.63->0.21 (fixed) — frozen base now in-distribution. Tests: test_wan_preprocessor_feeds_native_timestep_scale + all wan tests (10 CPU, 5 GPU) pass. Generation script already used native [0,1000] so it's now consistent with training."
closed_at: 2026-06-22
related:
  - "[[feat-flow-inference-sampler-eval]]"
  - "[[feat-wan21-backbone-integration]]"
  - "[[../30_Knowledge/tech/wan21-model-architecture]]"
---

# Wan flow timestep scale: feed the base its native [0,1000], not [0,1]

## Bug
`WanBatchPreprocessor` builds the flow latent with `σ∈[0,1]` AND feeds the
model `t=σ` (∈[0,1]). But the **pretrained Wan base was trained at t∈[0,1000]**
(`timesteps = sigmas·num_train_timesteps`, `num_train_timesteps=1000`), and the
proven `wan_generate_video.py` corgi run feeds raw `[0,1000]`. So during adapter
training the frozen base sees an **off-distribution timestep** —
`sinusoidal_embedding_1d(256, 0.3)` vs `(256, 300)` are completely different —
producing a wrong base velocity that the adapter then has to fight.

## Fix — native convention everywhere
Single knob `flow_timestep_scale` (default `1000`). Interpolation/velocity use
`σ∈[0,1]`; the **model input** is `t = σ · flow_timestep_scale`.

- `data/wan_batch_preprocessor.py`: sample `σ`, build `x_t=(1-σ)z0+σ·noise`,
  `target=noise-z0`, output `t = σ·scale` (was `t=σ`). New
  `WanBatchPreprocessConfig.timestep_scale`.
- `training/shortcut_targets.py:compute_self_consistency_target_v_flow`: the
  flow Euler micro-step on `x` uses the σ-step `d` (`x_mid=x-d·v`), but the
  model-input step must be scaled: `t_next = t - d·scale`. Add `timestep_scale`.
- `training/trainer.py`: thread `flow_timestep_scale` into the flow shortcut
  branch and the `FlowInferenceSampler`.
- `inference/flow.py`: feed the model `t = (timestep/num_train_timesteps)·scale`
  = raw scheduler timestep when scale==num_train_timesteps (native). Default
  flips from the old `[0,1]` to native `[0,1000]`.
- `scripts/train_wan_shortcut_metaworld.py`: pass the scale from
  `training.extra.flow_timestep_scale` to the preprocessor (single source).

## Acceptance
- Training feeds base t∈[0,1000]; the frozen base velocity is in-distribution
  (matches what the generation script feeds).
- Train, eval sampler, and standalone generation all agree on the convention.
- Tests updated/added; diffusion path untouched.
