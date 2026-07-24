---
type: decision
status: decided
created: 2026-06-17
decided_at: 2026-06-17
updated: 2026-06-17
target_date:
scope: conditioning
related:
  - "[[shortcut-anchor-schedule]]"
  - "config: configs/diffusion_avid_shortcut_metaworld.yaml"
  - "config: configs/diffusion_avid_shortcut_action_metaworld.yaml"
  - "config: configs/diffusion_avid_shortcut_affine_metaworld.yaml"
  - "config: configs/diffusion_hyperalign_shortcut_metaworld.yaml"
  - "config: configs/diffusion_unicon_metaworld.yaml"
  - "config: configs/multimodal_metaworld.yaml"
  - "config: configs/opensora_output_adapter.yaml"
  - "code: src/generative_flow_adapters/models/adapted_model.py:141"
  - "code: src/generative_flow_adapters/inference/diffusion.py:72-95"
---

# Decision: no classifier-free guidance (condition dropout) for the action/step-size adapters

## Status

**Decided 2026-06-17 — drop CFG.** `drop_condition_prob` set to `0.0`
across all action-conditioned adapter configs (8 configs; one,
`multimodal_dynamicrafter.yaml`, was already `0.0`). Condition dropout
is only worth its cost if CFG is used at inference, and we are not using
it.

## Context

Condition dropout is the training-side half of **classifier-free
guidance (CFG)**: each forward, a per-sample Bernoulli mask
(`AdaptedModel._sample_condition_drop_mask`,
`src/generative_flow_adapters/models/adapted_model.py:141`) replaces the
encoded condition with a learned `null_embedding`
(`conditioning/encoders.py:154-165`) for the dropped fraction. At
inference you then extrapolate `v = v_uncond + w·(v_cond − v_uncond)`
with `w>1` to amplify how strongly the sample obeys the condition.

The live MetaWorld / shortcut configs carried `drop_condition_prob:
0.05` (opensora `0.1`), inherited from the image/video-generation
lineage (DynamiCrafter/AVID), not chosen for this project.

Two observations made it a clear net negative as configured:

1. **Inference never uses guidance.** The CFG combination in the
   sampler is dead-coded — `src/generative_flow_adapters/inference/diffusion.py:93`
   is `if False: #TODO check if we need unconditional sampling`, and the
   default `guidance_scale=1.0`. So the unconditional prediction the
   dropout trains for is never consumed. Pure cost.

2. **It corrupted the shortcut self-consistency target.** The target is
   built under `model.eval()` (dropout off → always conditional) while
   the supervised student forward runs under `model.train()` (dropout on
   → ~5% unconditional). For that 5% the loss regressed the
   *unconditional* 2d-shortcut onto a *conditional* self-consistency
   target — which actively shrinks the CFG guidance gap, i.e. weakens
   action-following, the opposite of the intent. (The mask is also
   resampled per forward, so the consistency triple `v1/v2/student` need
   not even share a regime.) See the shortcut target path:
   `training/shortcut_targets.py:34-58`, `training/trainer.py` schedule
   branch.

## Decision

No CFG for this project. The conditioning here (action `a_t` + step-size
`d`) is meant to **constrain** the dynamics — `p(x_{t+d} | x_t, a_t, d)`
— not to nudge samples into semantically "more meaningful" regions the
way a text prompt does in T2I. Specific reasons:

- **World-model objective tension.** CFG with `w>1` deliberately
  distorts the sampling distribution toward over-adherence; for a model
  whose headline metric is prediction accuracy / calibrated dynamics for
  planning (D2/D4), a guidance-sharpened rollout is plausibly *less*
  accurate, not more.
- **Low-dim continuous action, not a prompt.** Extrapolating past a 4-D
  action embedding can leave the valid action manifold; the classic
  "model ignores a rich prompt" failure mode is weak here.
- **Fights the few-step goal (D3/D4).** CFG doubles forwards per step;
  the point of shortcut adapters is *cheaper* rollout.
- **The frozen base keeps its own CFG.** `drop_condition_prob` lives in
  the `conditioning:` block and drops the *adapter's action condition*,
  not the base's pretrained text/image conditioning — so turning it off
  does not touch the base.

## Consequences

- `drop_condition_prob: 0.0` in all action-conditioned configs (done
  2026-06-17).
- The shortcut-target / condition-dropout interaction bug is **moot**
  while dropout is off — no separate bug ticket needed unless CFG is
  reintroduced.
- If action-following later measures too weak and CFG is reconsidered, it
  must be a deliberate reversal: (1) wire up the inference CFG path
  (`diffusion.py:93`), (2) share one drop mask across the shortcut
  consistency triple (teacher + student), and (3) account for the 2×
  few-step inference cost. Supersede this note if so.

## Related

- [[shortcut-anchor-schedule]] — the other knob shaping the shortcut
  training signal.
- The shortcut-target faithfulness question (v-space vs x₀ averaging) is
  separate and still open — not resolved here.
