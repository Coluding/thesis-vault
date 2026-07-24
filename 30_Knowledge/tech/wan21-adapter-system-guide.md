---
type: tech-note
status: living
last_updated: 2026-06-23
sources:
  - "code: src/generative_flow_adapters/models/base/wan.py"
  - "code: src/generative_flow_adapters/adapters/output/wan.py"
  - "code: src/generative_flow_adapters/adapters/output/output_head.py"
  - "code: src/generative_flow_adapters/backbones/wan/modules/action_model.py"
  - "code: src/generative_flow_adapters/data/wan_batch_preprocessor.py"
  - "code: src/generative_flow_adapters/inference/flow.py"
  - "code: examples/wan_generate_video.py"
  - "code: scripts/train_wan_shortcut_metaworld.py"
relevance: D1 / D2 / D3 / D4  # the working Wan flow-matching adapter stack
---

# Wan2.1 adapter system — usage guide (with code)

End-to-end reference for the Wan2.1 flow-matching stack built into the repo:
frozen 1.3B base + trainable adapter, action + shortcut conditioning, MetaWorld
training, flow inference, and generation. See companion notes:
[[wan21-model-architecture]] (DiT internals), [[wan-vendoring-patches]],
[[wan21-vs-pyramid-flow-backbone]]. Tickets: [[../../20_Tickets/feat-wan21-backbone-integration]],
[[../../20_Tickets/feat-tiny-wan-action-adapter]],
[[../../20_Tickets/fix-wan-flow-timestep-scale]],
[[../../20_Tickets/feat-flow-inference-sampler-eval]].

## 0. Mental model

```
prediction = base(x_t, t)  +  adapter_Δ(x_t, t, action, step_d)
             └ frozen Wan ┘    └────── trainable ──────┘
```
- Base: `Wan21DiTWrapper`, `model_type="flow"`, `prediction_type="velocity"`.
  Sees only `(x_t, t)` + null text context. **No action.**
- Adapter: two interchangeable styles (below). All action/step conditioning
  lives here. Zero-init output ⇒ delta≈0 at init ⇒ `prediction == base`.
- Timestep convention: interpolate with `σ∈[0,1]`, **feed the model
  `t = σ·1000`** (Wan native). See §5.

## 1. Checkpoint

Downloaded to `ckpts/Wan2.1-T2V-1.3B/` (DiT safetensors + `Wan2.1_VAE.pth` +
umt5-xxl T5). `flash_attn` is NOT installed → SDPA fallback (vendored patch).

## 2. Build the frozen base

```python
from generative_flow_adapters.models.base.wan import Wan21DiTWrapper
import torch

base = Wan21DiTWrapper.from_config(
    wan_config_path="configs/base/wan2.1_t2v_1_3B.yaml",  # or wan2.1_tiny.yaml for CPU
    model_type="flow", prediction_type="velocity",
    checkpoint_path="ckpts/Wan2.1-T2V-1.3B",  # dir of *.safetensors; strict load
    dtype=torch.bfloat16,
).to("cuda").eval()

x_t = torch.randn(2, 16, 4, 32, 32, device="cuda")   # [B, 16ch, T, H/8, W/8]
t   = torch.tensor([120., 880.], device="cuda")       # NATIVE [0,1000], float OK
v   = base(x_t, t)                                     # velocity, same shape
```
Provider key `wan2.1` in `models/base/factory.py`; config knobs in
`model.extra` (`wan_config_path`, `allow_missing_checkpoint`, `dtype`).

## 3. Two adapter styles (same frozen base)

### (a) Lightweight transformer head — backbone-agnostic
DiT-style head over the latent; action via the condition encoder, step via its
own `step_level_embed`. ~13M params at `hidden_dim:512`.
```yaml
adapter:
  type: output
  hidden_dim: 512
  feature_dim: 16
  extra:
    backbone: transformer
    output_format: direct
    condition_on_base_outputs: true
    use_step_level_conditioning: true   # step embed added to action embed
    step_level_key: step_level
    step_level_transform: log2
```

### (b) Tiny-Wan AVID-style — a scaled Wan DiT as the delta
Structural Wan copy; conditioning injected via AdaLN
(`e = time_embed(t) + cond_proj(c) + step_embed(log2 d)`). **Modality-agnostic**:
`c` is the fused `[B, cond_dim]` embedding from the condition encoder (action
today; proprio/goal/language later by swapping to `StructuredConditionEncoder`,
no adapter change). Tiers:

| tier | dim / layers / heads | params | config |
|---|---|---|---|
| 11M | 256 / 10 / 4 | 11.3M | `configs/base/wan_adapter_11m.yaml` |
| 34M | 448 / 10 / 8 | 34.2M | `configs/base/wan_adapter_34m.yaml` |
| 150M | 768 / 16 / 12 | 157M | `configs/base/wan_adapter_150m.yaml` |

```yaml
adapter:
  type: output
  extra:
    backbone: wan                                      # -> Wan21OutputAdapter
    wan_adapter_config_path: configs/base/wan_adapter_11m.yaml
    condition_on_base_outputs: true
    use_step_level_conditioning: true
    step_level_transform: log2
```

Both compose identically:
```python
from generative_flow_adapters.config import load_config
from generative_flow_adapters.training.builders import build_experiment

model = build_experiment(load_config("configs/diffusion_wan_avid_shortcut_metaworld.yaml")).model
# model.base_model frozen (0 trainable); model.adapter is the trainable Δ.
out = model(x_t, t, {"action": torch.randn(2,4,device="cuda"),
                     "step_level": torch.full((2,), 0.25, device="cuda")})
```

## 4. How conditioning is injected

- **Action**: per-frame `act [T,A]` → summed to `[B,A]` in the preprocessor →
  `cond["action"]`. Transformer head: encoded by `MLPConditionEncoder`,
  broadcast-added to every token. Tiny-Wan: `action_embed` MLP added to the
  AdaLN time embedding (AVID-style).
- **Step size** `d`: injected by the trainer as `cond["step_level"]`
  (normalized σ-fraction ∈(0,1], e.g. {0.125, 0.25, 0.5, 1.0}); `log2`-embedded
  and summed with the action conditioning. The frozen base never sees it.

## 5. Flow training target + timestep scale (the important bit)

The flow trainer branch reads `x_t` and `target` straight from the batch, so
`WanBatchPreprocessor` builds the rectified-flow triple:
```python
# data/wan_batch_preprocessor.py (essence)
z0    = WanVAE.encode(video)              # clean 16-ch latent
noise = torch.randn_like(z0)
sigma = torch.rand(B).clamp_min(1e-5)     # interpolation coord in [0,1]
x_t   = (1 - sigma)*z0 + sigma*noise
target = noise - z0                       # velocity (sigma-independent)
t      = sigma * timestep_scale           # 1000 -> Wan NATIVE [0,1000]
```
Set `training.extra.use_batch_timesteps_for_flow: true` so the trainer uses
this `t`, and `flow_timestep_scale: 1000`. **Why scale matters:** feeding the
pretrained base raw `σ∈[0,1]` is off-distribution (it was trained at [0,1000]);
fixing it dropped the flow loss from ~2.3 to ~0.6 at step 1.

## 6. Shortcut self-consistency (flow-native)

DDIM is invalid for flow, so the trainer uses a straight-line Euler micro-step:
```python
# training/shortcut_targets.py
flow_micro_step_v(x, v, d) = x - d*v                  # d in sigma units
# target for step 2d = ½[ v(x_t,t,d) + v(x_t - d·v, t - d·scale, d) ]
```
Enable in config: `shortcut_direction_weight: 1.0`,
`shortcut_max_log2_steps: 3`, `shortcut_anchor_prob: 0.75`. The trainer auto-
branches on `model_type=="flow"`; diffusion path untouched.

## 7. Train on MetaWorld

```bash
python scripts/train_wan_shortcut_metaworld.py \
  --config configs/diffusion_wan_avid_shortcut_metaworld.yaml \  # or _wan_shortcut_ for the head
  --hdf5 ds/metaworld_corner2.hdf5 \
  --ckpt-dir ckpts/Wan2.1-T2V-1.3B \
  --steps 5 --batch-size 1
```
All args default (runs with none). Encodes MetaWorld pixels → 16-ch Wan
latents, builds the flow triple, trains only the adapter (~0.8–0.9% of params).

## 8. Flow inference (eval rollout)

`FlowInferenceSampler` (FlowUniPC, velocity, CFG) — the trainer auto-selects it
for flow models (`training/trainer.py:_build_inference_sampler`). Drop-in API
matching `DiffusionInferenceSampler`:
```python
from generative_flow_adapters.inference import FlowInferenceSampler
sampler = FlowInferenceSampler(model, num_train_timesteps=1000, shift=3.0,
                               timestep_scale=1000, amp_dtype=torch.bfloat16)
latent = sampler.sample_from_batch(batch, num_inference_steps=25, guidance_scale=5.0)
```

## 9. Generate a video (standalone)

```bash
# text-to-video
python examples/wan_generate_video.py --prompt "a corgi running, cinematic" \
  --frames 33 --height 256 --width 256 --steps 25 --out outputs/corgi.mp4

# conditioned on a real MetaWorld frame (SDEdit; 1.3B is T2V, no native I2V)
python examples/wan_generate_video.py \
  --prompt "a robotic arm manipulating an object on a table" \
  --cond-image-hdf5 ds/metaworld_corner2.hdf5 --strength 0.7 \
  --frames 33 --height 256 --width 256 --steps 25 --out outputs/mw_cond.mp4

# through the adapter (base + Δ)
python examples/wan_generate_video.py --through-adapter ...
```

## 10. Gotchas (all bit us; all fixed)

- **bf16 autocast** required around any Wan forward outside the main loop (the
  DiT's fp32 time-embedding needs it) — shortcut target, eval sampler,
  `ActionWanModel` modulation all force/wrap fp32 appropriately.
- **flash_attn absent** → SDPA fallback; the fallback had a dtype-restore bug we
  patched ([[wan-vendoring-patches]]).
- **T5 OOM**: build T5 on CPU, encode, free it *before* loading the DiT.
- **`4n+1` frames**: causal Wan-VAE compresses time 4× → `(F-1)/4+1` latent
  frames; our `temporal_length:8` gives only 2 latent frames (short — consider 17/33).
- **Tests**: `tests/test_wan_backbone.py` (CPU), `tests/test_wan_generation_gpu.py`
  (guarded GPU). 32+ pass.
