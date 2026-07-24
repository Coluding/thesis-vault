---
type: feat
scope: backbone
status: done
priority: high
created: 2026-06-18
updated: 2026-07-15
resolution: shipped
resolution_note: >
  P0-P6 all complete and GPU-verified (2026-06-22, see the "Phases / files /
  acceptance" section for the full per-phase evidence): frozen Wan2.1-T2V-1.3B
  wrapper, output adapter (reused the generic transformer head rather than a
  second full DiT — parameter-efficiency call made mid-implementation),
  MetaWorld training loop (real 1.43B base + real Wan-VAE, flow loss
  2.30->1.68 over 4 steps), generation (corgi T2V + MetaWorld frame-conditioned
  SDEdit), and shortcut wired for flow + the transformer head
  (shortcut_direction_loss firing, per-rung bucketing verified). 11+ guarded
  GPU tests pass. "Full Wan integration (base+adapter+data+train+generate) is
  end-to-end functional" per the ticket's own final phase note. Two minor
  "Open questions" (text-context handling for an action-only world model;
  a bf16/autocast note) remain as decisions, not blockers — left in the body
  below for whoever picks this up next. Note: this is Wan**2.1**-T2V-1.3B,
  superseded as the primary backbone by the Wan**2.2**-TI2V-5B diffusion-forcing
  integration (feat-wan22-ti2v-diffusion-forcing-i2v, done) that all the
  2026-07 debugging work is built on — 2.1 stays as a working, documented
  reference integration, not actively used for the current experiments.
closed_at: 2026-07-15
related:
  - "[[../30_Knowledge/tech/wan21-vs-pyramid-flow-backbone]]"
  - "[[../30_Knowledge/tech/concat-condition]]"
  - "[[../30_Knowledge/tech/frame-stride-conditioning]]"
---

# Integrate Wan2.1-T2V-1.3B as a frozen flow-matching backbone

Add Wan2.1-T2V-1.3B as a second frozen base model alongside DynamiCrafter,
mirroring the DynamiCrafter integration pattern. Wan is our first **clean
single-stage flow-matching** backbone (velocity prediction, linear path) —
see [[../30_Knowledge/tech/wan21-vs-pyramid-flow-backbone]] for why it was
chosen over Pyramid Flow.

## Why
D1/D2 need a modern flow-matching DiT to complement the DynamiCrafter U-Net,
so adapter mechanisms can be shown to transfer across diffusion↔flow and
U-Net↔DiT. Wan-T2V-1.3B is small (1.3B), Apache-2.0, and exposes a clean
`(x_t, t) → velocity` map.

## Decisions (2026-06-18)
- **Vendor the full `wan/` module** into `backbones/wan/` (incl. T5/VAE/
  distributed), closest to upstream.
- **First milestone = end-to-end MetaWorld run** (Phases 0–3).
- **No checkpoint assumed yet** — build with random weights via
  `allow_missing_checkpoint`; document HF download `Wan-AI/Wan2.1-T2V-1.3B`.

## What stays the same vs DynamiCrafter
| Aspect | DynamiCrafter | Wan2.1 |
|---|---|---|
| Wrapper implements `BaseGenerativeModel` | `DynamicCrafterUNetWrapper` | NEW `Wan21DiTWrapper` |
| `model_type` / `prediction_type` | diffusion / velocity | **flow / velocity** |
| Registered in `models/base/factory.py` | `provider == "dynamicrafter"` | `provider == "wan2.1"` |
| Vendored model code | `backbones/dynamicrafter/` | `backbones/wan/` |
| Output adapter | `adapters/output/dynamicrafter.py` | `adapters/output/wan.py` |
| Configs | `configs/base/...` + experiment yaml | parallel new yamls |
| Loss | `LossRegistry["diffusion"]` | `LossRegistry["flow"]` (already exists) |

## What is genuinely different (the real work)
1. **DiT, not U-Net** — see the wrapper-bridge note below. Frozen base stays
   a pure `(x_t,t)→v` map; action conditioning moves entirely into the Δ
   adapter.
2. **Text encoder / context dim** — umt5-xxl (T5, `text_len=512`), not
   open_clip. For an action-only world model, text is fixed/empty → see Open
   question.
3. **VAE** — Wan-VAE, `vae_stride=(4,8,8)`, `z_dim=16` (vs DynamiCrafter
   4-ch). Affects `latent_channels`/shape metadata + any pixel decode.
4. **Checkpoint format** — single safetensors/`.pth` DiT (no
   `model.diffusion_model.` LDM prefix nesting) → simpler prefix handling.

## Key architectural difference vs DynamiCrafter
Wan is a **DiT**, not a U-Net. `WanModel.forward(x, t, context, seq_len,
clip_fea=None, y=None)` takes a **list** of `(16,F,H,W)` latents plus a
packed `seq_len` (patch `(1,2,2)`), and umt5-xxl text context. The wrapper
must bridge our batched `(B,16,T,H,W)` → list + seq_len. The frozen base
stays a pure `(x_t,t)→v` map; **all action conditioning lives in the Δ
adapter** (DynamiCrafter baked `act`/`fs` into the U-Net forward; we don't).
VAE is Wan-VAE, `vae_stride=(4,8,8)`, `z_dim=16` (vs DynamiCrafter 4-ch).

## Phases / files / acceptance
**P0 — Vendor + skeleton.** Copy `external_repos/Wan2.1/wan/` →
`src/generative_flow_adapters/backbones/wan/` (full module incl. T5/VAE/
distributed). Add `__init__.py`, fix imports to the new package path.
*Done when:* `import ...backbones.wan.modules.model` succeeds.

**P1 — Frozen base wrapper.** NEW `models/base/wan.py`
`Wan21DiTWrapper(BaseGenerativeModel)` with `from_config(...)`,
`forward(x_t, t, cond)`, `freeze()`; `model_type="flow"`,
`prediction_type="velocity"`; the `(B,16,T,H,W)` → list + `seq_len` packing;
checkpoint load with `allow_missing_checkpoint` (random weights fallback).
EDIT `models/base/factory.py` → `elif provider == "wan2.1"`.
*Done when:* wrapper builds with random weights, forward returns correct
shape, and `base + 0·Δ == base` (composition sanity).

**P2 — Output adapter.** ✅ **Decision changed during impl (2026-06-19):** did
*not* build a second full Wan DiT adapter. The existing backbone-agnostic
`transformer` `OutputHeadAdapter` already handles 5D `[B,C,T,H,W]` latents and
injects action conditioning via `cond_dim` — and a second 1.3B DiT would defeat
the *parameter-efficient* goal. Wan composes with it out of the box (only a
`testing/fake_data.py` `wan2.1` branch was needed). Result: frozen base (0
trainable), ~438K-param adapter, exact identity composition at init (zero-init
head). A Wan-DiT-shaped delta remains a future option if ever wanted; it would
also require adding an action pathway to the vendored `WanModel`.
*Done:* `AdaptedModel(base=wan, transformer head)` trains on fake data;
`tests/test_wan_backbone.py::test_adapted_model_composition` passes.
Config: `configs/wan_output_adapter.yaml`.

**P3 — Configs + MetaWorld run.** NEW `configs/base/wan2.1_t2v_1_3B.yaml`
(dims: dim 1536 / 30 layers / 12 heads / 16-ch) + NEW
`configs/diffusion_wan_shortcut_metaworld.yaml` (experiment, parallel to the
avid/hyperalign metaworld configs). Flow loss already in `LossRegistry`.
*Done when:* `examples/training_test.py --config … --steps 3` passes, then a
real MetaWorld run trains and loss decreases.

**P4 — Wan-VAE decode + GPU generation.** ✅ Done (2026-06-22).
`examples/wan_generate_video.py`: full sampling pipeline through our
`Wan21DiTWrapper` (real weights) + vendored T5 encoder + FlowUniPC sampler
(CFG) + Wan-VAE decode → mp4. Produced a coherent, prompt-faithful video
(corgi running across a field) at 33f/256² in 25 steps on an RTX 3090.
`--through-adapter` (full `AdaptedModel`) is bit-identical to the base path at
every step (zero-init delta); the ~1% mp4 diff is x264 lossy-encode noise.
*Gotchas recorded:* (1) build T5 on CPU + encode-first-then-free — its fp32
construction peaks ~22GB on GPU and OOMs alongside the DiT; (2) `transformers`
is required for the umt5 tokenizer.

**Frame-conditioned generation (SDEdit) added (2026-06-22).** The 1.3B model is
T2V (no native image input — true I2V is the 14B model), so frame conditioning
= encode a MetaWorld frame via **Wan-VAE encode** → tile over time → add partial
noise at `--strength` → denoise the schedule tail. `wan_generate_video.py
--cond-image-hdf5 ds/metaworld_corner2.hdf5 --strength 0.7` produced a video
that **retains the MetaWorld scene** (table, red Sawyer arm, green object) with
added motion — stored at `outputs/wan_metaworld_cond.mp4` (+ `_cond_frame.png`,
`_strip.png`). This also proves the **Wan-VAE encode** path (the last gap for a
training run). New guarded GPU tests: `tests/test_wan_generation_gpu.py`
(VAE frame roundtrip + strict DiT load). So MetaWorld pixel→latent encoding is
now demonstrated end-to-end; a training run just needs that encode wired into
the data pipeline.

**Base-vs-adapter frame-conditioned comparison test (2026-06-22).**
`tests/test_wan_generation_gpu.py::test_wan_frame_conditioned_base_vs_adapter`
runs the SDEdit frame-conditioned rollout through both the frozen base and the
full AdaptedModel (zero-init adapter), stores both mp4s + a
`[cond | base | adapter]` strip, and asserts the latent trajectories match
(<1e-3). Result: the two videos are **byte-identical** — the untrained adapter
adds no delta even across a multi-step frame-conditioned rollout. Artifacts:
`outputs/test_wan_cond_{base,adapter}.mp4`,
`outputs/test_wan_cond_base_vs_adapter_strip.png`. (Fast variant: null text /
6 steps / 128² — low fidelity by design; the point is the equivalence.)

**P5 — Wan AVID training script (2026-06-22).** `scripts/train_wan_shortcut_metaworld.py`
(the Wan analogue of `train_avid_shortcut_metaworld.py`). Key difference is the
data layer: NEW `data/wan_batch_preprocessor.py` `WanBatchPreprocessor` encodes
MetaWorld pixels → **16-ch Wan-VAE** latents and builds the rectified-flow
triple `(x_t, t, target=noise−z0)` (the flow trainer branch reads x_t+target
directly, so the preprocessor must build them; config sets
`use_batch_timesteps_for_flow: true`). Conditioning is Wan-native: base sees a
null context, the adapter gets the per-clip aggregated action. Verified: trains
on `ds/metaworld_corner2.hdf5` with the real 1.43B base + real Wan-VAE, loss
2.30→1.68 over 4 steps, only the 13.2M-param adapter gets grads. Guarded test
`test_wan_metaworld_training_step`. This closes the loop the earlier P3/P4 notes
flagged — MetaWorld pixel→latent encode is now wired into a working train loop.
**P6 — Shortcut wired for flow + transformer head (2026-06-22).** Two parts:
1. *Conditioning* — `OutputHeadAdapter` now has the same `step_level_embed`
   path as the DynamiCrafter adapter (embeds the step size, adds it to the
   action embedding via the shared `encode_step_level_embedding` /
   `combine_adapter_embeddings` helpers). Honors `use_step_level_conditioning`/
   `step_level_key`/`step_level_hidden_dim`/`step_level_transform`; factory
   passes them through. Verified the adapter delta changes with step_level.
2. *Supervision* — the trainer's self-consistency target was DDIM-only (invalid
   for flow: `alphas_cumprod[t]` with t∈[0,1] indexes 0). Added
   `flow_micro_step_v` (straight-line Euler `x - d·v`) +
   `compute_self_consistency_target_v_flow`, and a `model_type=="flow"` branch
   in `_maybe_prepare_shortcut` (dyadic d=2^-k, supervise 2d vs avg of two
   d-steps). Diffusion path untouched (branch is flow-guarded). Gotcha: the
   target must run under `self._autocast()` — the Wan DiT's fp32 time-embedding
   needs autocast to reconcile with bf16 weights outside the main forward.
   Verified on GPU with the real base: `shortcut_direction_loss=0.1214` fires
   with per-rung bucketing (`N004`), loss finite. Config
   `diffusion_wan_shortcut_metaworld.yaml` now enables it
   (`shortcut_direction_weight: 1.0`, `shortcut_max_log2_steps: 3`). Tests:
   `test_flow_micro_step_and_self_consistency_target`,
   `test_flow_shortcut_training_step_fires`. Full Wan integration (base +
   step-conditioned shortcut adapter + Wan-VAE data + train + generate) is now
   end-to-end functional — D3 shortcut path included.

## Reference: DynamiCrafter integration touchpoints (the pattern to mirror)
- Wrapper: `src/generative_flow_adapters/models/base/dynamicrafter.py`
  (`DynamicCrafterUNetWrapper.from_config`, checkpoint prefix-strip ~L76-96).
- Factory dispatch: `src/generative_flow_adapters/models/base/factory.py`
  (`elif provider == "dynamicrafter"`, ~L51-65).
- `ModelConfig` dataclass: `src/generative_flow_adapters/config.py` (`type`,
  `provider`, `prediction_type`, `pretrained_model_name_or_path`, `freeze`,
  `pass_cond_to_base`, `extra`).
- Builder: `src/generative_flow_adapters/training/builders.py`
  (`build_experiment` → `build_base_model` → `build_adapter` → `AdaptedModel`).
- Adapter: `src/generative_flow_adapters/adapters/output/dynamicrafter.py`;
  factory `src/generative_flow_adapters/adapters/factory.py` (~L219-240).
- Wan source interface: `external_repos/Wan2.1/wan/modules/model.py`
  `WanModel.forward(x, t, context, seq_len, clip_fea=None, y=None)`;
  input prep in `wan/text2video.py` (~L160-195, `target_shape`/`seq_len`).

## Open questions
- Text context for an action-only world model: precompute a fixed null T5
  context once, or strip text conditioning from the vendored forward?
  (Leaning: precompute null context once — least invasive to vendored code.)
- bf16 via `training.extra.amp_dtype` (autocast), not model dtype — see the
  bf16/flash-attention memory note before training.
