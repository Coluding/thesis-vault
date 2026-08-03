# Search: flow-matching video model with NO temporal VAE compression

_Captured 2026-08-01. Overnight literature/checkpoint search. Inbox note — not promoted to `30_Knowledge/`._

**Question.** The three backbone cells (DC = diffusion + per-frame 2D VAE; Wan2.2 TI2V = flow + 4x causal 3D VAE; SkyReels = flow + 4x causal 3D VAE) confound *objective* with *tokenizer*. Find the missing cell: a **flow-matching / rectified-flow** latent video model whose **VAE does not compress time** (latent frame count == pixel frame count).

---

## 1. Verdict

**The intersection is not empty. There are four released hits — and the best one hands you the entire 2x2 inside a single codebase.**

> ### 🥇 **OpenDWM CTSD-3.5** — weights `wzhgba/opendwm-models`, code `SenseTime-FVG/OpenDWM`
>
> One pipeline class (`dwm.pipelines.ctsd.CrossviewTemporalSD`) with **three released checkpoint families that differ in exactly the variable you are trying to isolate**:
>
> | checkpoint family | base | objective | tokenizer | cell |
> |---|---|---|---|---|
> | `ctsd_21_*` | SD2.1 | **diffusion** (DDPM/DDIM branch) | per-frame 2D `AutoencoderKL` | **(diffusion, tc=1)** — DC's cell |
> | `ctsd_35_*`, `ctsd_unimlvg_*` | SD3.5-medium | **flow** (`FlowMatchEulerDiscreteScheduler`) | per-frame 2D `AutoencoderKL` | **(flow, tc=1) ← THE MISSING CELL** |
> | `ctsd_35_tvae_f17_*` | SD3.5-medium | **flow** | `AutoencoderKLCogVideoX`, **tc = 4** | **(flow, tc=4)** — Wan's cell |
>
> Same repo, same trainer, same data pipeline, same conditioning interface, same adapter hook points. **The tokenizer ablation `ctsd_35_*` vs `ctsd_35_tvae_f17_*` is a pure one-line config swap** (`"vae": "diffusers.AutoencoderKLCogVideoX"`) on the *same base model*. That is a far cleaner experiment than comparing DC vs Wan vs Vchitect across three unrelated codebases — it removes architecture, data, and training recipe as confounds all at once.
>
> Apache-2.0 weights (ungated), MIT code, start-frame conditioned by construction (`reference_frame_count`), autoregressive rollout already implemented.

**Second, and the strongest match for D3/D4 specifically: `arnavkj1995/WEAVER`.** A genuinely **action-conditioned** (DROID robot actions) rectified-flow world model on a per-frame SD3 VAE — and the release includes **`WEAVER-ReFlow`, a reflow-distilled few-step checkpoint (`val_steps: 4`)**. That is published prior art sitting directly on the D3 contribution surface, which the vault does not currently track. MIT, ungated, 8.26 GB.

The other two hits: **`Vchitect/Vchitect-2.0-2B`** (SD3-medium T2V, flow + per-frame VAE — the only *general-domain* model in the intersection, but gated, T2V-only with no I2V checkpoint or code, and abandoned since 2024-09), and **`tencent/Hy-Embodied-RxBrain-1.0`** (FLUX VAE + 6.2B multimodal transformer with a flow head — satisfies both criteria but is not a video DiT; deprioritise).

### Correction to my own method — read this before trusting the screening table

My first pass ran an automated screen over **612** HuggingFace models carrying the `text-to-video` / `image-to-video` pipeline tags (361 in `diffusers` format), fetching every `vae/config.json` and `scheduler/scheduler_config.json`. That screen returned **`{flow AND tc==1}` = empty**, and I initially concluded the intersection held only Vchitect.

**That conclusion was wrong, and the error is structural, not incidental.** All three additional hits are **world models** (driving / robotics / embodied) released as research repos — raw `.pth` or `.pt` files plus JSON/YAML configs, no `model_index.json`, no `vae/` subfolder, no HF pipeline tag. **A config-scan over HF pipeline tags cannot see them by construction.** The screen remains valid for what it covers — the *general* text-to-video space, where the intersection really is a near-vacuum — but it systematically misses exactly the corner of the literature this thesis lives in. Worth keeping as a search-methodology lesson: **the models closest to your problem are the least likely to be packaged as `diffusers` pipelines.**

The negative result that survives, narrowed to what was actually tested: *among general-purpose text-to-video models packaged for `diffusers`, no released model combines flow matching with an uncompressed temporal axis, and the minimum temporal compression among flow models is 4x — there is no 2x fallback.* World-model repos are the exception, and they are the exception **because they inherit a per-frame image VAE from SD3/SD3.5/FLUX rather than training a video tokenizer.**

### Recommended plan

1. **Integrate OpenDWM CTSD.** It resolves the confound directly and gives two extra control cells for free. Start with `ctsd_35_xs_df6v3_*` (9.21 GB, smallest SD3.5 variant) to de-risk, then the full `ctsd_35_tirda_bm_nwao` (17.22 GB).
2. **Run the pure tokenizer ablation**: `ctsd_35_*` (tc=1) vs `ctsd_35_tvae_f17_*` (tc=4), identical everything else. This is the cleanest available answer to "objective or tokenizer?".
3. **Read WEAVER properly for D3.** `WEAVER-ReFlow` is prior art for few-step distillation of an action-conditioned flow world model and belongs in related work regardless of whether the code is adopted.
4. Treat **Vchitect** as a fallback only. Its unique value (general-domain, not driving/robotics) is real, but gated + T2V-only + no training code + dead repo is a lot to absorb.
5. **Still do the cheap DC action-smearing control** (§4). Hours of work, and it partially answers the question before any new backbone lands.

**One caveat that must be checked before committing to Path A (§5.2):** CTSD is a *driving* world model conditioned on layout / 3D boxes / ego-action over multi-camera rigs, not on a low-dimensional robot action vector. Whether your `a_t` interface maps onto it cleanly is **unverified and is the biggest unknown in this recommendation.**

---

## 2. Screening table

`tc` = temporal compression ratio (1 = per-frame latent, no temporal mixing). **`Objective` and `tc` are verified for every row** from the exact config fields / source lines cited in §3 and §6, except where marked `_needs verification_`.

> **Sourcing caveat on `Size` / `I2V?` / `License`.** Verified only for the rows detailed in §3 (HF `?blobs=true` API and repo `LICENSE` files). On rejected rows these columns are model-card context supporting a "too big / already have this cell" verdict and were **not** independently checked — re-verify before quoting.

### 2a. The hits — flow AND tc = 1

| Model | Objective (sourced) | tc (sourced) | Start-frame cond.? | Weights | License | Verdict |
|---|---|---|---|---|---|---|
| **OpenDWM CTSD-3.5** (`ctsd_35_*`, `ctsd_unimlvg_*`) | **flow** — `SD3Transformer2DModel` branch → `FlowMatchEulerDiscreteScheduler` (`ctsd.py` L977-983) | **1** — default `diffusers.AutoencoderKL` (L952-954); `is_temporal_vae` true **only** for CogVideoX (L963-964) | **Yes** — `reference_frame_count`, autoregressive | 9.21 / 13.61 / 16.02 / 17.22 GB, ungated | weights Apache-2.0, code **MIT** | ✅ **BEST** — ships the whole 2x2 |
| **WEAVER / -FT / -ReFlow** | **rectified flow** — `class FlowWM`, `mse_loss(pred, x1 - x0)` (`model.py` L233, L1921), `xt = (1-t)x0 + t·x1` (L1970) | **1** — SD3 `AutoencoderKL` per-frame (`encoders.py` L169, L181, L223→L242) | **Yes** — 2 history + 6 memory frames | 3 × 8.26 GB, ungated | **MIT** (code); no HF license tag | ✅ **BEST for D3/D4** — action-conditioned + released few-step reflow ckpt |
| **Vchitect-2.0-2B** | **flow** — `FlowMatchEulerDiscreteScheduler` (`pipeline.py` L30, L247) | **1** — `AutoencoderKL` + per-frame decode loop (L29, L235, L956-962) | **No** — T2V only, no I2V ckpt or code | transformer 4.96 GB, **gated (auto)** | apache-2.0 | ⚠️ only *general-domain* hit; dead repo |
| **Hy-Embodied-RxBrain-1.0** | flow matching — `x_t=(1-t)x_0+t·noise`, Euler ODE _(sweep-reported, not independently checked)_ | **1** — FLUX VAE, "zero `nn.Conv3d`" _(sweep-reported)_ | Yes (`--frames obs.jpg`) | 12.41 GB, ungated | apache-2.0 | ⚠️ MoT VLM, not a video DiT — deprioritise |

### 2b. Same-codebase control cells (OpenDWM) — why this is the best option

| Checkpoint family | Base | Objective | tc | Cell |
|---|---|---|---|---|
| `ctsd_21_tirda_bm_nwa_30k` / `ctsd_21_tirda_nwao_30k` (7.69–8.03 GB) | stable-diffusion-2-1 | **diffusion** — `UNetSpatioTemporalConditionModel` branch → DDPM/DDIM (`ctsd.py` L968-976) | **1** | (diffusion, per-frame) |
| `ctsd_35_tirda_nwao_20k`, `ctsd_35_tirda_bm_nwao_40k`, `ctsd_35_df16_..._40k`, `ctsd_35_xs_df6v3_..._60k`, `ctsd_unimlvg_..._60k` | stable-diffusion-3.5-medium | **flow** | **1** | **(flow, per-frame) ← missing cell** |
| `ctsd_35_tvae_f17_tirda_bm_nwao_50k` (17.22 GB) | stable-diffusion-3.5-medium | **flow** | **4** (`"vae": "diffusers.AutoencoderKLCogVideoX"`, `THUDM/CogVideoX-2b`) | (flow, compressed) |

### 2c. Rejected — everything else checked

| Model | Objective (sourced) | tc (sourced) | I2V? | Size | License | Verdict |
|---|---|---|---|---|---|---|
| **CogVideoX-5b-I2V** (`zai-org/`, formerly `THUDM/`) | **diffusion** — `CogVideoXDDIMScheduler`, `prediction_type: v_prediction` | **4** — `temporal_compression_ratio: 4` | Yes, native | 5B; 21.6 GB | CogVideoX Licence (free academic) | Useful *fallback* (diffusion+compressed), ungated — but OpenDWM covers this cell better |
| CogVideoX1.5-5B-I2V / -2b / -5b | diffusion — `CogVideoXDDIMScheduler`, `v_prediction` | 4 | mixed | 2B/5B | CogVideoX / Apache-2.0 (2b) | Same cell |
| Wan2.1 / Wan2.2 (all) | flow — `prediction_type: flow_prediction`, `use_flow_sigmas: true` | 4 — `temperal_downsample: [false,true,true]` | Yes | 1.3–14B | Apache-2.0 | Already have this cell |
| SkyReels-V2 / V3 | flow — Wan scheduler | 4 — `AutoencoderKLWan` | Yes | 1.3–14B | — | Already have this cell |
| HunyuanVideo-1.5 | flow — `FlowMatchDiscreteScheduler` | 4 — `"ffactor_temporal": 4` | — | — | — | ❌ compressed |
| LTX-Video 0.9.x / LTX-2 / LTX-2.3 | flow — `FlowMatchEulerDiscreteScheduler` | 8 — `spatio_temporal_scaling: [t,t,t,f]` | Yes | 2–19B | LTXV licence | ❌ worst compression |
| Mochi-1 | flow — `FlowMatchEulerDiscreteScheduler` | **6** — `temporal_expansions: [1,2,3]` | No | 10B | Apache-2.0 | ❌ compressed |
| Pyramid-Flow (miniFLUX + MMDiT) | flow (pyramidal FM) | **8** — `encoder_temporal_down_sample: [t,t,t,f]` | Yes | 2/8B | MIT | ❌ compressed |
| EasyAnimate V5.1 | flow — `FlowMatchEulerDiscreteScheduler` | >1 — 3× `SpatialTemporalDownBlock3D` in `AutoencoderKLMagvit` | Yes (`-InP`) | 12B | Apache-2.0 | ❌ compressed |
| SANA-Video | flow (Flow-DPM-Solver) | 4 — DCAE-V is **F32T4**C32; 480p path uses Wan2.1-VAE **F8T4**C16 | — | 2B | — | ❌ compressed |
| Lumina-Video | flow (Next-DiT) | 4 — adopts **CogVideoX 3D causal VAE** | — | 2B | Apache-2.0 | ❌ compressed |
| Kandinsky 5.0 Lite I2V | flow | 4 — `vae: name: "hunyuan"`; `HunyuanVideoCausalConv3d` | Yes (`visual_cond: true`) | ~2B | MIT | ❌ compressed (cleanest small MIT flow I2V if Wan ever needs replacing) |
| LongCat-Video, ContentV-8B, Motif-Video-2B, Cosmos3-Super-Image2Video, Helios, LongLive-2.0 | flow (various FM schedulers) | 4 — all `AutoencoderKLWan` | mixed | 2–14B | — | ❌ compressed (all reuse Wan-VAE) |
| Allegro / Allegro-TI2V | **diffusion** — `EulerAncestralDiscreteScheduler`, `epsilon` | 4 — `temporal_compression_ratio: 4` | Yes (TI2V) | 3B | Apache-2.0 | Weaker CogVideoX alternative |
| Cosmos-1.0-Diffusion-7B | **diffusion** — `EDMEulerScheduler` | 8 — `AutoencoderKLCosmos` | — | 7B | NVIDIA OM | ❌ wrong objective |
| Ruyi-Mini-7B | **diffusion** — `DDPMScheduler` | >1 — class *labelled* `AutoencoderKL` but `down_block_types` are `SpatialTemporalDownBlock3D` | Yes | 7B | Apache-2.0 | ❌ mislabelled config; **not** per-frame |
| Open-Sora v1.0 / v1.1 | **diffusion** — `scheduler = dict(type="iddpm")` | **1** — `VideoAutoencoderKL` from `sd-vae-ft-ema` | No | ~0.7B | Apache-2.0 | Wrong objective — note repo already has an `opensora` provider |
| Open-Sora v2.0 | flow (rectified, FLUX-lineage) | 4 — loads `hunyuan_vae.safetensors` | Yes (`stage{1,2}_i2v.py`) | 11B | Apache-2.0 | ❌ compressed + large |
| Latte-1 | **diffusion** — `DDIMScheduler` | **1** — `AutoencoderKL` | No | 0.7B | — | Wrong objective |
| Stable Video Diffusion (xt / xt-1-1) | **diffusion (EDM)** — `EulerDiscreteScheduler` | **1** — `AutoencoderKLTemporalDecoder` (2D encoder; temporal layers in *decoder* only) | Yes, native | 1.5B | SVD NC | Wrong objective |
| I2VGen-XL / Hotshot-XL / AnimateDiff / Cinemo / Zeroscope / ModelScope T2V | **diffusion** — `DDIM`/`EulerDiscrete`/`PNDM` | **1** — `AutoencoderKL` | mixed | 0.4–1.7B | — | Wrong objective |
| WorldGym (`world-model-eval`, arXiv 2506.00613) | *defines* `class FlowMatching` (`diffusion.py` L181) but released 9 GB ckpt is **DDPM v-pred** _(sweep-reported)_ | 1 — SD3 per-frame VAE _(sweep-reported)_ | — | 9 GB | — | ❌ only viable if retraining |
| MiniWAM | frozen `sd-vae-ft-mse` + joint video/action flow matching _(sweep-reported)_ | 1 _(sweep-reported)_ | — | — | — | ❌ **no released weights** — recipe is close to this thesis; check for scoop risk |
| Step-Video-T2V | _needs verification_ | _needs verification_ | — | 30B | MIT | ❌ excluded on size |
| MAGI-1 | _needs verification_ | _needs verification_ | — | 24B | Apache-2.0 | ❌ excluded on size |
| Cosmos-Predict2-2B-Video2World | _needs verification_ — **gated**, configs unreadable | _needs verification_ | Yes | 2B | NVIDIA OM | Unresolved; family shows no per-frame flow variant |
| Qwen-RobotWorld (arXiv 2606.17030), KAM-WM (arXiv 2607.04652) | _needs verification_ | _needs verification_ | — | — | — | No weights repo found |

---

## 3. Per-candidate evidence (exact URLs and quoted lines)

§3.1 and §3.2 I **re-read myself** at the cited line numbers rather than relaying — the verdict turned on them.

### 3.1 OpenDWM CTSD — the recommended candidate

Code: <https://github.com/SenseTime-FVG/OpenDWM> (MIT, "Copyright (c) 2024 SenseTime Research") · weights: <https://huggingface.co/wzhgba/opendwm-models> (Apache-2.0, `gated: False`)

**(a) Objective branches on the base model class.** <https://raw.githubusercontent.com/SenseTime-FVG/OpenDWM/main/src/dwm/pipelines/ctsd.py>:

```python
# L968-976 — SD2.1 / UNet path: DIFFUSION
if isinstance(self.model, diffusers.UNetSpatioTemporalConditionModel):
    train_scheduler_type = dwm.common.get_class(
        self.training_config.get("scheduler", "diffusers.DDPMScheduler"))
    ...
# L977-983 — SD3.5 / MMDiT path: FLOW
elif isinstance(self.model, diffusers.SD3Transformer2DModel):
    self.train_scheduler =\
        diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler")
```

**(b) VAE defaults to the per-frame 2D `AutoencoderKL`; temporal handling is CogVideoX-only.** Same file:

```python
# L952-954
vae_type = dwm.common.get_class(
    self.common_config.get("vae", "diffusers.AutoencoderKL"))
# L963-964
self.is_temporal_vae = isinstance(
    self.vae, diffusers.models.autoencoders.autoencoder_kl_cogvideox.AutoencoderKLCogVideoX)
```

**(c) tc = 1 on the default path — frame count preserved through encode *and* decode.** `train_step`, L1200-1226:

```python
batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
...
if self.is_temporal_vae:
    latents = einops.rearrange(latents, "(b v) c t h w -> b t v c h w", v=view_count)
else:
    latents = einops.rearrange(
        latents, "(b t v) c h w -> b t v c h w",
        t=sequence_length, v=view_count)
```

Inference decode, L1628-1633 — same branch, per-frame flatten:

```python
if self.is_temporal_vae:
    cur_latents = einops.rearrange(latents, "b t v c h w -> (b v) c t h w")
else:
    cur_latents = latents.flatten(0, 2)
```

**(d) Exactly one config overrides the VAE.** I fetched **all 20** `configs/ctsd/**/*.json` and grepped each for a `"vae"` key. 19 inherit the default per-frame `AutoencoderKL`; the sole override is <https://raw.githubusercontent.com/SenseTime-FVG/OpenDWM/main/configs/ctsd/multi_datasets/ctsd_35_tvae_f17_tirda_bm_nwao.json>:

```json
"vae": "diffusers.AutoencoderKLCogVideoX",
"vae_pretrained_model_name_or_path": "THUDM/CogVideoX-2b",
"sequence_length_per_iteration": 17,
"reference_frame_count": 1,
```

Base-model and frame settings across the family (same configs):

| config | `pretrained_model_name_or_path` | `sequence_length_per_iteration` | `reference_frame_count` |
|---|---|---|---|
| `ctsd_21_tirda_bm_nwa` | `stable-diffusion-2-1` | 15 | 3 |
| `ctsd_35_tirda_bm_nwao` | `stable-diffusion-3.5-medium` | 19 | (sampled dict) |
| `ctsd_35_tvae_f17_tirda_bm_nwao` | `stable-diffusion-3.5-medium` | 17 | 1 |
| `ctsd_35_df16_tirda_bm_nwao` | `stable-diffusion-3.5-medium` | 16 | 15 |

**(e) The detail most relevant to D3/D4.** `ctsd_35_df16_tirda_bm_nwao.json` declares:

```json
"scheduler": "dwm.schedulers.temporal_independent.FlowMatchEulerDiscreteScheduler",
```

and that class really exists — <https://raw.githubusercontent.com/SenseTime-FVG/OpenDWM/main/src/dwm/schedulers/temporal_independent.py>, **L173** `class FlowMatchEulerDiscreteScheduler(...)` with **L176** `def step_by_indices(...)` (197-line file, also defining temporal-independent `DDPMScheduler` L6 and `DDIMScheduler` L48). This is **per-frame-independent flow timesteps over per-frame latents** — diffusion-forcing on rectified flow, with `reference_frame_count: 15` of 16. It is the closest released system to your D3/D4 setup found anywhere in this search, and it exists *because* tc = 1 makes per-frame timesteps meaningful at all.

**(f) Feasibility.** Ungated Apache-2.0 weights, from <https://huggingface.co/api/models/wzhgba/opendwm-models?blobs=true>: `ctsd_35_xs_df6v3_tirda_bm_nwao_60k.pth` **9.21 GB**, `ctsd_unimlvg_tirda_bm_nwa_60k.pth` **13.61 GB**, `ctsd_35_tirda_nwao_20k.pth` **16.02 GB**, `ctsd_35_tirda_bm_nwao_40k.pth` / `ctsd_35_df16_..._40k.pth` / `ctsd_35_tvae_f17_..._50k.pth` **17.22 GB** each, `ctsd_21_*` **7.69–8.03 GB**. (Repo also holds unrelated LiDAR checkpoints.) Code MIT → clean to vendor. Inference entry point `examples/ctsd_generation_example.py`. **Parameter counts are not published and I did not download multi-GB checkpoints to count tensors — `_needs verification_`.**

### 3.2 WEAVER — the D3/D4 match

Code: <https://github.com/arnavkj1995/WEAVER> (MIT, "Copyright (c) 2026 Arnav Kumar Jain") · weights: <https://huggingface.co/arnavkj1995/WEAVER> (`gated: False`; **no HF license tag** — MIT asserted by the in-repo `LICENSE`)

**Objective = rectified flow.** <https://raw.githubusercontent.com/arnavkj1995/WEAVER/main/weaver/wm/model.py>: L233 `class FlowWM(nn.Module)`; L1970 linear interpolant `xt[k] = (1 - t_) * x0[k] + t_ * x1[k]`; L1921 velocity target `loss = F.mse_loss(pred, x1 - x0, reduction='none')` (L1938 `target = x1 - x0`).

**tc = 1.** <https://raw.githubusercontent.com/arnavkj1995/WEAVER/main/weaver/wm/encoders.py>: L163 docstring _"Wrapper around Stable Diffusion 3's VAE encoder (AutoencoderKL)"_; L169 `model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers"`; L181 `self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")`; L223 `x = rearrange(x, "b t c h w -> (b t) c h w")` → L242 `latents = rearrange(latents, "(b t) n d -> b t n d", b=B)` — T preserved exactly. (The file also defines a second encoder on `AutoencoderKLTemporalDecoder` at L71 — SVD's VAE, whose *encoder* is likewise 2D per-frame, so tc = 1 either way.)

**Why it matters for D3.** `WEAVER-ReFlow/config.yaml` (<https://huggingface.co/arnavkj1995/WEAVER/raw/main/WEAVER-ReFlow/config.yaml>): `loss_target: v-pred`, **`val_steps: 4`**, `horizon: 8`, `eval_horizon: 5`, `n_history: 2`, `n_memory_frames: 6`, `relabel_actions: true`, `n_embed: 1536`, `n_layers: 32`. A **released, reflow-distilled, few-step, action-conditioned flow world model** — published prior art on the D3 contribution surface. Three checkpoints (`WEAVER`, `WEAVER-FT`, `WEAVER-ReFlow`) at **8.26 GB** each. Actions: DROID. Scale: 8-frame chunks at ~192×320, 2 cameras — small, which is an advantage for iteration and a limitation for fidelity. Param count not published — `_needs verification_`.

### 3.3 Vchitect-2.0-2B — the only general-domain hit

<https://huggingface.co/Vchitect/Vchitect-2.0-2B> (gated) · <https://github.com/Vchitect/Vchitect-2.0> · arXiv:2501.08453

Objective: `models/pipeline.py` L30 `from diffusers.schedulers import FlowMatchEulerDiscreteScheduler`, L247-249 `self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(load_path, subfolder="scheduler")`. Paper §VI-A: _"we trained our 2B model, **initialized from SD3-medium**, through a five-stage process"_.

tc = 1: L29 `from diffusers.models.autoencoders import AutoencoderKL`; `prepare_latents` L638-645 carries the author's own shape comment `#1, 60, 16, 32, 32` above `shape = (batch_size, frames, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)`; decode L956-962 is a Python loop `for v_idx in range(latents.shape[1]): image = self.vae.decode(latents[:,v_idx], ...)`. Paper §V-B: _"utilize **frame-wise DP-VAE** for inference"_. Transformer `patchify_and_embed` (`VchitectXL.py` L294-304): `B, F, C, H, W = x.size()` → `rearrange(x, "b f c h w -> (b f) c h w")`.

Feasibility: `license: apache-2.0`, **`gated: "auto"`** (auto-approved on accepting terms; **someone must click through on the model page** — the token on this machine returns `Access to model Vchitect/Vchitect-2.0-2B is restricted and you are not in the authorized list`). Sizes from `?blobs=true`: transformer **4.963 GB**, VAE fp16 **0.168 GB** (consistent with SD3's ~84M-param 2D VAE, far too small for a 3D video VAE), T5-XXL 9.525 GB; repo total 27.6 GB, minimal fp16 working set ≈16.3 GB (T5 droppable for a world model). Native `768×432`, `frames=40` at 8 fps. **No I2V checkpoint and no I2V code** — only `sd3_sparse_i2v_pipeline.cpython-310.pyc` in `__pycache__` with no matching `.py`, in both the GitHub repo and the HF Space. No training code. Last commit ≈2024-09.

---

## 4. Recommended integration path

The seam already exists. `models/base/interfaces.py` defines `BaseGenerativeModel(model_type, prediction_type)`; `models/base/video_model.py` defines `BaseVideoModel` with abstract `encode` / `decode` / `denoise` / `forward` / `generate`. Existing implementations are small — `dynamicrafter_video.py` 325 lines, `wan_ti2v.py` 315, `skyreels_video.py` 401 — and registration is one branch in `models/base/factory.py`.

**Repo context.** There is already an `opensora` provider (`backbones/opensora/`) whose docstring says _"MMDiT ... based on Flux ... Flow matching (velocity prediction)"_ — Open-Sora **v2**. But `configs/opensora/opensora_output_adapter.yaml` is a **toy config** (`hidden_size: 256`, `depth: 2`, "Reduced from 3072/19 for testing") with no VAE wired, and Open-Sora v2's real checkpoint lives in HunyuanVideo-VAE latent space (4x). Completing it yields **another (flow, 4x) cell — not a new one.**

**Path A — OpenDWM CTSD (recommended).**
1. Vendor `src/dwm/` (MIT) under `src/external_deps/opendwm/` — at minimum `pipelines/ctsd.py`, `schedulers/temporal_independent.py`, plus the `functional` and `common` helpers. Flag the boundary in `10_Now/architecture.md` per Part 12.
2. New `models/base/ctsd_video.py` implementing `BaseVideoModel`. `model_type="flow"`, `prediction_type="velocity"` for `ctsd_35_*` — **genuine flow-matching velocity, matching the existing Wan path exactly**, so no vocabulary extension is needed. `encode`/`decode` are per-frame 2D `AutoencoderKL` calls, structurally identical to the DynamiCrafter path.
3. The base is a stock `diffusers.SD3Transformer2DModel`, so LoRA and hidden-state adapters should attach via standard `diffusers` module paths. *(analysed estimate — input: the verified `isinstance(self.model, diffusers.SD3Transformer2DModel)` check at L977; reasoning: an unmodified diffusers class exposes the usual `attn`/`ff` submodule names. Not confirmed against the checkpoint's state dict.)*
4. **Register `ctsd_21_*` and `ctsd_35_tvae_f17_*` as sibling providers.** Same pipeline class, different configs — marginal cost after step 2 is near zero, and it buys two extra 2x2 cells with every non-target variable held fixed.
5. Start with `ctsd_35_xs_df6v3_*` (9.21 GB) before the 17 GB variants.
6. **Open a decision note** on whether CTSD's driving-domain conditioning (layout / 3D boxes / ego-action, multi-camera) can carry the `a_t` interface, or whether you condition only through `Δ_φ` and bypass the native conditioning. This is the real unknown (§5.2).

**Path B — WEAVER (in parallel, for D3).** Read it before building anything. `WEAVER-ReFlow` is a published few-step reflow-distilled action-conditioned flow world model, i.e. prior art on the D3 contribution surface. **Whether it changes the D3 framing is a positioning question, not an engineering one — it belongs in `50_Decisions/open/` and in the advisor conversation, not in a ticket.** As a backbone it is MIT, ungated, 8.26 GB, small enough to iterate on; as a baseline it is directly comparable.

**Path C — CogVideoX-5b-I2V (fallback).** Ungated, native I2V, 5B, first-class `diffusers`. If OpenDWM's conditioning turns out not to fit, this still completes the (diffusion, compressed) cell. **Gotcha:** CogVideoX's `v_prediction` is the **DDPM v-parameterisation** (`v = α_t·ε − σ_t·x_0`), *not* flow-matching velocity (`v = x_1 − x_0`). `infer_prediction_type()` only knows `noise`/`velocity`, so this needs a new enum value plus a matching target in `losses/diffusion.py` — **a closed-vocabulary extension requiring a decision note first** (Part 3 hard rule 3, Part 11). Exactly the Part-12 diffusion/flow conflation trap.

**Path D — Vchitect-2.0-2B (last resort).** Accept the gate, vendor 7 files (Apache-2.0), implement `models/base/vchitect_video.py` (`model_type="flow"`, `prediction_type="velocity"`). The real work is I2V conditioning, which does not exist. Because tc = 1, latent frame 0 corresponds exactly to pixel frame 0 with no temporal mixing, so first-frame conditioning is implementable by RePaint-style latent replacement with no base retraining — or folded into `Δ_φ`. **This is a design judgement, not a verified fact**: inputs are tc = 1 and the per-frame decode loop (both verified §3.3); reasoning is that with no temporal mixing, clamping frame 0 is exact rather than approximate. **Not demonstrated on this checkpoint.** Prototype on 8 frames before committing.

**Cheap control that needs no new backbone (do this regardless, ~2 hours).** On the DC cell, temporally smear the action signal (hold `a_t` piecewise-constant over 4-frame blocks) before it reaches the adapter, mimicking what a causal 4x VAE does to per-frame actions. If DC's adapter degrades toward the Wan failure mode, the confound is partly resolved for the cost of a data-loader flag. Highest information-per-GPU-hour experiment available, entirely inside code you already own.

---

## 5. Explicit gaps — what I could NOT verify

1. **Parameter counts for CTSD and WEAVER.** Neither publishes a figure; I did not download multi-GB checkpoints to count tensors. File sizes only.
2. **Whether CTSD's conditioning interface fits this thesis.** CTSD is a *driving* world model over multi-camera rigs conditioned on layout / 3D boxes / ego-action, not a low-dimensional robot action vector. **This is the single biggest unknown in the Path A recommendation** and should be settled by reading the conditioning code before any download.
3. **RxBrain was not independently verified by me.** The "zero `nn.Conv3d` / 13 `nn.Conv2d`" and FLUX-AE-constants findings are relayed from the delegated sweep. I confirmed only the repo, gating (`False`), Apache-2.0 licence and 12.41 GB across 3 shards. Deprioritised anyway.
4. **Vchitect's gated configs.** `vae/config.json`, `scheduler/scheduler_config.json`, `model_index.json`, `transformer/config.json` all unreadable. §3.3 rests on public GitHub code plus the paper. The HF API does expose `"config": {"diffusers": {"_class_name": "StableDiffusion3Pipeline"}}`, corroborating the SD3 (2D VAE) layout. **Re-verify after accepting the gate.**
5. **Vchitect's stated training objective.** The paper's §III-A Preliminaries gives generic DDPM background and **never states flow matching or rectified flow**. Classification rests on the inference scheduler plus SD3-medium initialisation. Sampling a noise-prediction model with a flow ODE solver would not work, so this is near-conclusive — but the prose does not confirm it.
6. **Vchitect param count / dtype.** Card says 2B; 4.963 GB implies ≈2.5B in bf16 — unverified. The `VchitectXL.py.__init__` defaults (`num_layers=18`, `num_attention_heads=18`) are **code defaults that may not match the checkpoint config** — do not quote as architecture.
7. **Whether a private Vchitect I2V variant exists.** An I2V pipeline was clearly written and not released. Worth one email before writing off Path D.
8. **Step-Video-T2V, MAGI-1** — excluded on size (30B / 24B); objective and tc unverified.
9. **Cosmos-Predict2-2B-Video2World** — gated, configs unreadable. The rest of the Cosmos family shows no per-frame-latent flow variant.
10. **Qwen-RobotWorld (arXiv 2606.17030), KAM-WM (arXiv 2607.04652)** — surfaced in search, no config-level verification, no weights repo found.
11. **MiniWAM** — reported as frozen `sd-vae-ft-mse` + joint video/action flow matching, i.e. **close to this thesis's recipe**, with **no released weights**. Not verified by me. **Worth checking for scoop risk regardless of the backbone decision.**
12. **No measured numbers anywhere in this note.** Every figure is a file size or a config field. No FLOP, latency, VRAM or quality claim here is measured, and nothing has been run.

---

## 6. Method and coverage

**Automated screen (primary evidence for the *general* T2V space; blind to research repos).** Enumerated HF models tagged `text-to-video` / `image-to-video`, top 200 by downloads and by likes for each tag → 612 unique, 361 in `diffusers` format. Fetched every model's `vae/config.json` and `scheduler/scheduler_config.json` and classified:

- **flow** iff scheduler `_class_name` contains `FlowMatch`, or `prediction_type == "flow_prediction"`, or `use_flow_sigmas == true`;
- **tc** from, in order: `temporal_compression_ratio`, `temperal_downsample`/`temporal_downsample` (2^count), `spatio_temporal_scaling` (2^count), `temporal_expansions` (product), else 1 if the VAE class is `AutoencoderKL` / `AutoencoderKLTemporalDecoder` / `AutoencoderTiny` / `AutoencoderDC`.

`{flow AND tc == 1}` came back **empty**, and exactly one of the 612 (Vchitect-2.0-2B) is built on an SD3/FLUX per-frame-VAE pipeline. The class-name fallback misfired once — Ruyi-Mini-7B declares `"_class_name": "AutoencoderKL"` but its `down_block_types` are `SpatialTemporalDownBlock3D`; caught by hand.

**Why that screen was insufficient — see §1.** It cannot see world-model research repos (raw `.pth` + JSON configs, no HF pipeline tag, no `vae/` subfolder). All three additional hits live there. **Do not reuse this screen alone for a similar question.**

**Hand-verified additions.** Open-Sora v1.0/v1.1 (<https://raw.githubusercontent.com/hpcaitech/Open-Sora/v1.0.0/configs/opensora/inference/16x256x256.py>: `vae = dict(type="VideoAutoencoderKL", from_pretrained="stabilityai/sd-vae-ft-ema")`, `scheduler = dict(type="iddpm")` → per-frame + diffusion); Open-Sora v2.0 (`model = dict(type="hunyuan_vae", ...)` → 4x, flow, has I2V train configs); Kandinsky 5.0 Lite I2V (`vae: name: "hunyuan"` + `class HunyuanVideoCausalConv3d` → 4x flow, MIT, `visual_cond: true`); NVIDIA Cosmos (`Cosmos-1.0-Diffusion-7B` = `AutoencoderKLCosmos` + `EDMEulerScheduler`, 8x diffusion; `Cosmos3-Super-Image2Video` = `AutoencoderKLWan` + UniPC, 4x flow).

---

## Follow-ups to consider (not actioned — inbox note)

- [ ] **Read OpenDWM's conditioning code** and settle gap 2 before anything else. It gates the whole Path A recommendation.
- [ ] Open `20_Tickets/experiments/exp-backbone-opendwm-ctsd-cell.md` for Path A.
- [ ] Open `20_Tickets/experiments/exp-eval-ctsd35-vs-tvae-f17-tokenizer-ablation.md` — the pure one-variable tokenizer swap. This is the experiment the whole search was for.
- [ ] Open `20_Tickets/experiments/exp-data-action-temporal-smearing-control.md` for the cheap DC control.
- [ ] Create `30_Knowledge/related-work/weaver.md` — **`WEAVER-ReFlow` is published prior art on the D3 surface** and needs an explicit positioning note. Consider `50_Decisions/open/d3-positioning-vs-weaver-reflow.md`.
- [ ] Check **MiniWAM** for scoop risk (frozen 2D VAE + joint video/action flow matching, no weights released).
- [ ] Open a decision on extending `prediction_type` to cover DDPM v-parameterisation — only needed if Path C is taken.
- [ ] Consider whether "the field has welded flow matching to temporal compression, *except* in world-model repos that inherit an image VAE" earns a paragraph in the discussion. It is sourced, non-obvious, and explains why the confound arose.
