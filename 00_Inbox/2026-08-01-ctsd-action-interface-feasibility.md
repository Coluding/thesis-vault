# OpenDWM CTSD — action-interface feasibility check (go/no-go before download)

_Date: 2026-08-01 · Author: Claude Code (research agent) · Status: decision input, not a vault fact-note_

Primary sources: OpenDWM cloned at commit `b0ecc3d4020612376ea5a87500f98bc76893428f`
(<https://github.com/SenseTime-FVG/OpenDWM>), code license MIT (`LICENSE` L1-3,
"Copyright (c) 2024 SenseTime Research"); weights
<https://huggingface.co/wzhgba/opendwm-models> (Apache-2.0, 146 GB total).
All `ctsd.py` / `crossview_temporal_dit.py` line numbers below are from that commit.

---

## 1. Verdict — **NO-GO**

> **Deciding reason:** CTSD's checkpoints only close the confound *inside the driving
> domain* — their forward pass structurally requires a calibrated 6-camera rig,
> rasterised HD-map/3D-box layout images and SD3.5's three text encoders — so using
> them means acquiring nuScenes/Waymo/Argoverse and rebuilding OpenDWM's multi-view
> layout+calibration pipeline, a domain pivot away from the robot-manipulation
> evidence base that costs far more than the confound is worth.

Notably, the gating risk as originally posed turns out to be **the wrong question**.
Our repo deliberately keeps the base a pure `f_base(x_t, t)` map and puts *all* action
conditioning in the adapter — `src/generative_flow_adapters/models/base/wan.py:40-43`:

> "The base stays a pure `(x_t, t)` map: text context is optional (a null zero-context
> is synthesised when absent), and **all action conditioning is expected to live in the
> adapter**, not here. That keeps the frozen base identical to the pretrained checkpoint."

So we would never use CTSD's own action input. What kills it is not `a_t` mismatch but
that **CTSD is not a pure `(x_t, t)` map** and is not a useful prior for robot video.

---

## 2. Verified checkpoint families + corrections to the reported anchors

### 2.1 All five reported source anchors are CONFIRMED

| Reported anchor | Status | Evidence |
|---|---|---|
| L952-954 VAE defaults to `diffusers.AutoencoderKL` | ✅ confirmed (exactly L953-954) | `vae_type = dwm.common.get_class(self.common_config.get("vae", "diffusers.AutoencoderKL"))` |
| L963-964 `is_temporal_vae` true only for CogVideoX | ✅ confirmed | `self.is_temporal_vae = isinstance(self.vae, ...autoencoder_kl_cogvideox.AutoencoderKLCogVideoX)` |
| L977-983 SD3 branch → `FlowMatchEulerDiscreteScheduler` | ✅ confirmed | L977 `elif isinstance(self.model, diffusers.SD3Transformer2DModel):`, L979 train, L983 test |
| L1220-1226 / L1628-1633 temporal dim preserved through encode/decode | ✅ confirmed | `is_temporal_vae` branches at L1206, 1220, 1332, 1611, 1628, 1641 |
| `ctsd_35_df16` uses `dwm.schedulers.temporal_independent.FlowMatchEulerDiscreteScheduler` | ✅ confirmed | `configs/ctsd/multi_datasets/ctsd_35_df16_tirda_bm_nwao.json:175` |

The pipeline class is `dwm.pipelines.ctsd.CrossviewTemporalSD` — "Crossview" is in the
class name, which is the first hint at the structural finding in §3.

### 2.2 The three families exist — with four corrections to the premise

Verified 1:1 checkpoint↔config mapping from `README.md` L76-80 and the HF file listing:

| Checkpoint (.pth) | Size | Config | Base | Objective | VAE (temporal compression) |
|---|---|---|---|---|---|
| `ctsd_21_tirda_nwao_30k.pth` | 7.69 GB | `ctsd_21_tirda_nwao.json` | SD 2.1, `UNetCrossviewTemporalConditionModel` | **diffusion** (DDPM train / DPMSolverMultistep infer) | `AutoencoderKL` per-frame (**1**) |
| `ctsd_21_tirda_bm_nwa_30k.pth` | 8.03 GB | `ctsd_21_tirda_bm_nwa.json` | SD 2.1, UNet | diffusion | `AutoencoderKL` (1) |
| `ctsd_35_tirda_nwao_20k.pth` | 16 GB | `ctsd_35_tirda_nwao.json` | SD 3.5-medium, `DiTCrossviewTemporalConditionModel` | **flow matching** | `AutoencoderKL` per-frame (**1**) |
| `ctsd_35_tirda_bm_nwao_40k.pth` | 17.2 GB | `ctsd_35_tirda_bm_nwao.json` | SD 3.5-medium, DiT | flow matching | `AutoencoderKL` (1) |
| `ctsd_35_tvae_f17_tirda_bm_nwao_50k.pth` | 17.2 GB | `ctsd_35_tvae_f17_tirda_bm_nwao.json` | SD 3.5-medium, DiT | flow matching | **`AutoencoderKLCogVideoX`** from `THUDM/CogVideoX-2b` (**4**) |
| `ctsd_35_df16_tirda_bm_nwao_40k.pth` | 17.2 GB | `ctsd_35_df16_tirda_bm_nwao.json` | SD 3.5-medium, DiT | flow, *temporal-independent* timesteps | `AutoencoderKL` (1) |

**Correction 1 — `ctsd_35_*` is not one family.** `ctsd_35_tirda_*` (frame_prediction_style
`"ctsd"`) and `ctsd_35_df16_*` (frame_prediction_style `"diffusion_forcing"`, DFoT, per-frame
independent flow timesteps) are *different training regimes*. Treating them as one family
would silently confound the comparison.

**Correction 2 — the families are NOT matched on training data or step count.** The
`ctsd_21` "bm" variant is trained on `nwa` (nuScenes/Waymo/Argoverse) while the `ctsd_35`
"bm" variant is `nwao` (+ OpenDV). The best-matched pairs are:

- *diffusion ↔ flow*: `ctsd_21_tirda_nwao_30k` vs `ctsd_35_tirda_nwao_20k` — same config
  family, same dataset mix, but **30k vs 20k steps**.
- *2D VAE ↔ 3D VAE*: `ctsd_35_tirda_bm_nwao_40k` vs `ctsd_35_tvae_f17_tirda_bm_nwao_50k` —
  identical base, objective, frame-prediction style, conditioning-dropout ratios and dataset
  mix; **only the VAE differs**. Residual confound: **40k vs 50k steps**.

The second pair is a genuinely clean 1-variable instrument. The first is not: SD2.1→SD3.5
changes base *and* architecture (UNet→DiT) *and* objective simultaneously — these are
inseparable, since there is no flow-matching SD2.1.

**Correction 3 — `cond_with_action` is a dead config key.** It appears `false` in 20 config
files but `grep -rn "cond_with_action" src/` returns **zero hits**. The real switch is
`added_time_ids`.

**Correction 4 — the reported "ego-action" conditioning is absent from all three target
checkpoints.** Only `ctsd_35_df16` sets `added_time_ids: "fps_camera_transforms_action"`;
`ctsd_21_*`, `ctsd_35_tirda_*` and `ctsd_35_tvae_f17_*` all use `"fps_camera_transforms"`
(no action at all). So the trio that closes the confound has **zero** action interface, and
the one checkpoint that has an action interface belongs to a different training regime.

---

## 3. The conditioning interface, in detail

### 3.1 Tensor layout — view count is a first-class dimension

`crossview_temporal_dit.py:415-416`:
```python
batch_size, sequence_length, view_count, _, height, width = hidden_states.shape
```
Everything is **6-D `[B, T, V, C, H, W]`**. Our repo's adapter path is 5-D `[B, C, T, H, W]`
(`ActionWanModel.patch_embedding` is a `Conv3d`). That rank mismatch propagates through the
entire adapter interface.

### 3.2 What the model conditions on

| Input | Shape | Source |
|---|---|---|
| `encoder_hidden_states` (text) | `[B, T, V, L, D]`, flattened to `(B·T·V, L, D)` | CLIP-L + CLIP-G + **T5-XXL** for SD3.5 (`self.text_encoders` list, `ctsd.py:948`) |
| `pooled_projections` | `[B, T, V, D]` | pooled text |
| `condition_image_tensor` | 3dbox + hdmap RGB layout images concatenated on channel dim (`ctsd.py:304-305`, `torch.cat(condition_image_list, -3)`) | rasterised HD-map / 3D boxes |
| `added_time_ids` | **`[B, T, V, 11]`** (or 13 with action) | fps(1) + camera intrinsics(4) + camera extrinsics(6) |
| `camera_intrinsics`, `camera_transforms` | `[B, T, V, 4, 4]` | calibrated rig |
| `crossview_attention_mask` | per-view mask | rig topology |

The `added_time_ids` width is confirmed by exact arithmetic against the configs:
`camera_intrinsic_embedding_indices` = `[0,4,2,5]` (4) + `camera_transform_embedding_indices` =
`[2,6,10,3,7,11]` (6) + fps (1) = **11 scalars × 256 channels = 2816** =
`projection_class_embeddings_input_dim` in all three target configs. For `df16`, 3328 =
**13 × 256** — exactly **2** extra scalars. This independently confirms the action vector is 2-D.

### 3.3 Single-camera mode — mechanically possible, structurally discouraged

`view_count` is a runtime tensor dimension, so `V=1` runs. But:

- **Every one of the 20+ CTSD configs sets `enable_crossview: true`**, and all use 6 cameras
  (`sensor_channels`: `["LIDAR_TOP","CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT"]`).
- **Decisive**: `view_cam_emb` — the *sole* carrier of `added_time_ids` (fps + camera + action)
  — is consumed at exactly two sites, and **both are gated on `self.enable_crossview`**:
  - `crossview_temporal_dit.py:536-537` → `if self.enable_crossview and not self.disable_view_emb_on_temporal_module: sequence_emb = sequence_emb + view_cam_emb`
  - `crossview_temporal_dit.py:558`/`568` → `if self.enable_crossview and i in self.crossview_block_layers: ... view_emb = view_emb + view_cam_emb`

  (Full reference list: L433, 437, 457, 537, 568 — nothing else.)

  **With cross-view disabled, the entire numeric conditioning path — including the ego action
  — becomes dead code.** The action conditioning structurally rides on the multi-camera branch.

### 3.4 The low-dimensional action vector — it exists, and it is 2-D and car-specific

`ctsd.py:98-156`, `get_action_ids`. Derived from `ego_transforms` `[B,T,V,4,4]` SE(3):

```python
relative_pose = torch.linalg.solve(current_pose[:, :-1], current_pose[:, 1:])
moving_distance = torch.norm(relative_pose[..., :3, 3], dim=-1, keepdim=True)
mps_to_kmph = 3.6
speed = mps_to_kmph * moving_distance * batch["fps"]...
rotation_angles = torch.atan2(relative_pose[..., 1, 0:1] - relative_pose[..., 0, 1:2],
                              relative_pose[..., 0, 0:1] + relative_pose[..., 1, 1:2])
wheel_base = 2.7
steering_ratio = 14
steering = torch.where(torch.abs(moving_distance) > 0.01,
                       rotation_angles / moving_distance * wheel_base * steering_ratio,
                       -1000.0 * torch.ones_like(rotation_angles))
action_ids = torch.cat([speed, steering], -1)
```

- **Dimensionality: 2** — `(speed, steering)`.
- **Normalisation: none learned.** Physical units: km/h, and a steering angle passed through a
  hard-coded **kinematic bicycle model** (`wheel_base = 2.7` m, `steering_ratio = 14`). These
  are car constants, not general robot kinematics.
- **Unconditional sentinel: `-1000.0`** (not a learned null embedding).
- **Where it enters** (correcting a likely assumption): it is **not** adaLN. `temb` is computed
  separately at L431 (`self.time_text_embed(timestep.flatten(), pooled_projections)`). The
  action goes: `added_time_ids` → `view_cam_proj` (`diffusers...Timesteps(num_channels=256)`,
  L163-164) → `view_embedding` (`TimestepEmbedding(in_channels=proj_dim, time_embed_dim=inner_dim)`,
  L165-167) → **added to the temporal and cross-view *positional* embeddings** (L537, L568).
  So it modulates the temporal/cross-view attention branches, not the global AdaLN stream.

### 3.5 Layout / 3D-box / HD-map can be zeroed — yes, cleanly

Condition dropout is first-class and trained:

- `training_config` in all four target configs: `3dbox_condition_ratio: 0.8`,
  `hdmap_condition_ratio: 0.8`, `text_prompt_condition_ratio: 0.8`.
- Masked pixels are set to `uncondition_image_color` (`0.1255`): `ctsd.py:266-269`, `285-288`.
- CFG at inference (`guidance_scale: 4`) prepends an all-`uncondition_image_color` branch:
  `ctsd.py:271-277`, `290-296`, combined at L1549-1552.
- The action gets its **own** CFG channel — `ctsd.py:339-343` replaces only the last 2 dims with
  `-1000` ("`# action is allowed to be guidance scaled`").

So layout conditioning is genuinely optional. This is the one part of the interface that is
friendly to us.

### 3.6 Start-frame / image conditioning (I2V) — **yes, hard requirement satisfied**

- `frame_prediction_style` ∈ `{None, "diffusion_forcing", "ctsd"}` (`ctsd.py:642-646, 670`).
  All three target checkpoints use `"ctsd"`; `df16` uses `"diffusion_forcing"`.
- `try_make_input_for_prediction` (`ctsd.py:619+`) builds `reference_frame_indicator` and keeps
  reference latents clean while noising future frames (L728-732).
- `get_reference_latent_count` (`ctsd.py:1120-1132`) from `training_config.reference_frame_count`;
  `inference_config.reference_frame_count: 3`.

Future-from-current-frame prediction is supported.

---

## 4. How `a_t` would map on

**Their action path is irrelevant to us**, because our composition rule puts the action in
`Δ_φ` and keeps `f_base` a pure `(x_t, t)` map (`models/base/wan.py:40-43`, quoted in §1). We
would attach our own adapter (`ActionWanModel`, `action_injection: cross_attention`,
`conditioning.input_dim: 7`) and feed CTSD nulls on its own conditioning ports.

That is *mechanically* plausible — and if we ran in the driving domain, `a_t = (speed, steering)`
would actually be a decent 2-D continuous action, well matched to our `MLPConditionEncoder`.

**But `f_base` cannot be made pure cheaply.** To call CTSD at all we must supply:

1. `encoder_hidden_states` + `pooled_projections` — requires loading CLIP-L, CLIP-G **and T5-XXL**
   for SD3.5 (three text encoders, `ctsd.py:948`). Our Wan path synthesises a null zero-context;
   CTSD has no equivalent null path for SD3.5 pooled projections.
2. `added_time_ids` — 11 scalars of *real camera calibration*. We would pass identity/dummy
   values that are far outside the training distribution.
3. `crossview_attention_mask` + `V=1` — never trained; all checkpoints are 6-view.
4. Layout images — droppable (§3.5). ✅ the only clean one.

And then the fundamental problem: **a frozen driving world model is not a useful prior for
tabletop robot manipulation.** The thesis premise is that `f_base` contributes pretrained
dynamics that the adapter steers. On RT-1/DROID/MetaWorld video, a nuScenes/Waymo-specialised
model contributes essentially nothing, so `Δ_φ` would have to learn the whole dynamics — which
destroys the D1/D2 story rather than supporting it.

The alternative — pivoting the empirical domain to driving — requires nuScenes/Waymo/Argoverse
acquisition plus OpenDWM's custom `dwm.fs.czip.CombinedZipFileSystem` layout, layout
rasterisation, and calibration pipeline. That is a much larger project than the confound merits,
and it would fragment the thesis away from the robot-manipulation evidence base.

---

## 5. Integration cost estimate

**_Analysed estimate_** (inputs: the wan/DynamiCrafter/SkyReels wiring measured in this repo;
CTSD's interface as read above). Not a measured number.

Reference points measured in `/home/lukas/projects/generative-flow-adapters/`:

| Backbone | First-party glue (excl. vendored internals) |
|---|---|
| SkyReels (most recent, reused the Wan adapter) | ~1,090 lines |
| Wan2.2 | ~2,319 lines impl + 1,050 lines tests |
| DynamiCrafter | ~1,298 lines + 397 tests |

Checklist a new backbone must touch: (1) `models/base/factory.py` `elif provider` branch;
(2) new `models/base/ctsd.py` implementing `BaseVideoModel`'s four abstract methods
(`encode`/`decode`/`denoise`/`generate`); (3) `data/ctsd_batch_preprocessor.py`;
(4) `scripts/train_ctsd_*.py`; (5) `configs/base/ctsd_*.yaml`; optionally
(6) `adapters/factory.py:_build_output`.

CTSD-specific surcharges beyond the SkyReels baseline:

- **6-D `[B,T,V,C,H,W]` vs our 5-D `[B,C,T,H,W]`** — a rank mismatch through the whole adapter
  path, not a reshape at the boundary.
- **Three text encoders including T5-XXL** just to produce a null context.
- **`dwm.common.get_class` config-driven instantiation** — a whole parallel config system to
  bridge to our dataclass configs.
- **`dwm.fs` CombinedZipFileSystem** data layer + layout rasterisation, if driving data is used.
- **Two VAE regimes** to support (`AutoencoderKL` and `AutoencoderKLCogVideoX`), which is the
  point of the exercise but doubles the preprocessor work — note our repo has **no** VAE
  abstraction on the older interface; it is per-backbone special-cased.

**Size: ~2-3 weeks of engineering** for a working frozen-base integration, *plus* driving-dataset
acquisition — and that is before a single adapter is trained. Well past "a week". On the
requested scale: **"not worth it."**

---

## 6. If GO — download list (NOT executed)

Not applicable (NO-GO). Recorded for completeness, and for the narrow side-study in §7.

| File | Size |
|---|---|
| `ctsd_21_tirda_nwao_30k.pth` | 7.69 GB |
| `ctsd_35_tirda_nwao_20k.pth` | 16 GB |
| `ctsd_35_tirda_bm_nwao_40k.pth` | 17.2 GB |
| `ctsd_35_tvae_f17_tirda_bm_nwao_50k.pth` | 17.2 GB |
| **CTSD subtotal** | **~58.1 GB** |
| + `stabilityai/stable-diffusion-2-1` (base, VAE, CLIP) | ~5 GB _needs verification_ |
| + `stabilityai/stable-diffusion-3.5-medium` (incl. T5-XXL) | ~15 GB _needs verification_ |
| + `THUDM/CogVideoX-2b` (VAE subfolder only) | ~0.4 GB _needs verification_ |
| **Total** | **~80-90 GB** |

Destination, matching `ckpts/Wan2.2-TI2V-5B`: `/scratch-shared/$USER/ckpts/OpenDWM-CTSD/`.

**Scratch headroom (verified, `ssh snellius 'df -h /scratch-shared'`):**
```
Filesystem      Size  Used Avail Use% Mounted on
wstor_scratch1  2.5P  612T  1.9P  25% /gpfs/scratch1
```
User is `lbierling`; `/scratch-shared/lbierling` exists. Filesystem-level headroom is ample;
**per-user usage/quota could not be measured** — `du -sh /scratch-shared/lbierling` did not
return within ~10 min across three attempts, and no `myquota`/`quota` tool responded. Check
this before any large download.

---

## 7. Recommended alternative

### 7.1 WEAVER as a backbone — assessed explicitly: **NO**

Weights *are* released and usable in principle: `arnavkj1995/WEAVER`
(<https://huggingface.co/arnavkj1995/WEAVER>), 3 × 8.26 GB (`WEAVER`, `WEAVER-FT`,
`WEAVER-ReFlow`), code MIT (`LICENSE` L1, "Copyright (c) 2026 Arnav Kumar Jain"), last push
2026-06-26. Paper: *"WEAVER, Better, Faster, Longer: An Effective World Model for Robotic
Manipulation"*, arXiv:2606.13672 (preprint, not peer-reviewed). Rectified flow confirmed in
code (`weaver/wm/model.py:1920-1922`, `loss = F.mse_loss(pred, x1 - x0)`; interpolant
`xt = (1-t)*x0 + t*x1` at L1958-1971). I2V satisfied (history frames pinned clean at `t=1`;
`collapse_prob: 0.1` explicitly trains single-frame context).

**But it fails on two counts that matter more:**

1. **It breaks the composition rule's premise.** The dynamics transformer is trained **from
   scratch** on DROID (`weaver/pretrain.py:198` loads state only for *resume*; no pretrained
   init anywhere). Only the SD3 **VAE** and a CLIP text encoder are frozen pretrained parts.
   And it is **already action-conditioned by architecture** — the action is a per-frame token
   (`model.py:765-772`, `inp_prj['actions'] = MLP(n_actions, 512, 1536)`), 8-D = 7 joint-position
   deltas + 1 gripper delta, z-scored. So there is no action-free `f_base(x_t, t)` to adapt;
   D1/D2 evaporate and you would be adapting an already-solved problem.
2. **It closes neither confound.** Flow-only (no diffusion sibling on the same base), and
   per-frame VAE only — SD3 `AutoencoderKL`, 16 ch, **temporal compression 1**
   (`encoders.py:222-229` flattens `b t c h w -> (b t) c h w`). Its predecessor Ctrl-World uses
   `AutoencoderKLTemporalDecoder`, whose **encoder is also 2-D (temporal compression 1)** — so
   the WEAVER↔Ctrl-World pair changes four variables at once and does *not* vary VAE temporal
   compression at all. No 3D-VAE WEAVER variant exists.

**WEAVER's real value is as a baseline and a D3 target**, not a backbone: it is a strong frozen
released flow world model on DROID, and `WEAVER-ReFlow` is a plain 2-rectification ReFlow
(`weaver/reflow.py:589-592`), **not** step-size-conditioned — so the shortcut-adapter
contribution (D3) is *not* scooped, and ReFlow is a fair few-step comparator.

Vchitect-2.0-2B (gated, T2V-only, no I2V, dead since 2024-09) and MiniWAM (no released weights)
were not investigated further — both fail on stated blockers.

### 7.2 What I actually recommend

**Do not adopt a third backbone to close this confound.** Instead, split it:

1. **For the VAE-temporal-compression axis** — if this confound must be closed empirically, the
   best public instrument I found is the CTSD pair
   `ctsd_35_tirda_bm_nwao_40k` vs `ctsd_35_tvae_f17_tirda_bm_nwao_50k` (34.4 GB): identical base,
   objective, frame-prediction style, conditioning ratios and dataset mix, differing **only** in
   `AutoencoderKL` vs `AutoencoderKLCogVideoX` (residual confound: 40k vs 50k steps). Run this as
   a **bounded, inference-only side-study in the driving domain** — OpenDWM ships a small demo
   layout package (`nuscenes_scene-0627_package.zip`, `README.md:112`) that plausibly avoids full
   nuScenes acquisition (_analysed estimate — the package is documented for the inference demo;
   whether it supports a quantitative comparison is unverified_). This is a figure/appendix
   result, **not** a backbone adoption, and needs no adapter training.
2. **For the main thesis line** — keep DynamiCrafter (diffusion, 2D VAE) and Wan2.2 (flow, 3D VAE)
   and treat the confound as **acknowledged and bounded in the write-up** rather than resolved by
   a third integration. Two variables move together; say so plainly in the limitations section
   and cite the §7.2.1 side-study if it gets run.

Recommend opening `50_Decisions/open/` on "how to close the diffusion-vs-flow / 2D-vs-3D-VAE
confound" rather than acting on this unilaterally.

---

## 8. What I could not verify

- **Per-user scratch usage/quota** — `du -sh /scratch-shared/lbierling` did not return within
  ~10 min across three attempts; no `myquota`/`quota` tool responded. Filesystem-level
  headroom (1.9P avail) is verified; per-user is not.
- **Exact contents of the `.pth` files** — whether they hold only the trained DiT/UNet or also
  VAE/text-encoder weights. `README.md` L76-80 links the `.pth` *and* separately links the
  SD2.1 / SD3.5 / CogVideoX-2b HF repos, which implies DiT/UNet-only, but I did not open a file.
  This changes the download math in §6.
- **Sizes of the SD2.1 / SD3.5-medium / CogVideoX-2b base repos** — estimated, not measured.
- **Whether `view_count=1` inference produces coherent video** — mechanically it runs, but no
  checkpoint was trained at `V=1`; this needs an actual run to settle.
- **SD3.5-medium gating status on HF** (SD3-medium *is* gated, verified for the WEAVER path;
  SD3.5-medium not separately checked).
- **No FID/FVD/quality numbers appear anywhere in this note** — none were verified from a run,
  and none should be quoted downstream.
