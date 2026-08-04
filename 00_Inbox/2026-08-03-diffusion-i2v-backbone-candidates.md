# Diffusion-objective I2V backbone candidates — screening for the objective-vs-flow decisive experiment

> **Created** 2026-08-03 · inbox note, not yet promoted.
> **Question asked:** find the best *diffusion*-trained I2V base model matching Wan2.2 TI2V-5B on
> **size (~5B)** and **3D temporally-compressed VAE (~4x)**, so that diffusion-vs-flow is the only
> variable that moves.
> **Nothing here is a result.** No run has been launched. All numbers are sourced config/code/
> checkpoint facts, or explicitly labelled as arithmetic/inference.

---

## 0. Read this before spending the budget — the premise is in tension with two vault findings

The brief states the hypothesis as *"flow matching learns a near-deterministic transport … so an
adapter's perturbation is pulled back toward the same solution; diffusion leaves more room."* Two
experiment notes dated **2026-08-02** already push against that framing. Flagging them because they
change what this backbone buys you, not because they kill the experiment.

1. **[[../30_Knowledge/experiments/20260802-avid-wan-cleanroom-perframe-causal]]** — in an AVID
   clean-room on the *same frozen Wan2.2 TI2V-5B*, per-frame action conditioning produced
   `action_effect_rel` (shuffle) **0.017474 ± 0.001531** vs pooled **0.010192 ± 0.001587**
   (ratio 1.71, Welch t=3.30), and *at matched step 5000 the Wan faithful arm exceeded the
   AVID/DynamiCrafter reference* (0.017474 vs 0.012541). Its own `notes:` field states:
   *"Wan is NOT the harder substrate"* and *"What differs is purity … not size."*
   → On that evidence the Wan flow base **is** steerable; what failed was our conditioning
   pathway (global pooled AdaLN), not the objective.

2. **[[../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]]** — carries an
   explicit in-note **`🛑 CORRECTION`** banner: the cross-base flow-vs-diffusion claim is
   *confounded* by consistency target (`endpoint_inversion` vs `v_average`) and by depth
   (DC step 400 vs Wan step 800). And it points the *opposite* way — the shortcut objective was
   learnable on the **flow** base (cos 0.302 vs control 0.034) and not on the diffusion base.

3. **[[../30_Knowledge/experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]**
   — same adapter is a strong *domain* adapter on Wan (FVD 1118.4 → 406.3, −63.7%) with
   `steering_cos 0.00`, and an *action* conditioner on DC (`steering_cos 0.117`,
   `temporal_alignment 1.000`). Its note field: *"On Wan the actions enter faithfully and drown in
   the residual stream"* — a **loss-weighting / residual-competition** account, not an
   objective-geometry account.

**Consequence for this decision.** A third backbone cell is still worth buying — but the outcome
that would actually be *decisive* has changed. If CogVideoX (diffusion, 5.6B, 4x temporal) steers,
that is consistent with both "objective matters" *and* "concat/per-frame conditioning matters"
(CogVideoX conditions by channel-concat exactly like DynamiCrafter — see §3.1, §6). **Consider
spending part of the budget on the per-frame-conditioning arm on Wan instead**, which finding (1)
says is already causal and costs no new backbone integration. This note answers the question as
asked; the prioritisation call is the user's.

---

## 1. Recommendation

### ★ Primary: **EasyAnimate V5-7b-InP (diffusion) vs V5.1-7b-InP (flow)** — a matched pair that beats *any* single model

The brief asked for a diffusion model matching Wan2.2 "so that diffusion-vs-flow is the only
variable that moves". There is a strictly better construction available: **the same team shipped the
same architecture twice, once trained as diffusion (V5) and once as flow matching (V5.1).** Using
that pair, the objective is the only variable *by construction* — you no longer have to argue that
CogVideoX's VAE is "close enough" to Wan's.

| | `alibaba-pai/EasyAnimateV5-7b-zh-InP` | `alibaba-pai/EasyAnimateV5.1-7b-zh-InP` |
|---|---|---|
| **objective** | **DIFFUSION** — `DDIMScheduler`, `prediction_type: "v_prediction"`, `beta_schedule: "scaled_linear"` 8.5e-4→1.2e-2, `rescale_betas_zero_snr: true` | **FLOW MATCHING** — `FlowMatchEulerDiscreteScheduler`, `shift: 3.0`, `use_dynamic_shifting: true` |
| params (BF16, HF safetensors) | **6,813,097,536** | **6,814,670,912** |
| VAE class | `AutoencoderKLMagvit` | `AutoencoderKLMagvit` |
| `latent_channels` | 16 | 16 |
| `block_out_channels` | `[128, 256, 512, 512]` | `[128, 256, 512, 512]` |
| VAE temporal / spatial compression | **4 / 8** | **4 / 8** |
| `scaling_factor` | 0.7125 | 0.7125 |
| DiT class | `EasyAnimateTransformer3DModel` | `EasyAnimateTransformer3DModel` |
| `in_channels` / `out_channels` | **33** / 16 | **33** / 16 |
| `num_layers` | 36 | 36 |
| `num_attention_heads` × `attention_head_dim` | 48 × 64 = **3072** | 48 × 64 = **3072** |
| `patch_size` | 2 | 2 |
| licence | Apache-2.0 (shipped `LICENSE`), ungated | Apache-2.0, ungated |

**The VAE and DiT configs are field-for-field identical.** The parameter delta is
**1,573,376 (0.023%)**, and it is *exactly the same delta* in the 12b class
(11,797,692,480 → 11,799,265,856) — consistent with the single extra config key V5.1 carries,
`add_norm_text_encoder: True`. Everything the adapter touches — latent geometry, token count,
width, depth, I2V channel layout — is held fixed.

**Why this is the right shape of experiment.** It also happens to match Wan2.2 well
(**VAE temporal compression 4 = 4**; **DiT width 3072 = 3072**), but that no longer has to carry
the argument, because the contrast is internal to the pair. If the diffusion arm steers and the flow
arm does not, the objective is implicated with essentially no architectural confound left. If
*both* steer, the Wan failure is a Wan/conditioning problem, not an objective problem — which is
what §0 already suspects.

**What stays confounded — state this plainly, it is the one real weakness.** V5 and V5.1 are two
*different training runs*, not a controlled ablation: training data, data volume, and schedule
length almost certainly differ, and the **text encoder changed** (`BertModel` →
`Qwen2VLForConditionalGeneration`, hence `add_norm_text_encoder`). So the honest claim is
"objective + training-recipe generation", not "objective alone". It is nonetheless a far tighter
control than any cross-vendor pairing available. `_needs verification_`: I did not find a source
quantifying the V5→V5.1 data/schedule differences.

**Integration is cheaper than it looks: one wrapper serves both arms.** `AutoencoderKLMagvit`
(`diffusers/models/autoencoders/autoencoder_kl_magvit.py`) and `EasyAnimateTransformer3DModel`
(`diffusers/models/transformers/transformer_easyanimate.py:316`) are both **upstream in diffusers**.
The two arms differ only in the scheduler object, so a single `easyanimate_video.py` loads both and
the objective is a config switch — which also means the diffusion and flow arms cannot silently
diverge in wrapper code, a failure mode the vault has already been bitten by (the `🛑 CORRECTION`
in [[../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]] was exactly a
mismatched-target-between-arms bug).

**Cost.** ~6.8B × 2 arms. *Analysed estimate:* at 256×256/49 frames both arms run 3,328 tokens
(same arithmetic as §4b), so ≈1.21x CogVideoX's per-step cost each, ≈2.9x the single-model
option overall. If the budget only affords one arm, use the fallback below.

---

### Fallback (single-model, plugs into the existing DC/Wan comparison): **CogVideoX-5b-I2V**

Choose this if only **one** new run fits in the budget, or if reusing the existing Wan/DC
evaluation harness matters more than the tightest control. It is the best *single* model on the
criteria as literally stated in the brief.

| axis | Wan2.2 TI2V-5B (flow cell) | **CogVideoX-5b-I2V** | match? |
|---|---|---|---|
| objective | rectified flow | **VP diffusion, v-prediction, zero-terminal-SNR** | **the intended variable** |
| denoiser params | **4.9998 B** (measured, §3.1) | **5.6251 B** (HF safetensors metadata) | +12.5% — close |
| VAE temporal compression | **4** (`vae_stride=(4,16,16)`) | **4** (`temporal_compression_ratio: 4`) | **exact** |
| temporal patch size | 1 (`patch_size=(1,2,2)`) | 1 (no `patch_size_t`) | **exact** → token-level temporal compression 4 both |
| transformer hidden dim | **3072** (`dim: 3072`) | **3072** (48 heads × 64) | **exact** |
| VAE spatial compression | 16 | 8 | ✗ residual |
| latent channels | 48 | 16 | ✗ residual |
| I2V conditioning | frame-replace / diffusion-forcing | channel-concat (`in_channels: 32`) | ✗ residual |
| native res / frames | 704×1280 / 121 | 480×720 / 49 | ✗ (cheaper — good for budget) |

**Which confounds it breaks.** Against Wan2.2 it holds **parameter count (within 12.5%), VAE
temporal compression (exactly 4), and transformer width (exactly 3072)** fixed while flipping the
objective. That is the diffusion-vs-flow contrast the brief asked for, and no other model found
does it.

**A second, unrequested confound it also breaks — this is the strongest argument for it.**
DynamiCrafter-512, sourced from the *local* config `configs/base/dynamicrafter512.yaml`, uses
`parameterization: "v"`, `rescale_betas_zero_snr: True`, `linear_start: 0.00085`,
`linear_end: 0.012`, `timesteps: 1000`. CogVideoX-5b-I2V's scheduler config is
`prediction_type: "v_prediction"`, `rescale_betas_zero_snr: true`, `beta_start: 0.00085`,
`beta_end: 0.012`, `beta_schedule: "scaled_linear"`, `num_train_timesteps: 1000`.
**These are the same noise schedule and the same parameterisation.** So CogVideoX is
simultaneously a *scale-and-tokenizer* sibling of the DynamiCrafter cell that already works, with
the diffusion schedule held exactly constant. That yields a three-point chain that identifies the
axis rather than a single A/B:

| cell | objective | params | token-level temporal compression | status |
|---|---|---|---|---|
| DynamiCrafter-512 | v-diffusion, betas 8.5e-4→1.2e-2, zeroSNR | 1.4B | **1** | steering **works** (0.117) |
| **CogVideoX-5b-I2V** | **v-diffusion, identical schedule** | **5.6B** | **4** | **to run** |
| Wan2.2 TI2V-5B | rectified flow | 5.0B | 4 | steering **fails** (0.00) |

- CogVideoX **steers** → size and 3D-causal tokenizer are exonerated; the objective (or the
  concat-vs-frame-replace conditioning path) is what moves. Objective hypothesis survives.
- CogVideoX **fails** → the objective hypothesis is dead, and size and/or the compressed tokenizer
  is the cause. That is an equally publishable negative and it is cheap to interpret.

**What stays confounded (state this in the thesis).** VAE spatial compression (8 vs 16), latent
channel count (16 vs 48), and the I2V conditioning mechanism (channel-concat vs frame-replace).
The last one is *not* cosmetic given finding (1) above — CogVideoX-vs-Wan moves objective **and**
conditioning pathway together. There is no released model that fixes this: nothing pairs Wan's
frame-replace conditioning with a diffusion objective.

**Replication arm (if budget allows a second diffusion point): `rhymes-ai/Allegro-TI2V`** —
diffusion with **epsilon**-prediction and plain-linear betas (a *different* diffusion flavour, so a
positive result there is not an artefact of DynamiCrafter's specific schedule), temporal
compression **4**, Apache-2.0, ungated, and stochastic-by-default sampling. Loses as the primary
arm on size (~2.8B) and `latent_channels: 4`. Details §3.3.

**Runner-up: `nvidia/Cosmos-Predict1-7B-Video2World`** — genuine EDM diffusion, purpose-built
world model, permissive licence, and it ships an *action-conditioned* Video2World training config.
Worse on the ranking axes (7B not 5B; **8x** temporal compression not 4x). Pick it only if the
robotics-native framing and the existing action-conditioning code are worth more than the
tokenizer match. Details §3.2.

**⚠ Do not use Cosmos-Predict2 or Cosmos-Predict2.5.** Both are **rectified flow** despite paper
prose saying "EDM denoising score matching". This is exactly the trap the brief warned about —
the receipts are in §3.3.

---

## 2. Screening table

Objective column reflects what was verified in **config/scheduler/training code**, never prose.
Rows marked ‡ were verified by a delegated agent and spot-checked by me; unmarked rows I verified
directly.

### 2a. The shortlist — DIFFUSION objective **and** a released I2V checkpoint

| model | I2V repo id | objective + prediction type | params (BF16) | VAE temporal comp. | latent ch | licence | verdict |
|---|---|---|---|---|---|---|---|
| **EasyAnimate V5-7b-InP** | `alibaba-pai/EasyAnimateV5-7b-zh-InP` | `DDIMScheduler`, **v_prediction**, scaled_linear, zeroSNR | **6,813,097,536** | **4** | 16 | Apache-2.0 file (HF meta `other`) | ★★ **PRIMARY** — pairs with V5.1 flow twin |
| EasyAnimate V5-12b-InP | `alibaba-pai/EasyAnimateV5-12b-zh-InP` | same | 11,797,692,480 | 4 | 16 | same | Same pairing, 12b class — use only if 7b proves too small |
| **CogVideoX-5b-I2V** | `THUDM/CogVideoX-5b-I2V` | `CogVideoXDDIMScheduler`, **v_prediction**, `ZeroSNRDDPMDiscretization` | **5,625,087,552** | **4** | 16 | CogVideoX License (research free) | ★ **FALLBACK** — closest single model to Wan2.2 |
| CogVideoX1.5-5B-I2V | `THUDM/CogVideoX1.5-5B-I2V` | same | 5,571,594,880 | 4 in VAE, **`patch_size_t: 2` → token-level 8** | 16 | CogVideoX License | Worse: token temporal comp. 8 ≠ Wan's 4; 1360×768×81 far costlier |
| **Allegro-TI2V** | `rhymes-ai/Allegro-TI2V` | `EulerAncestralDiscreteScheduler`, **epsilon**, linear betas, no zeroSNR | 2,787,950,608 (repo-wide) | **4** | **4** | **Apache-2.0** | ★ Replication arm — different diffusion flavour, **stochastic default sampler** |
| **Ruyi-Mini-7B** ‡ | `IamCreateAI/Ruyi-Mini-7B` | `DDPMScheduler`, **v_prediction** | **7,244,452,992** | **4** ‡ | 16 | **Apache-2.0** | ★ Strong alt: cleanest licence at 7B. **No text encoder at all** — conditioning is CLIP-on-image, which suits a world model but is a real interface change. ⚠ its temporal ratio is a **config trap**: 3 `SpatialTemporalDownBlock3D` entries but `add_downsample=not is_final_block` ⇒ **4**, not 8 |
| CogVideoX-Fun-V1.5-5b-InP ‡ | `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP` | `CogVideoXDDIMScheduler`, **v_prediction**, scaled_linear, zeroSNR | 5,571,094,144 | **4** | 16 | other | ★ Notable: CogVideoX objective/VAE **plus multi-resolution + native I2V** — the obvious hedge against CogVideoX's fixed-resolution constraint (§4b) |
| CogVideoX-Fun-V1.1-5b-InP ‡ | `alibaba-pai/CogVideoX-Fun-V1.1-5b-InP` | same | 5,570,491,968 | 4 | 16 | other | Same, earlier revision |
| Open-Sora-Plan v1.3.0 ‡ | `LanguageBind/Open-Sora-Plan-v1.3.0` @ `any93x640x640_i2v` | `DDPMScheduler`, **v_prediction** + ZeroSNR | 2,771,981,600 | 4 | **8** | MIT | Viable small arm; MIT is the cleanest licence |
| Open-Sora-Plan v1.2.0 ‡ | `LanguageBind/Open-Sora-Plan-v1.2.0` @ `93x480p_i2v` | `DDPMScheduler`, **epsilon** | 2,771,907,856 | 4 | 4 | MIT | Viable small arm |
| EasyAnimate V4-XL-InP ‡ | `alibaba-pai/EasyAnimateV4-XL-2-InP` | `DDPMScheduler`, **v_prediction** | 1,982,894,112 | 4 | 4 | ⚠ `other` / tencent-hunyuan-community | Licence ambiguous — avoid |
| EasyAnimate V3-XL-InP ‡ | `alibaba-pai/EasyAnimateV3-XL-2-InP-*` | `DPMSolverMultistepScheduler`, **epsilon** + learned σ | 1,802,605,856 | 4 | 4 | Apache-2.0 | Too small; note V3 is **already 3D-VAE**, contrary to the brief's prior |
| SVD / SVD-XT | `stabilityai/stable-video-diffusion-img2vid-xt` | `EulerDiscreteScheduler` (EDM-style) | 1,524,623,082 (F32) | **1** — `AutoencoderKLTemporalDecoder` | 4 | SAI non-commercial research | ❌ Size *and* tokenizer land in DynamiCrafter's corner — cell already occupied |
| Cosmos-Predict1-7B-Video2World | `nvidia/Cosmos-Predict1-7B-Video2World` | **EDM**, `EDMScaling` + `EDMSDE(p_mean=-1.2, p_std=1.2, σ∈[0.002,80])`, x0-pred | 7B (card, **not measured** — gated) | **8** | 16 | NVIDIA Open Model | ★ Runner-up — broadest noise distribution, but tokenizer 8 ≠ 4 |
| Cosmos-Predict1-14B-Video2World | `nvidia/Cosmos-Predict1-14B-Video2World` | same EDM | 14B | 8 | 16 | NVIDIA Open Model | Too large |

### 2b. Ruled out

| model | reason | evidence |
|---|---|---|
| **EasyAnimate V5.1** (7b & 12b) | **FLOW** — but keep it: it is the *flow twin* of the primary recommendation | `model_index.json` → `FlowMatchEulerDiscreteScheduler`; `train.sh --loss_type="flow"` ‡ |
| Cosmos-Predict2 (2B/14B) | **RECTIFIED FLOW** despite paper saying "EDM" | `RectifiedFlowScaling`, `RectifiedFlowAB2Scheduler`, `sigma_data=1.0` — §3.5 |
| Cosmos-Predict2.5 | Flow matching (NVIDIA: FM backbone "distinct from EDM used in Predict1") | vendor statement |
| Open-Sora v1.2 / v1.3 / v2.0 ‡ | Rectified flow | `scheduler = dict(type="rflow")`; v2.0 `train.py`: `v_t = (1 - sigma_min) * x_1 - x_0` |
| Vchitect-2.0 ‡ | Flow matching **and** no I2V | `FlowMatchEulerDiscreteScheduler` |
| Open-Sora v1.0 / Open-Sora-Plan v1.0, v1.1 / Latte-1 ‡ | Diffusion, but **no I2V checkpoint** (Latte & OS v1.0 also 2D VAE, temporal comp. 1) | see §3.6 |
| Open-Sora v1.1 ‡ | IDDPM/epsilon and does I2V by masking — but **VAE is 2D, temporal ratio 1** | `self.patch_size = (1, 8, 8)`, `sd-vae-ft-ema`, per-frame `rearrange` |
| HunyuanVideo-I2V ‡ | **FLOW**, 12,821,012,544, temporal 4, latent 16 | ⚠ **licence excludes the EU** — see §2c |
| SkyReels-V1-Hunyuan-I2V ‡ | **FLOW**, 12,821,209,152, temporal 4 | inherits Tencent terms ⇒ ⚠ EU exclusion |
| SkyReels-V2-I2V-14B / 1.3B ‡ | **FLOW**, 16,394,878,784 / 1,564,428,608, temporal 4, latent 16 | Skywork Community licence. Also: **V2 is *not* a Wan2.1 fine-tune** — paper §3.3: *"we adopt the model architecture from Wan2.1 and only train the DiT from scratch while retaining the pretrained weight of other components including VAE and text encoder"* |
| Step-Video-TI2V ‡ | **FLOW**, 29,382,205,504, temporal 8, latent 64 | far too large |
| LTX-Video 2B / 13B ‡ | **FLOW**, 1,923,385,472 / 13,042,569,344, temporal 8, latent 128 | LTXV OWL licence, $10M-revenue commercial gate |
| Mochi-1 ‡ | **FLOW**, 10,027,677,744 — and **no I2V checkpoint exists** (`demos/cli.py` starts from pure `torch.randn`) | ruled out twice over |
| PixelDance ‡ | **weights never released** (org has one repo, the website) | not obtainable |
| VideoCrafter2 ‡ | Diffusion epsilon, ~1.41B, but **T2V only** | no I2V |
| DynamiCrafter-1024 ‡ | Diffusion **v-pred**, UNet **1,438,854,980** (measured), temporal comp. **1**, latent 4, Apache-2.0 | ❌ per-frame VAE — same corner as DC-512, already occupied |
| ConsistI2V ‡ | Diffusion epsilon, 1,249,975,241, temporal **1**, MIT | ❌ 2D VAE, too small |
| SEINE ‡ | Diffusion epsilon, ~909M, temporal **1** | ❌ 2D VAE, too small |
| I2VGen-XL ‡ | Diffusion v-pred, ~1.42B, temporal **1** | ❌ 2D VAE; licence conflict (HF MIT vs repo non-commercial) |
| VideoCrafter1-I2V ‡ | Diffusion epsilon, params `_needs verification_`, temporal **1** | ❌ 2D VAE; Apache + §10 non-commercial rider |

### 2c. ⚠ Licensing landmines — you are at a European institution

Verbatim from the HunyuanVideo / SkyReels-V1 LICENSE ‡:

> **"THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA."**

- **HunyuanVideo-I2V and SkyReels-V1-Hunyuan-I2V are therefore not licensed for use here.**
  SkyReels-V1's HF card claims `apache-2.0`, which **understates** the inherited Tencent terms —
  do not rely on the card.
- ⚠ **Action item for the existing results table:** the thesis currently reports a *SkyReels* cell.
  `external_repos/SkyReels-V2/` is vendored, and V2 carries the *Skywork Community* licence rather
  than the Tencent one — so this is probably fine. **But confirm which SkyReels was actually run
  before the thesis publishes that row.** If any V1/Hunyuan-derived checkpoint was used, that is a
  licensing problem independent of the science. `_needs verification_` — I did not determine which
  SkyReels checkpoint produced the reported result.
- **LTX-Video ≥0.9.6** adds a $10M-revenue commercial gate with a double-fees liquidated-damages
  clause; the "OpenRAIL-M" claim covers only ≤0.9.5.
- Cleanest licences among models that actually fit: **Ruyi-Mini-7B (Apache-2.0)**,
  **EasyAnimate V5/V5.1 (Apache-2.0)**, **Allegro-TI2V (Apache-2.0)**,
  **Open-Sora-Plan v1.3 (MIT)**.

| model | objective (sourced) + prediction type | params | VAE temporal compression | I2V ckpt? | license | verdict |
|---|---|---|---|---|---|---|
| **CogVideoX-5b-I2V** | **VP diffusion**, `CogVideoXDDIMScheduler`, `prediction_type: v_prediction`, `ZeroSNRDDPMDiscretization` | **5.6251B** (HF safetensors) | **4** (`temporal_compression_ratio: 4`) | ✅ `THUDM/CogVideoX-5b-I2V`, ungated | CogVideoX License (research free) | ★ **RECOMMENDED** |
| CogVideoX1.5-5B-I2V | VP diffusion, same scheduler + `v_prediction` | 5.5716B (HF safetensors) | 4 in VAE, but **`patch_size_t: 2`** → token-level **8** | ✅ `THUDM/CogVideoX1.5-5B-I2V`, ungated | CogVideoX License | Viable, strictly worse: token temporal compression 8 ≠ Wan's 4, and 1360×768×81 costs far more |
| Cosmos-Predict1-7B-Video2World | **EDM diffusion**, `EDMScaling` + `EDMSDE(p_mean=-1.2, p_std=1.2, σ∈[0.002,80])`, x0-pred | 7B (`faditv2_7b`) | **8** (`temporal_compression_factor = 8`) | ✅ Video2World = image/video→video | NVIDIA Open Model License | ★ Runner-up — size & tokenizer both off |
| Cosmos-Predict1-14B-Video2World | EDM diffusion, same | 14B | 8 | ✅ | NVIDIA Open Model | Too large |
| Cosmos-Predict2-2B / 14B-Video2World | **RECTIFIED FLOW** — `RectifiedFlowScaling`, `RectifiedFlowAB2Scheduler`, `sigma_data=1.0` | 2B / 14B | 4 (Wan2.1 VAE) | ✅ | NVIDIA Open Model | ❌ **flow** — does not answer the question (see §3.3) |
| Cosmos-Predict2.5 | Flow matching (NVIDIA states FM backbone, "distinct from EDM used in Predict1") | 2B/14B | 4 | ✅ | NVIDIA Open Model | ❌ flow |
| **Allegro-TI2V** | **Diffusion**, `EulerAncestralDiscreteScheduler`, `prediction_type: "epsilon"`, `beta_schedule: "linear"` 1e-4→2e-2, no zeroSNR | 2.788B (HF safetensors, repo-wide) | **4** (`temporal_compression_ratio: 4`), `patch_size_t: 1` → token-level 4 | ✅ `rhymes-ai/Allegro-TI2V`, ungated | **Apache-2.0** | ★ **Viable 3rd option** — best licence, **stochastic default sampler**; loses on size (2.8B) and `latent_channels: 4` |
| SVD / SVD-XT | Diffusion, `EulerDiscreteScheduler` (EDM-style, v-pred) | **1.5246B** (HF safetensors, F32) | **1** — `AutoencoderKLTemporalDecoder` is a 2D encoder + temporal decoder | ✅ `stabilityai/stable-video-diffusion-img2vid-xt` | SAI non-commercial research | ❌ Both size and tokenizer land in DynamiCrafter's corner — the cell already occupied |
| Wan2.2 TI2V-5B *(incumbent)* | Rectified flow | **4.9998B** (measured from local ckpt) | 4 (`vae_stride=(4,16,16)`) | ✅ | Apache-2.0 | current flow cell |
| DynamiCrafter-512 *(incumbent)* | VP diffusion, `parameterization: "v"`, betas 8.5e-4→1.2e-2, zeroSNR | 1.4B | **1** (2D `AutoencoderKL`, `z_channels: 4`) | ✅ | Apache-2.0 | current diffusion cell |

*(Screening of the remaining candidate list — Open-Sora v1.0–v2.0, Open-Sora-Plan, EasyAnimate,
Allegro, Latte, Vchitect, Ruyi, HunyuanVideo-I2V, Step-Video-TI2V, LTX-Video, Mochi, SkyReels,
VideoCrafter, I2VGen-XL, SEINE, ConsistI2V, PixelDance, DynamiCrafter-1024 — was delegated and is
**not yet merged into this note**; see §7. None of them can displace the recommendation on the
stated criteria unless one turns out to be diffusion at ~5B with 4x temporal compression, which
would be a surprise given the 2024→2026 industry shift to flow matching.)*

---

## 3. Deep dive

### 3.0 EasyAnimate V5-7b-InP (diffusion) / V5.1-7b-InP (flow) — PRIMARY

**Objective, diffusion arm.**
<https://huggingface.co/alibaba-pai/EasyAnimateV5-7b-zh-InP/raw/main/scheduler/scheduler_config.json>
```json
"_class_name": "DDIMScheduler",
"beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
"num_train_timesteps": 1000,
"prediction_type": "v_prediction",
"rescale_betas_zero_snr": true,
"timestep_spacing": "trailing"
```
Note this is **the same schedule as DynamiCrafter-512 and CogVideoX-5b-I2V** (§1) — so the
diffusion arm sits in the same schedule family as the cell that already works.

**Objective, flow arm.**
<https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh-InP/raw/main/scheduler/scheduler_config.json>
```json
"_class_name": "FlowMatchEulerDiscreteScheduler",
"num_train_timesteps": 1000,
"shift": 3.0, "use_dynamic_shifting": true,
"base_shift": 0.5, "max_shift": 1.15
```
No `prediction_type` key at all — the flow parameterisation has no ε/v choice to make.
Corroborated by the training script ‡: `--loss_type="flow"` with
`noisy_latents = (1.0 - sigmas) * latents + sigmas * noise; target = noise - latents`.
`model_index.json` for V5 → `["diffusers", "DDPMScheduler"]`; for V5.1 →
`["diffusers", "FlowMatchEulerDiscreteScheduler"]`.
*(V5's two files disagree on which* diffusion *scheduler — `model_index.json` says `DDPMScheduler`,
`scheduler/scheduler_config.json` says `DDIMScheduler`. Both are VP-diffusion samplers over the
same betas, so the diffusion-vs-flow conclusion is unaffected; it only means the sampler must be
pinned explicitly rather than taken from `model_index.json`. V5's `model_index.json` also lists
stale generic classes `Transformer2DModel`/`AutoencoderKL` while the component configs give the
real `EasyAnimateTransformer3DModel`/`AutoencoderKLMagvit` — **load components explicitly, not via
`DiffusionPipeline.from_pretrained`**.)*

**VAE — identical, and the temporal ratio is derived from source, not guessed.**
Both `vae/config.json` files return byte-identical values:
`"_class_name": "AutoencoderKLMagvit"`, `"latent_channels": 16`,
`"block_out_channels": [128, 256, 512, 512]`, `"scaling_factor": 0.7125`,
`"down_block_types": ["SpatialDownBlock3D", "SpatialTemporalDownBlock3D" ×3]`.
There is **no `temporal_compression_ratio` field**, so the ratio must be computed. From
<https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/autoencoders/autoencoder_kl_magvit.py>
lines 732–733:
```python
self.spatial_compression_ratio  = 2 ** (len(block_out_channels) - 1)
self.temporal_compression_ratio = 2 ** (len(block_out_channels) - 2)
```
With `len(block_out_channels) == 4` ⇒ **spatial 8, temporal 4**. (The naive read — "three
`SpatialTemporalDownBlock3D` ⇒ 8x temporal" — is wrong, because the encoder applies
`add_downsample=not is_final_block` at line 485. This is the same trap the agent flagged for Ruyi.)
**Temporal compression 4 matches Wan2.2's `vae_stride=(4,16,16)` exactly.**

**DiT — identical.** Both `transformer/config.json`:
`"_class_name": "EasyAnimateTransformer3DModel"`, `"in_channels": 33`, `"out_channels": 16`,
`"num_layers": 36`, `"num_attention_heads": 48`, `"attention_head_dim": 64` (⇒ width **3072**,
same as Wan2.2's `dim: 3072`), `"patch_size": 2`. The **only** differing key is V5.1's
`"add_norm_text_encoder": True`.

**I2V mechanism.** `in_channels: 33 = 16 noisy latent + 16 masked-video latent + 1 mask channel` ‡.
This is **masked-frame conditioning with an explicit mask channel** — architecturally closer to
Wan2.2's frame-replace/diffusion-forcing scheme than CogVideoX's plain channel-concat, which is a
further point in its favour. `_needs verification_`: the 16+16+1 decomposition is the delegated
agent's reading of the EasyAnimate pipeline, not a line I read myself.

**Params (HF safetensors metadata).**

| | 7b class | 12b class |
|---|---|---|
| V5 (diffusion) | 6,813,097,536 | 11,797,692,480 |
| V5.1 (flow) | 6,814,670,912 | 11,799,265,856 |
| **Δ** | **1,573,376** | **1,573,376** |

The delta is *bit-identical across both size classes* — strong evidence it is a fixed
architectural addition (`add_norm_text_encoder`) rather than any difference in the backbone.

**Licence.** `https://huggingface.co/alibaba-pai/EasyAnimateV5-7b-zh-InP/raw/main/LICENSE` returns
the **Apache License 2.0** text, although HF `cardData.license` says `other` — a metadata bug, and
the shipped file governs. V5.1 is `apache-2.0` in metadata. Both `gated: false`.
(⚠ EasyAnimate **V4** is the genuinely ambiguous one — HF frontmatter points at
`tencent-hunyuan-community`. Avoid V4.)

### 3.1 CogVideoX-5b-I2V — FALLBACK

**Objective — verified three independent ways, none of them a blog or an abstract.**

1. Inference scheduler —
   <https://huggingface.co/THUDM/CogVideoX-5b-I2V/raw/main/scheduler/scheduler_config.json>
   ```json
   "_class_name": "CogVideoXDDIMScheduler",
   "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
   "num_train_timesteps": 1000,
   "prediction_type": "v_prediction",
   "rescale_betas_zero_snr": true,
   "timestep_spacing": "trailing"
   ```
2. **Official training config** (strongest source) —
   <https://raw.githubusercontent.com/THUDM/CogVideo/main/sat/configs/cogvideox_5b.yaml>
   ```yaml
   denoiser_config:
     target: sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser
     num_idx: 1000
     weighting_config:      { target: ...EpsWeighting }
     scaling_config:        { target: ...VideoScaling }
     discretization_config: { target: ...ZeroSNRDDPMDiscretization }
   loss_fn_config:
     target: sgm.modules.diffusionmodules.loss.VideoDiffusionLoss
     sigma_sampler_config:
       target: ...DiscreteSampling
       uniform_sampling: True
       num_idx: 1000
       discretization_config: { target: ...ZeroSNRDDPMDiscretization }
   ```
   `DiscreteDenoiser` + `num_idx: 1000` + `ZeroSNRDDPMDiscretization` + `VideoDiffusionLoss`
   = discrete-time **VP/DDPM** diffusion. There is **no** rectified-flow or flow-matching class
   anywhere in this config.
3. Pipeline class — <https://huggingface.co/THUDM/CogVideoX-5b-I2V/raw/main/model_index.json>
   `"scheduler": ["diffusers", "CogVideoXDDIMScheduler"]`.

**VAE temporal compression = 4.**
<https://huggingface.co/THUDM/CogVideoX-5b-I2V/raw/main/vae/config.json>
```json
"_class_name": "AutoencoderKLCogVideoX",
"latent_channels": 16,
"temporal_compression_ratio": 4,
"down_block_types": ["CogVideoXDownBlock3D", ×4],
"sample_height": 480, "sample_width": 720, "scaling_factor": 0.7
```
Spatial compression 8 is *derived* (my arithmetic): 480/8 = 60 and 720/8 = 90, matching the
transformer's `sample_height: 60`, `sample_width: 90`. Corroborated by the CogVideoX literature
describing an "8×8×4 compression ratio along the spatial and temporal axes"
(<https://arxiv.org/html/2411.17459v2>, WF-VAE related-work). Compare Wan2.2's `vae_stride=(4,16,16)`.

**I2V support.**
- Transformer config <https://huggingface.co/THUDM/CogVideoX-5b-I2V/raw/main/transformer/config.json>:
  `"in_channels": 32, "out_channels": 16, "num_layers": 42, "num_attention_heads": 48,
  "attention_head_dim": 64, "sample_frames": 49, "patch_size": 2,
  "use_rotary_positional_embeddings": true` — `in_channels` is **2× `out_channels`**, i.e. the
  image latent is concatenated to the noisy latent.
- Confirmed in the diffusers pipeline
  (<https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py>):
  `latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)`.
- Confirmed in the SAT training config
  (<https://raw.githubusercontent.com/THUDM/CogVideo/main/sat/configs/cogvideox_5b_i2v.yaml>):
  `noised_image_input: true`, `noised_image_dropout: 0.05`, `in_channels: 32`.
  → **Same conditioning mechanism as DynamiCrafter** (`in_channels: 8 = 4 latent + 4 image`).

**Parameters = 5,625,087,552 (BF16).** From the HF model-info API
(`https://huggingface.co/api/models/THUDM/CogVideoX-5b-I2V` → `safetensors.parameters.BF16`).
Cross-check: the transformer safetensors index reports `total_size: 11,250,175,104` bytes;
11,250,175,104 / 2 = 5,625,087,552 (my arithmetic, consistent with BF16 storage).

**License.** `THUDM/CogVideoX-5b-I2V/blob/main/LICENSE` — "The CogVideoX License":
*"This license allows you to freely use all open-source models in this repository for academic
research."* Commercial use requires registration and is capped at 1M visits/month. `gated: false`
on the HF API — no access request needed. Fine for a thesis.

### 3.2 Cosmos-Predict1-7B-Video2World — runner-up

**Objective = EDM diffusion**, verified in code:
- <https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict1/main/cosmos_predict1/diffusion/modules/denoiser_scaling.py>
  contains **only** `class EDMScaling` with the Karras preconditioning
  (`c_skip = σ_data²/(σ²+σ_data²)`, `c_out = σ·σ_data/√(σ²+σ_data²)`, `c_noise = 0.25·log σ`).
  There is no `RectifiedFlowScaling` in this file — contrast Predict2, §3.3.
- <https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict1/main/cosmos_predict1/diffusion/training/modules/edm_sde.py>
  `class EDMSDE: p_mean=-1.2, p_std=1.2, sigma_max=80.0, sigma_min=0.002` — the textbook
  Karras et al. EDM log-normal σ sampler.
- Loss is x0-prediction: `L(D_θ,σ) = E‖D_θ(x₀+n;σ,c) − x₀‖²` (Cosmos Policy paper §3,
  <https://arxiv.org/html/2601.16163v1>).

*Analysed note (my inference, labelled as such):* EDM's variance-exploding path with σ ∈ [0.002, 80]
is a **broader** distribution of noised states than DynamiCrafter's or CogVideoX's VP schedule.
If the brief's "diffusion leaves more room for a conditioning signal" mechanism is real, EDM should
show it most strongly. That makes Cosmos a good *second* diffusion point — but a poor *first* one,
because it moves the tokenizer too.

**Tokenizer temporal compression = 8** (this is why it loses to CogVideoX):
<https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict1/main/cosmos_predict1/diffusion/config/base/tokenizer.py>
```python
@tokenizer_register("cosmos_diffusion_tokenizer_comp8x8x8")
temporal_compression_factor = 8
spatial_compression_factor  = 8
latent_ch = 16
```
Confirmed by the model's own latent shape —
<https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict1/main/cosmos_predict1/diffusion/config/inference/cosmos-1-diffusion-video2world.py>:
`latent_shape=[16, 16, 88, 160]` with tokenizer override
`cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624` → C=16, T=16 latent frames from 121
pixel frames, 88×160 from 704×1280.

**I2V.** `{"override /conditioner": "video_cond"}`, net `VideoExtendGeneralDIT`. Model card:
*"When image or video is provided as input, their latent frames are concatenated with the generated
frames along the temporal dimension. Augment noise is added to conditional latent frames."*
Input image 1280×704; output 121 frames @ 24fps.

**Params.** `{"override /net": "faditv2_7b"}` → 7B (model card states 7 billion). HF safetensors
metadata unavailable (repo is gated), so the count is from the card/config name, not measured.

**License.** NVIDIA Open Model License — commercially usable, derivative models permitted.
HF `gated: "auto"` → auto-approved click-through, satisfies "easily gated".

**Bonus for D2.** The repo ships `cosmos_predict1/diffusion/training/config/video2world_action/`
(experiment.py + registry.py) and `cosmos_predict1/diffusion/inference/video2world_action.py` —
an existing **action-conditioned** Video2World path. Directly relevant prior art regardless of
which backbone is chosen.

### 3.3 Allegro-TI2V — viable third option, best licence, stochastic by default

The only *other* model I personally verified as **diffusion + 4x temporal + released I2V**.

- **Objective — diffusion, epsilon-prediction.**
  <https://huggingface.co/rhymes-ai/Allegro-TI2V/raw/main/scheduler/scheduler_config.json>
  ```json
  "_class_name": "EulerAncestralDiscreteScheduler",
  "prediction_type": "epsilon",
  "beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02,
  "num_train_timesteps": 1000, "rescale_betas_zero_snr": false
  ```
  Note this is a **different diffusion flavour** from both incumbents: plain-linear betas and
  **epsilon** (not v), no zero-SNR. That is a virtue if you want to show the result is not an
  artefact of the specific DynamiCrafter schedule — and a vice if you want a single controlled axis.
- **VAE temporal compression = 4.**
  <https://huggingface.co/rhymes-ai/Allegro-TI2V/raw/main/vae/config.json> —
  `"_class_name": "AutoencoderKLAllegro"`, `"temporal_compression_ratio": 4`,
  `"latent_channels": 4`, `"temporal_downsample_blocks": [true, true, false, false]`.
  `latent_channels: 4` is **far** from Wan's 48 (and CogVideoX's 16) — the biggest strike against it.
- **I2V.** `model_index.json` → `"transformer": ["diffusers", "AllegroTransformerTI2V3DModel"]`.
  Transformer config: `"in_channels": 4, "out_channels": 4, "patch_size_t": 1,
  "num_layers": 32, "num_attention_heads": 24, "attention_head_dim": 96` (→ width 2304),
  `"sample_size": [90, 160], "sample_size_t": 22` (≈720×1280, 88 frames).
  Because `in_channels == out_channels` there is **no channel-concat**, so conditioning is
  presumably masked-frame/inpainting-style — *closer to Wan's frame-replace than CogVideoX is*.
  ⚠ `_needs verification_` — I did not read the conditioning code, only inferred from the channel counts.
- **Params 2,787,950,608 (BF16)** from the HF model-info API. ⚠ This is the **repo-wide**
  safetensors total (transformer + VAE), not a transformer-only count — so the DiT is somewhat
  under 2.8B. Either way it is ~2.8B, not ~5B, so **size stays partly confounded**.
- **Licence Apache-2.0, `gated: false`** — the cleanest licence of any candidate here, better
  than CogVideoX's bespoke licence.
- **Sampler is stochastic by default**: `EulerAncestralDiscreteScheduler` is an *ancestral*
  sampler (injects noise each step). Directly useful for the §5 stochasticity angle.

*Judgement:* if the objective hypothesis survives on CogVideoX, Allegro is the natural **replication**
arm — different diffusion parameterisation, different licence, independent codebase. As the
*primary* arm it is worse than CogVideoX because 2.8B leaves the size confound half-standing and
`latent_channels: 4` moves the tokenizer further from Wan, not closer.

### 3.4 Cosmos-Predict2 — DISQUALIFIED, and the reason is the exact trap in the brief

The Cosmos Policy paper (<https://arxiv.org/html/2601.16163v1>, §3 "Cosmos video model") says
Cosmos-Predict2 is *"trained using the EDM denoising score matching formulation"* with loss
`L(D_θ,σ) = E‖D_θ(x₀+n;σ,c) − x₀‖²`, over the *"Wan2.1 spatiotemporal VAE tokenizer"* with
`T/4, H/8, W/8, 16` channels. Read at face value that is a ~perfect candidate: **diffusion +
Wan VAE at 4x temporal**. It is wrong.

The shipped code contradicts the prose. In
<https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict2/main/cosmos_predict2/pipelines/video2world.py>:
```python
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling      # line 39
from cosmos_predict2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler  # line 41
```
and every Video2World model config (2B *and* 14B) in
<https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict2/main/cosmos_predict2/configs/base/config_video2world.py>
sets
```python
rectified_flow_t_scaling_factor=1.0,
rectified_flow_loss_weight_uniform=True,
sigma_data=1.0,
state_ch=16, state_t=24,
```
`sigma_data=1.0` is the tell: `denoiser_scaling.py` defines
`class RectifiedFlowScaling` with `assert abs(sigma_data - 1.0) < 1e-6`, and its preconditioning is
the **linear interpolant** `t = σ/(σ+1); c_skip = 1−t; c_out = −t; c_in = 1−t` — rectified flow.
The co-resident `class EDMScaling` (σ_data default 0.5) is *not* the one Video2World selects.

**Resolution of the contradiction (my inference, labelled):** the paper's "EDM denoising score
matching" describes only the **x0-MSE loss form**; the **noise-to-data path is rectified flow**.
The `edm_loss` variable name in `video2world_model.py` (lines 451–466) is legacy naming and is
*not* evidence of an EDM path. NVIDIA's own materials also state Predict2.5 uses a Flow Matching
backbone *"distinct from Elucidated Diffusion Models used in Cosmos-Predict1"* — consistent with
Predict1 = EDM, Predict2/2.5 = flow.

**Had this been taken from the paper abstract, ~20,000 SBU would have been spent on a second flow
model.**

---

## 4. Integration estimate

**The contract.** `src/generative_flow_adapters/models/base/video_model.py` defines
`BaseVideoModel` with exactly four abstract methods — `encode`, `decode`, `denoise`, `generate` —
plus a `ComposeFn = Callable[[Tensor, object, Tensor], Tensor]` seam that the backbone must invoke
inside its **own** native sampling loop. Benchmarks in-repo:

| file | lines |
|---|---|
| `models/base/wan_ti2v.py` | 333 |
| `models/base/wan.py` | 223 |
| `models/base/wan2_2.py` | 126 |
| `models/base/dynamicrafter_video.py` | 325 |
| `models/base/skyreels_video.py` | 401 |
| `models/base/factory.py` | 241 |

**For the primary (EasyAnimate V5 + V5.1): ~350–450 lines total for BOTH arms.**
*Analysed estimate.* Both checkpoints instantiate the **same** two diffusers classes —
`AutoencoderKLMagvit` and `EasyAnimateTransformer3DModel` (upstream at
`diffusers/models/transformers/transformer_easyanimate.py:316`) — so one
`easyanimate_video.py` covers both and the objective is selected by which scheduler is
constructed. Concretely: `model_type="diffusion", prediction_type="velocity"` for V5 versus
`model_type="flow", prediction_type="velocity"` for V5.1, both already legal values in
`BaseVideoModel.__init__`. **This is the single most valuable property of the primary
recommendation**: the two arms physically cannot diverge in wrapper code, which is the exact
failure mode that forced the `🛑 CORRECTION` on
[[../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]].

**Note on a concern raised during screening.** One agent warned that
*"`prediction_type: "noise"` will not cover v-pred/EDM — your interface needs a v-pred branch"*.
For **v-pred this is already handled**: `losses/diffusion.py:104` accepts `{"velocity", "v"}` on
the diffusion side and returns `sqrt_alphas * noise - sqrt_one_minus * x_start` (line 113), the
correct diffusion v-target. It is **only EDM** (Cosmos-Predict1) that would need new loss code —
continuous σ, x0-prediction, Karras preconditioning. Another reason EDM ranks below the
VP/v-prediction candidates here.

**Estimate for a `cogvideox_video.py` (fallback path): ~300–400 lines** — i.e. the same order as `wan_ti2v.py`.
*Analysed estimate; inputs = the five existing wrappers above and the fact that CogVideoX's pipeline
is a stock `diffusers` pipeline.* Breakdown:

- **`_ComposedDiT` equivalent** (~50 lines). `wan_ti2v.py:54–121` wraps the DiT so `compose_fn`
  fires per step and delegates `.to/.cpu/.parameters` via `__getattr__`. For CogVideoX the wrapped
  object is `CogVideoXTransformer3DModel`; its call signature takes
  `(hidden_states, encoder_hidden_states, timestep, ofs=..., image_rotary_emb=...)`.
- **`encode`/`decode`** (~40 lines). `AutoencoderKLCogVideoX` with `scaling_factor: 0.7`; note
  `invert_scale_latents: false` for 1.0 vs `true` for 1.5 — do not mix them up.
- **`denoise`** (~60 lines). Must channel-concat the conditioning image latent to reach
  `in_channels: 32`, and map the repo's `t` to CogVideoX's discrete `timestep`.
- **`generate`** (~120 lines). Delegate to `CogVideoXImageToVideoPipeline` with the `compose_fn`
  injected at the transformer call — mirrors `wan_ti2v.py:248–318`.
- **`factory.py` provider branch** (~15 lines) alongside the existing
  `elif provider in ("wan2.1", "wan", "wan2.2"):` at line 103.

**Vendoring: substantially cheaper than Wan or DynamiCrafter.**
- Wan required vendoring `external_repos/Wan2.2/` (`wan/` alone is 1,803 lines across
  `image2video.py`, `text2video.py`, `first_last_frame2video.py`, `vace.py`) plus
  `src/external_deps/lvdm/` for DynamiCrafter.
- CogVideoX needs **no research-repo vendoring** for the inference path — the transformer, VAE and
  both schedulers are upstream in `diffusers` (`CogVideoXTransformer3DModel`,
  `AutoencoderKLCogVideoX`, `CogVideoXDDIMScheduler`, `CogVideoXDPMScheduler`,
  `CogVideoXImageToVideoPipeline`). `factory.py:44` already has a `provider == "diffusers"` branch
  and `models/base/diffusers.py` exists (though it currently only loads `UNet2DConditionModel` /
  `UNet2DModel`, so it needs a 3D sibling — it is not reusable as-is).
- Vendor the SAT tree only if the **stochastic VP-SDE sampler** is wanted (§5).

**Training-loss side: ZERO new code.** This is the strongest integration argument and it is
verifiable in-repo:
- `losses/diffusion.py:104` — `if prediction_key in {"velocity", "v"}` returns
  `sqrt_alphas * noise - sqrt_one_minus * x_start` (line 113), which is exactly the diffusion
  v-target `v = α_t·ε − σ_t·x₀`. CogVideoX's `v_prediction` is already supported.
- `losses/diffusion.py:41` calls `make_beta_schedule(schedule=beta_schedule, ...)`; in
  `backbones/dynamicrafter/models/utils_diffusion.py:33–34`,
  `schedule == "linear"` computes `linspace(linear_start**0.5, linear_end**0.5, n)**2` — which
  **is** diffusers' `"scaled_linear"`.
- `DiffusionTrainingObjective` already exposes `rescale_betas_zero_snr`
  (`losses/diffusion.py:25`) and imports `rescale_zero_terminal_snr` (line 7).

So CogVideoX's schedule is reproduced bit-for-bit by:
```python
DiffusionTrainingObjective(
    timesteps=1000, beta_schedule="linear",
    linear_start=0.00085, linear_end=0.012,
    rescale_betas_zero_snr=True,
)   # + prediction_type="velocity"
```
— identical to the DynamiCrafter arm's settings in `configs/base/dynamicrafter512.yaml`.

**Cost comparison.** *Analysed estimate.* Cosmos-Predict1 would be **more** work: no diffusers
pipeline, so `cosmos_predict1/diffusion/` must be vendored (its own `res_sampler.py`,
`VideoExtendGeneralDIT`, tokenizer wrapper, guardrail/prompt-upsampler stack to stub out), plus a
new **EDM/VE** objective in `losses/` (continuous σ, x0-prediction, Karras preconditioning) —
none of which the current VP-only `DiffusionTrainingObjective` covers. Rough order **600–900
lines + a new loss module**, versus ~350 lines and no loss work for CogVideoX.

---

## 4b. ⚠ Compute risk — the one thing that could actually blow the 20,000 SBU

**CogVideoX's VAE compresses spatially 8x; Wan2.2's compresses 16x.** At matched pixel resolution
CogVideoX therefore emits **4x the latent cells** and, after the identical `patch_size: 2`,
**~2x the tokens** — and quadratically more attention. This is a direct consequence of the
config fields in §3.1 and it is easy to miss because the *temporal* axis matches perfectly.

Token counts (my arithmetic, from the sourced config fields; latent frames = `(F−1)/vae_t + 1`):

| config | latent (T,H,W) | tokens | vs current Wan run |
|---|---|---|---|
| **Wan2.2 TI2V-5B — as actually run today** (`configs/wan22/...acwm_robotarm.yaml`: `latent_height: 16`, `latent_width: 16`, `temporal_length: 97` → 256×256) | 25 × 16 × 16 | **1,600** | 1.0x |
| CogVideoX-5b-I2V @ 256×256, 49 frames (**off-spec**) | 13 × 32 × 32 | 3,328 | 2.1x tokens |
| CogVideoX-5b-I2V @ **native 480×720**, 49 frames | 13 × 60 × 90 | **17,550** | **11.0x tokens** |
| Allegro-TI2V @ 256×256, 49 frames (off-spec) | 13 × 32 × 32 | 3,328 | 2.1x tokens |
| Cosmos-Predict1-7B @ 256×256, 121 frames | 16 × 32 × 32 | 4,096 | 2.6x tokens |

*Analysed estimate of wall-clock, with reasoning shown.* For CogVideoX-5b (42 layers, width 3072):
per-token-per-layer linear FLOPs ≈ 12·d² ≈ 113M; attention FLOPs ≈ 2·N·d. At N=1,600 attention is
~10M (linear-dominated); at N=17,550 attention is ~108M, i.e. **attention ≈ linear**. So the
native-resolution run is roughly **11x tokens × ~2 for the attention crossover ≈ 20x** the current
Wan step cost, times a further 1.125 for the larger parameter count. **That would very likely
consume the whole budget.** The 256×256 off-spec run is ~2–3x — affordable.

**EasyAnimate is materially better on this axis than CogVideoX — a second reason to prefer the
primary recommendation.** The V5-7b-InP card states it *"supports multi-resolution (512, 768, 1024)
video prediction, trained at 49 frames, 8 fps"*
(<https://huggingface.co/alibaba-pai/EasyAnimateV5-7b-zh-InP/raw/main/README.md>), and lists
22 GB inference memory. So **512×512 is an officially supported resolution**, whereas for
CogVideoX every resolution except 720×480 is officially unsupported. Cost at 512×512/49 frames:
latent 13 × 64 × 64, patch 2 → **13,312 tokens** (8.3x the current Wan run) per arm; at an
off-spec 256×256 it drops to 3,328. *If the budget is tight, the honest ordering is: try 256×256
first and check base-only generation is not degenerate; fall back to the supported 512×512 only if
it is.*

**Hedge if the resolution constraint bites on CogVideoX: `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`** ‡
— verified to carry the identical CogVideoX objective (`CogVideoXDDIMScheduler`,
`v_prediction`, `scaled_linear` 8.5e-4→1.2e-2, `rescale_betas_zero_snr: true`), the same
`AutoencoderKLCogVideoX` and `CogVideoXTransformer3DModel`, 5,571,094,144 params — but it is an
explicitly **multi-resolution, native-I2V** adaptation. Licence is `other` and it is a third-party
fine-tune, so it is a hedge, not a first choice.

**The tension.** The CogVideoX-5b-I2V model card states the resolution is
*"720 x 480, no support for other resolutions (including fine-tuning)"*
(<https://huggingface.co/THUDM/CogVideoX-5b-I2V/raw/main/README.md>). So the cheap option is the
one the vendor explicitly says is unsupported. **This is the single biggest execution risk in the
recommendation and it should be settled with a short smoke test before the real run is queued:**
freeze the base, encode→decode a handful of robot frames at 256×256, and check base-only native
generation is not degenerate. If it is degenerate, the choice becomes (a) pay for 480×720, or
(b) fall back to **Allegro-TI2V**, whose `sample_size: [90, 160]` is likewise native-locked, or
(c) reconsider spending the budget on the per-frame-conditioning arm on Wan instead (§0).

Note this also partly explains a *pre-existing* asymmetry: the DynamiCrafter cell that works runs a
2D VAE at `temporal_length: 16`, so it has never been compared to Wan at matched token budget
either. Token count is a live, unlogged covariate across all three cells.

## 5. Sampler note — stochastic vs deterministic

| backbone | shipped samplers | default | stochastic available? |
|---|---|---|---|
| **CogVideoX-5b-I2V (SAT)** | `VPSDEDPMPP2MSampler`, `VPODEDPMPP2MSampler`, `VideoDDIMSampler`, `VideoDDPMSampler`, `Image2VideoDDIMSampler` | **`VPSDEDPMPP2MSampler` — STOCHASTIC** | ✅ **matched SDE/ODE pair on one model** |
| CogVideoX-5b-I2V (diffusers) | `CogVideoXDDIMScheduler`, `CogVideoXDPMScheduler` | `CogVideoXDDIMScheduler` | deterministic (DDIM `eta=0`) |
| Cosmos-Predict1-7B | `res_sampler.py` (RES / DDIM / DEIS / DPM-Solver / EDM), solver options `2ab`/`2mid`/`1euler` | deterministic | ✅ via `S_churn` (code comment: *"following parameters control stochasticity, see EDM paper. BY default, we use deterministic with no stochasticity"*, `s_churn: float = 0.0`) |
| **Allegro-TI2V** | `EulerAncestralDiscreteScheduler` | **STOCHASTIC (ancestral)** | ✅ stochastic *is* the default |
| **EasyAnimate V5-7b** (diffusion arm) | `DDIMScheduler` (+ Euler, Euler-A, DPM++, PNDM in the repo ‡) | DDIM | ✅ via `eta`, or swap to Euler-A |
| **EasyAnimate V5.1-7b** (flow arm) | `FlowMatchEulerDiscreteScheduler` | deterministic ODE | ✗ — flow arm is ODE-only |
| **DynamiCrafter-512/1024** *(incumbent)* | DDIM | **`ddim_eta = 0.0` in this repo** (upstream default is `1.0`) | ✅ via `eta` |

**A confound I chased and then ruled out — recording it so nobody re-chases it.** Upstream
DynamiCrafter's inference default is **`--ddim_eta = 1.0`**, i.e. fully ancestral/*stochastic*
sampling despite the "DDIM" name (the same trap exists in VideoCrafter1/2) ‡. Had the DC cell been
sampled stochastically while Wan was sampled deterministically, sampler stochasticity would have
been confounded with the objective **in the results already on the table**.

**It is not.** This repo overrides the upstream default:
`src/generative_flow_adapters/models/base/dynamicrafter_video.py:235` declares
`ddim_eta: float = 0.0` and passes it at line 281 (`eta=ddim_eta`) — **deterministic DDIM**. Wan's
TI2V path likewise defaults to the deterministic `FlowUniPCMultistepScheduler`
(`external_repos/Wan2.2/wan/textimage2video.py:356`). **Both incumbent cells are sampled
deterministically, so the existing DC-vs-Wan gap is not a sampler artefact.**
⚠ Caveat: this covers the *framework* path only. The AVID clean-room runs under
`external_repos/avid/` use AVID's own sampling code, which I did **not** check — if any headline
number came from there, re-verify its `eta` separately.
| Wan2.2 TI2V-5B *(incumbent)* | `FlowUniPCMultistepScheduler`, `FlowDPMSolverMultistepScheduler` | `unipc` (deterministic ODE) | ✅ **already vendored** — see below |
| DynamiCrafter-512 | DDIM (`make_ddim_timesteps`) | DDIM | eta>0 |

**CogVideoX gives a clean SDE-vs-ODE pair on one checkpoint.** In
<https://raw.githubusercontent.com/THUDM/CogVideo/main/sat/sgm/modules/diffusionmodules/sampling.py>,
`VPSDEDPMPP2MSampler.sampler_step` injects noise explicitly:
```python
mult_noise = append_dims((1 - next_alpha_cumprod_sqrt**2) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5, x.ndim)
x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
```
while its twin `VPODEDPMPP2MSampler` has no `randn_like` and uses
`mult1 = ((1 - next_ᾱ_sqrt²)/(1 - ᾱ_sqrt²))**0.5`. Same model, same schedule, differing **only**
in the noise-injection term — as controlled a stochasticity ablation as exists.

**The cheap experiment on the *existing* flow model is a one-kwarg change and needs no new
download.** In the already-vendored
`external_repos/Wan2.2/wan/utils/fm_solvers.py`, `FlowDPMSolverMultistepScheduler.__init__`
accepts `algorithm_type` ∈ `{"dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"}`
(lines 141, 156–158). The TI2V-5B path at `external_repos/Wan2.2/wan/textimage2video.py:364`
instantiates it **without** `algorithm_type`, so it falls back to the deterministic
`"dpmsolver++"`. Passing `algorithm_type="sde-dpmsolver++"` yields a stochastic sampler on the
current Wan flow backbone. Given the 11-day deadline this should be run **before** any new
backbone is downloaded — it tests the stochasticity limb of the hypothesis for ~zero integration
cost.

---

## 6. What I could not verify

- **CogVideoX paper section/equation numbers.** `arxiv.org/pdf/2408.06072` exceeds the fetch size
  limit and `ar5iv`/`arxiv.org/html` both fail to render it ("Conversion to HTML had a Fatal error").
  The objective claims in §3.1 rest on the **official SAT training config and the diffusers
  scheduler config**, which I consider stronger evidence than paper prose — but the paper's own
  wording on v-prediction / zero-SNR / explicit uniform sampling is `_needs verification_` at the
  section-number level. (Secondary corroboration only: search summaries and the WF-VAE
  related-work section, not the primary text.)
- **Cosmos-Predict1-7B exact parameter count.** The HF repo is gated, so
  `safetensors.parameters` is unavailable. "7B" comes from the model card and the `faditv2_7b`
  config name — **not measured**, unlike CogVideoX (5,625,087,552) and Wan (4,999,800,000-ish,
  measured at 4.9998B from the local checkpoint headers).
- **Wan2.2 TI2V-5B "5B" as an official figure.** My 4.9998B is *measured* by summing safetensors
  header shapes in `ckpts/Wan2.2-TI2V-5B/` — I did not find a vendor statement of the exact count.
- **Cosmos-Predict2's VAE identity.** "Wan2.1 spatiotemporal VAE" comes from the Cosmos Policy
  paper §3, not from a Cosmos-Predict2 config field I read directly. Moot — Predict2 is
  disqualified on objective anyway.
- **Allegro's I2V conditioning mechanism.** Inferred from `in_channels == out_channels == 4`
  (i.e. *not* channel-concat), not read from code. `_needs verification_`.
- **Allegro's transformer-only parameter count.** 2.788B is the repo-wide safetensors total
  (transformer + VAE); the DiT alone is smaller. Not separated.
- **The V5 → V5.1 training-recipe delta.** I found no source quantifying how the *data* and
  *schedule* differ between the two arms of the primary recommendation. The architectural identity
  is measured; the training identity is **not**, and cannot be assumed. This is the recommendation's
  main scientific weakness and should be stated in the thesis.
- **Rows marked ‡** were verified by delegated agents, not by me personally. I spot-checked
  EasyAnimate V5/V5.1, Allegro, CogVideoX-Fun and DynamiCrafter's `ddim_eta` directly; the rest
  (Ruyi's VAE ratio, Open-Sora/Open-Sora-Plan objectives, the EU licence text, Mochi's missing
  I2V, SkyReels-V2 provenance) rest on the agents' quoted evidence.
- **Cosmos-Predict1-7B and VideoCrafter1-I2V param counts** — `_needs verification_` (gated repo
  shipping `model.pt`, and a single bundled pickle, respectively). Neither ships a readable
  safetensors manifest.
- **Which SkyReels checkpoint produced the existing result row** — unresolved, and it has a
  licensing consequence (§2c).
- **Vchitect-2.0 latent channels / VAE config** ‡ — both HF repos are gated (401); the reported
  values are analysed estimates, not config-confirmed. Moot: it is flow and has no I2V.
- **Open-Sora v1.0 param count (724M)** ‡ — from the repo's own report, not a tensor recount.
- **CogVideoX-5b-I2V on robot data.** No evidence either way that it fine-tunes well at
  MetaWorld/RT-1 resolutions. Its native 480×720 / 49 frames is a fixed constraint — the model card
  states *"720 x 480, no support for other resolutions (including fine-tuning)"*, which is a real
  risk for low-res robot datasets and should be checked before committing the budget.
