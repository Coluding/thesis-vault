# Wan-based action-conditioned world models — who exists, and how they inject actions

**Date:** 2026-08-02
**Type:** literature sweep (inbox — not yet promoted)
**Question:** Which papers/projects use Wan (Wan2.1 / Wan2.2) as the base for an
action-conditioned world model, and exactly how do they inject actions? Does anyone
report our residual-drowning failure, or a fix for it?

**Our finding this is testing against:**
[[../30_Knowledge/experiments/20260731-wan-action-trace-value-pathway-drowns]] —
action tokens survive cross-attention (44–56% action-driven output across all 10
blocks) then drown at the residual add: xattn output RMS ~0.01 against a stream of
1.8–3.0, `drel` 0.44 → 0.0085 in one addition. qk_norm rescues the logits; the
**value pathway is unnormalised**.

---

## 1. Verdict

**Genuine Wan-based action-conditioned world models exist and are numerous — this
is now a crowded field, not an empty one.** The sweep found Wan used as the base
across robot manipulation, teleoperation, gaming/camera-navigation, and as a
compared-against baseline in dexterous manipulation.

**On the residual-drowning problem specifically:**

1. **Our *conclusion* has been published — twice, on Wan — but never *measured*.**
   **GigaWorld-1 §5.3 Finding 9** (§2a): *"Cross-attention provides only a marginal
   improvement… suggesting that **attention-side action tokens are easily overwhelmed
   by appearance and semantic tokens**."* Their Table 3 shows cross-attention scoring
   **0.1620 Traj. Acc. against 0.1576 for no control at all** on Wan2.1-1.3B — our
   "sensitivity without control" result, in someone else's table.
   **GameFactory §6.2** names the same mechanism from the softmax side.
   **What is still ours:** nobody has *instrumented* it. No paper reports
   cross-attention-output RMS vs residual-stream RMS inside a video DiT. GigaWorld-1
   infers "overwhelmed" from a downstream metric; we measured the actual magnitudes
   (0.01 vs 1.8–3.0; `drel` 0.44 → 0.0085 in one addition). **Csordás et al. 2025**
   supplies the published "healthy" sublayer-to-residual ratio (~0.10–0.15), making
   our ~0.005 quantifiable as **20–30× below** a healthy branch. So we should reframe
   the contribution from *discovery* to **mechanism + measurement for a conclusion
   the field has reached empirically but never explained.**
2. **But the fix is convergent and near-universal.** Every Wan-based system I could
   verify at code or equation level puts *something* on the conditioning pathway to
   control its magnitude before it reaches the residual stream: distribution
   alignment + learnable gate (RynnWorld-Teleop, Eq. 3), zero-init output projection
   + qk-RMSNorm + riding the base adaLN gate (DreamX-World, code), learned per-token
   event gate (EA-WM, Eqs. 5–6), or per-dimension normalisation + explicit constant
   scale-up (DexAC-WM, Eq. 4).
3. **One paper states our problem almost verbatim, from the other end.** DexAC-WM:
   *"raw action values are numerically very small, we multiply each dimension by a
   constant scaling factor before tokenization to enhance conditioning strength."*
4. **The same paper independently reproduces our caveat**: scale alignment alone is
   **necessary but insufficient**. Their naive global ×200 rescale is *worse* than
   their structured fix (FVD 371.18 vs 284.40). This is the external mirror of our
   own `action_token_norm` result (6–10× gain, no unlocked steering) flagged as
   provisional in [[../30_Knowledge/writing/writing-plan-2026-08]].

**On a frozen Wan — the verdict changed twice during this sweep, and the honest
answer is: frozen-Wan-plus-gated-adapter is a MATURE, CROWDED pattern.**

At least **twelve** systems condition a frozen or near-frozen Wan: VACE, Uni3C,
**DepthDirector (Wan2.2-TI2V-5B — our exact base, frozen)**, VerseCrafter, UCPE,
PostCam, InfCam, SymphoMotion, DriveCtrl, DriVerse/WanControl (control signals), plus
**Micro-World**, **DreamX-World 1.0** (Wan2.2-TI2V, our base) and **LingBot-World**
(actions/camera). **We should stop framing "adapter on a frozen Wan" as novel.**

**The correlation that reframes our whole failure** (§2d): *whether a work gates its
conditioning pathway is predicted almost perfectly by whether its base is frozen.*
Every fully-fine-tuned work injecting by concat uses **no gate**; every frozen work
**gates**. We are in the frozen regime running an **ungated** design borrowed from the
fine-tuned regime. That is the one-sentence diagnosis.

**And Wan's own source contains the smoking gun:** its I2V image cross-attention has a
learned zero-init gate **shipped commented out** (`wan/modules/model.py:199`:
`# self.alpha = nn.Parameter(torch.zeros((1, )))`); the image branch adds un-gated.
Our action adapter imitates precisely that ungated path.

**What is genuinely unoccupied:** **frozen Wan + continuous, high-dimensional
robot-action conditioning.** The robot sweep is explicit: *"in the Wan-robot-world-model
literature, essentially nobody freezes the base and trains only an adapter."* The four
partial exceptions (Kinema4D LoRA r=64, GigaWorld-1, ABot-PhysWorld's A2V stage,
PAVXploreRL's RL phase) mostly full-fine-tune in a prior SFT stage. Everyone keeping
Wan frozen conditions on **camera / geometry / dense control maps**, not actions. That
intersection — not "frozen Wan", not "adapters" — is our slot, and it is narrow but
real.

**The single cleanest gap statement available to us**, from the robot sweep's targeted
search: **a LayerNorm/RMSNorm applied to robot-action *tokens themselves* on a Wan
backbone does not exist in the literature.** RynnWorld-4D has exactly that machinery
(per-modality LN + qk-RMSNorm + `tanh(g)` gate, frozen Wan2.2-TI2V-5B) **but its
injected branch carries depth/flow, not actions.** So the machinery is proven on Wan,
and applying it to actions is unclaimed.

**Closest prior art to our composition rule: RynnWorld-4D**, which is functionally
`f_base(x,t) + g·Δ_φ(x,t,·)` on a **frozen** Wan2.2-TI2V-5B (§2a). Cite it as
validation of the gate design, and differentiate on modality (actions vs depth/flow).

---

## 2. The table

Legend for "base frozen?": **F** = base DiT frozen, adapter-only; **L** = LoRA on
base; **FT** = full fine-tune; **—** = not Wan-based (listed for contrast).

| Work | Base + frozen? | Action type | Injection mechanism | Action-pathway normalisation | Action-following metric | Dataset | Link |
|---|---|---|---|---|---|---|---|
| **DreamX-World 1.0** | Wan2.2-TI2V (5B), **F** for camera stage; **FT** for later event-instruction stage | Keyboard WASD/IJKL → per-frame camera pose (viewmats + K) | **Parallel PRoPE self-attention branch** per DiT block; output added to the *self-attn output* **before** the adaLN gate `e[2]` (code L487–489). Camera enters via **projective positional encoding on q/k/v**, not as residual-stream tokens | **zero-init `out_proj` weight+bias** ("for stable residual training", code L218–220); **qk RMSNorm** on branch (L215–216); `attn_compress=4`; all 30 layers | Camera control score **73.75** on OmniWorldBench (vs HY-WorldPlay 1.5 = 65.12, LingBot-World = 71.73) | UE5, SpatialVID, RealEstate10K, Sekai, DL3DV, OmniWorld-Game | [paper](https://arxiv.org/abs/2606.16993) · [code](https://github.com/AMAP-ML/DreamX-World) · [HF](https://huggingface.co/GD-ML/DreamX-World-5B) |
| **RynnWorld-Teleop** | Wan2.2-TI2V-5B, **L** (LoRA r=64) or **FT** | 21-joint hand skeleton rendered as depth-modulated video → VAE latent `c ∈ R^{C×T×H×W}` | **Additive patch-embedding** (Eq. 3): `x = PatchEmbed^z(z_t) + α·PatchEmbed^c(c̃)` | **Distribution alignment**: `c̃ = (c−μ_c)/σ_c·σ_z + μ_z` with running estimates; **`PatchEmbed^c` zero-initialised**; **learnable scalar gate α init 0.1** | No action-following metric; PSNR/SSIM/LPIPS/FVD + downstream policy success | VITRA (1.23M), EgoDex (0.91M), 1.8K teleop episodes | [arXiv 2607.06558](https://arxiv.org/abs/2607.06558) |
| **EA-WM** | Wan2.2-TI2V, **L** (two-stage LoRA) | KVAF — kinematics rendered to camera view as H×W×3 images (arm skeleton, joint landmarks, gripper, EE heatmap) | **Dual-stream bidirectional cross-attention** at sparse fusion layers (Eqs. 5–6): `H̃_v = H_v + G_ℓ ⊙ CA_{v←k}(H_v,H_k)` and the symmetric `H̃_k = H_k + G_ℓ ⊙ CA_{k←v}(...)` | **Learned per-token event gate `G_ℓ`** multiplying the cross-attn output before the residual add; gate supervised by Event-Difference Latent Supervision (EDLS). No zero-init stated | **Instruction Following 0.792** (vs CogVideoX 0.727); Trajectory Accuracy 0.430; Interaction Quality 0.682 — WorldArena | RoboTwin | [arXiv 2605.06192](https://arxiv.org/html/2605.06192) |
| **τ₀-WM** | Wan2.2-TI2V-5B + 0.5B action DiT (5.5B), **FT** | Continuous action chunks, horizon `H_a` (dim not stated) | **Separate action-expert DiT**; action tokens self-attend over the horizon then **cross-attend into intermediate video features** at matched stages. *Direction: action reads video (action decoding), not video reads action* | None stated; `λ_z = λ_a = 1` | Task success rates only; no action-prediction/trajectory metric | 27.3K h (17.8K real-robot teleop + 6.5K UMI + 3.0K egocentric) | [arXiv 2606.01027](https://arxiv.org/html/2606.01027v1) |
| **Light-WAM** | Wan2.1-T2V-1.3B, **F** (1,545.79M frozen, Table 6) + LoRA | Outputs 24×14 action chunk (RoboTwin 2.0) | **Read-out, not conditioning.** Residual bottleneck MLP adapters at layers {8,16,24} (Eq. 4): `H_ℓ = U_ℓ + A_ℓ(U_ℓ)`, `A_ℓ(x)=γ·W^up σ(W^down x)`, bottleneck 256, `γ=1.0`. Then query-pooling → action head | `LN` after mean-pooling (Eq. 9); `LN` before output head (Eq. 11). No zero-init/gate stated | LIBERO 97.2% avg; RoboTwin 2.0 76.4% avg (task success, not action-following) | LIBERO, RoboTwin 2.0, real dual-arm | [arXiv 2606.08242](https://arxiv.org/html/2606.08242v1) |
| **Fast-WAM** | Wan2.2-5B video DiT + 1B action expert (~6B); frozen status **_needs verification_** | Action tokens from action expert | Mixture-of-Transformer with **shared attention**; **structured attention mask** stops action tokens attending to future video tokens | _needs verification_ | Task success (RoboTwin, LIBERO, real towel folding) | _needs verification_ | [arXiv 2603.16666](https://arxiv.org/html/2603.16666v1) |
| **DreamZero** | Wan2.1-I2V-14B-480P, **FT** (all DiT blocks + state/action encoders; text/image encoders + VAE frozen) | Actions **predicted jointly with video**, not injected. Conditions on proprio `q_l`, past obs, language | Joint flow matching over concatenated `[video latent; normalised action]`; shared timestep within a chunk | Action normalised before concatenation | Task success / policy metrics | Large robot corpus | [arXiv 2602.15922](https://arxiv.org/html/2602.15922v1) · vault: [[../30_Knowledge/related-work/dreamzero-wam]] |
| **Wan2.1-Fun-Control / Wan2.2-Fun-Control** (official Alibaba/PAI) | Wan2.1-1.3B / Wan2.2-5B & 14B, **FT** (patch-embed in_dim changes → base not frozen, *[inference]*) | Dense control video (Canny / depth / OpenPose / MLSD / trajectory) + camera Plücker (24-dim) | **(a) Channel concatenation** of control latents into the DiT input before patch embedding (`pipeline_wan2_2_fun_control.py` L804–822 builds `control_latents_input`, passed as `y=` L858 → `wan_transformer3d.py` L906 `torch.cat([u,v], dim=0)`, then L909 `patch_embedding`). **(b) Camera**: `SimpleAdapter` (PixelUnshuffle→Conv→ResBlocks) output **added raw** to patch-embedded latent, `wan_transformer3d.py` L912–914: `x = [u + v for u,v in zip(x, y_camera)]` | **NONE on either path** — no gate, no zero-init, no normalisation (verified: `wan_camera_adapter.py` full file, 62 lines, contains no init call) | Not reported by the repo | Internal | [code](https://github.com/aigc-apps/VideoX-Fun) |
| **DexAC-WM** ("Not All Actions Are Equal") | **—** Cosmos-Predict2.5 (2B); Wan2.1 used only as **frozen visual encoder**. Compares *against* Wan2.1-Fun-1.3B-Control and Wan2.2-Fun-5B-Control | 57-DoF dexterous (15/hand fingers, 9/wrist 6D rot, 9 camera) | **Dual pathway**: local cross-attn (Eq. 7) `Z^local = Z + Attn(Q=Z, K=A_tok, V=A_tok)` **+** global AdaLN modulation (Eqs. 8–10) from a learnable summary query | **Per-dimension z-score** (Eq. 4) `ã = (a−μ_i)/(σ_i+ε)` **+ explicit constant scale-up** "to enhance conditioning strength" | **PCK@10 / PCK@20** + per-action-family diagnostics (wrist / finger / head) | EgoDex (829 h), EgoVerse (1,362 h) | [arXiv 2606.27325](https://arxiv.org/html/2606.27325v1) |
| **AV DiT World-Action Model** | **—** from-scratch 4-block DiT (5.4M), frozen SD-VAE | 2D ego (steer, accel) | Learned **Fourier features** (Eq. 2, N_f=64/dim) summed into conditioning vector `c` with timestep emb + pooled latent → **adaLN-Zero** modulation (Eq. 1) | **z-score on actions**; adaLN-**Zero** (gate zero-init by construction) | **Steering sweep**: 5th–95th pct with fixed noise, measure horizontal scene displacement at t+15; **Spearman ρ = +0.81** (regression: −0.18); 100% sign-correct on 18/40 scenes above detection threshold | nuScenes v1.0 | [arXiv 2606.12987](https://arxiv.org/html/2606.12987) |
| **Vid2World** | **—** DynamiCrafter 1.1B U-Net, **fine-tuned** (100k steps, 4×A100, ~7 days) | per-frame action, dim not stated | **Additive to the latent at the matching temporal position**: *"When predicting o_t, the embedding of a_{t-1} is added to the model's latent representation at temporal position t"*, via *"a lightweight multi-layer perceptron"* | **None described** | No action-specific metric — FVD/FID/SSIM/PSNR/LPIPS/DreamSim only. Separately: **causal action guidance** (§4.2) gives FVD 29.4→25.8 | **RT-1**, CS:GO, RECON | [2505.14357v3](https://arxiv.org/html/2505.14357v3) |
| **AVID** | Not Wan (see vault note); **F** frozen base | Action-conditioned | Trainable correction on frozen base output; **learned mask-mix** at the **output level** | Mask gate, `init_mask_bias: 0.0` (σ=0.5) | — | MetaWorld etc. | vault: [[../30_Knowledge/related-work/avid]] |

### 2a. The robot-manipulation cluster — **the most important section in this note**

Sixteen further Wan-based, robot-action-conditioned systems. Two of them state our
finding; one of them is functionally our composition rule.

**🔴 Another group has published our diagnosis.** GigaWorld-1 (§5.3, Finding 9),
ablating injection mechanisms on Wan2.1-1.3B:
> "Cross-attention provides only a marginal improvement… suggesting that
> **attention-side action tokens are easily overwhelmed by appearance and semantic
> tokens**."

**Table 3 (Wan2.1-1.3B, control-condition ablation)** — the cleanest injection
ablation on Wan anywhere:

| Control type | Traj. Acc.↑ | Dynamic↑ | Smooth↑ | Flow↑ | Subject↑ |
|---|---|---|---|---|---|
| I2V, no control | 0.1576 | 0.2429 | 0.4997 | 0.0971 | 0.5568 |
| **Cross-attention** | **0.1620** | 0.1049 | 0.4525 | 0.0624 | 0.3573 |
| ControlNet | 0.2566 | 0.3083 | 0.5197 | 0.1412 | 0.7212 |
| **Channel concat** | **0.3528** | 0.3566 | 0.5747 | 0.2179 | 0.8600 |

**Cross-attention scores 0.1620 against 0.1576 for no control at all** — i.e. on Wan,
low-dim action cross-attention is worth ~nothing, and is *worse than no control* on
every other axis. That is our "sensitivity without control" result, in someone else's
table. Their winner is "not merely explicit, but **spatially aligned with the noisy
latent from the beginning of denoising**."
**Independent confirmation:** World Action Planner Tab. 1 rebuilds WPE, IRASim
(AdaLN-Zero) and Ctrl-World (cross-attn) **on the same Wan-T2V-1.3B with 2× the
training steps**, and cross-attn is worst of three on PSNR/LPIPS.

**🔴 RynnWorld-4D is functionally our composition rule, on Wan2.2-TI2V-5B, with a
frozen backbone.** [2607.06559](https://arxiv.org/html/2607.06559v1) §3.3, §4.1 —
Joint Cross-Modal Attention every 3 blocks (layers 0,3,…,27 = 10 modules):
- **Eq. 3:** `z̃^m_l = LN^m(z^m_l + e^m)` — modality embedding **zero-initialised**
  "so the module starts as a pure residual", plus **per-modality LayerNorm "to align
  numerical scales across branches"**.
- **Eq. 4:** `Q = RMSNorm_q(QProj(z̃))`, `K ← RMSNorm_k(K)` — explicit **qk-norm on the
  injected pathway**.
- **Eq. 6:** `ẑ^m_l = z^m_l + tanh(g^m_l)·OutProj^m_l(A^m_l)` — and the rationale is a
  direct argument against the default fix:
  > "**Instead of the double zero-initialization used in ControlNet — which we found to
  > introduce a saddle-point deadlock** — we combine a **zero-initialized output
  > projection** with a **learnable gate g initialised to 1**… `tanh(1) ≠ 0` ensures
  > non-zero gradients flow into the gate so that it can decrease, increase, or change
  > sign, **preventing the joint pathway from being trapped at the origin**."
- **Stage 2 freezes the entire Wan backbone**; only JA projections, RMSNorms,
  per-modality LayerNorms, tanh gates and modality embeddings train (lr 5e-5).

This is `f_base(x,t) + g·Δ_φ(x,t,·)` on a frozen Wan, with normalisation *and* a
learned gate. **It is the closest published prior art to our composition rule**, it
independently validates the gate design, and its saddle-point argument is exactly the
collapsed-gating risk from §5.6. ⚠️ **Caveat: its injected branch carries depth/flow,
not actions** — so the gap for robot-action tokens stands.

| Work | Base + frozen? | Action | Injection | Norm / gate | Metric |
|---|---|---|---|---|---|
| **ABot-PhysWorld** (AMAP/Alibaba) [2603.23376](https://arxiv.org/abs/2603.23376) | Wan2.1-I2V-14B-480P; full SFT, but **A2V stage `TRAINABLE_MODELS=vace`** → backbone frozen, side branch trained | 16-dim/frame (dual-arm xyz+quat+gripper) → **rendered 3-ch RGB trajectory map** → VAE | **VACE parallel context blocks** cloned at layers (0,2,…,28), `vace_in_dim=96`; `x = x + vace_hint * vace_scale` (default 1.0) | **Double zero-init**: `vace_patch_embedding` and `before_proj`/`after_proj` weight+bias all zeroed (`wan_video_vace.py:106-133`) | **Trajectory Consistency 0.8522** vs EnerVerse 0.8157 |
| **GigaWorld-1** [2607.02642](https://arxiv.org/html/2607.02642v1) | Wan 1.3B + 5B; LoRA + control pathways, frozen scope partly ambiguous | EE pose map + ray map → unified control map → VAE | **Channel concat** of control latent with noisy video latent (Eq. 24) | — | **Trajectory Accuracy** = SAM arm boxes → centers → **normalised DTW** vs reference. Backbone comparison Tab. 9 |
| **RynnWorld-4D** [2607.06559](https://arxiv.org/html/2607.06559v1) | **Wan2.2-TI2V-5B, backbone FROZEN in stage 2** | depth/flow (not actions) | JCMA every 3 blocks | **zero-init OutProj + `tanh(g)`, g=1; per-modality LN; qk-RMSNorm** | — |
| **X-WAM** [2604.26694](https://arxiv.org/html/2604.26694v1) | Wan2.2-TI2V-5B, full FT | **14-dim relative** (Δpos, Δaxis-angle, gripper) ×2 | **Sequence concat**, bidirectional attention, same temporal RoPE as video. Action-conditioned mode = denoise actions first (p=0.5) | **Quantile norm, scaling only, no bias** — "preserving the semantics that a zero action corresponds to no movement" | RGB PSNR / depth AbsRel under action-conditioned denoising |
| **BWM / Boundless** (code only) | **Wan2.2-TI2V-5B, fully fine-tuned** (`trainable_models: dit`) | **14-dim** dual-arm | **Dual, both reusing existing Wan pathways**: (a) `context = action_emb` — action tokens occupy **Wan's T5 text slot**, text disabled; (b) `t = t + action_mod_emb` → adaLN | **NONE in-network** — three bare MLPs, no norm/gate/zero-init. Data-level p01/p99 min-max only | PSNR/SSIM only |
| **LingBot-VA** [2601.21998](https://arxiv.org/abs/2601.21998) | Wan2.2-5B, **nothing frozen** | **30-dim** universal dual-arm | Action tokens are **first-class noised tokens interleaved into the causal sequence** (τ=4); separate adaLN pathway for actions; text stays in frozen T5 cross-attn | **α = √(d_v/d_a) variance-preserving init**, with the stated failure mode: *"action tokens' output distribution initially diverges significantly from the video distribution, **disrupting the joint attention mechanism**"* (ablated Fig. 7). Separate SNR shift per modality (video 5.0 vs **action 0.05**) | LIBERO avg **98.5** |
| **PAVXploreRL** [2607.16602](https://arxiv.org/html/2607.16602v2) | Wan2.2-TI2V-5B, full SFT then **RL updating only action encoder + LoRA** | **14D** 6-DoF ×2 | *"injecting action embeddings into the temporal pathway via **AdaLN** and time embeddings"* | none stated | **flow EPE + COS** (directional consistency) |
| **World Action Planner** [2607.27599](https://arxiv.org/html/2607.27599) | Wan-T2V-1.3B, fine-tuned | 7–24 D → FK → **pose skeleton image** | VAE-encode pose images, **concat tokens with video tokens**, kept noise-free. Uses a **dummy text embedding** — does not reuse the text slot | — | Tab. 1: beats Ctrl-World / IRASim / WPE rebuilt on the same base |
| **Kinema4D** [2603.16669](https://arxiv.org/html/2603.16669v1) | **Wan2.1-14B, LoRA r=64 only** | URDF FK → **4D pointmap** | channel-wise concat of input+noisy latents+robot masks. **Text-slot hijack**: *"we intentionally replace the text embedding with the VAE latents of robot sequences"* | **latent norm μ=−0.17, σ=1.36**; **1×1 zero-init conv on the robot latent** (cites ControlNet) | — |
| **iMaC** [2606.09813](https://arxiv.org/abs/2606.09813) | Wan2.2 IT2V DiT, variant/freeze `_needs verification_` | joint actions → motion + contact videos | **Latent-wise addition via new control-specific patchify layers** parallel to Wan's `P_v` (Eq. 4) | zero-init **not reported** despite ControlNet shape | MSE 0.028, FID 36.96 vs Ctrl-World / ABot-PhysWorld |
| **RoboWorld** [2607.01060](https://arxiv.org/html/2607.01060) | Wan2.1-T2V-1.3B, full FT | EE Cartesian position | *"encode actions with a two-layer MLP and **inject them into each frame through cross-attention**"* — new-vs-text-slot `_needs verification_` | none | correlation with real RoboArena leaderboard |
| **Action Images** [2604.06168](https://arxiv.org/html/2604.06168v2) | ⚠️ text says Wan2.2, **App. says Wan2.1-I2V-14B-480P** — flag | 7-DoF → 3 semantic 3D points → **RGB Gaussian heatmap** | temporal concat + latent masking | **zero-init encoder proj + identity-init final projector** — but on the *camera* path | — |
| **Masked Visual Actions** [2607.19343](https://arxiv.org/html/2607.19343) | **Wan-Fun-Control 2.2 14B** | masked conditioning video | **concat** — *"appropriate as the conditioning signal is spatially aligned with the desired output"* | — | — |
| **Fast-WAM / Motus / SANTS** | Wan2.2-5B, MoT family | action expert `d_a=1024` | shared/joint attention. **Motus**: *Action-Dense Video-Sparse* — video downsampled ~6× because token imbalance *"causes the model to overfit to video prediction, thereby weakening its action prediction"* | `_needs verification_` | Fast-WAM 91.8 vs LingBot-VA 80.6 vs Motus 77.3 on RoboTwin, all from raw Wan2.2 |
| **DreamZero** [2602.15922](https://arxiv.org/html/2602.15922v1) | Wan2.1-I2V-14B, **all DiT blocks updated**; *"experimented with LoRA but found it led to suboptimal results"* | relative joint positions | jointly-denoised tokens, block-causal mask | none reported | task progress / success only |

**Wan DiT base but NOT action-conditioned** (don't miscount these): HarmoWAM, Vidar,
EVA, PhysisForcing, DreamGen/GR00T-dreams (Wan2.1, LoRA r=4).

**⚠️ Cosmos is not a Wan-DiT work.** Cosmos-Predict2/2.5 runs in the **WAN2.x
tokenizer's** latent space but uses NVIDIA's own DiT. State this precisely in the
thesis. Within that family: **DreamDojo** (ICML 2026) zero-inits the last layer of its
action MLP *"to avoid perturbing the pretrained model state at the beginning of
training"*, citing ControlNet, and *"empirically found [it] leads to improved
physics"*; **Cosmos-Predict2** applies **RMSNorm to (timestep + action) before AdaLN**
(`video2world_action_dit.py:113`) — **but the adaln_lora branch is not normed**;
**IRASim's fixed per-dim action scaler `[20,20,20,20,20,20,1.0]`** (rot/trans ×20,
gripper ×1) is copied verbatim into Cosmos-Predict2 and OSCAR.

**MiraBench** is the action-following benchmark to position against: TCR/OPS/GEN judged
by InternVL3-78B, 906 videos, 16,704 human judgments. **Wan2.1-14B TCR 58.0 / OPS 74.0;
Wan2.2-5B TCR 92.0 / OPS 90.0.** Headline: *"increasing model scale does not reliably
improve action following."*

### 2b. The driving cluster — 10 more Wan-based, ego-action-conditioned systems

Driving is where Wan-based action conditioning is most mature. All mechanism claims
below are **paper-sourced** (arXiv HTML Method sections); **no public code was
available** for CompoSIA, X-World, DriveVA, DriveWAM, Metis or HorizonDrive at scan
time.

| Work | Base + frozen? | Action rep | Injection | Action-pathway norm/gate | Metric | Link |
|---|---|---|---|---|---|---|
| **CompoSIA** | Wan2.1-T2V-1.3B, **FT** | `a = (Δx, Δy, Δyaw) ∈ R^{F×3}` (residual/delta) | **adaLN gating** `f_zero(φ(a)) ∈ R^{f×6×d}`, φ = sinusoidal freq embedding; + PRoPE camera attention in reduced subspace injected via **zero-convolutions** | **`f_zero` = zero-init projector**; 6 channels split into shift/scale/**post-layer residual gate** for self-attn and FFN. Decoupled LR: **2e-4 action projector vs 1e-5 elsewhere** | RotErr **0.55°**, TransErr **7.37e-3**; ablation: w/o residual motion 2.84/15.80 | [2603.12864](https://arxiv.org/html/2603.12864v1) |
| **HorizonDrive** | Wan2.1-1.3B T2V, VAE temporal compression 4→1; DiT split `_needs verification_` | same `(Δx,Δy,Δyaw) ∈ R^{F×3}` | Identical `f_zero(φ(a)) ∈ R^{f×6×d}` adaLN gating; layout via zero-init projector | Same 6-channel shift/scale/residual-gate split | **ARE** (geodesic rot err) + **DTW**; poses via VGGT | [2605.11596](https://arxiv.org/html/2605.11596) |
| **X-World** | **Wan2.2-TI2V-5B**, partial (Wan params loaded, new modules random) | velocity, curvature, roll, pitch | **symlog → Fourier features → MLP → adaLN-Zero** | **symlog** scalar-range normalisation (unique in this set); adaLN-Zero | **None reported** | [2603.19979](https://arxiv.org/html/2603.19979v1) |
| **CausalDrive** | Wan2.1-1.3B, multi-stage; split `_needs verification_` | camera poses → **Plücker** → MLP | **AdaLN for geometric action** ("hard spatial constraint") + **cross-attention for sociology prompt** (Eqs. 3–4) | **None stated** — "AdaLN, unspecified init" | ADE 0.45/0.52; **Yielding Compliance Rate** 82.0% | [2606.15341](https://arxiv.org/html/2606.15341v1) |
| **DriveVA** | **Wan2.2-TI2V-5B**, **FT** (ablated: full FT wins, PDMS 90.9, Table 7) | K=8 future actions, each 3-dim (x,y,yaw), MLP→1 token each | **Token concatenation** into the denoised sequence (Eq. 7); coupling purely via **shared self-attention** | **None stated** | L2@1/2/3s, collision rate, PDMS | [2604.04198](https://arxiv.org/html/2604.04198) |
| **UNIVERSE** | Wan2.2-TI2V-5B (text encoder confirmed frozen; **DiT status `_needs verification_`**) | (x, y, yaw), K-step | Modality-specific input projections + **modality embeddings** + temporal pos emb; **DiT self-attn and FFN shared by both modalities** (Eq. 3) | `_needs verification_` | L2 + NAVSIM NC/DAC/TTC/Comfort/EP | [2607.05133](https://arxiv.org/html/2607.05133v1) |
| **DriveWAM** | Wan2.2-TI2V-5B, **FT** (full DiT + new action/ego modules) | **normalised** ego-frame translation + yaw increments, MLP→tokens | **Two separate cross-attention pathways**: temporally-localised guidance xattn + separate ego-state xattn (Eq. 2). New-vs-text-slot `_needs verification_` | Input normalisation only; no zero-init/gate stated | ADE/FDE @3s, @4s | [2605.28544](https://arxiv.org/html/2605.28544v1) |
| **Metis** | Wan2.2-5B (VAE + T5 reused; **DiT frozen status `_needs verification_`**) + **separate 1B action-expert DiT** (d_a=1024, mirrors VGE depth) | NAVSIM h=8 (x,y,θ); CityWalker h=5 (x,y) | **Structured asymmetric attention mask**: action tokens attend **only to current observation**; future video tokens attend to current obs **and all future action tokens** | `_needs verification_` ("expert-specific projections") | EPDMS; MAOE + avg L2. **Backbone scaling (Table 6):** Wan2.1-1.3B 88.5 / Wan2.2-5B 88.2 / **Wan2.2-14B 89.1** navtest PDMS | [2606.15869](https://arxiv.org/html/2606.15869v1) |
| **PhyGenesis** | Wan2.1 (**variant/size never stated** `_needs verification_`), FT | 2D waypoints → 6-DoF | Trajectories **rendered to camera-view control images** → VAE-encoded → **channel-concat** with video latents; + deformable spatial cross-attn (Eqs. 1–4). **No numeric action embedding at all** | `_needs verification_` | **CtrlErr** (geometric mean rot/trans err), poses via ViPE | [2603.24506](https://arxiv.org/html/2603.24506) |
| **DriVerse** | Wan2.1 (confirmed via companion repo **WanControl**; the "I2V-14B" claim is secondary-source, `_needs verification_`), **F** — ControlNet-Transformer, `control_layers=15`, "most parameters being frozen" | (a) 360° split into **12 angular sector tokens `<Tk>` embedded into the natural-language prompt** → rides the **frozen text cross-attention slot**; (b) Trajectory-Guided Spatial Anchors via trainable ControlNet branch | ControlNet-Transformer side branch (PixArt-δ style) + text-slot tokenisation | Only anchor transparency decay `α = exp(−λδ)`; no zero-init/gate documented | **GAE** (ORB-SLAM + Sim(3) alignment, MSE metres) | [2504.18576](https://arxiv.org/html/2504.18576) · [WanControl](https://github.com/shalfun/WanControl) |
| **WoVR** *(robotics, not driving — listed for its mechanism)* | **Wan2.2-TI2V-5B** | robot actions | **Dual pathway: (a) timestep-conditioned normalization modulation, (b) actions REPLACE the text embeddings in cross-attention** — "preserves the original DiT structure while enabling frame-level control" | `_needs verification_` | `_needs verification_` | [2602.13977](https://arxiv.org/pdf/2602.13977) |
| **DriveCtrl** *(not action-conditioned)* | **Wan2.2-Fun-A14B-Control**, **F** frozen backbone + **LoRA** on 4 self-attn + 4 cross-attn + 2 FFN projections | depth maps / text / ref image — **no ego action** | Structural condition video "injected as the primary condition **before patch embedding**" (Eq. 1) | — | none | [2605.15116](https://arxiv.org/html/2605.15116v1) |

**Matrix-Game 2.0 base, precisely:** **SkyReels-V2-I2V-1.3B weights in a Wan2.1
*architecture*** (§5.1; its `config.json` `dim=1536, ffn_dim=8960, num_heads=12,
num_layers=30` is byte-identical to official
[`wan_t2v_1_3B.py:21-25`](https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/wan/configs/wan_t2v_1_3B.py),
with `in_dim=36` marking I2V). So it is Wan-*lineage*, **not** a Wan checkpoint —
record that precisely. *"We remove the text injection modules from the released
checkpoint"*, fine-tune 5k steps, add action modules → 1.8B, 120k steps. Its action
module carries the literal docstring `"""action module from
https://arxiv.org/pdf/2501.08325"""` (`wan/modules/action_module.py:40`) — i.e. it is
a direct descendant of GameFactory's design. Data ~1200 h @360p (UE 615 h, Minecraft
153 h, Sekai 85 h; GTA 574 h + Temple Run 560 h for fine-tunes); 25 FPS on one H100.
[arXiv:2508.13009](https://arxiv.org/html/2508.13009v1) ·
[HF](https://huggingface.co/Skywork/Matrix-Game-2.0)

### 2c. The interactive / game cluster — including **two more frozen-Wan systems**

This cluster is the richest source of *code-verified* gating detail, and it overturns
the "only one frozen Wan" reading.

| Work | Base + frozen? | Action | Injection | Action-pathway norm/gate | Metric |
|---|---|---|---|---|---|
| **Micro-World** (AMD) | Wan2.1-T2V-1.3B **and** Wan2.1-I2V-14B-480P — **FROZEN, code-verified** (`train_game_action_i2w.py:787-790` `requires_grad_(False)` on VAE/text-enc/transformer/CLIP; only `--trainable_modules "action"` re-enabled) | keyboard **7-dim** (W,A,S,D,Ctrl,Shift,Space), mouse **2-dim** | **Two released variants, one codebase.** (a) **adaLN**: `action_adaLN_modulation = Sequential(SiLU, Linear(action_dim, 6*dim))`, then `kwargs["e"] = kwargs["e"] + action_feat` — action's 6-way modulation **added to the timestep modulation vector** (`wan_adaln_action.py:131-142`). (b) **ControlNet**: side branch, `x = x + hints[block_id] * context_scale` (`wan_controlnet_action.py:175`) | **(a) adaLN: NONE** — grep of the file shows **zero** occurrences of `zeros_`/`zero_`. **(b) ControlNet: `before_proj` and `after_proj` both zero-init weight+bias, plus explicit scalar `context_scale`** (exposed at inference as `action_context_scale`, default 1.0) | Keyboard Precision **0.7224**, Camera Precision **0.5055** (I2W), via an IDM. 6,000 Minecraft clips |
| **LingBot-World** | **Wan2.2 I2V 14B — FROZEN** | camera (Plücker) | Per-block **FiLM**: `(1+cam_scale)·x + cam_shift` | **xavier_uniform_ on `cam_scale_layer`/`cam_shift_layer` weights, zeroing only biases** → **NOT identity at init**; perturbs the frozen backbone from step 0 | camera control 71.73 (per DreamX-World's table) |
| **Matrix-Game 3.0** | Wan2.2 (`Wan2.2_VAE.pth`); frozen split `_needs verification_` | mouse + keyboard + Plücker camera | MG2 ActionModule **+ separate camera FiLM** between self-attn and cross-attn (`model.py:601-608`) | **Zero-init weight AND bias of both `cam_scale_layer` and `cam_shift_layer`** (`model.py:1120-1123`) → identity at init; plus zero-init `proj_mouse`/`proj_keyboard` (`model.py:1098-1103`) | — |
| **Matrix-Game 2.0** | SkyReels-V2-I2V-1.3B in **Wan2.1 architecture** (config byte-identical to `wan_t2v_1_3B.py:21-25`); **full FT** | mouse `dim_in=2` continuous; keyboard multi-hot (4 universal / 2 GTA / 7 Temple Run) | ActionModule **between text cross-attn and FFN** (`causal_model.py:292-295`); mouse concat→MLP→temporal self-attn; keyboard cross-attn. All 30 blocks (foundation) / blocks 0–14 (distilled) | **Zero-init `proj_mouse` and `proj_keyboard` weight AND bias** in every block (`causal_model.py:906-914`) — **no learned gate**; `nn.LayerNorm(1024)` terminating the mouse MLP (`action_module.py:83-88`); RMSNorm QK **that silently never applies its `self.weight`** (`action_module.py:30-35`) | **Keyboard 0.91 / Mouse 0.95** (GameWorld Score) |
| **WorldPlay / HY-World 1.5** | WAN variant undisclosed (`_needs verification_`; DiT defaults are 14B-class, so the "5B" label is unreconciled); 3-stage FT | discrete keys (`action_embed_dim=8`) + camera | Keys go through the **same sinusoidal `Timesteps` embedder as the diffusion timestep**, then `temb = temb + action` (`arwan.py:415`) — rides the adaLN modulation path. Camera = PRoPE second attention branch, `hidden_states_rope + hidden_states_prope` (`:225`) | **Zero-init on BOTH**: `action_embedder.linear_2` weight+bias (`:1019-1027`); `attn1.to_out_prope[0]` weight+bias (`:1029-1039`). Paper §3.2: *"PE and a zero-initialized MLP to encode discrete keys and incorporate it into the timestep embedding"* | R_dist / T_dist (Table 2) |
| **BiWM** | Wan2.1-T2V-1.3B, **Wan2.2-TI2V-5B** (+HY1.5, LTX); fine-tuned | camera, as an **81-class vocabulary rendered as text phrases** | **Reuses the existing text cross-attention slot — zero new parameters** | none by construction | — |
| **Yume 1.0** | SkyReels-V2-14B-540P (= Wan2.1-I2V-14B config); full FT | 9 keyboard × 9 mouse states | **Action → English sentence → umT5 → existing cross-attention.** Zero new params | N/A — no action module | — |
| **minWM** | Wan2.1-T2V-1.3B (+HY1.5-TI2V-8B); fine-tuned | camera | **PRoPE inside self-attention — no new parameters** | none specified | — |
| **PlayerOne** | Wan2.1-T2V-1.3B ("Wanx2.1 1.3B", §4.1); **LoRA r=128 + last 6 blocks** | egocentric motion | 3D-conv motion encoders, **channel-concat**; camera latents added | not stated, no code (`_needs verification_`) | — |

**The cleanest natural experiment in the entire sweep — and it is unablated.**
LingBot-World and Matrix-Game 3.0 ship **line-for-line identical FiLM camera-injector
code on Wan backbones and initialise it oppositely**: MG3 zero-inits scale and shift
(identity at init); LingBot xavier-inits them (nonzero at step 0, over a *frozen* 14B
backbone). Neither paper ablates the choice. Combined with **Micro-World**, which
gives a **frozen Wan2.1 base plus both an adaLN variant (no zero-init, no gate) and a
ControlNet variant (zero-init + explicit scalar gate) in one released codebase with a
published controllability metric**, there is a **well-posed, unclaimed ablation over
{zero-init projection, learned scalar gate, norm on action embeddings} against real
baselines on a frozen Wan.** That is arguably the single most valuable experimental
opening this sweep found for D1/D2.

**Not Wan-based, from this cluster** (verified negatives): **GameFactory** — internal
undisclosed T2V, no model code ever released (repo has only `detection.py`,
`visualize.py`, dataset assets). ⚠️ *Size discrepancy: my own fetch of §6.1 read
"internal **11B**-sized"; the cluster sweep read "internal **1B**-sized". Unresolved —
check by hand before citing a size. Both agree it is internal and not Wan.*
**Matrix-Game 1.0** — HunyuanVideo I2V 17B (the switch to Wan happened at 2.0).
**Hunyuan-GameCraft 1.0/2** — HunyuanVideo-13B / 14B MoE; token-wise addition of
camera tokens before block 0 (`models.py:579`); its `CameraNet` is a strong gating
precedent — `GroupNorm(2,C)`, **zero-init final `Conv2d(96→16)`**, and
`self.scale = nn.Parameter(torch.ones(1))` applied as `camera_states * self.scale`
(the paper advertises only the scalar and never mentions the zero-init).
**GameGen-X** — own MSDiT, base frozen in stage 2, ControlNet-style InstructNet, FiLM
`ẑ = γ(f_O)⊙(z−μ)/σ + β(f_O)`; no code released. **Oasis / Oasis 2.0** — from-scratch;
25-dim action (23 binary keys + 2 camera) via a single `nn.Linear(25,1024)` **added to
the timestep embedding** (`dit.py:308-311`), no norm/gate/zero-init there; **the only
work that explicitly normalises action *inputs*** (camera bucketised to [−1,1],
`utils.py:60-81`). **WORLDMEM** — Open-Oasis; routes actions to **temporal adaLN only**
while spatial adaLN sees timestep alone (`dit.py:249-260`) — a choice absent from the
paper text. **AlayaWorld** (LTX-2.3), **Vidu**, **Seedance**, **Kling** — none Wan.

**Two corrections worth recording:** **CausVid is not Wan-based** (its widely-used
*reimplementation* is — a common misattribution). **Self-Forcing++ uses a
Wan2.1-T2V-1.3B teacher, not 14B** (§8.1); several secondary sources state otherwise.
**Odyssey's Starchild-1** is only *journalistically* reported to adapt Ovi (whose video
branch is Wan2.2-TI2V-5B) — do not cite as fact.

### 2d. The camera / drag / control-signal cluster — where the gating evidence actually lives

Not robot-action, but this is the **richest source of code-verified gating detail** in
the sweep, and it changes the verdict: **frozen-Wan-plus-zero-init-adapter is a mature,
crowded engineering pattern.**

**The single most important correlation found anywhere in this document:**

> **Whether a work gates its conditioning pathway is predicted almost perfectly by
> whether its base is frozen.** Every work that fully fine-tunes the backbone *and*
> injects by sequence/latent concatenation — FullDiT, Phantom, Wan-Move, ATI, OmniVCus,
> OmniHuman, UniVideo, SkyReels-A2 — uses **no zero-init and no gate**. Every frozen or
> near-frozen work — VACE, Uni3C, EPiC, AC3D, PostCam, UCPE, InfCam, VerseCrafter,
> RealCam-I2V — **gates**.

The structural reason: when the whole backbone moves, there is no pretrained function
to preserve at step 0, so identity-at-init buys nothing. **This is the direct,
citable justification for our `g(d)` gate — it is well-motivated *precisely because*
`f_base` is frozen.** It also reframes our failure: we are in the frozen regime but
inherited an ungated design from the fine-tuned regime.

**🔴 The most striking single find in the whole sweep — in Wan's own source:**
Wan's I2V **image cross-attention has a learned zero-init gate that ships commented
out**:
```python
# raw.githubusercontent.com/Wan-Video/Wan2.1/main/wan/modules/model.py:199
# self.alpha = nn.Parameter(torch.zeros((1, )))
```
The shipped image branch adds **un-gated**. Wan-Move's fork carries the same
commented-out line. So Wan's own authors built a gate for an additive conditioning
branch and disabled it — and *that* ungated image-conditioning path is the design our
action adapter imitates.

| Work | Base + frozen? | Signal | Injection | Gate / norm |
|---|---|---|---|---|
| **VACE** (Alibaba) [2503.07598](https://arxiv.org/html/2503.07598v2) | Wan2.1-T2V-1.3B/14B, **FROZEN** (§3.3.2: *"parameters of DiT are frozen. Only the Context Embedder and Context Blocks are trainable"*) | VCU `[T;F;M]`, `vace_in_dim=96` | Parallel side branch, hints added after strided main blocks (`vace_layers=[0,5,…,35]`, 8 of 40) | **Full ControlNet recipe**: `zeros_` on weight+bias of both `before_proj` and `after_proj` (`vace_model.py:26-31`); `x = x + hints[id] * context_scale`, `context_scale` a **user scalar, not learned** |
| **Uni3C / PCDController** (DAMO) [2504.14899](https://arxiv.org/html/2504.14899v2) | **Wan-I2V-14B, FROZEN**; branch is 0.95B vs 14B | point-cloud render (VAE) + **Plücker 6-ch** + mask | *"a simplified DiT rather than copying modules and weights from the main backbone"*, hidden 1024 vs 5120, **first 20 layers**; readout `hidden_states += controlnet_states[i]` — plain add, no scale | **zero-init ×2**: `zero_module` on all 20 `proj_out` (`controlnet.py:272-277`) and on `mask_zero_proj`; branch uses `qk_norm="rms_norm_across_heads"` |
| **DepthDirector** [2601.10214](https://arxiv.org/html/2601.10214v1) | **Wan2.2-TI2V-5B, FROZEN** (§4.1) — **our exact base** | warped depth video + occlusion mask | view tokens → linear projector → **added to noise latent**; source frame-dim concat; **LoRA rank 32** | No explicit zero-init/gate. RotErr/TransErr/CamMC |
| **VerseCrafter** [2601.05138](https://arxiv.org/html/2601.05138v1) | **Wan2.1-14B, FROZEN** — *"only the GeoAdapter is updated"* | 4D control maps | DiT-style branch, **every 5th** Wan block, *"linearly projected back to the backbone width and added as a residual modulation"* | **Copy-init, not zero-init** — *"GeoAdapter blocks initialized from weights of paired DiT blocks"* |
| **UCPE** [2512.07237](https://arxiv.org/html/2512.07237v1) | Wan, **frozen, <1% trainable (~35.6M)** | Relative Ray Encoding (not Plücker) | parallel camera-conditioned attention branch per block, tokens compressed to 1/8 dim | **YES** — *"linear projection layer with zero-initialized weights, ensuring the pretrained model remains unaltered at initialization"* |
| **PostCam** [2511.17185](https://arxiv.org/html/2511.17185v1) | **Wan2.1, "almost all parameters frozen. Only the self-attention modules are updated"** | 6-DoF extrinsics + rendered video | **New cross-attention in every block**, Q from noised latent, **K/V from both conditions concatenated along token dim** (Eqs. 4-5) | **YES** — *"projection layer is initialized to zero, effectively suppressing the influence of the camera-control signal"* |
| **InfCam** [2512.17040](https://arxiv.org/html/2512.17040v1) | Wan2.1, *"freeze the pretrained weights… train only the camera encoder and homography-guided self-attention"* | infinite homography + 16-dim camera vector | warped source latent added via residual through a conv | **YES** — warping residual *"initialized to zero"* |
| **Wan-Fun-Control** | Wan2.1/2.2-Fun, **full FT** (`--trainable_modules "."`) | control video | channel concat into widened patch-embed (`in_dim` 48 / 52 / 148) | Identity-at-init by **weight surgery** — new channels zeroed at checkpoint load (`wan_transformer3d.py:1394`) |
| **Wan-Fun-Control-Camera** | Wan2.1/2.2-Fun, **full FT** | Plücker ×4 temporal pack = **24 ch** | `SimpleAdapter` added after patch-embed | **NONE** — grep of `wan_camera_adapter.py` for `zeros_/zero_module/Norm/gate/scale` → **zero hits**; `initialize_missing_parameters` routes it to **`xavier_uniform_`**, not zeros |
| **ATI** (ByteDance) [2505.22944](https://arxiv.org/html/2505.22944v1) | Wan2.1-I2V-14B, **full FT** | 2D point trajectories | **Parameter-free latent splat** that *overwrites* Wan-I2V's 36-ch conditioning tensor (`motion_patch.py:77-150`) | **N/A — no parameters.** Acc@0.05 59.0% |
| **Wan-Move** (ali-vilab) [2512.08765](https://arxiv.org/html/2512.08765v1) | Wan2.1-I2V-14B, **full FT** | dense point trajectories | **Scatter-write of first-frame features into the conditioning latent, zero new parameters**; *"without any architecture change or extra motion modules"* | **None** (gate commented out at `model.py:199`). **Table 5 — the cost number: direct concat FVD 83.5 @ +3 s vs ControlNet FVD 84.6 @ +225 s** |
| **ReCamMaster** [2503.11647](https://arxiv.org/abs/2503.11647) | release = Wan2.1-T2V-1.3B; only `cam_encoder`/`projector`/`self_attn` trainable | 3×4 extrinsics → 21×12 | source video **frame-dim** concat; camera `Linear(12→dim)` broadcast, **added after adaLN modulation, before self-attention** | **Zero-init encoder + IDENTITY-init projector** (`torch.eye`) — ⚠️ **paper claims both are zero-init; the code disagrees** |
| **FlashMotion** `arXiv:2603.12146` | **Wan2.2-TI2V-5B** + DMD distillation to **4 steps** | trajectory maps → VAE | Trajectory Adapter with **one block per DiT block**; *"output … passed through a **zero-initialized convolution layer** and added to the corresponding DiT block"* (§3.2) | zero-init conv per block |
| **SkyReels-V2** [2504.13074](https://arxiv.org/abs/2504.13074) | Wan2.1 T2V-1.3B/14B + I2V-14B, full continued training | diffusion-forcing timestep + **FPS scalar** | FPS: `nn.Embedding(2,dim)` → MLP → **added into the 6-way adaLN modulation vector** | **YES** — `nn.init.zeros_` on `fps_projection[-1]` weight+bias (`transformer.py:835-836`); plus `zero_init_i2v_cross_attn()` zeroing every block's `cross_attn.v_img` (`:534-538`) — **added by SkyReels, absent upstream** |

**Not Wan-based, from this cluster** (mechanism still instructive): Tora/Tora2
(CogVideoX-5B in the released code; **SPADE-style adaptive-norm fuser**, explicitly
rejecting concat and cross-attn — Fig. 3, Eqs. 6/7/8 — with zero-init on the *temporal*
conv only) · MotionPro (SVD; character-for-character the same fusion module as Tora) ·
Go-with-the-Flow (CogVideoX; **no architectural injection at all** — control lives
entirely in warped initial noise) · DragAnything, RealCam-I2V, Image Conductor,
MotionCtrl (SVD/DynamiCrafter/LVDM) · EPiC, AC3D, TrajectoryCrafter, MagicMotion
(CogVideoX) · FullDiT/FullDiT2, UNIC, CameraCtrl-II (Kuaishou internal) ·
HunyuanCustom, UniVideo (HunyuanVideo) · OminiControl (**FLUX, image-only**) ·
Kling, Seedance (proprietary). **"Any2Video" does not exist — do not cite.**

**Not Wan-based** (useful negatives): Vista (SVD) · MagicDrive (SD1.5) · Panacea (SD) ·
GeoDrive, AutoAWG, WorldDrive, ReSim (CogVideoX) · DriveLaW, MAD (LTX-Video) ·
OmniDreams (Cosmos-Predict2.5) · Cosmos-Drive (Cosmos) · InstaDrive (OpenSora V1.1) ·
OmniDrive (SD3) · Orbis 2 (from scratch) · Epona / DrivingWorld / DrivingGPT (from
scratch AR) · GAIA-1/2 (proprietary).

---

## 3. Deep dive — top 3

> **Read §2a first.** This section was written before the robot-manipulation sweep
> returned. On the final evidence the three most important works are
> **RynnWorld-4D** (our composition rule, on a frozen Wan2.2-TI2V-5B),
> **GigaWorld-1** (publishes our conclusion with the cleanest injection ablation on
> Wan) and **DreamX-World** (§3.1) — all covered in §2a. The three deep dives below
> remain accurate and are the most *mechanistically detailed* of the set, but they
> are no longer the top 3 by relevance.

### 3.1 DreamX-World 1.0 — frozen-Wan conditioning on our exact base model

**Why it matters most:** it is Wan2.2-TI2V (**our exact base**), the DiT is **frozen**
during the camera-conditioning stage, and it achieves a real, measured control score.
Of the three verified frozen-Wan systems it is the only one on our base model, and its
code is the most instructive. (Micro-World is the better *experimental instrument* —
see §4 e0 — but it is on Wan2.1.)

**Frozen claim, sourced (paper, Method):**
> "we train PRoPE modules by freezing the DiT backbone and only backpropagating the
> gradients to the PRoPE parameters."

(Later, in *Event Instruction Tuning*, §3.3: "we fine-tune the full DiT while keeping
the architecture unchanged" — so the frozen claim is stage-specific, and the camera
pathway is the frozen stage.)

**The mechanism — and why it structurally cannot drown.** This is the important part
for us. Verified in code, `models/wan_transformer3d.py` (branch `master`):

```python
# L480–489, WanAttentionBlock.forward
temp_x = self.norm1(x) * (1 + e[1]) + e[0]        # adaLN-modulated block input
y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype)

if hasattr(self, 'cam_self_attn') and cam_emb is not None:
    y = y + self.cam_self_attn(temp_x, cam_emb, seq_lens=seq_lens)   # L487

x = x + y * e[2]                                   # L489 — adaLN gate on the SUM
```

Three design choices, each of which independently attacks our failure mode:

1. **The conditioning output is merged into the self-attention output *before* the
   adaLN gate `e[2]`, not added to the residual stream directly** (L487 vs L489). It
   therefore inherits the base model's own learned output scale. Our design adds the
   cross-attention output straight to the stream, where it must supply its own
   magnitude — and doesn't.
2. **Zero-initialised output projection**, with an explicit comment naming residual
   stability (`models/wan_transformer3d.py` L218–220):
   ```python
   # zero-initialize out_proj for stable residual training
   nn.init.zeros_(self.out_proj.weight)
   nn.init.zeros_(self.out_proj.bias)
   ```
3. **qk-RMSNorm on the conditioning branch** (L215–216):
   ```python
   self.norm_q = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()
   self.norm_k = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()
   ```
   Note this is qk only — the **value pathway is unnormalised here too**, exactly as
   in our diagnosis. They get away with it because of (1) and (2).

**Camera never becomes residual-stream content at all.** `prope_qkv` applies the
projective camera transform to q/k/v *inside* attention (`models/prope_utils.py`,
253 lines; called at L248), with an inverse transform applied to the output (L263).
Camera is a **geometry on the attention operator**, not an additive token signal.
This is a categorically different injection route from ours.

**Config** (`configs/dreamx-ar/config.json`): `model_type: ti2v`, `dim: 3072`,
`num_layers: 30`, `in_dim: 48`, `add_control_adapter: true`, `cam_method: "prope"`,
`attn_compress: 4`, `cam_self_attn_layers: [0…29]` (**all 30 layers**).

**They deliberately dropped the additive path.** `wan_transformer3d.py` L672–673 sets
`self.control_adapter = None` and `self.camera_projection = None` — the
Fun-Control-style `SimpleAdapter` additive injection is present in the upstream Wan
codebase but disabled here. *[inference]* This is consistent with the additive route
having been found inferior, but the repo does not say so — **`_needs verification_`**
as to their reason.

### 3.2 RynnWorld-Teleop — the explicit statistical-alignment fix, on Wan2.2-TI2V-5B

The single most direct corroboration of the *scale-matching* idea behind
`action_token_norm`, and on our exact base model.

**Eq. 3 (§3.3):**
```
x  = PatchEmbed^z_{C→D}(z_t) + α · PatchEmbed^c_{C→D}(c̃)
c̃ = (c − μ_c)/σ_c · σ_z + μ_z
```

**Their stated rationale** (quoted from §3.3):
> "Since c and z originate from different modalities, they exhibit distinct
> statistics. We maintain running estimates of the mean and standard deviation (μ,σ)
> for both signals and align c to the video latent distribution before
> patchification."

> "This ensures the additive control signal remains statistically compatible with the
> pretrained video stream throughout training."

That sentence is, in substance, our residual-drowning diagnosis stated as a design
principle: an additive conditioning signal must be *statistically compatible with the
stream it is added to*.

**Plus a gate and a zero-init**, same paragraph:
> "To preserve the generative prior, PatchEmbed^c is zero-initialized and the gating
> scalar α is initialized to a small value (e.g., 0.1), allowing the network to
> gradually incorporate the pose signal without destabilizing the pretrained weights."

**Ablation (Table 4)** — conditioning strategy:
- Concatenation fusion: **FVD 1191**
- Distribution-aligned additive: **FVD 585**
- Their comment: *"Concatenation disrupts the pre-trained latent distribution of the
  base DiT, leading to unstable synthesis."*

**Caveats for us:** (a) the base is **not frozen** — LoRA r=64 or full SFT; (b) they
report **no action-following metric**, only PSNR/SSIM/LPIPS/FVD plus downstream
policy success, so the ablation shows *fidelity*, not *control*; (c) the action is
rendered as a **depth-modulated skeleton video**, i.e. a dense pixel-space signal
already in the video modality — a much easier alignment problem than our
low-dimensional action vector.

### 3.3 DexAC-WM — the magnitude problem stated outright, and the "necessary but insufficient" result

Not Wan-based for generation (Cosmos-Predict2.5 2B; Wan2.1 is only a *frozen visual
encoder* — quote: *"each RGB frame is encoded by a frozen Wan2.1 visual encoder to
extract spatial features"*). Included because it is the **only paper found that
directly analyses conditioning-signal magnitude as the failure cause**, and because
it benchmarks the Wan Fun-Control variants.

**The problem statement (§3):** actions *"span multiple orders of magnitude"*, with a
*"10⁵ scale gap between macro-movements (wrist/camera) and micro-movements
(fingers)"*, and:
> "In a standard MLP-based action encoder, all action dimensions are projected into a
> single shared embedding before interacting with visual features … large-magnitude
> wrist and head dimensions contribute more strongly to the shared action
> representation, while low-magnitude finger dimensions are easily weakened."

**Explicit magnitude fix:**
> "raw action values are numerically very small, we multiply each dimension by a
> constant scaling factor before tokenization to enhance conditioning strength."

**The result that matters most to us (Table 4):**

| Variant | FVD |
|---|---|
| Full method | **284.40** |
| − normalisation | 418.53 |
| Naive global ×200 rescale | 371.18 |

Their conclusion: *"scale alignment is necessary but insufficient"*, because
> "Even after scaling, a vanilla MLP still compresses wrist pose, finger
> articulation, and ego motion into one shared embedding, mixing semantically
> different motion factors."

**Their injection** is a two-pathway design worth noting: local cross-attention (Eq. 7,
`Z^local = Z + Attn(Q=Z, K=A_tok, V=A_tok)`) **plus** a global AdaLN modulation
(Eqs. 8–10) derived from a learnable summary query. I.e. they use cross-attention
*and* modulation, not either alone.

**Their metric** — PCK@10 / PCK@20 on keypoints, broken out **per action family**
(wrist-dominant / finger-dominant / head-dominant, Table 3). That per-family
breakdown is a directly borrowable evaluation idea for us.

**Wan baselines underperform:** Wan2.1-Fun-1.3B-Control and Wan2.2-Fun-5B-Control
both score **FVD > 1200**; the paper notes generative baselines *"obtain moderate
reconstruction quality but show weak temporal coherence."* It does **not** describe
how those baselines inject actions — `_needs verification_` from the paper, though I
verified it from the Fun-Control source directly (table row above).

---

## 4. What this implies for us

**(a) Does anyone use a normalisation/gain fix on the action pathway? Yes —
universally, and it corroborates `action_token_norm` while also reproducing its
limitation.**

Four independent systems put explicit magnitude control on the conditioning pathway:

| System | The control |
|---|---|
| RynnWorld-Teleop | running-statistics distribution alignment to the video-latent moments + zero-init + scalar gate α=0.1 |
| DreamX-World | zero-init `out_proj` + qk-RMSNorm + rides base adaLN gate |
| EA-WM | learned per-token gate `G_ℓ` on the cross-attn output |
| DexAC-WM | per-dimension z-score + explicit constant scale-up |

Nobody presents this as a *discovery*; they present it as *engineering hygiene*. That
is itself the finding: **the field treats conditioning-pathway scale control as
mandatory, and our failing configuration omits it.** Our contribution is therefore
not "normalisation helps" — it is the **measurement** (RMS 0.01 vs 1.8–3.0, `drel`
0.44 → 0.0085 in one addition), which nobody has published.

**Crucially, DexAC-WM independently reproduces our caveat.** Our vault already flags
that `action_token_norm` gave 6–10× gain without unlocking steering, and that the
gate control moved similarly ([[../30_Knowledge/writing/writing-plan-2026-08]],
[[../00_Inbox/2026-08-01-red-team-audit]]). DexAC-WM's Table 4 says the same thing
with different numbers: naive rescaling (×200) recovers only part of the gap and is
beaten by a structured fix. **This is strong external support for the honest version
of our claim** — "scale is a real and necessary barrier, but not the whole
mechanism" — and it means we should *stop* trying to sell `action_token_norm` as the
fix and instead cite DexAC-WM as independent evidence that scale alone is
insufficient.

**(a2) The precise shape of the gap — norms on the *token* vs gates on the *output*.**

A careful distinction the sweep makes possible, and it is the sharpest version of our
novelty claim:

- **Gates and zero-inits on the conditioning branch's *output*: everywhere.**
  Zero-init is the near-universal default — `f_zero` (CompoSIA, HorizonDrive),
  adaLN-Zero (X-World, DiT), zero-convolution (ControlNet, CompoSIA's PRoPE branch),
  zero-init `out_proj` (DreamX-World), zero-init `PatchEmbed^c` (RynnWorld-Teleop),
  zero-init output projection (OmniDreams).
- **Normalisation applied to the *action representation before* it becomes tokens:
  common** — z-score (DexAC-WM Eq. 4, DiT-WAM), symlog (X-World), running-moment
  distribution alignment (RynnWorld-Teleop Eq. 3), sinusoidal/Fourier embedding
  (CompoSIA, HorizonDrive, X-World, DiT-WAM).
- **Normalisation applied to the *action tokens themselves inside the block*
  (LayerNorm/RMSNorm on the action token stream): rare, undocumented, and where it
  exists, incomplete.** The driving sweep found **zero** instances across 10 Wan-based
  driving works plus comparators. No paper in any cluster discusses whether new action
  cross-attention layers inherit Wan's internal qk-norm.
- **Learned gates are almost absent.** Across everything read, only two exist:
  Hunyuan-GameCraft's single global `nn.Parameter(ones(1))` scalar (degenerate at init
  — `0 × 1 = 0`, since the zero-init conv does the real work, and the paper advertises
  only the scalar), and Micro-World's inference-time `action_context_scale`.

**Two counter-examples, both found in code and neither mentioned in any paper — and
both incomplete in exactly our direction:**

1. **DreamX-World** applies `WanRMSNorm` to q and k on its conditioning branch
   (`models/wan_transformer3d.py` L215–216) — **qk only, value pathway unnormalised**,
   precisely the asymmetry our trace identifies.
2. **Matrix-Game 2.0** is the only work found that normalises action embeddings at
   all: `nn.LayerNorm(1024)` terminating the mouse MLP (`action_module.py:83-88`)
   plus RMSNorm QK on both action attentions — **but its local `WanRMSNorm.forward`
   (`action_module.py:30-35`) returns `x * rsqrt(mean(x²)+eps)`, declaring
   `self.weight` and never applying it**, unlike the backbone's `model.py:87` which
   does `* self.weight`. Matrix-Game 3.0 makes this explicit with a commented-out
   `# fast_rms_norm(x, self.weight, self.eps)`. **The action QK-norm is therefore
   unlearned — the one place the field does normalise the action pathway, it does so
   with a latent bug.**

So the practice exists in implementations, is undocumented in papers, and where it
exists it is **partial (qk only) or unlearned (dropped weight)**. That is a
substantially stronger version of our contribution claim than "nobody normalises".

That is a clean, defensible statement of our contribution surface: **the field
controls the conditioning branch at its output (gates) and at its input
(normalised/embedded actions), but not in the value pathway between them — which is
where we measured the loss.**

**(a3) The one dissent on zero-init, worth citing.** Orbis 2
([arXiv:2607.15898v1](https://arxiv.org/html/2607.15898v1), App. A.6 — not Wan-based,
from-scratch 512M ST-Transformer) deliberately uses "**small non-zero-initialized**
linear projections" and a "gated spatial cross-attention layer (**small non-zero-
initialized gate**)", rationale: "the pretrained STDiT backbone is minimally perturbed
at initialization." So zero-init vs small-init is a **live, under-ablated choice** —
and given the collapsed-gating result in §5.6, a gate that starts at exactly zero with
a starved gradient is precisely the failure Orbis 2's small-init hedges against.

**(b) Does anyone condition a frozen Wan? Yes — three, and none of them on robot
actions. Our slot is narrower than "frozen Wan" but still real.**

| System | Frozen Wan variant | Action | Why it isn't us |
|---|---|---|---|
| **Micro-World** (AMD) | Wan2.1-T2V-1.3B, Wan2.1-I2V-14B-480P | keyboard 7-dim + mouse 2-dim | **Discrete/low-dim game actions**, Minecraft only (6,000 clips) |
| **DreamX-World 1.0** | Wan2.2-TI2V | camera pose from keyboard | Camera is a *geometric* signal expressible as an attention transform; frozen only for the camera stage |
| **LingBot-World** | Wan2.2-I2V-14B | camera (Plücker) | Camera only |

Plus two frozen-Wan **non-action** control systems: **DriveCtrl** (LoRA on
Wan2.2-Fun-A14B-Control, depth maps) and **DriVerse/WanControl**
(ControlNet-Transformer, `control_layers=15`, "most parameters being frozen").

**So the honest framing is:** frozen-Wan-plus-adapter is an *established engineering
pattern*, not a novel one, and we should stop claiming it as such. What is genuinely
unoccupied is **frozen Wan + continuous, high-dimensional robot-action conditioning**
— and §5.4 explains why: everyone who has tried continuous actions has moved them
*off* the cross-attention path, and everyone doing robot actions on Wan has *unfrozen*
the base.

**Micro-World deserves special attention — it is the closest available baseline AND a
ready-made experimental instrument.** It ships a frozen Wan2.1 base with **two**
injection variants in one released codebase: an **adaLN** variant with (grep-verified)
**no zero-init and no gate at all**, and a **ControlNet** variant with zero-init on
both ends plus an explicit inference-time scalar `action_context_scale`. It reports
Keyboard Precision 0.7224 / Camera Precision 0.5055 via an IDM. That is a real,
runnable frozen-Wan baseline with a controllability metric — we currently have no
external baseline of that kind.

**Counterweight we must address honestly:** DriveVA ablates frozen-vs-full on a Wan2.2-
TI2V-5B driving base and finds **full fine-tuning wins** (Table 7, PDMS 90.9),
concluding "effective transfer requires adapting the video prior under joint
video-level supervision". That is the strongest published argument *against* our
frozen-base premise and should be cited and rebutted (our premise is
plug-and-play/composability and inference cost, not peak accuracy), not ignored.

This is our **closest competitor**, and it is on Wan2.2-TI2V — our base family. It is
also a partial threat to the novelty framing, so it needs a clean differentiator.
Available differentiators, all defensible:

- **Action type.** DreamX-World conditions on **camera pose** derived from keyboard
  input. Camera is a *geometric* signal that can be expressed as a transform on
  attention (PRoPE). **Robot actions cannot** — there is no projective operator for
  "close gripper". Our problem is strictly harder and their mechanism does not
  transfer.
- **Frozen scope.** Their freeze is **stage-specific**: frozen for the camera stage,
  then *"we fine-tune the full DiT"* for event-instruction tuning. Ours is frozen
  throughout.
- **Their design is evidence for our diagnosis, not against it.** They avoid drowning
  by never making conditioning a residual-stream addend (geometry-in-attention) and
  by zero-initialising the one projection that does write to the stream.

**(c0) The strongest architectural verdict in this whole sweep: continuous actions
should probably not go through cross-attention at all.**

Four independent sources, three of them with numbers, say cross-attention is the wrong
path for a *continuous* action vector — which is exactly what we feed it:

- **GameFactory** (ICCV 2025) §6.2 + Table 2. **Correction: its base is an internal
  11B T2V model, NOT Wan** — *"Our experiments are based on an internal 11B-sized
  transformer-based text-to-video diffusion model … distilled from a larger
  pretrained video diffusion model."* It is included for its mechanism, not as a Wan
  work. Its Phase 2 is structurally our setup — §5.2, verified: *"We freeze both
  pre-trained parameters and LoRA, only training the action control module with game
  videos and action signals."*
  Verified §6.2 direction: *"For discrete keyboard inputs, cross-attention outperforms
  concatenation. This suggests that category-based signal control benefits from
  similarity-based cross-attention."* Table 2 keyboard: cross-attn Cam 0.0527 / Flow
  **8.67** vs concat Cam 0.0853 / Flow **22.37**. For **continuous mouse control the
  ordering reverses and concatenation wins** (Cam 0.0685 vs 0.0798).
  The stated cause — *"cross-attention's similarity computation, which tends to reduce
  the influence of the control signal's magnitude, thereby affecting the final
  result"* — is the closest thing in print to our diagnosis.
  ⚠️ A second set of Table 2 figures (mouse-large: cross-attn 325.18 vs concat 258.93)
  came from a secondary read and I could **not** reconcile it with my own fetch —
  **cite only the keyboard numbers above until Table 2 is checked by hand.**
- **Matrix-Game 2.0**: independently splits by type — continuous mouse → **concat +
  MLP + temporal self-attn**; discrete keyboard → cross-attn. Reports Keyboard Acc
  0.91 / Mouse Acc 0.95.
- **Nano World Models** Table 3, **on RT-1** (our dataset): FiLM 40.62 < additive
  42.27 < adaLN 43.62 < **cross-attention 51.12 (worst)**.
- **DiT** Fig. 5: adaLN-Zero ≈9.6 FID vs **cross-attention ≈26**.

**And two more, both directly on Wan and both with robot actions:**
- **GigaWorld-1** Tab. 3 (Wan2.1-1.3B): cross-attn **0.1620** vs no-control **0.1576**
  vs ControlNet 0.2566 vs **channel-concat 0.3528**.
- **World Action Planner** Tab. 1: WPE, IRASim (AdaLN-Zero) and Ctrl-World (cross-attn)
  rebuilt **on the same Wan-T2V-1.3B with 2× the training steps** — cross-attn worst.

Our configuration is the intersection of every one of these warnings: **continuous
action vector, through cross-attention, into a frozen deep DiT.** Six independent
groups, four with numbers, two of them on Wan with robot actions. *[inference]* Our
residual-drowning measurement is plausibly the mechanistic explanation for a result
these groups observed empirically but did not instrument.

**Recommendation, stated plainly:** stop repairing the cross-attention arm. Run
adaLN / FiLM / channel-concat arms. The literature's own verdict — GigaWorld-1's
"spatially aligned with the noisy latent from the beginning of denoising" and the
fact that **the classic MLP(a)→timestep→adaLN recipe barely exists on Wan** while
pixel-space and concat conditioning dominate — says the winning move on Wan is to get
the action signal into the *latent stream*, not into attention.

**(c00) The "reuse the text cross-attention slot" family is a scale fix in disguise —
and it is the cheapest thing we could try.**

Five independent systems inject actions by **replacing the text context** rather than
adding a new cross-attention layer:

| System | Base | How |
|---|---|---|
| **WoVR** | Wan2.2-TI2V-5B | actions *"replace text embeddings in cross-attention"* + timestep-conditioned normalization modulation; *"preserves the original DiT structure"* |
| **BiWM** | Wan2.1-1.3B, **Wan2.2-TI2V-5B** | camera as an 81-class vocabulary rendered as text phrases; **zero new parameters** |
| **Yume 1.0** | SkyReels-V2-14B (Wan2.1-I2V config) | 9 keyboard × 9 mouse states → English sentences → umT5 → existing cross-attn |
| **DriVerse** | Wan2.1 | 360° → 12 angular sector tokens `<Tk>` embedded into the natural-language prompt |
| **Ctrl-World** | SVD | `encoder_hidden_states=action_hidden` (`ctrl_world.py:205`) |

*[inference, but I think it is the right read]*: **this sidesteps residual drowning
without anyone naming it.** The base model's text cross-attention is *already
calibrated* — its output projection was trained to contribute at a magnitude the
residual stream actually responds to. Feeding actions through that pathway inherits
that calibration for free, whereas a freshly-initialised cross-attention layer must
discover the right output scale on its own and (per our trace) does not.

This predicts something testable and cheap on our stack: **route action tokens through
Wan2.2-TI2V-5B's existing text cross-attention (replacing/augmenting the text context)
and the output RMS should land near the base's own text-conditioning RMS rather than
at 0.01.** That is a one-line diagnostic before it is even an architecture change —
measure the base's text cross-attention output RMS on the same stream and compare with
our adapter's 0.01. If the base's text path sits near 1.8–3.0 × 0.1, that is the
target magnitude, and it also gives us a *principled* value for a fixed gain instead
of a learned gate.

**(c) The single most actionable architectural lead.** DreamX-World L487–489 puts the
conditioning output **inside the adaLN-gated branch** (`x = x + (self_attn + cam_attn) * e[2]`)
rather than adding it to the residual stream on its own. That makes the conditioning
inherit the base model's *own learned per-block output scale* — which is exactly the
1.8–3.0-magnitude quantity our pathway fails to match. This is a concrete, cheap
variant to try: **route the action cross-attention output through the base block's
existing `e[2]` gate instead of adding it post-hoc.** Related open ticket:
[[../20_Tickets/feat-adapter-wan-per-frame-adaln]].

**(d) A borrowable metric.** The AV DiT world-action model (arXiv 2606.12987) defines
a clean controllability protocol we should adopt or cite: sweep the action across its
5th–95th percentile **with diffusion noise held fixed**, measure the induced
displacement, report **Spearman ρ** plus **sign-correctness** on the subset above a
detection threshold (they get ρ=+0.81 vs −0.18 for a regression baseline). This is
the same construct as our steering-cosine test but with an established framing, and
their "only score scenes above a detection threshold" move is a useful robustness
detail. DexAC-WM's **per-action-family PCK breakdown** is the second borrowable idea.

**The strongest protocol found, and the one I'd actually adopt:** Orbis 2's
**counterfactual action scaling** — scale ground-truth speeds and yaw rates by 0.5×
and 1.5×, then run an off-the-shelf pose estimator on the *generated* video and check
whether recovered ego-motion tracks the **counterfactual** command (they report ADE
1.20 m nominal vs 2.18 m at speed×1.5). This tests **action-following** rather than
reconstruction, which is exactly the sensitivity-vs-control distinction we are trying
to measure. The general driving recipe it belongs to — *run VGGT / ViPE / MonST3R /
ORB-SLAM on the generated video and compare recovered motion to the commanded
trajectory* — is a mature, citable evaluation family (CompoSIA RotErr/TransErr,
HorizonDrive ARE+DTW, PhyGenesis CtrlErr, DriVerse GAE) that our D2 eval currently
does not draw on.

**🔴 And here is a real, exploitable hole in the robot literature.** Action-following
is almost never measured directly. The complete list of genuine action-following
metrics found across the entire robot cluster:
- **Trajectory Accuracy** — SAM arm boxes → centre trajectories → **normalised DTW**
  vs reference (GigaWorld-1 §4.3)
- **Trajectory Consistency** (ABot-PhysWorld, 0.8522)
- **flow EPE + COS** directional consistency (PAVXploreRL; also A2World)
- **TCR / OPS / GEN** — VLM-judged (MiraBench)
- **PCK@10/20** per action family (DexAC-WM)

**There is no predicted-vs-commanded trajectory-error metric anywhere in the
Wan-robot-world-model literature.** Everyone else reports PSNR/SSIM/LPIPS/FVD or
downstream task success. Given MiraBench's own headline — *"visual fidelity is a poor
proxy for action fidelity"*, and *"increasing model scale does not reliably improve
action following"* (Wan2.1-14B TCR 58.0 vs Wan2.2-5B TCR 92.0) — **defining a
commanded-vs-recovered action-error metric is a cheap, defensible D2 contribution on
its own**, independent of whether our adapter works.

**(e0) A well-posed, unclaimed experiment falls straight out of this sweep.**

Three facts line up:
1. **LingBot-World and Matrix-Game 3.0 ship line-for-line identical FiLM
   camera-injector code on Wan backbones and initialise it oppositely** — MG3
   zero-inits scale and shift (identity at init), LingBot xavier-inits them (nonzero
   at step 0, over a *frozen* 14B backbone). **Neither paper ablates the choice.**
2. **Micro-World gives us a frozen Wan2.1 base with both an adaLN variant (no
   zero-init, no gate) and a ControlNet variant (zero-init + explicit scalar gate) in
   one released codebase**, with a published controllability metric.
3. Orbis 2 argues for **small-non-zero** init as a third option (§4 a3), and §5.6 says
   learned gates can collapse from gradient starvation.

⇒ **The ablation {zero-init projection · small-non-zero init · learned scalar gate ·
fixed RMS-matched gain · norm on action embeddings}, run on a frozen Wan with an
action-following metric, is unclaimed and directly serviced by existing public code
and baselines.** That is a concrete D1/D2 experiment with a built-in external
comparison, and it converts our diagnostic finding into a positive contribution.
Candidate ticket, and it supersedes further repair of the current cross-attention arm.

**(e1) Two direct hits on D3/D4 that were not on anyone's radar.**

**A working, published template for injecting the step-size `d` into Wan already
exists.** DiffSynth-Studio's `WanMotionControllerModel` takes a **scalar** bucket →
`sinusoidal_embedding_1d(256, id*10)` → 3-layer MLP → `dim*6`, and adds it **straight
into the adaLN modulation vector**:
```python
# diffsynth/pipelines/wan_video.py:1393
t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
```
with the **last linear zero-initialised** (`wan_video_motion_controller.py`, `init()`).
It hooks Wan's per-block `self.modulation = nn.Parameter(torch.randn(1,6,dim)/dim**0.5)`
chunked into shift/scale/gate ×2 (`wan_video_dit.py:226-234`). **SkyReels-V2
independently does exactly this for an FPS scalar** with the same zero-init
(`transformer.py:835-836`). This is the shape our `g(d)·Δ_φ` wants, it is
scalar-conditioned, it is zero-init, and it is already validated on Wan — **strong
prior art to build on and to cite for D3.**

**FlashMotion is the closest published work to D4, and its central result is a
warning for us.** `arXiv:2603.12146` puts a trajectory adapter on **Wan2.2-TI2V-5B**
and DMD-distils to **4 steps**. Their finding: naively distilling the generator while
leaving the trained adapter unchanged (*"SlowAdapter"*) **degrades both quality and
trajectory accuracy** (Fig. 1b) — the adapter must be **re-aligned to the few-step
generator** (hybrid diffusion + adversarial objective). Table 1: 50-step Wan2.2
FID 16.93 / FVD 152.04; 4-step DMD 24.38 / 228.33; **4-step FlashMotion 15.81 /
108.96** — recovering to better than the 50-step baseline.
**Implication:** if our shortcut adapters are trained against a multi-step teacher and
then run few-step, expect the same degradation. Budget for an adapter re-alignment
stage in D4, and cite FlashMotion as the precedent.

**(e2) Two published numbers that cut *against* the adapter framing — engage, don't
ignore.** Wan-Move Table 5: direct concat **FVD 83.5 at +3 s** vs ControlNet
**FVD 84.6 at +225 s** — a 75× inference-cost difference for a marginally *worse* FVD.
FullDiT Table 6: token-concat beats an adapter by only ~6% RotErr on the same base,
while their Table 5 shows the *training curriculum* moves RotErr 2.2×. Both are direct
evidence that architecture choice is second-order to data/curriculum in this
literature. Our D1 cost argument should lean on **composability and frozen-base reuse**,
not on a claim that adapters win on quality.

**(e3) A genuine gap in the adapter taxonomy (D1).** Across the seven-work camera
survey, **nobody uses LoRA for camera control.** LoRA appears only for motion transfer
(Follow-Your-Motion), depth-warped novel view (DepthDirector, r=32), noise warping
(Go-with-the-Flow, r=2048) and image control (OminiControl, r=4). And **four distinct
identity-at-init mechanisms exist with no published ablation between them**: zero-init
projection (VACE), identity-init `torch.eye` (ReCamMaster), copy-init from the paired
base block (VerseCrafter, FullDiT), and zero-padded channel growth (Wan-Fun). A
**copy-init vs zero-init vs learned-gate vs RMS-matched-gain ablation on a
step-size-conditioned adapter** is defensible, unclaimed, and squarely D1+D3.

**(e4) Two gate designs on axes we haven't considered, both relevant to `g(d)`.**
**AC3D** applies camera conditioning **only over the first 40% of the reverse
trajectory** (§3.3), with train-time noise from a truncated normal on [0.6, 1] — a hard
on/off schedule over `t`, structurally a `g(t)` rather than a `g(d)`. **EPiC**
multiplies its ControlNet output by a **data-derived binary visibility mask** — spatial
gating from geometry rather than learning. Both suggest `g` need not be a learned
scalar.

**(e5) Two published gate designs on frozen Wan that DISAGREE — that disagreement is
itself a publishable ablation, and it is the core of our `g(d)`.**

| | ABot-PhysWorld (Wan2.1-I2V-14B, VACE stage) | RynnWorld-4D (Wan2.2-TI2V-5B, frozen) |
|---|---|---|
| Design | **Double zero-init** — `vace_patch_embedding`, `before_proj`, `after_proj` all zeroed | **zero-init OutProj + learnable gate `tanh(g)`, g init 1** |
| Rationale | "VACE output is initially zero, so the DiT backbone produces the same output as before injection" | double zero-init causes a **"saddle-point deadlock"**; `tanh(1) ≠ 0` keeps gradients flowing "preventing the joint pathway from being trapped at the origin" |
| Deployed at | 14B, real robot data, Traj. Consistency 0.8522 | frozen Wan2.2-TI2V-5B, stage-2 adapter-only |

They contradict each other on the same backbone family. **Combined with §5.6's
collapsed-gating result** (gate gradients 2–3 orders of magnitude smaller than the
parameters they gate), RynnWorld-4D's saddle-point argument is almost certainly the
right one — and it is *precisely* the failure mode our own gate-saturation observation
records. **This settles the design of `g(d)`: non-zero-at-init gate over a zero-init
projection, not double zero-init.**

Add **LingBot-VA's `α = √(d_v/d_a)` variance-preserving initialisation** as a third
option with an explicitly stated scale-matching rationale — *"action tokens' output
distribution initially diverges significantly from the video distribution, disrupting
the joint attention mechanism"* — plus its **per-modality SNR shift** (video 5.0 vs
action 0.05). That is the closest thing in the literature to a principled derivation
of the action-pathway gain, and it belongs in our D1 taxonomy.

**(e6) An unclaimed D1 ablation nobody has run: what to do with Wan's text slot.**
Three incompatible strategies are deployed, each with an explicit rationale:
- **Hijack it** — BWM (`context = action_emb`, text disabled), Kinema4D (*"we
  intentionally replace the text embedding with the VAE latents of robot sequences"*),
  WoVR, BiWM (*"it leaves the pretrained input distribution intact and makes
  fine-tuning substantially more stable and data-efficient"*).
- **Zero it** — A2World (`c = 0`), Cosmos (`video2world_action.py:199`).
- **Leave a dummy prompt and inject elsewhere** — World Action Planner (App. D.1.1).

Per §4 c00, the hijack option is the one that inherits the base's *calibrated* output
scale, so it is the cheapest test of our drowning hypothesis. **No one has ablated
these three against each other.**

**(e) Negative-space contribution.** Official `Wan2.x-Fun-Control` — Alibaba's own
control variants — inject control by **raw channel-concat** and, for camera, a
**completely ungated, un-normalised additive adapter** (`wan_camera_adapter.py` has
no initialisation call at all; `wan_transformer3d.py` L912–914 is a bare `u + v`).
They also require changing `patch_embedding.in_dim`, so the base cannot be frozen.
And DexAC-WM measures them at **FVD > 1200**, far behind. *[inference — the causal
link between "no gating" and "FVD>1200" is not established by any source]*, but the
juxtaposition is a legitimate motivating observation for the thesis.

---

## 5. The mechanism literature — why this happens

Not Wan-based, but this is the evidence that turns our measurement from an anecdote
into an instance of a known law. Four independent strands.

### 5.1 The residual stream inflates with depth — so any fixed-scale branch loses

**Rethinking Cross-Layer Information Routing in Diffusion Transformers**
(arXiv:2605.20708, [v2](https://arxiv.org/html/2605.20708v2)), §3 / Fig. 2, on
SiT-XL/2 at 600K iters over 4096 ImageNet samples:

> "The forward hidden-state magnitude grows monotonically from ∼15.5 at block 1 to
> ∼1576 at block 28, corresponding to roughly 100× inflation."

> "This growth forces deeper blocks to produce ever-larger raw outputs in order to
> retain influence over the residual stream, echoing the *PreNorm dilution*
> phenomenon characterized in LLMs."

**This is the law our adapter is losing to, measured in a DiT.** Their fix (DAR:
learnable timestep-adaptive softmax aggregation over sublayer history) gets SiT-XL/2
to FID 7.56 vs 9.67 and matches converged baseline quality with 8.75× fewer
iterations. They never discuss adapters or injected conditioning.

### 5.2 The calibration number — what a *healthy* branch contributes

**Do Language Models Use Their Depth Efficiently?** (Csordás, Manning, Potts,
[arXiv:2505.13898v2](https://arxiv.org/html/2505.13898v2)) measures **exactly our
quantity**: ‖a_l‖₂/‖h_l‖₂ (attention-sublayer output over residual stream). Fig. 2a,
Llama 3.1 70B on GSM8K: **early layers ≈ 0.10–0.15**, with "a sharp drop near the
middle".

**This is the most defensible way to state our number.** Our branch: ~0.01 / ~2.0 ≈
**0.005** — roughly **20–30× below** the published healthy range. *[inference — the
comparison across architectures (LLM vs video DiT) is ours, not theirs.]*

### 5.3 Adapters are *generically* quieter than the stream they write to

**The Hidden Space of Transformer Language Adapters** (Alabi, Mosbach, Eyal, Klakow,
Geva; ACL 2024 Long, pp. 6588–6607,
[aclanthology](https://aclanthology.org/2024.acl-long.356/) ·
[arXiv:2402.13137](https://arxiv.org/abs/2402.13137)), §4.1 / Fig. 2:

> "adapters introduce updates that are substantially smaller in magnitude compared to
> the residual stream representations."

> "the norm of adapter outputs in the last layers is larger for Arabic and Hebrew
> than for German and French, suggesting that larger updates are needed to steer
> predictions to more distant languages."

Method: average L2 norm of adapter output vs FFN vs full layer, 6,500 FLORES-101
tokens, 24-layer GPT-NeoX-arch decoder + mBERT. **The second quote is the useful
one:** bigger required behavioural change ⇒ bigger required adapter norm. Action
control is a large behavioural change; a 0.5% relative norm is prima facie
inconsistent with achieving it. *[inference]*

### 5.4 Cross-attention is independently the *worst* conditioning path in a DiT

This is the cluster that most directly indicts our architecture choice.

| Source | Setting | Result |
|---|---|---|
| **GameFactory** (ICCV 2025), [arXiv:2501.08325](https://arxiv.org/pdf/2501.08325), §6.2 + Table 2 | action-controllable video gen; **Phase 2 freezes pretrained params *and* LoRA, training only the action module** (§5.2) — structurally our setup | **Flow metric, continuous (mouse-large) control: cross-attention 325.18 vs concatenation 258.93.** Recommend cross-attn for *discrete*, concat for *continuous* |
| **DiT** (Peebles & Xie, ICCV 2023), [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.09748), §3.2 + **Fig. 5** | class conditioning, ImageNet 400K iters | adaLN-Zero ≈**9.6 FID**; adaLN ≈25; **cross-attention ≈26**; in-context ≈35 (values read off Fig. 5, approximate). Cross-attn also ~15% Gflops overhead |
| **Nano World Models** ([arXiv:2605.23993v2](https://arxiv.org/html/2605.23993v2)), **Table 3** | **action injection on RT-1** and PushT | RT-1 FID: FiLM **40.62** (best), additive 42.27, adaLN 43.62, **cross-attention 51.12 (worst)**. PushT: additive 23.89 best, cross-attn 28.64 |
| Hunyuan-GameCraft ([arXiv:2506.17201](https://arxiv.org/abs/2506.17201)) | camera/action control | actions injected by **token addition after patchification**, not cross-attention (search-level; `_needs verification_`) |
| From Virtual Games to Real-World Play ([arXiv:2506.18901](https://arxiv.org/abs/2506.18901)), Table 4 | action-injection ablation | adaLN **90.0%** success vs self-attn 78.3% vs **cross-attn 77.3%** — ⚠️ `_needs verification_`, search summary only |
| IOI ([arXiv:2606.23296](https://arxiv.org/abs/2606.23296)) | action-injection ablation | FVD: additive 87.31 → concat 67.89 → cross-attn 62.56 → **AdaLN 56.47** — ⚠️ `_needs verification_`, PDF fetch failed |

**GameFactory §6.2, verbatim — the closest thing to our diagnosis in print:**
> "This may be due to cross-attention's similarity computation, which tends to reduce
> the influence of the control signal's magnitude, thereby affecting the final
> result."

Note their named mechanism is **softmax convexity** — the attention output is a convex
combination of V, so its scale is bounded by ‖V‖ regardless of how important the
condition is. That is *complementary to and compounds with* our residual-scale
finding. And their recommendation maps onto us precisely: **continuous action vectors
are exactly the case they say cross-attention handles badly.**

### 5.5 AVID used the strong path; we used the weak one

Re-read of [AVID v2](https://arxiv.org/html/2410.12822v2) (Rigter, Gupta, Hilmkil,
Ma; RLC/RLJ 2025) sharpens our own vault note:

- **Action injection is adaLN, not cross-attention** (§3.3): actions embedded per
  timestep (embedding table for discrete, linear for continuous), concatenated with
  the diffusion-step embedding, *"processed by an MLP to compute the scale and shift
  parameters, γ_τ and β_τ"*, applied after each 3D conv. **AVID deliberately used the
  strong conditioning path.**
- **Fusion is an output-level convex mask** (Eq. 5), *not* a residual-scale gate:
  `ε_final = ε_pre ⊙ m + ε_adapt ⊙ (1−m)`, m sigmoid-bounded, `m ∈ R^{T×h×w×1}`.
- **No mention anywhere of conditioning magnitude, scale mismatch, or the adapter
  being too quiet** — confirmed absent.
- **Table 3 ablation:** RT1 Large (145M) AVID Action-Err-Ratio **1.609** / FVD 39.3;
  No Mask 1.769 / 44.1; No Conditioning 1.775 / 36.2. On Coinrun 500k Large the mask
  is marginally *worse* than No Mask on every metric (1.154 vs 1.136).
- Fig. 4(d) + §4.3: *"On RT1 AVID has a higher mask value, and therefore uses the
  pretrained model more heavily than on Coinrun."*
- *[inference]* AVID's high RT1 mask value **is** a measurement of the frozen model
  dominating the adapter — but they read it as a feature (strong prior) rather than a
  pathology. Their Action-Err-Ratio 1.609 means action error is 60% worse than the
  ground-truth-action reference *even with* the mask. That is arguably AVID's own
  "sensitivity without control", never named as such.

### 5.6 Counterweight — learned gates can collapse

**When Adaptation Fails: A Gradient-Based Diagnosis of Collapsed Gating in
Vision-Language Prompt Learning** ([arXiv:2605.09549v1](https://arxiv.org/html/2605.09549v1)),
§IV-B1 / Fig. 2 / Table IV:

> "gate gradients remain 2-3 orders of magnitude smaller than prompt gradients across
> datasets and seeds"

Magnitude gap 2.60±0.02 (AdaptiveBiMaPLe). They define **collapsed gating** — gates
that "converge to stable, near-constant values, showing no input-dependent
variation" — and **propose no working fix**; gradient-balancing (App. D-2) and
reviving adaptive gating (App. D-3) both failed. Measured only on frozen
CLIP-ViT-B/16 prompt learning; **no diffusion models**.

**Direct operational warning for us** *[inference]*: if we respond to the drowning
finding by bolting a *learnable* scalar gate onto the cross-attention output, this is
prior evidence the gate may simply sit at its initialisation because its own gradient
is starved. A **fixed** gain, or **normalising the cross-attention output to a target
RMS**, is the more robust intervention; a learned gate must be *monitored*, not
assumed. This is consistent with our own gate-saturation observation in
[[../00_Inbox/2026-08-01-red-team-audit]].

### 5.7 The asymmetry nobody has addressed

Every published gating fix — adaLN-Zero (DiT §3.2, "initializes the full DiT block as
the identity function"), ControlNet zero-convolution ([§3.1, Eq. 3](https://arxiv.org/html/2302.05543):
"1×1 convolution layer with both weight and bias initialized to zeros"), Flamingo
tanh gating (α init 0, **Table 3 row (iii): 70.7 → 66.5 without it**), LLaMA-Adapter
zero-init attention gating ([Eq. 7](https://arxiv.org/html/2303.16199v3), **Table 5:
Random-Init 40.77% → Zero-Init 83.85%**) — is designed to keep the injected branch
**quiet at initialisation**.

**None of them guarantees it becomes loud enough at convergence.** No paper found
proposes "diagnose that the branch is too quiet at convergence and explicitly
renormalise it". *That asymmetry is the seam our contribution sits in.*

One inference-time exception worth citing: **Vid2World**
([arXiv:2505.14357v3](https://arxiv.org/html/2505.14357v3), §4.2) uses **causal
action guidance** — train with action dropout prob p, sample with
`ε_guided = (1+λ)ε_cond − λ·ε_ucond` where ε_ucond drops the most recent action.
**Table 2: FVD 29.4 → 25.8, FID 7.07 → 6.84, SSIM 0.824 → 0.840.** This *amplifies*
action conditioning at inference without ever diagnosing why it was weak.

---

## 6. Gaps / not verified

- **Fast-WAM** (arXiv 2603.16666): frozen-vs-finetuned status of the Wan2.2-5B video
  DiT, action-pathway normalisation, and dataset all **`_needs verification_`** —
  the alphaXiv overview did not state them and I did not reach the full text.
- ~~Ctrl-World base model~~ — **RESOLVED, and it is not Wan.** Verified from
  [arXiv:2510.10125v1](https://arxiv.org/html/2510.10125v1): *"We initialize our model
  with the pretrained 1.5B Stable-Video-Diffusion (SVD) model"*; **not frozen** —
  *"we only newly initialize an action-projection MLP for the input actions and keep
  other parameters unchanged at initialization. Then this action-conditioned world
  model is fine-tuned with diffusion loss."* Action = **7-dim Cartesian robot-arm
  pose** → 3-layer MLP → **1024-dim** embedding; injected by **frame-wise
  cross-attention in the spatial transformer** (*"allowing the visual tokens of each
  frame to attend to its associated pose embedding"*). **No zero-init or gating.**
  Metrics are PSNR/SSIM/LPIPS/FID/FVD only — no action-specific controllability
  metric. DROID (95,599 trajectories, 564 scenes).
  **Why it matters to us:** this is the closest published system to our design
  (continuous robot action → MLP → cross-attention, no gating) — and it **fine-tunes
  the whole backbone**. It is evidence that this injection route is viable *only*
  when the base is trainable, which is precisely the frozen-base constraint we are
  fighting. *[inference]*
- **MiraBench** (arXiv 2605.29360): I read only the abstract. Its "Action-Following
  Fidelity" metric definition and its list of evaluated base models are
  **`_needs verification_`** — but it is likely the right benchmark to position our
  action-following numbers against, and its headline claim (*"visual fidelity is a
  poor proxy for action fidelity"*, plus pervasive "optimism bias") directly supports
  our sensitivity-vs-control distinction.
- **DreamX-World's reason for disabling the additive `control_adapter`** — inferred,
  not sourced.
- **Wan Fun-Control "base cannot be frozen"** — inferred from the channel-concat
  changing `patch_embedding` input dim; not stated in any doc I read.
- **τ₀-WM action dimensionality** — not stated in the paper.
- **EA-WM zero-init** — not mentioned; only the learned gate is described.
- I did **not** find any paper reporting a cross-attention-output-RMS vs
  residual-stream-RMS measurement inside a video DiT. If that holds up, the diagnostic
  itself is novel.
  - **Verified negative worth recording:** a search-engine summary attributed to the
    Motif-Video 2B technical report ([arXiv:2604.16503v2](https://arxiv.org/html/2604.16503v2))
    a quantitative analysis — "cross-attention signal accounts for 7.6% of the
    self-attention residual magnitude on average, max 21.7%, min 5.2%", cosine
    ≈ −0.008. **I fetched the document and those numbers are not in it.** Motif-Video
    has only qualitative attention heatmaps (Figs. 4–5) and the token-imbalance
    argument (§3.3). **Do not cite those figures** — they appear to be a search-summary
    fabrication. This was the single closest published match to our protocol, and it
    does not exist.
- **GameFactory base-model size:** my fetch of §6.1 read "internal **11B**-sized"; the
  cluster sweep read "internal **1B**-sized". **Unresolved — check by hand before
  citing a size.** Both readings agree it is internal, undisclosed, and **not Wan**,
  and that no model code was ever released (repo has only `detection.py`,
  `visualize.py`, dataset assets), so its action module cannot be code-verified.
- **GameFactory Table 2 numbers:** two irreconcilable readings (see §4 c0). Only the
  keyboard row (cross-attn Flow 8.67 vs concat 22.37) is verified by my own fetch.
- **Matrix-Game 3.0**: frozen-vs-trained split unknown; and its released-5B config is
  inconsistent — README claims a 5B release + 2×14B MoE, but the checked-in
  `wan/configs/config.py` reads `dim=5120, num_layers=40, num_heads=40,
  ffn_dim=13824, in_dim=48` (A14B-class, **not** the TI2V-5B shape `dim=3072`,
  30 layers). `_needs verification_`.
- **WorldPlay / HY-World 1.5**: exact Wan variant unreconciled — repo names only
  `wan_transformer` / `wan_distilled_model`, and DiT defaults
  (`num_attention_heads=40, attention_head_dim=128, ffn_dim=13824, num_layers=40`) are
  14B-class, so the "5B" label does not match.
- **GameGen-X and PlayerOne** action-path initialisation — neither released model code.
- **Micro-World**: the HuggingFace "18B" figure for an I2V-14B base is unexplained.
- **No public code** was available for CompoSIA, X-World, DriveVA, DriveWAM, Metis or
  HorizonDrive, so every driving mechanism claim above is **paper-sourced, not
  code-verified**.
- **All five cluster sweeps have now reported.** Verified **not** Wan: Vid2World
  (DynamiCrafter), Ctrl-World / Persistent-RWM / Mem-World / Mask2Real-WM (SVD),
  EnerVerse + EVAC (DynamiCrafter), RoboTransfer (SVD), WorldVLA + RynnVLA
  (Chameleon, autoregressive VQ), Genie-Envisioner / GE-Sim (LTX-Video or Cosmos),
  BridgeV2W (CogVideoX-5B-I2V), WEAVER (from scratch), UWM (from scratch), IRASim
  (from scratch Latte-DiT), iVideoGPT (from scratch), RoboDreamer (AVDC/Imagen),
  DSWAM and HiMem-WAM (no Wan mention).
- **Unresolved freeze scopes:** τ₀-WM (fine-tuned by inference from lr/scale, not
  stated), iMaC (Wan2.2 variant and freeze policy never stated), GigaWorld-1 (stages
  1–2 read as full training, stage-3 LoRA described as optional — exact frozen scope
  ambiguous), Fast-WAM, Matrix-Game 3.0, UNIVERSE, Metis.
- **iMaC** is ControlNet-shaped but **does not report zero-init** of its control
  patchifiers — notable omission, `_needs verification_`.
- **Action Images** contradicts itself on the base: main text §4 says Wan2.2, App.
  Training Details says Wan2.1-I2V-14B-480P. Treat the appendix as authoritative.
- **LingBot-VA**: released code is the *shared-backbone* variant; the paper describes
  a *separated dual-stream MoT*. Its RoboTwin baseline numbers are truncated in the
  HTML — `_needs verification_`.
- **Kinema4D** action-following metric: tables did not render in the available HTML.
- **RoboWorld**: whether its action cross-attention uses new layers or Wan's text slot
  is not stated.

### ⚠️ Paper-vs-code discrepancies — verified, and directly relevant to our citations

**Do not cite any of these mechanisms from the paper alone.** All verified in source:

| Work | Paper says | Code does |
|---|---|---|
| **ReCamMaster** | both encoder and projector "zero-initialized" | encoder `zero_()`, **projector `torch.eye`** (identity-init) |
| **SkyReels-A2** | supplementary cross-attention layers | `zero_module` + `TagAttention` exist but are **never called** — dead code |
| **Phantom** | two-branch VAE + CLIP design | release uses **only** the VAE branch |
| **Follow-Your-Pose** | zero-initialised encoder | **no zero-init anywhere** in `unet.py` |
| **Tora** | Eq. 7 is `γ·h + β + h` | code is `h + GroupNorm(h)·γ + β` |
| **Wan-S2V** | marketing copy claims AdaIN | paper describes **cross-attention only** |
| **VACE** | Context Embedder mask-weights zero-init, latent-weights copied from base | released `vace_patch_embedding` is a fresh `Conv3d` with **default init** |
| **DreamX-World** | no mention of zero-init or qk-norm | **both present** (`wan_transformer3d.py:215-220`) |
| **Matrix-Game 2.0** | RMSNorm on action QK | local `WanRMSNorm.forward` **declares `self.weight` and never applies it** |

This cuts both ways for us: it means (a) our own claims must be code-verified, and
(b) **the "nobody normalises the action pathway" claim is a claim about papers; the
code record is messier and partially contradicts it** (see §4 a2).

### ⚠️ Metric-comparability warning — do not build a single leaderboard

At least **four incompatible RotErr/TransErr conventions** are in play: RE10K-GT-pose
style (RotErr ~0.4–1.6: RealCam-I2V, CamCloneMaster, EPiC, CamI2V), GLOMAP-on-WebVid
degrees (ReCamMaster 1.22), ParticleSfM with normalised translation (AC3D 0.035), and
VGGT/ViPE-estimated (Uni3C ATE/RPE/RRE, PostCam, UCPE). Trajectory metrics are equally
fragmented: TrajError (Tora) ≠ ObjMC (MotionCtrl 28.9 / DragAnything 305.7) ≠
MD-Img/MD-Vid (MotionPro) ≠ EPE (Wan-Move 2.6) ≠ Acc@0.05 (ATI) ≠ mask/box IoU
(FlashMotion) ≠ PCK (DexAC-WM). **Any comparison table in the thesis must state
tracker, pose estimator, resolution and frame count per block**, or it will be wrong.

---

## Sources

- DreamX-World 1.0 — https://arxiv.org/abs/2606.16993 · https://github.com/AMAP-ML/DreamX-World · https://huggingface.co/GD-ML/DreamX-World-5B
- RynnWorld-Teleop — https://arxiv.org/abs/2607.06558
- EA-WM — https://arxiv.org/html/2605.06192
- τ₀-WM — https://arxiv.org/html/2606.01027v1
- Light-WAM — https://arxiv.org/html/2606.08242v1
- Fast-WAM — https://arxiv.org/html/2603.16666v1 · https://www.alphaxiv.org/overview/2603.16666
- DreamZero — https://arxiv.org/html/2602.15922v1
- VideoX-Fun (Wan Fun-Control source) — https://github.com/aigc-apps/VideoX-Fun
- DexAC-WM — https://arxiv.org/html/2606.27325v1
- AV DiT World-Action Model — https://arxiv.org/html/2606.12987
- Ctrl-World — https://arxiv.org/abs/2510.10125
- MiraBench — https://arxiv.org/abs/2605.29360
- Awesome-WAM reading list — https://github.com/OpenMOSS/Awesome-WAM

**Interactive / game cluster:** Matrix-Game 2.0 https://arxiv.org/html/2508.13009v1 ·
https://huggingface.co/Skywork/Matrix-Game-2.0 — Matrix-Game 3.0 https://arxiv.org/html/2604.08995 —
Micro-World (AMD) — LingBot-World — WorldPlay / HY-World 1.5 — BiWM — Yume 1.0 —
minWM — PlayerOne — GameFactory https://arxiv.org/html/2501.08325v2 ·
https://github.com/KwaiVGI/GameFactory — Hunyuan-GameCraft https://arxiv.org/abs/2506.17201 —
GameGen-X — Oasis / WORLDMEM

**Driving cluster:** CompoSIA https://arxiv.org/html/2603.12864v1 · HorizonDrive
https://arxiv.org/html/2605.11596 · X-World https://arxiv.org/html/2603.19979v1 ·
CausalDrive https://arxiv.org/html/2606.15341v1 · DriveVA https://arxiv.org/html/2604.04198 ·
UNIVERSE https://arxiv.org/html/2607.05133v1 · DriveWAM https://arxiv.org/html/2605.28544v1 ·
Metis https://arxiv.org/html/2606.15869v1 · PhyGenesis https://arxiv.org/html/2603.24506 ·
DriVerse https://arxiv.org/html/2504.18576 · WanControl https://github.com/shalfun/WanControl ·
DriveCtrl https://arxiv.org/html/2605.15116v1 · WoVR https://arxiv.org/pdf/2602.13977 ·
Orbis 2 https://arxiv.org/html/2607.15898v1

**Robot cluster:** ABot-PhysWorld https://arxiv.org/abs/2603.23376 ·
https://github.com/amap-cvlab/ABot-PhysWorld — GigaWorld-1 https://arxiv.org/html/2607.02642v1 —
RynnWorld-4D https://arxiv.org/html/2607.06559v1 — X-WAM https://arxiv.org/html/2604.26694v1 —
LingBot-VA https://arxiv.org/abs/2601.21998 · https://github.com/robbyant/lingbot-va —
PAVXploreRL https://arxiv.org/html/2607.16602v2 — World Action Planner https://arxiv.org/html/2607.27599 —
Kinema4D https://arxiv.org/html/2603.16669v1 · https://github.com/mutianxu/Kinema4D —
iMaC https://arxiv.org/abs/2606.09813 — RoboWorld https://arxiv.org/html/2607.01060 —
Action Images https://arxiv.org/html/2604.06168v2 — Masked Visual Actions https://arxiv.org/html/2607.19343 —
BWM https://github.com/boundless-large-model/boundless-world-model — Motus https://arxiv.org/abs/2512.13030 —
SANTS https://arxiv.org/abs/2605.27947 — BiWM https://arxiv.org/html/2606.10135v1 —
DreamDojo https://arxiv.org/abs/2602.06949 — OSCAR https://arxiv.org/abs/2606.04463 —
A2World https://arxiv.org/abs/2606.29501 — Mask2Real-WM https://arxiv.org/abs/2607.04546 —
Persistent Robot World Models https://arxiv.org/abs/2603.25685

**Mechanism literature:** Xu et al. cross-layer routing https://arxiv.org/html/2605.20708v2 ·
Csordás et al. depth efficiency https://arxiv.org/html/2505.13898v2 · Alabi et al. ACL 2024
https://aclanthology.org/2024.acl-long.356/ · Nano World Models https://arxiv.org/html/2605.23993v2 ·
DiT https://ar5iv.labs.arxiv.org/html/2212.09748 · ControlNet https://arxiv.org/html/2302.05543 ·
Flamingo https://ar5iv.labs.arxiv.org/html/2204.14198 · LLaMA-Adapter https://arxiv.org/html/2303.16199v3 ·
collapsed gating https://arxiv.org/html/2605.09549v1 · Vid2World https://arxiv.org/html/2505.14357v3 ·
Motif-Video 2B https://arxiv.org/html/2604.16503v2 (verified negative)
