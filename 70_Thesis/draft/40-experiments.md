---
section: experiments
status: drafting
deliverable: D2/D3
last_updated: 2026-07-24
sources:
  - "[[../../30_Knowledge/writing/ablation-axes]]"
  - "[[../../50_Decisions/open/second-dataset-action-informativeness]]"
  - "[[../../50_Decisions/decided/param-matched-adapter-comparison-definition]]"
  - "[[../../50_Decisions/decided/per-sample-frame-stride-sampling]]"
  - "[[../../30_Knowledge/tech/frame-stride-conditioning]]"
  - "[[../../30_Knowledge/experiments/20260724-metaworld-cap-shift-triangle-base-parity]]"
---

# 4. Experiments

> Setup and protocol only. Observed numbers live in ch. 5 and must cite
> runs. Planned runs are tickets, not results (Part 12).

## 4.1 Datasets and preprocessing

The study uses **two dataset families, chosen for opposite
action-informativeness**, and evaluates the adapter on **three frozen base
backbones of differing strength/paradigm** (§4.1.4). The full axis rationale
is in [[../../30_Knowledge/writing/ablation-axes]].

- **MetaWorld** (§4.1.1) — scripted manipulation demonstrations. The
  redundant-action **control**: given the observation, the future is largely
  determined *without* the actions, so a strong frozen base predicts it well
  and the adapter has little to gain from conditioning. This is the anchor for
  the base-parity diagnosis (§5.1.x).
- **ACWM-Phys** (§4.1.2) — a simulated physical-interaction benchmark where
  the commanded action *determines* the future (pusher target, joint
  torques). The action-**informative** counterpart, and the primary vehicle
  for the D2 claim that adapters can make a frozen base action-following. The
  move from MetaWorld to ACWM-Phys is the pre-registered dataset decision
  [[../../50_Decisions/open/second-dataset-action-informativeness]], triggered
  by the measured base-parity on MetaWorld (§5.1.x).

### 4.1.1 MetaWorld clip windowing and the `fs` anchor (drafting)

> The `fs` convention below is **specific to the DynamiCrafter base**
> (§4.1.4); the Wan2.2 base uses diffusion forcing with no `fs` channel
> (§4.1.3). It is retained here as the preprocessing for the DynamiCrafter
> arm of the base-backbone axis.

We train on a vendored MetaWorld HDF5 collection
(`<task>/<episode>/{pixels, action, ...}` with `pixels ∈ [T, H, W, 3]`
uint8 and `action ∈ [T, A]` float32). Each item is a sliding window of
`traj_len` consecutive frames sampled at a contiguous index range
`[start, start + traj_len)`; only the clip **start index** is
randomised per sample. We do **not** subsample frames within a clip
(no slice stride), and we do not vary the frame-stride / fps
conditioning per sample.

The frozen DynamiCrafter UNet exposes an `fs` scalar conditioning
channel, embedded via its own `fps_embedding` MLP and added to the
time embedding. Its pretrained anchor is `default_fs = 10` (config
`external_repos/avid/.../dynamicrafter_512.yaml`). We feed
`fs = fps = frame_stride = 10` unconditionally — matching the
convention used by the AVID team's MetaWorld data module, which writes
the same constants and reads frames contiguously. This places the
base's fps channel **at its trained anchor** by construction and
sidesteps the `fps_condition_type` (`"fs"` vs `"fps"`) semantic
ambiguity: under either interpretation, `10` is the value the base
was tuned to operate at.

This is a deliberate design choice rather than a fallback. We
considered (i) per-sample stride sampling with summed action
aggregation, and (ii) per-sample stride with a principled
in-distribution mapping `fs(k) = (source_fps_DC / fps_MetaWorld) · k`,
and rejected both for the baseline because D2's contribution surface
is *action-conditioned dynamics*, not *temporal-resolution
robustness*. The temporal variation that the random clip start index
provides is sufficient at this stage. Revisit triggers (e.g. observed
brittleness to inference-time fps changes; evidence that a constant
`fs` masks an identifiable `Δ_φ` failure mode) and the full reasoning
are recorded in
[[../../50_Decisions/decided/per-sample-frame-stride-sampling]].

A direct consequence of this choice is that the action interface
remains the simple per-step convention: each predicted transition
consumes the single action that produced it, so the `(B, T, A)`
action tensor consumed by the action-conditioning path of `Δ_φ` is
shape-stable across the dataset. No stride-induced action dropping or
aggregation is needed for the baseline.

The complementary design question — whether `Δ_φ` should ever
explicitly consume `fs` — is treated as a framework-level boundary in
§3.3 and is held outside the composition rule for the baseline.

### 4.1.2 ACWM-Phys (the action-informative benchmark) (drafting)

We add ACWM-Phys (HF `t1an/ACWM-Phys`; Xue et al., arXiv:2605.08567) as the
action-informative counterpart to MetaWorld. Its environments are built so the
commanded action determines the outcome, giving high mutual information
between the action sequence and the future *given* the anchor frame — the
property MetaWorld's scripted demonstrations lack. We use **three
environments spanning action dimensionality**: Push Cube (`da=2`, rigid-body
pusher target), Robot Arm (`da=7`, per-joint angle deltas), Reacher (`da=2`,
two-link joint torques). Each provides ~1500 in-distribution training
trajectories plus 100 in-distribution and 100 out-of-distribution test
trajectories, at a fixed 66-frame horizon, released as per-episode `mp4`
(1024²) with a `metadata.pt` of per-step actions. Push Cube (da=2) vs Robot
Arm (da=7) doubles as the action-dimensionality probe for the injection
choice (§4.4); Reacher is a cheap second-kinematic control. Particle and
deformable environments are omitted — physics-regime breadth is the source
benchmark's contribution, not ours.

An `ACWMPhysTranslator` decodes clip windows from the `mp4`s on demand and
emits the **same clip contract** as the MetaWorld loader (video tensor,
per-window summed actions, and the latent-cache identity keys), so the entire
downstream pipeline — windowing, latent caching, the diffusion-forcing
preprocessor, evaluation — is shared unchanged across both dataset families.

### 4.1.3 Wan2.2 diffusion-forcing preprocessing (the primary base) (drafting)

For the Wan2.2-TI2V-5B base (§4.1.4) we use **diffusion forcing**: within a
clip, the first observation frame is kept clean (per-latent-frame timestep 0)
while the future frames are corrupted at a per-sample noise level σ; the model
predicts the rectified-flow velocity `v = ε − z0` and the loss is masked to
the predicted (future) frames only. Frames are encoded by the frozen Wan-VAE
(48-channel latent; 4× temporal, 16× spatial compression) to `z0`. We train on
**65-frame windows** (17 latent frames — the near-full 66-frame ACWM episode)
at **768² square** (`max_area 589824`), which is inside the base's native
resolution regime; on this domain the frozen base produces coherent video
directly, so no aspect-ratio letterboxing is applied (a base-coherence check
performed per new visual domain). Actions enter as **per-frame tokens**
(`action_seq`) consumed by the adapter's cross-attention; text conditioning
uses precomputed T5 prompt embeddings so no text encoder runs during training.
Because the frozen-VAE encode dominates the training step (§3.5), latents are
precomputed once into an on-disk cache keyed on the clip identity + geometry.

### 4.1.4 Base backbones (the base-strength axis) (drafting)

The frozen base is itself an experimental axis (§4.4): the base-parity
diagnosis (§5.1.x) attributes the adapter's collapse to the *strength* of the
base — a base that already predicts the future well leaves the adapter nothing
to earn beyond cloning it, whereas a weaker base leaves a genuine residual. We
therefore compare the same output adapter across bases of differing strength
and generative paradigm:

| Base | paradigm | ~size | role |
|---|---|---|---|
| Wan2.2-TI2V-5B | flow matching (velocity) | 5B | strong — primary |
| SkyReels-V2-1.3B | flow matching (velocity), Wan-VAE lineage | 1.3B | weak, same paradigm |
| DynamiCrafter | diffusion (noise) | ~1.5B | weak, older paradigm (AVID-native) |

Wan-5B vs SkyReels-1.3B isolates **base strength** with no paradigm confound
(both flow, both diffusion-forcing); SkyReels-1.3B vs DynamiCrafter isolates
**paradigm** (flow vs diffusion) at matched weak strength; Wan-5B vs
DynamiCrafter is the strong-flow vs weak-diffusion contrast for which
early gate-health evidence already exists. SkyReels-V2 shares the Wan VAE and
diffusion-forcing conditioning, so its integration reuses the Wan preprocessing
path (§4.1.3); a capability probe on the target domain precedes integration,
as for every base.

## 4.2 Protocol: parameter / FLOPs-matched comparison
_Stub. Source:
[[../../50_Decisions/decided/param-matched-adapter-comparison-definition]]._

## 4.3 Metrics and baselines
_Stub — prediction accuracy, stability, inference cost; action-following._

## 4.4 Ablation design (the axes)

_Full design + rationale + run plan:
[[../../30_Knowledge/writing/ablation-axes]]._

The study is organised as a **dataset axis run in full** (the
action-informative-vs-redundant contrast — MetaWorld anchor + three ACWM-Phys
envs: Push Cube da=2, Robot Arm da=7, Reacher da=2) crossed with a **search
toolbox** of interventions pulled adaptively to find a configuration that
escapes base-parity: gate-cap, σ-shift, and an AVID-style pure-adapter warmup
(`pretrain_steps`), plus single mechanism-probe cells (base-output
conditioning off; AdaLN vs cross-attention). The adapter-family axis is held
to the output adapter empirically; the hidden-state and hypernetwork families
are treated as a computational-complexity discussion (D1 as a software +
cost-analysis contribution). Every cell reports the same five metrics
(action-sensitivity, prediction accuracy, generation quality, base-parity
cosine, cost), so the matrix is a result whether or not action-following
emerges (§5.1.x is the diagnostic anchor).
