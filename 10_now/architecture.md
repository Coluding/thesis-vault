---
last_updated: 2026-07-15
status: living
---

# Codebase Architecture

> Living document. Reflects the state of the implementation repo as it
> stands today. Previous versions live in Git history. AI overwrites this
> freely when changes happen. Do not append change logs here — log decisions
> in `50_Decisions/`.
>
> **Honesty contract:** cells marked _not yet built_ describe planned
> components from the thesis proposal, not deployed reality. When a cell is
> filled with a concrete value, that value reflects something that actually
> exists in code at the path shown. See [[../CLAUDE]] hard rule 7.
>
> **Source of truth:** `/home/lukas/projects/generative-flow-adapters/` at
> the current branch `main`. The repo is a research scaffold, not a
> product — see [[../../README]] in that repo for the entry-point pitch.

## What this repo is

A research scaffold for the thesis. It implements the composition rule

```
f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)
```

over diffusion and flow-matching backbones, with the adapter `Δ_φ` as the
trainable module. The repo's stated design goals (from its README):

- Treat adapters as first-class modules.
- Keep diffusion and flow matching under one `BaseGenerativeModel` interface.
- Wrap existing backbones instead of reimplementing them.
- Make configuration (YAML) the primary control surface.
- Leave clear extension points for shortcut conditioning, multimodal inputs, planning, and RL.

## Top-level layout

| Path | Contents |
|---|---|
| `src/generative_flow_adapters/` | The library. Subpackages: `adapters/`, `models/`, `conditioning/`, `losses/`, `training/`, `data/`, `inference/`, `testing/`, `backbones/` |
| `src/external_deps/` | Vendored third-party code (e.g. `lvdm/`, `avid_utils/`) — clearly fenced ownership boundary |
| `configs/` | Example experiment YAMLs (~22 files at HEAD) |
| `scripts/` | Standalone entrypoints — currently only `train_hyperalign_metaworld.py` |
| `examples/` | Small demonstration scripts (`build_model.py`, `training_test.py`) |
| `tests/` | pytest suite — architecture tests, integration tests, dataset tests, video logging |
| `docs/` | Thesis-adjacent material — `thesis-plan/`, `paper/` (8 vendored PDFs), architecture reports |
| `external_repos/` | Original third-party repos kept for reference (not on import path) |
| `ckts/`, `ds/`, `wandb/` | Checkpoints, datasets, wandb local cache |

## Core abstractions

### `BaseGenerativeModel` — `models/base/interfaces.py`

Common interface for all backbones.

```python
forward(x_t, t, cond=None) -> prediction
model_type: "diffusion" | "flow"
prediction_type: "noise" | "velocity"
```

`model_type` determines the loss semantics used by the loss registry.
`prediction_type` keys the scheduler / sampler behaviour.

### `Adapter` — `adapters/base.py`

All adapters implement:

```python
forward(x_t, t, cond, base_output=None) -> delta
attach_base_model(base_model) -> None  # called by AdaptedModel
```

`attach_base_model` gives adapters access to backbone internals (hidden
states, hooks, weight references) — this is how hidden-state and LoRA-style
adapters reach inside the frozen base.

### `AdaptedModel` — `models/adapted_model.py`

Owns the composition logic. Handles condition encoding, condition dropout,
and output composition modes (`add`, `replace`, `mask_mix`).

## Adapter families

Located in `src/generative_flow_adapters/adapters/`. Registered and built
through `adapters/factory.py:build_adapter()`.

| Family | Factory key | Subfolder | Concrete adapters |
|---|---|---|---|
| Output | `output` | `adapters/output/` | unified `dynamicrafter` (default; `backbone`/`output_format` knobs) + legacy `affine`, `shortcut_direction` |
| Hidden State | `hidden_state` | `adapters/hidden_states/` | `residual`, `unicon`, `replace_decoder`, `full_skip_controlnet` |
| Hypernetwork | `hyper` | `adapters/hypernetworks/` | `hyperalign`, `hyper_lora_simple` |
| LoRA | `lora` | `adapters/low_rank/` | `LoRAAdapter` (with `PAPER_HYPERALIGN_TARGET_MODULES`) |

This is the **D1 deliverable surface** — the taxonomy from the proposal
mapped 1-to-1 onto the codebase. See [[positioning]] for how these tie to
the four thesis deliverables.

### Unified output adapter (merged 2026-05-29)

`adapter.type: output` now resolves to **one** unified adapter, built by
`factory._build_output`, with the *form* chosen by `extra` — there is no longer
a separate `output_v2` class to remember (the key still works as a legacy
alias; `affine`/`shortcut_direction` and the hidden-state architectures remain
explicit legacy keys). The default form is the DynamiCrafter UNet, so existing
`architecture: dynamicrafter` configs are unchanged.

Knobs (all in `adapter.extra`):

- `backbone`: `unet` **(default — the DynamiCrafter 3D UNet)**,
  `transformer` (DiT-style over patchified latents), or `mlp` (lightweight FiLM
  head ≈ the original `affine`). `unet` → `DynamicCrafterOutputAdapter`;
  `transformer`/`mlp` → `OutputHeadAdapter`.
- `output_format`: `direct` **(default)** (emit the delta `C`) vs `affine`
  (emit `2C` → `delta = base*scale + shift`, realised as `base*(1+scale)+shift`
  under `add`). Both return `output_kind="delta"`, so the affine-vs-direct
  comparison is apples-to-apples (and dense affine is a strict superset of
  direct — see [[../30_Knowledge/tech/affine-output-granularity]]).
- `affine_granularity`: `dense` (per-element scale/shift maps) vs `channel`
  (pooled per-channel FiLM) — [[../30_Knowledge/tech/affine-output-granularity]].
- `gate_kind`: `none` | `channel` | `dense` — emit a gate for the gated
  composition modes (head backbones only; the `unet` affine path doesn't emit a
  gate yet).
- `gate_kind`: `none` | `channel` | `dense` — when set, the head emits a gate
  alongside the contribution so the composition layer can blend it (see below).

**Composition / blending (mixing layer, `AdaptedModel._compose`).** Gating is a
separate abstraction from the adapter recipe — the adapter emits a full-size
contribution `Δ` (+ optional gate); mixing owns all interaction with `base`.
Four modes, two of them gated:

| mode | formula | `Δ` is | identity at init |
|---|---|---|---|
| `add` | `base + Δ` | contribution | needs `Δ→0` |
| `gated_residual` *(added 2026-05-29)* | `base + σ(gate)·Δ` | contribution | automatic |
| `mask_mix` (AVID) | `base·σ(gate) + Δ·(1−σ(gate))` | standalone prediction | needs gate_bias |
| `replace` | `Δ` | standalone prediction | needs `Δ→base` |

`gated_residual` is the recipe-agnostic gate for *contributions* (works for
direct or affine Δ, matches the thesis core `f = base + g·Δ`); `mask_mix`
treats Δ as a competing prediction. See [[../30_Knowledge/tech/mask-mix-gate]].

Shared format/gate math lives in `adapters/output/format.py`; the new backbones
in `adapters/output/output_head.py`. All backbones zero-init their final
projection (identity residual at init). Configs:
`configs/diffusion_output_v2_{affine,direct}_metaworld.yaml`. Theory:
[[../30_Knowledge/theory/unicon-output-adapters-detached-backward]] (this is a
new size point on the same detached-output family). Decision/experiment:
[[../50_Decisions/open/output-format-affine-vs-direct]],
[[../20_Tickets/experiments/exp-adapter-output-format-affine-vs-direct]].

## Multimodal output adapters (added 2026-06-10)

A new `multimodal/` subpackage implements the multi-stream world model from
[[../50_Decisions/open/multimodal-adapter-broadening]] — the "optional
extension" `(x^video_{t+d}, x^prop_{t+d}) = f(x_t, t, a_t)` elevated to a
parallel research line. **This is a separate class tree from `AdaptedModel`,
not an extension of it** — the single-modality shortcut/`d` path is untouched
and dormant (shortcut is shelved for this line per the decision note).

Landed in commit `b09e8d5` ("cleaned configs and added multimodal model",
2026-06-10), ~1000 LoC across `src/generative_flow_adapters/multimodal/`:

| File | Role |
|---|---|
| `model.py` | `MultiModalAdaptedModel` — sibling to `AdaptedModel`; `forward(x_t: dict, t: dict, cond) -> dict`. Frozen base predicts the video stream; modality streams (proprio/tactile/…) have **no frozen prior** and are predicted whole by per-modality heads. |
| `fusion.py` | Video-stream fusion: `TrivialFusion` (`ε_video = ε_pre + Σ contributions`, additive substrate / degenerate baseline) and `LearnedMaskFusion` (compositional — softmax mask `m ∈ ℝ^{n+2}` over {base, action, modality₁…ₙ}, base-biased init). |
| `modality_adapter.py` | Per-modality prediction heads (vector / map kinds). |
| `modality_encoder.py` *(added 2026-06-26)* | Real-backbone compositional coupling: `ModalityEncoder` (**video←m** — encodes `z_t^m` + `t_m` into `context_dim` tokens injected into *that modality's own* AVID adapter's cross-attention `context`) and `VideoReadout` (**m←video** — pools the frozen base video prediction into the modality heads' conditioning). |
| `spec.py` | `OutputModalitySpec` — output-side dual of `ConditionSpec`; kinds `video`/`vector`/`map`, per-stream `loss_weight`, codec selection. |
| `codecs.py` | `IdentityCodec` (raw + per-channel norm), `ResizeCodec` (map downsample ↔ restore) — raw clip data ↔ diffusion target. |
| `trainer.py` | `MultiModalTrainer` — forks the diffusion branch. Each stream noised at its **own** timestep `t_m ~ U(0,T)` (the UWM scheme); one summed objective `Σ w_m · L_m`. Logs total + **per-modality (unweighted) denoising losses** (`loss_video/loss_proprio/...`) to stdout, JSONL, and W&B (`--wandb`). |
| `eval.py` *(added 2026-06-26)* | `MultiModalEvaluator` — eval-time **modality rollout** (full reverse diffusion via `DiffusionInferenceSampler` on a per-modality head wrapper, teacher-forcing the GT video for the m←video readout) + predicted-vs-GT viz. Plugs into `trainer.train(on_step=…)` at a config cadence; modalities flagged `visualize: true`. Vectors → W&B `line_series`; maps → images; always an `.npz` dump. No matplotlib. |
| `preprocessor.py`, `config.py`, `builders.py` | Batch prep, config partition (`MultiModalExperimentConfig`), and `build_multimodal_experiment` wiring (dummy base **or** the real output-adapter factory for the video stream). |

**Design specifics that are now code, not plan:** independent per-modality
timesteps + shared summed denoising objective (UWM scheme, sub-decisions 1 & 3);
default `fusion` knob is `compositional`; modality streams composition-from-scratch.

**Status — compositional now wired to the real DynamiCrafter backbone, still no
training run.** Two layers:
- *Substrate (2026-06-10):* validated on `DummyVectorField` by
  `tests/test_multimodal_substrate.py` (7 tests): multi-stream contract, codec
  roundtrips, spec/config, per-stream noising, overfit of `TrivialFusion` +
  `LearnedMaskFusion`.
- *Real-backbone TRUE compositional (2026-06-26):* on a non-dummy provider +
  `fusion: compositional`, `builders.py` wires **one AVID adapter per modality**
  (separate weights) + a `ModalityEncoder` each + `VideoReadout` +
  `LearnedMaskFusion(1+n)`; `model.py` runs `_forward_compositional` —
  `ε_video = m₀·ε_pre + m₁·ε_adj + Σ_i m_{i+1}·Δ_i`, each Δ_i from an adapter that
  sees only its own modality tokens (appended to `context`, preserving the fixed
  77-token text/image split). Contract-tested by
  `tests/test_multimodal_real_backbone.py` (3 tests): per-modality token routing,
  text boundary preserved, mask is a normalised n+2 softmax, bidirectional + mask
  grad flow. (An initial single-shared-adapter cut was corrected to this.)

Both layers are unit/contract checks plus a random-weight smoke run (35.2M
trainable params, runs end-to-end on the real DynamiCrafter UNet) — **not** a
DynamiCrafter *training* run and **not** an experimental result. The *fused*
variant (modality↔modality self-attention) is not yet built.

**Variant coverage vs the plan.** The decision note's contribution
(`LearnedMaskFusion`, compositional) is built; the additive `TrivialFusion`
substrate is built. The two intended *comparison baselines* — channel-stack and
single-joint — are **not yet** present as distinct fusion strategies.

**Flow matching is deferred** — `MultiModalTrainer` raises `NotImplementedError`
for non-diffusion bases.

## Backbone providers

`models/base/factory.py` resolves model configs into concrete backbones.

| Provider | File | Status | Notes |
|---|---|---|---|
| `dummy` | `models/base/dummy.py` | Live | Lightweight MLP for local smoke tests |
| `diffusers` | `models/base/diffusers.py` | Live (soft integration) | Wraps Hugging Face `diffusers` pipelines; optional dep |
| `dynamicrafter` | `models/base/dynamicrafter.py` | Live | Video U-Net from DynamiCrafter / AVID — vendored under `backbones/dynamicrafter/` and `src/external_deps/lvdm/` |
| `opensora` | _via configs_ (`opensora_output_adapter.yaml`) | _Partial_ | OpenSora vendored under `external_repos/` per commit `3572c82`; full provider wiring _needs verification_ |
| `wan2.2` | `scripts/train_wan22_i2v_metaworld.py` | ⚠️ **Vendored, unverified prior** | Loads the hand-copied `Wan22DiTWrapper`. Confirmed (2026-07-14, wandb run metadata) as the base used by a run whose adapter had to learn entirely from scratch (no useful prior) — see [[../20_Tickets/bug-infra-wan-script-provider-mismatch]]. Also has **no wiring** for `action_per_frame`/`action_seq_len`. Most `diffusion_wan22_*` config headers still point here — treat as a landmine until that ticket is resolved. |
| `wan2.2_external` | `scripts/train_wan22_i2v_metaworld_external.py` | **Preferred — real pretrained weights** | Loads the real upstream `wan.WanTI2V` (`external_repos/Wan2.2`). Confirmed used by both the proper-prior xattn run and the DC-UNet smoke-validated runs. The only script wired for `action_per_frame`/`action_seq_len` (added 2026-07-14). Use this for any real-weights run. |

## Conditioning

`conditioning/encoders.py` plus the registry in
`conditioning/__init__.py` produce a single conditioning vector consumed by
adapters. Supports:

- Tensor conditions (e.g. actions, goals)
- Multimodal conditions (declared via `ConditioningConfig.modalities` /
  `conditions`)
- Step-size conditioning for shortcut adapters
  (`ConditioningConfig.include_step_size`, key `step_size`)
- Horizon conditioning (`include_horizon`, `horizon_dim`)
- Condition dropout via `drop_condition_prob`
- Fusion modes via `fuse_mode` (default `concat_mlp`)

### Action injection: aggregated vs per-frame (2026-07-14)

**Finding:** across the WAN runs the action reaching the adapter's AdaLN/additive
path was **aggregated** — the preprocessor summed the per-frame delta-actions over
the whole clip into one `[B, A]` vector (`wan_batch_preprocessor._aggregate_action`,
`action_aggregation: sum`), which the MLP encoder broadcast identically to every
latent frame. This **departs from the original AVID** action head
(`external_repos/avid/.../openaimodel3d.py:737-747`), which conditions **per-frame**:
`act` is `[B, T, A]`, embedded per frame and added to each frame's time-embedding.
The only per-frame signal previously wired anywhere was the **cross-attention**
`action_seq` token path — i.e. the xattn arm that
[[../30_Knowledge/experiments/20260712-wan-xattn-action-no-improvement]] showed
does **not** help. So AVID-style per-frame *additive* conditioning existed nowhere;
a failed per-frame *cross-attention* result does not speak to it. This aggregation
is a candidate cause of the standing weak-action-signal finding.

**Fix (flag `action_per_frame`, default False):** when on, the preprocessor bins
the per-frame delta-actions onto the **latent** temporal grid (`action_seq_len` =
latent frame count, 11 for a 41-frame/stride-4 clip, summing deltas within each
bin) and routes that `[B, L, A]` to the adapter's action encoder → `[B, L, C]` →
per-frame `emb = time_emb + act_emb`, exactly AVID's mechanism. The rank-3 embedding
flows through `_prepare_adapter_embedding` and combines with the (broadcast)
step-level embedding. Default False preserves the aggregated AdaLN broadcast, so
**aggregated-vs-per-frame is a clean one-switch ablation** of the action-signal
question. Smoke-validated per-frame on the DC-UNet adapter (training steps run,
`action=per-frame[B,11,A]`). Touched: `wan_batch_preprocessor.py` (config flag +
routing), `scripts/train_wan22_i2v_metaworld_external.py` (plumb + align
`action_seq_len` to latent frames), `diffusion_wan22_dcunet_output_metaworld.yaml`.

## Losses

`losses/registry.py:LossRegistry` selects the loss by `model_type`:

| Key | Loss | File |
|---|---|---|
| `diffusion` | `diffusion_loss` | `losses/diffusion.py` |
| `flow` / `flow_matching` | `flow_matching_loss` | `losses/flow_matching.py` |

Shortcut consistency losses are kept separate (composable add-ons):

- `shortcut_direction`
- `local_consistency`
- `multistep_self_consistency`

All three are referenced from `TrainingConfig` weights
(`shortcut_direction_weight`, `local_consistency_weight`,
`multistep_consistency_weight`) and live in `losses/consistency.py`.

This is the **D3 deliverable surface**.

## Training

**Hyperparameter setup (optimizer, LR schedule/warmup, grad accumulation,
precision, EMA status, per-config values) lives in its own living doc:**
[[training-hyperparameters]] — this section covers code layout only.

| Layer | Where |
|---|---|
| Config dataclasses | `config.py` — `ExperimentConfig`, `ModelConfig`, `AdapterConfig`, `ConditioningConfig`, `TrainingConfig`. Unknown YAML fields land in `extra` dicts. |
| Builders | `training/builders.py:build_experiment(config)` resolves YAML → instantiated `AdaptedModel` + optimizer + data |
| Trainer | `training/trainer.py` — currently modified working tree (`M` in git status); thin loop. |
| Data | `data/` — batch preprocessor (also modified working tree), dataset loaders |
| CLI | `scripts/train_hyperalign_metaworld.py` is the only standalone training entrypoint at HEAD. Working tree also modified. |
| Quality metrics *(added 2026-07-01)* | `training/quality_metrics.py` — `QualityMetricSuite`, a **native** implementation over `torchmetrics` (PSNR/SSIM/LPIPS/FID) + `cd-fvd` (FVD). **Does not import `external_deps`** (hard rule) — the vendored AVID `metrics.py` is untouched. Scored on decoded eval rollouts for **both** the adapted and frozen-base sampler (base-vs-adapted delta). **Two-tier cadence:** paired per-frame metrics (`psnr/ssim/lpips/mse`, `TrainingConfig.quality_metrics`) every eval cycle; distribution metrics (`fid/fvd`, `quality_dist_metrics`) on a separate rarer `quality_dist_every_n_steps` (off by default — they load Inception/i3d and need many samples; both accumulate correctly across batches). Requires a VAE decoder on the wandb logger (`decode_to_uint8`) + inference sampler; else silently skipped. Core deps: `torchmetrics[image]` (pulls torch-fidelity + lpips) + `cd-fvd`. |

## Inference

`src/generative_flow_adapters/inference/` handles rollout / sampling. Video
inference recently fixed (commit `88e4430`: "fixed video generation
inference"). Video logging added in the prior commit (`44b214b`).

## Tests (`tests/`)

| File | Covers |
|---|---|
| `test_hyperalign_architecture.py` | HyperAlign adapter wiring + forward shapes |
| `test_dynamicrafter_checkpoint_sanity.py` | DynamiCrafter weight loading |
| `test_dynamicrafter_integration.py` | End-to-end DynamiCrafter + adapter |
| `test_hyper_step_size_conditioning.py` | Step-size-conditioned hypernetwork (D3 territory) |
| `test_batch_preprocessor.py` | Data pipeline |
| `test_metaworld_dataset.py` | MetaWorld dataloader |
| `test_null_caption.py` | Empty / null caption handling for text-conditioned backbones |
| `test_video_logging.py` | Video logging utilities |
| `test_multimodal_substrate.py` *(added 2026-06-10)* | Multimodal substrate: multi-stream contract, codecs, spec, config partition, per-stream noising, overfit of trivial + compositional fusion (dummy base, no GPU) |
| `test_multimodal_real_backbone.py` *(added 2026-06-26)* | Real-backbone compositional wiring: per-modality token routing (each adapter sees only its own tokens; text/image split preserved), learned mask m∈ℝ^{n+2} normalised, bidirectional + mask grad flow (fake per-modality adapters, no checkpoint/GPU) |

No CI runner wired. Tests run locally via `pytest`.

## Vendored code boundary (`src/external_deps/`)

Clearly fenced third-party code under a single subpackage:

- `lvdm/` — Latent Video Diffusion Modules from DynamiCrafter
- `avid_utils/` — AVID evaluation utilities

`backbones/dynamicrafter/` contains adapted DynamiCrafter model code (also
vendored, with the noted modifications to allow adapter injection).

**Vendored DynamiCrafter changes to run the DC 3D-UNet as an output adapter on a
WAN diffusion-forcing flow base (2026-07-13, three edits — flagged per hard-rule
Part 12).** These make the DC UNet composable on a base it was never written for
(WAN's per-frame-timestep diffusion forcing, no CLIP text/image context). All are
guarded so the standard DC-base path is unchanged:

- `backbones/dynamicrafter/modules/networks/openaimodel3d.py` — `forward` now
  accepts **per-frame timesteps** `[b, t]` (flatten → embed → skip the scalar-case
  `repeat_interleave(t)`), in addition to the scalar `[b]` case. WAN diffusion
  forcing holds the leading obs frame(s) clean at t=0 and noises the rest, so the
  timestep is per-frame; the DC UNet previously assumed one shared timestep.
- Same file — the `context` reshape block is guarded for `context is None`
  (no cross-attn context → SpatialTransformer self-attends).
- `backbones/dynamicrafter/modules/attention.py` — `BasicTransformerBlock._forward`
  **skips `attn2` (cross-attention) when `context is None`**: attn2's `to_k/to_v`
  are sized for `context_dim` (1024) so it cannot self-attend, and with no text/image
  tokens there is nothing to attend to — the block reduces to self-attn + FF.
- `adapters/output/dynamicrafter.py` — the adapter drops a **non-tensor** base
  `context` (the WAN base passes text context as a *list*) → `None`, so the adapter
  self-attends rather than consuming the frozen base's context.

Smoke-validated: a training step runs end-to-end (145M-tier DC-UNet adapter, 5.1B
frozen WAN base, `loss≈0.48` at step 1). See
[[../20_Tickets/feat-adapter-dynamicrafter-output-on-wan-base]]. This confirms the
D1 claim of a **heterogeneous** adapter (DC-UNet architecture ≠ WAN-DiT base).

**`external_repos/avid/` MetaWorld reference run (2026-07-14) — config +
one-file data-module fix, no model/training code touched.** Preparing a run of
the **real, unmodified `AVIDAdapter` + `train_avid.py`** on our MetaWorld
frames instead of RT1
([[../20_Tickets/experiments/exp-adapter-avid-native-reference-run]]), so the comparison is
decoupled from anything in our own trainer/preprocessor:

- Stale hardcoded checkpoint path
  (`/host_home/avid/dynamicrafter_512/model.ckpt`) in
  `configs/train/act_cond_diffusion_{11M,34M,145M}.yaml` → local
  `ckts/dynami512.ckpt`.
- **Fixed a real bug** in the (pre-existing, not built this session)
  `src/ldwma/lightning/data_modules/metaworld.py`: it emitted the action
  tensor under key `"action"`, but `LatentVisualDiffusion.get_batch_input`
  reads `batch["act"]` — action conditioning would have silently been dropped,
  no error.
- **New configs** (additive, no vendored file behaviour changed):
  `configs/train/act_cond_diffusion_11M_metaworld.yaml` (adds
  `action_dims: 4` — the UNet's default is `7`, RT1's dim; would have crashed
  on MetaWorld's 4-dim actions) and
  `configs/train/avid/avid_11M_metaworld.yaml` (points at the MetaWorld data
  module + the new UNet config; `base_config_file`/`adapter_params`/trainer
  block untouched — the real reference composition, just on our data).
- Not yet smoke-tested (no local Poetry/TF env) — code-read-verified only.

This repo remains a genuinely separate Poetry/TF toolchain, not part of our
pipeline. Also surfaced a real, externally-validated finding while reading
`AVIDAdapter.apply_model`: the reference implementation's `init_mask_bias: 0.0`
(50/50 gate at init) vs. our `gate_bias: 4.0` (98/2) — see
[[../20_Tickets/bug-adapter-gate-saturation-mask-mix]].

Per the thesis-writing plan: this boundary must be described explicitly so
the contribution surface is clean — see [[positioning]] for what counts as
ours vs. theirs.

## External dependencies

| Layer | Choice |
|---|---|
| Language | Python (uv / pip editable install) |
| ML framework | PyTorch |
| Diffusers integration | Hugging Face `diffusers` (optional dep, `pip install -e .[diffusers]`) |
| DynamiCrafter stack | Optional dep, `pip install -e .[dynamicrafter]` |
| Experiment tracking | Weights & Biases (`wandb/` local dir present) |
| Lint | `ruff` |
| Tests | `pytest` |

## Current working-tree state

Files with modifications at the start of this session (relevant to thesis
work in flight):

- `M scripts/train_hyperalign_metaworld.py`
- `M src/generative_flow_adapters/data/batch_preprocessor.py`
- `M src/generative_flow_adapters/training/trainer.py`
- `M .gitignore`
- Deleted: ~50 generated sanity-test PNG/MP4 artefacts under
  `tests/_outputs/dynamicrafter_sanity/` (test outputs, not source)

These should land as a commit (`exp-` or `chore-` ticket) once the
in-flight HyperAlign / MetaWorld changes settle — or split into a chore for
the gitignore + sanity artefacts cleanup. Recent commits:

- `88e4430 UPDATE: fixed video generation inference`
- `44b214b UPDATE: added video logging to trianing`
- `8db744b UPDATE: added hyperalign adjsutments`
- `459ff9d UPDATE`
- `3572c82 Added opensora vendor`

## Open architectural questions

Surface-level open decisions that should be opened as
`50_Decisions/open/{slug}.md` when they are prioritised:

- **Which backbone scales to the headline experiments.** DynamiCrafter for
  video is wired; OpenSora is partially vendored; diffusers covers smaller
  image cases. The thesis needs at least one backbone where the four
  deliverables connect end-to-end. → candidate
  [[../50_Decisions/open/primary-backbone]].
- **Which adapter family is the "default" for D2's headline ablation.**
  HyperAlign hypernetwork has the most working configs and the dedicated
  CLI; output/affine is the simplest; UniCon hidden-state has its own
  tests. Pick one and run the others as ablations. → candidate
  [[../50_Decisions/open/d2-default-adapter]].
- **Shortcut target method** — `shortcut_target_method` defaults to
  `linear` but `two_step` is also implemented. The D3 ablation needs both
  cleanly compared. → candidate
  [[../50_Decisions/open/shortcut-target-method]].
- **Multimodal scope.** No longer just conditioning-side: the multi-stream
  **output** substrate now exists (`multimodal/` subpackage, see above) and the
  compositional learned-mask adapter is built and overfit-tested on a dummy
  base. Open parts: (1) the channel-stack and single-joint *baseline* variants
  aren't built; (2) no real-backbone (DynamiCrafter) run has happened; (3) the
  go/no-go on whether multimodal becomes the thesis headline vs. shortcut is
  still open. → tracked in [[../50_Decisions/open/multimodal-adapter-broadening]].
- **Whether to keep both `external_repos/` and `src/external_deps/`** as
  separate vendored zones. The first is reference-only, the second is on
  the import path — easy to confuse. → trivial enough to be a chore ticket
  (`chore-infra-document-vendored-zones`).

## Related

- [[product-state]] — what's actually run and what came out
- [[positioning]] — the four deliverables and contribution surface
- [[setup-status]] — vault coverage gaps
- [[../30_Knowledge/related-work/_MOC|Related work]]
- The implementation repo's own README at
  `/home/lukas/projects/generative-flow-adapters/README.md`
- The proposal at `docs/thesis-plan/Updated_Thesis_Proposal.pdf`
