---
last_updated: 2026-06-10
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
[[../20_Tickets/exp-adapter-output-format-affine-vs-direct]].

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
| `spec.py` | `OutputModalitySpec` — output-side dual of `ConditionSpec`; kinds `video`/`vector`/`map`, per-stream `loss_weight`, codec selection. |
| `codecs.py` | `IdentityCodec` (raw + per-channel norm), `ResizeCodec` (map downsample ↔ restore) — raw clip data ↔ diffusion target. |
| `trainer.py` | `MultiModalTrainer` — forks the diffusion branch. Each stream noised at its **own** timestep `t_m ~ U(0,T)` (the UWM scheme); one summed objective `Σ w_m · L_m`. |
| `preprocessor.py`, `config.py`, `builders.py` | Batch prep, config partition (`MultiModalExperimentConfig`), and `build_multimodal_experiment` wiring (dummy base **or** the real output-adapter factory for the video stream). |

**Design specifics that are now code, not plan:** independent per-modality
timesteps + shared summed denoising objective (UWM scheme, sub-decisions 1 & 3);
default `fusion` knob is `compositional`; modality streams composition-from-scratch.

**Status — substrate built and tested, no real-backbone run yet.** Validated
on the lightweight `DummyVectorField` base by `tests/test_multimodal_substrate.py`
(7 tests, all passing): multi-stream contract, codec roundtrips, spec validation,
config partition, per-stream noising, and **overfit tests for both `TrivialFusion`
and `LearnedMaskFusion`** (both learn end-to-end; the compositional mask receives
gradient and stays a normalised softmax). These are unit/overfit checks on a toy
base — **not** a DynamiCrafter run and **not** an experimental result. The real
video backbone path is wired in `builders.py` but not yet exercised end-to-end.

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

| Layer | Where |
|---|---|
| Config dataclasses | `config.py` — `ExperimentConfig`, `ModelConfig`, `AdapterConfig`, `ConditioningConfig`, `TrainingConfig`. Unknown YAML fields land in `extra` dicts. |
| Builders | `training/builders.py:build_experiment(config)` resolves YAML → instantiated `AdaptedModel` + optimizer + data |
| Trainer | `training/trainer.py` — currently modified working tree (`M` in git status); thin loop. |
| Data | `data/` — batch preprocessor (also modified working tree), dataset loaders |
| CLI | `scripts/train_hyperalign_metaworld.py` is the only standalone training entrypoint at HEAD. Working tree also modified. |

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

No CI runner wired. Tests run locally via `pytest`.

## Vendored code boundary (`src/external_deps/`)

Clearly fenced third-party code under a single subpackage:

- `lvdm/` — Latent Video Diffusion Modules from DynamiCrafter
- `avid_utils/` — AVID evaluation utilities

`backbones/dynamicrafter/` contains adapted DynamiCrafter model code (also
vendored, with the noted modifications to allow adapter injection).

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
