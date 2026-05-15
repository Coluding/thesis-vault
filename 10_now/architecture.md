---
last_updated: 2026-05-15
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
| Output | `output` | `adapters/output/` | `affine`, `dynamicrafter`, `shortcut_direction` |
| Hidden State | `hidden_state` | `adapters/hidden_states/` | `residual`, `unicon`, `replace_decoder`, `full_skip_controlnet` |
| Hypernetwork | `hyper` | `adapters/hypernetworks/` | `hyperalign`, `hyper_lora_simple` |
| LoRA | `lora` | `adapters/low_rank/` | `LoRAAdapter` (with `PAPER_HYPERALIGN_TARGET_MODULES`) |

This is the **D1 deliverable surface** — the taxonomy from the proposal
mapped 1-to-1 onto the codebase. See [[positioning]] for how these tie to
the four thesis deliverables.

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
- **Multimodal scope.** The proposal lists a multimodal extension
  (video + proprio) as optional. Architecturally the conditioning system
  supports `modalities`, but no config or run uses both yet. → candidate
  [[../50_Decisions/open/multimodal-scope]].
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
