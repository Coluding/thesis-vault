---
date: 2026-06-17
category: progress
deliverable: exploratory
meeting:
sources:
  - "[[2026-06-10-multimodal-substrate-landed]]"
  - "[[../../50_Decisions/open/multimodal-adapter-broadening]]"
  - "repo: configs/multimodal_dynamicrafter.yaml"
  - "repo: configs/multimodal_metaworld.yaml"
  - "repo: examples/multimodal_training_test.py"
  - "repo: scripts/train_multimodal_metaworld.py"
---

# Multimodal multi-stream model runs on the real DynamiCrafter backbone

## What

Closed the "not yet run on a real backbone" gap from
[[2026-06-10-multimodal-substrate-landed]]. Added two artifacts:

- `configs/multimodal_dynamicrafter.yaml` — a real DynamiCrafter experiment YAML
  (the AVID-style frozen 512 UNet + output adapter + structured action
  conditioning) extended with the `output_modalities:` section: `video`
  (`has_frozen_prior`, `vae` codec), `proprio` (`vector`, identity codec), and
  `tactile` (`map`, resize codec).
- `examples/multimodal_training_test.py` — a smoke runner that builds the real
  base via `build_multimodal_experiment`, wires the video `VaeCodec` to the
  base's `first_stage_model`, synthesises one clip batch (pixel video
  `(B,3,16,128,128)` + proprio + tactile + action), and runs `MultiModalTrainer`
  steps. No dataset/checkpoint needed (`allow_missing_checkpoint`, random UNet
  weights).

The forward path executes end-to-end: frozen base denoises the VAE latent, the
real DynamiCrafter output adapter contributes the video delta, and the
per-modality heads predict proprio/tactile. **12.4M trainable params**, all three
per-stream losses drop over 3 steps on a single overfit batch.

## Why it matters

This is the first time the multimodal substrate has touched a real frozen
diffusion prior (not the `DummyVectorField` toy base) — the multi-stream
contract, the VAE codec, the per-stream UWM noising, and the summed loss all hold
against the actual DynamiCrafter latent shape and cross-attention conditioning.
It de-risks the broadening line as a viable thesis headline.

## Gap found — compositional fusion is dummy-base-only

`fusion: compositional` does **not** yet run on a real backbone. The builder
(`multimodal/builders.py`) wires the per-modality video-adjustment heads as
`ModalityPredictionHead(video_spec.feature_shape, …)`, which assumes the dummy
"video-as-vector" substrate and has **no notion of the `(B,C,T,H,W)`
DynamiCrafter latent layout** — `video_spec.feature_shape` is empty there, so it
raises immediately. The smoke config therefore uses `fusion: trivial` (standard
additive composition, video adapter only). **To run `LearnedMaskFusion` on a real
backbone, the video-adjustment contributions need a latent-shaped head (a small
DynamiCrafter-style adapter or a conv head over the latent), not the flat MLP.**
This is the next implementation step for the compositional contribution on real
data.

## Real-data training script (full pipeline)

Beyond the synthetic `examples/` smoke, added `scripts/train_multimodal_metaworld.py`
— the real-data sibling of `scripts/train_avid_shortcut_metaworld.py`, same skeleton
(argparse, VAE checkpoint load, OpenCLIP null-prompt + image embedder + Resampler,
`build_metaworld_clip_dataset`, JSONL/checkpoint IO). It plugs the real
`DynamiCrafterBatchPreprocessor` in as the *video* preprocessor inside
`MultiModalBatchPreprocessor`, which codec-encodes the remaining streams from the
clip batch. Trimmed vs. the AVID trainer: no shortcut/step-level, no eval loop, no
resume (none implemented on `MultiModalTrainer` yet).

`configs/multimodal_metaworld.yaml` matches the streams the corner2 HDF5 actually
emits: **video + force_torque (vector[6]) + depth (map, resized 32×32)** — `proprio`/
`tactile` are not present in this dump (see `METAWORLD_OPTIONAL_KEYS`).

### Second gap found + fixed — modality heads couldn't take per-frame conditioning

`ModalityPredictionHead` assumed per-sample conditioning `(B, cond_dim)` and broke on
real data, where the action is per-frame so the structured encoder emits `(B, T, 512)`
— a `torch.cat` dim mismatch. The substrate tests never hit this (they pass `act` as
`(B, 4)`). **Fixed** by `_align_cond`: when `cond_emb`'s leading dims already match the
stream's `(B, T)` lead it is flattened directly to `(n, cond_dim)`; per-sample cond
still broadcasts via `repeat_interleave`. Backward-compatible — all 7 substrate tests
still pass.

### Caveat — unnormalised vector streams

`force_torque` with the `identity` codec (no mean/std) has raw magnitudes ~O(100s), so
its denoising loss starts ~4e4 and the summed objective is dominated by it until the
head fits. A real run should set `codec_kwargs: {mean, std}` (or per-channel stats) for
the vector streams so the per-stream losses are commensurate before weighting.

## Evidence / sources

- `python examples/multimodal_training_test.py --config
  configs/multimodal_dynamicrafter.yaml --steps 3` → 3 steps, `loss` 0.93→0.59,
  per-stream losses all decreasing. Smoke/overfit only, random base weights — not
  a data result.
- `python scripts/train_multimodal_metaworld.py --config configs/multimodal_metaworld.yaml
  --hdf5 ds/metaworld_corner2.hdf5 --steps 3 --batch-size 1` → real DynamiCrafter VAE
  (248 tensors loaded), 490 windows, **12.5M trainable / 1.54B total (0.82%)**, all
  three streams stepping. Wiring smoke only (3 steps, single overfit-ish batch).

## Next

- Implement a latent-shaped video-adjustment head so `fusion: compositional`
  works on DynamiCrafter (the actual contribution, vs. the trivial baseline).
- Wire a real video preprocessor (VAE-encode + DynamiCrafter conditioning) so the
  smoke runner can swap synthetic context/fs for a real MetaWorld clip + caption.
- Add per-channel normalisation stats to the vector-stream codecs so per-stream
  losses are commensurate before loss-weighting (see caveat above).
- Then a real overfit/short training run with a real checkpoint for a first
  "does it learn coupled dynamics" signal.
