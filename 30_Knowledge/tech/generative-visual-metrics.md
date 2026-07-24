---
last_updated: 2026-07-01
status: living
---

# Generative-visual quality metrics (image & video)

Reference note for the metrics we score during training-time eval. Covers what
each metric *means*, how it is *computed as wired in our codebase*, and the
gotchas. Implemented **natively** in
`training/quality_metrics.py::QualityMetricSuite` (in `generative-flow-adapters`)
over `torchmetrics` (PSNR/SSIM/LPIPS/FID) and `cd-fvd` (FVD). It **does not**
import `external_deps` — the vendored AVID `metrics.py` is untouched (hard rule).
See [[../../20_Tickets/feat-eval-training-quality-metrics]] and the training row
in [[../../10_now/architecture]].

## The setup they run in

This is a **world model**: eval has the aligned ground-truth future frame, so
we can use *paired* metrics (pred vs. the true next frame), not only
distribution metrics. Each eval batch:

1. Sample the **adapted** rollout and the **frozen-base** rollout from the
   *same* starting noise → the base-vs-adapted delta is attributable to the
   adapter, not to noise drift.
2. VAE-decode latents → `uint8 (B, T, 3, H, W)`.
3. Score both adapted-vs-GT and base-vs-GT.

Representations fed to the metrics (`quality_metrics.py`): image metrics get
frames flattened to `(B*T, 3, H, W)` as float in `[0, 1]` (`_frames01`); FVD
gets clips as uint8 `(B, T, H, W, C)`.

Two tiers (see [[../../20_Tickets/feat-eval-training-quality-metrics]]):
**paired** metrics every eval cycle (cheap, reliable); **distribution** metrics
(FID/FVD) on a rarer opt-in cadence (load feature nets, noisy on small sets).

---

## Paired metrics — per-frame, vs. ground truth

Every frame `(B*T)` is treated as an image; the metrics aggregate over all
frames of all eval batches into a single scalar per metric (no per-timestep
breakdown in the current native version — a possible later enhancement if we
want to watch error grow over the rollout horizon).

### MSE

- **Idea:** raw pixel error. `mean((pred − real)²)`.
- **As wired:** `_MSE` in `quality_metrics.py` — accumulates squared error over
  frames in `[0,1]` space (`frames = uint8/255`). Diagnostic only, not
  perceptual.
- Lower = better.

### PSNR — Peak Signal-to-Noise Ratio

- **Idea:** log-scaled pixel fidelity; the standard "how close in dB".
- **As wired:** torchmetrics `PeakSignalNoiseRatio(data_range=1.0)` on `[0,1]`
  frames; `10·log₁₀(1/MSE)`. (Returns `inf` for a pixel-identical pair — a
  non-issue for real rollouts.)
- Higher = better. Sensitive to blur/shifts, weakly correlated with perceived
  quality — but cheap and monotone, a good training curve.

### SSIM — Structural Similarity

- **Idea:** compares local luminance, contrast, and structure in sliding
  windows rather than raw pixel differences; closer to human judgement than
  PSNR.
- **As wired:** torchmetrics `StructuralSimilarityIndexMeasure(data_range=1.0)`
  on `[0,1]` frames.
- Range ≈ `[-1, 1]`, higher = better. **This + PSNR are the per-cycle
  defaults.**

### LPIPS — Learned Perceptual Image Patch Similarity

- **Idea:** distance in the feature space of a pretrained CNN (deep features
  track perceptual similarity far better than pixels). "Do these *look* alike
  to a network trained on images."
- **As wired:** torchmetrics
  `LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)` — the
  `normalize=True` flag means we feed `[0,1]` frames and it maps to the `[-1,1]`
  LPIPS expects internally. Downloads VGG-16 weights on first use.
- Lower = better. **Wired but off by default** (we chose PSNR/SSIM per-cycle).

---

## Distribution metrics — realism, not per-sample accuracy

These do **not** compare a prediction to its paired GT. They compare the
*distribution* of generated samples to the *distribution* of real samples in a
feature space — "do generated frames/clips look like they came from the real
data." Need many samples to be meaningful; noisy on small eval sets → opt-in,
rare cadence.

### FID — Fréchet Inception Distance

- **Idea:** embed real and generated images with InceptionV3, fit a Gaussian to
  each set, measure the Fréchet (Wasserstein-2) distance between the two
  Gaussians: `‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½)`.
- **As wired:** torchmetrics `FrechetInceptionDistance(feature=2048,
  normalize=True)` (via the `torchmetrics[image]` → torch-fidelity backend). We
  feed **every frame as an independent image** in `[0,1]`; it resizes to 299×299
  and accumulates real (`real=True`) + fake (`real=False`) Inception features
  across all batches, distance computed at the end.
- Lower = better. **Caveat: this is frame-level FID** — measures per-frame
  realism, ignores temporal coherence. Do not read it as a video-quality
  number.

### FVD — Fréchet Video Distance

- **Idea:** the video analogue of FID — same Fréchet-distance-of-Gaussians, but
  features come from a **video** network so motion/temporal dynamics count.
- **As wired:** `cd-fvd` (`cdfvd.fvd.cdfvd`), backbone **i3d** for clips of ≥10
  frames, **videomae** for fewer. We push clips as uint8 `(B,T,H,W,C)` through
  `add_real_stats` / `add_fake_stats` (running feature stats) and read the score
  with `compute_fvd_from_stats` — so it accumulates over **all** batches this
  cycle. cd-fvd handles its own resize/preprocess.
- Lower = better. The one metric here that captures temporal coherence.

---

## Gotchas (read before trusting a number)

1. **FID is frame-wise, not video.** Realistic frames ≠ coherent motion; that's
   FVD's job. Report them together, never FID alone for video quality.
2. **All six accumulate correctly across batches.** The native rewrite uses
   torchmetrics running state (PSNR/SSIM/LPIPS/FID) and cd-fvd's
   `add_real_stats`/`add_fake_stats` → `compute_fvd_from_stats` (FVD), so
   `quality_dist_num_batches > 1` genuinely grows the sample pool. (This fixes a
   bug in the vendored AVID FVD, which overwrote its score each batch and
   reported only the last one — that path is no longer used.)
3. **Paired vs. distribution answer different questions.** PSNR/SSIM/LPIPS:
   "is this the *right* future?" (accuracy — only meaningful because we have GT
   futures). FID/FVD: "does this look *real*?" (fidelity). A model can win one
   and lose the other; the thesis wants both.
4. **Small-set noise.** Distribution metrics want thousands of samples; on a
   MetaWorld training eval they're jittery. Treat FID/FVD as milestone signals
   over a larger budget, trust PSNR/SSIM for the per-cycle curve.

## Sign / direction cheat-sheet

| Metric | Type | Higher or lower better | Captures |
|---|---|---|---|
| MSE | paired | lower | raw pixel error |
| PSNR | paired | higher | pixel fidelity (dB) |
| SSIM | paired | higher | local structure |
| LPIPS | paired | lower | deep-feature perceptual |
| FID | distribution (frame) | lower | per-frame realism |
| FVD | distribution (video) | lower | temporal realism |
