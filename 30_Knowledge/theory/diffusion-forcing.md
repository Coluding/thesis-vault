---
type: theory
last_updated: 2026-06-24
sources:
  - "code: src/generative_flow_adapters/data/wan22_batch_preprocessor.py"
  - "code: src/generative_flow_adapters/inference/flow.py (FlowInferenceSampler diffusion-forcing path)"
  - "code: src/generative_flow_adapters/backbones/wan/modules/model2_2.py (per-token timestep)"
  - "vendored: external_repos/wan22/wan/textimage2video.py (i2v), wan/utils/utils.py (masks_like)"
  - "paper: Chen et al., 'Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion', NeurIPS 2024"
  - "[[prediction-objectives]]"
  - "[[../../20_Tickets/feat-wan22-ti2v-diffusion-forcing-i2v]]"
  - "[[../../memory/multimodal-adapter-broadening]]"
---

# Diffusion forcing

> **One line:** standard diffusion gives every frame the *same* noise level;
> diffusion forcing gives every frame its *own* noise level. Once each frame's
> noise level is independent, the noise level becomes a per-frame **conditioning
> knob** — a frame at `t = 0` is *given* (observed), a frame at `t = max` is
> *generated*. Image-to-video, autoregressive rollout, and text-to-video then
> become the *same* model run with different per-frame noise schedules, with no
> CLIP encoder and no concat channel.

This is the mechanism behind the Wan2.2-TI2V-5B world-model base
([[../../20_Tickets/feat-wan22-ti2v-diffusion-forcing-i2v]]). It is also the
concrete realisation of the "UWM independent-timestep" idea sketched in
[[../../memory/multimodal-adapter-broadening]].

## Convention

Rectified-flow / flow-matching convention (matches Wan2.2):

- latent video `z ∈ R^{C×F×h×w}`; index frames by `f = 0 … F-1`.
- interpolation coordinate `σ ∈ [0, 1]`: `σ = 0` is clean data, `σ = 1` is pure
  noise. Forward path `x = (1-σ)·z₀ + σ·ε`, velocity target `v = ε - z₀`
  (sigma-independent — a straight line; see [[prediction-objectives]]).
- the model is fed `t = σ · timestep_scale` (Wan native `timestep_scale = 1000`,
  so `t ∈ [0, 1000]`).

The single new ingredient: **`σ` (hence `t`) is per-frame, not per-clip.**

## 1. The problem it solves

Ordinary full-sequence video diffusion couples all frames to one global `σ`. You
cannot say "frame 0 is known, frames 1…F are unknown" — there is no place to put
that asymmetry. To condition on an observation you must add machinery *outside*
the noise process: an image cross-attention (CLIP, as in Wan2.1-I2V) or a
masked-latent concat channel (DynamiCrafter, Wan2.1-I2V). Both change the
network's inputs, so they cannot be bolted onto a frozen text-to-video base.

Our Wan2.1-T2V world model hit exactly this wall: training was action-only with
`x_t` = the noised *whole* clip, so generation had to hallucinate the entire
scene from pure noise + a 4-dim action — under-determined, hence washed-out eval
videos (see [[../../20_Tickets/fix-wan-flow-eval-video-grid-never-fired]]).

## 2. The core idea: per-token noise levels

Assign each frame an independent noise level `σ_f` and noise each frame by its
own `σ_f`:

```
x_f = (1 - σ_f)·z₀,f + σ_f·ε_f ,      ε_f ~ N(0, I)   independently per frame
```

The denoiser is trained to predict the velocity of **each** frame given **its
own** noise level and the (independently, partially noised) **context of the
other frames**. Because the backbone is a full-sequence transformer, frame `f`'s
prediction attends to every other frame at whatever noise level that frame
currently sits at.

```
standard diffusion:   σ = [0.9, 0.9, 0.9, 0.9, 0.9]    (one global level)
diffusion forcing:    σ = [0.0, 0.6, 0.75, 0.85, 0.9]  (independent per frame)
                            ^observed (forced)  ^generated (free)
```

## 3. Training objective

Per-frame noise levels are *sampled* during training (each frame i.i.d., or with
structure — see §7). The loss is the usual flow-matching velocity MSE, but
**masked to the frames that are actually being predicted**. A frame held at
`σ_f = 0` is *given*; it carries no learning signal (the model is not asked to
denoise a clean frame, and its "velocity" `ε_f - z₀,f` references a noise draw
the model can never see). In our code the I2V special case is:

```python
# data/wan22_batch_preprocessor.py
frame_mask = [0, 1, 1, …]                 # 0 = observation, 1 = predicted
x_t  = (1-fm)·z0 + fm·((1-σ)·z0 + σ·noise)  # obs clamped clean, future noised
t    = frame_mask · (σ · timestep_scale)    # obs at 0, future at σ·scale
target = noise - z0                          # velocity
# trainer._flow_loss masks the MSE by frame_mask -> loss only on predicted frames
```

**Implementation note — the future frames share ONE noise level.** In our
preprocessor `σ` is a *single scalar sampled per training step* and applied to
**every** predicted frame (`t > 0`) at once. The per-frame schedule is therefore

```
σ = [0, σ, σ, σ, …]     # frame 0 forced at 0; all of t>0 share one sampled σ
```

**not** the fully general diffusion-forcing form `σ = [0, σ₁, σ₂, σ₃, …]` with an
*independent* level per future frame. So while the model uses the per-token-
timestep *machinery* of diffusion forcing, the I2V schedule collapses to ordinary
image-conditioned video denoising: one given frame, the whole future denoised
together at a shared level. The independent-per-future-frame regime — the part
that actually buys autoregressive rollout, sliding-window infinite video, and
monotone-in-horizon uncertainty — is **not exercised in v1** (see §7).

The general diffusion-forcing objective samples *all* `σ_f` independently (not
just `0` vs one shared σ); the I2V case is the special schedule "frame 0 forced,
the rest share one σ".

## 4. What it unifies: teacher forcing ↔ full-sequence diffusion

The name generalises **teacher forcing** in sequence models (feed the
ground-truth past, predict the next token). Teacher forcing is binary:
a token is either *given* (clean) or *absent*. Diffusion forcing makes that axis
**continuous** — the noise level `σ_f ∈ [0, 1]` is "how forced" a token is:

- `σ_f = 0`  → fully forced (the clean ground-truth frame is given).
- `σ_f = 1`  → fully free (generate from noise).
- `0 < σ_f < 1` → partially forced (a noisy hint).

So one trained network spans the continuum from autoregressive next-frame
prediction (clean past, noisy future) to joint full-sequence denoising (all
frames at the same σ). Chen et al. (NeurIPS 2024) show this also yields a
sampling scheme with monotone-in-horizon uncertainty and stable long rollouts.

## 5. Sampling: the noise schedule *is* the conditioning

At inference you pick **any** per-frame σ trajectory. The *same weights* then do:

| Task | Per-frame schedule at sampling |
|---|---|
| Text-to-video | all frames share one σ, anneal `1 → 0` together |
| **Image-to-video** (ours) | **frame 0 pinned at σ=0 (clamped to obs); rest anneal `1 → 0`** |
| Autoregressive / infinite video | past frames at σ=0, future noised; slide the window |
| Interpolation / inpainting | known frames at σ=0, gaps noised |

This is why Wan2.2 is a *unified* **T**I2V model: T2V and I2V are not separate
networks — they are the same network with a different inference-time per-frame
schedule. No CLIP, no concat: "observed" is encoded as nothing more than "noise
level zero."

## 6. How Wan2.2 TI2V implements it (and how we reuse it)

`in_dim == VAE z_dim == 48` — the DiT input is *just* the latent, no extra
channels. Two pieces make conditioning work, both **outside** the network:

1. **Latent clamp (re-applied every step).** The observed frame's clean latent
   `z` is written into the masked region each denoising step so it never drifts:
   `latent = (1-mask)·z + mask·latent`
   (`textimage2video.i2v`; our `FlowInferenceSampler` diffusion-forcing path).
2. **Per-token timestep.** `model2_2.WanModel.forward` takes a per-token
   timestep `[B, seq_len]` (it flattens it into a per-token time embedding). The
   observation tokens get `0`, the rest get the scheduler timestep. Our
   `Wan22DiTWrapper` accepts the ergonomic per-latent-frame form `[B, F']` and
   expands it across each frame's patch tokens.

`masks_like(..., zero=True)` builds the `(keep, noise)` mask pair: `mask = 0` on
frame 0, `1` elsewhere — driving both the clamp and the per-token timestep.

Adapter-first is preserved: the **frozen** base does the observation conditioning
natively; the **trainable adapter** still injects the action (its global time
conditioning takes a per-sample σ — the max over frames, since obs=0).

## 7. Subtleties & caveats

- **Loss must be masked.** Supervising the clean observation frame is wrong:
  its velocity target references an unobservable noise draw. We mask with
  `frame_mask` (trainer `_flow_loss`).
- **Re-clamp every step.** The scheduler step nudges *all* frames; without
  re-clamping the observation each step it would drift off the given frame.
- **Conditioning-frame noise augmentation.** Wan2.2's `masks_like` has a `p≈0.2`
  branch that puts the observation at a *small* non-zero σ instead of exactly 0
  during training (robustness to imperfect observations / train-test gap on the
  clamp). Not yet wired in our preprocessor (clean obs only) — a TODO.
- **Independent vs structured σ.** Fully independent per-frame σ is the general
  form (best for autoregressive rollout). Our I2V uses a *structured* schedule
  (one forced frame, one shared future σ). Richer schedules (monotone-in-horizon
  σ, sliding-window AR) are the natural next experiments for long rollouts.
- **Error accumulation.** Autoregressive use (clamp generated frames as the next
  step's observation) can drift; the paper's stabilisation relies on training
  with independent noise so the model is robust to imperfect (noisy) context.
- **History length.** `cond_frames > 1` (multi-frame observation history)
  is supported by the preprocessor but untested; it is the lever for giving the
  world model velocity/context, not just a single snapshot.

## 8. Why it is the right fit here

A world model should answer "given the current observation and these actions,
what happens next?" Diffusion forcing expresses that **as a noise schedule**:
present = σ 0, future = σ rising — no architectural conditioning added to the
frozen base. It turns the ill-posed "generate everything from noise" task into a
well-posed "predict the consequences of actions from a known state" task, which
is exactly what was missing from the Wan2.1-T2V setup.

## See also

- [[prediction-objectives]] — velocity / `v` parameterisation used here.
- [[../../20_Tickets/feat-wan22-ti2v-diffusion-forcing-i2v]] — the integration.
- [[../../memory/multimodal-adapter-broadening]] — the UWM independent-timestep
  plan this realises.
