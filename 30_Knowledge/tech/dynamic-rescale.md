---
type: tech-note
status: living
last_updated: 2026-05-15
sources:
  - "code: src/generative_flow_adapters/losses/diffusion.py"
  - "code: src/generative_flow_adapters/training/trainer.py"
  - "code: src/generative_flow_adapters/inference/diffusion.py"
  - "code: src/external_deps/lvdm/models/ddpm3d.py"
  - "code: src/generative_flow_adapters/models/base/dynamicrafter.py"
commit: 3a235c8
relevance: D2  # action-conditioned diffusion world models
---

# Dynamic rescale ("the dynamic scaler") in diffusion training

> Documents the `use_dynamic_rescale=True` code path: a DynamiCrafter-style
> per-timestep data-side SNR shaping applied during diffusion training and
> reversed at inference. Often referred to in conversation as "the dynamic
> scaler"; in the codebase it is called `use_dynamic_rescale` / `scale_arr` /
> `scale_x_start`.

## TL;DR

Before noising, the clean latent `x_0` is multiplied by a per-timestep
factor `scale_arr[t]` that ramps from `1.0` at `t=0` down to `base_scale`
(default `0.7`) by `t = turning_step` (default `400`) and stays at
`base_scale` for the rest of the schedule. The network is then trained on
the standard `q_sample(x_0_scaled, t, ε)` and predicts the target (noise /
velocity / `x_0`) **in that attenuated space**. At inference, the custom
DDIM step un-attenuates by `prev_scale / cur_scale` between consecutive
steps so the final sample lands back in unscaled, VAE-compatible latent
space. The mechanism is a no-op when `use_dynamic_rescale=False` (the
default).

## Where it lives

| Piece | File | Lines |
|---|---|---|
| `scale_arr` construction, `scale_x_start` method | `src/generative_flow_adapters/losses/diffusion.py` | 27–79 |
| Training-step call site (scales `x_0` before `q_sample` and `get_target`) | `src/generative_flow_adapters/training/trainer.py` | 91–104 |
| Objective construction (reads `use_dynamic_rescale`, `base_scale`, `turning_step` from the schedule config) | `src/generative_flow_adapters/training/trainer.py` | 32–43 |
| Inference reversal (custom DDIM step) | `src/generative_flow_adapters/inference/diffusion.py` | 121–185 |
| Config plumbing (YAML → `DiffusionScheduleConfig`) | `src/generative_flow_adapters/models/base/dynamicrafter.py` | 179–195 |
| Upstream reference (LVDM / DynamiCrafter) | `src/external_deps/lvdm/models/ddpm3d.py` | 506, 531, 549–554, 765–766 |

All line numbers are at commit `3a235c8` on `main`.

## Training-time mechanism

`DiffusionTrainingObjective.__init__` builds the schedule when the flag is
set (`losses/diffusion.py:61–66`):

```python
if use_dynamic_rescale:
    ramp = torch.linspace(1.0, base_scale, turning_step)   # length 400
    flat = torch.full((timesteps,), base_scale)            # length 1000
    self.scale_arr = torch.cat([ramp, flat]).to(torch.float32)
else:
    self.scale_arr = None
```

`scale_arr` is `(turning_step + timesteps,)` long. With the defaults
(`turning_step=400`, `timesteps=1000`) it is `(1400,)`. Only indices `[0,
timesteps)` are ever read at training time (because `sample_timesteps`
draws `t ∈ [0, timesteps)`), so the trailing `turning_step` entries are
dead buffer space — kept here to match the reference LVDM construction at
`external_deps/lvdm/models/ddpm3d.py:549–554`.

Schedule values at the defaults:

- `scale_arr[0]   = 1.0`        (clean end of the schedule — no attenuation)
- `scale_arr[399] = 0.7`        (`base_scale`; ramp finishes here)
- `scale_arr[400…999] = 0.7`    (flat region)

`scale_x_start(x_start, t)` (`losses/diffusion.py:68–79`) is the per-batch
gather:

```python
scale = extract_into_tensor(self.scale_arr, t, x_start.shape)
return x_start * scale
```

The training step calls it *before* both `q_sample` and `get_target`
(`trainer.py:91–104`):

```python
target_scaled = self.diffusion_objective.scale_x_start(target, t)
x_t = self.diffusion_objective.q_sample(x_start=target_scaled, t=t, noise=noise)
prediction = self.model(x_t, t, batch.get("cond"))
target_tensor = self.diffusion_objective.get_target(
    prediction_type=prediction_type or "noise",
    x_start=target_scaled,    # <-- scaled x_0 passed through to v / eps / x0 target
    x_t=x_t,
    t=t,
    noise=noise,
)
```

The consequence: `x_t`, the regression target (`v`, `ε`, or `x_0`), and
the model's prediction all live in the same `scale_arr[t]`-attenuated
space. The loss is plain MSE between prediction and target
(`losses/diffusion.py:10–11`); no extra reweighting term is introduced —
the "scaling" is entirely in the data, not in the loss.

## Inference-time reversal (DDIM)

`DiffusionInferenceSampler._scheduler_step`
(`inference/diffusion.py:121–129`) routes to a custom DDIM step when both
conditions hold:

```python
if getattr(self.objective, "use_dynamic_rescale", False) and self.scheduler_name.lower() == "ddim":
    return self._dynamic_rescale_ddim_step(scheduler, model_output, timestep, sample)
```

`_dynamic_rescale_ddim_step` (`inference/diffusion.py:131–185`) converts
the model output to `(pred_x0, pred_eps)` regardless of parameterisation
(handles `v_prediction`, `epsilon`, `sample`), then re-buckets the
predicted `x_0` between adjacent scale-arr entries before the standard
DDIM update:

```python
scale_arr = self.objective.scale_arr.to(...)
cur_scale  = scale_arr[timestep_int]
prev_scale = scale_arr[max(prev_timestep, 0)]
pred_x0    = pred_x0 * (prev_scale / cur_scale)
return sqrt_alpha_t_prev * pred_x0 + sqrt_one_minus_alpha_t_prev * pred_eps
```

`pred_eps` is **not** rescaled — the docstring (lines 164–166) notes this
matches the reference DDIM exactly. Because `scale_arr[0] = 1.0`, the
final denoising step (which lands at `prev_timestep ≤ 0`, so
`prev_scale = scale_arr[0] = 1.0`) multiplies `pred_x0` by `1.0 / cur_scale`,
bringing the output back into unscaled, VAE-decodable latent space.

The reversal is the LVDM `p_sample_ddim` routine, ported into this repo's
diffusers-based sampler — the docstring at `inference/diffusion.py:143–144`
points back to `external_deps.lvdm.models.samplers.ddim.DDIMSampler.p_sample_ddim`
as the reference.

**Important scope limit:** the reversal is **only wired for DDIM**. If
`use_dynamic_rescale=True` is set with `inference_scheduler="ddpm"`, the
custom branch is not taken (`inference/diffusion.py:122`), the default
diffusers DDPM step runs, and the predicted latents will stay in
`scale_arr[t]`-attenuated space — i.e. silently mis-scaled at decode. Use
DDIM, or extend the branch, when this flag is on.

## Config plumbing

The flags travel via the per-backbone `diffusion_schedule_config`. For
DynamiCrafter (`models/base/dynamicrafter.py:179–195`),
`_load_diffusion_schedule_config` reads the YAML at
`model.params.{use_dynamic_rescale, base_scale, turning_step}` and packs
them into a `DiffusionScheduleConfig` attached to the model. The trainer
then pulls those values off the model (`trainer.py:32, 40–42`):

```python
diffusion_schedule = getattr(model, "diffusion_schedule_config", None) or {}
...
use_dynamic_rescale=bool(diffusion_schedule.get("use_dynamic_rescale", False)),
base_scale=float(diffusion_schedule.get("base_scale", 0.7)),
turning_step=int(diffusion_schedule.get("turning_step", 400)),
```

So the flag is per-backbone (set in the upstream model's checkpoint YAML)
rather than per-experiment, which matches the intent that it preserves the
input-distribution contract the frozen base model was trained under.

## Why it exists (analysed estimate)

The flag is `False` by default in this repo's objective, but `True` for
the DynamiCrafter backbone the adapters wrap. The rationale, as best as
the code comments and upstream construction support:

1. **Data-side SNR shaping.** Reducing the magnitude of `x_0` at high `t`
   reduces the effective signal-to-noise ratio at those timesteps without
   touching the β schedule. This is functionally an alternative to (or
   stack-on with) `rescale_betas_zero_snr`, which the same objective also
   supports (`losses/diffusion.py:48–49`). The training comment at
   `trainer.py:91–94` describes it as "DynamiCrafter-style data SNR shaping".

2. **Preserving the base model's input contract.** The frozen
   DynamiCrafter weights were trained with this scaling on. If we drop the
   adapter into the same backbone and train without it, the model sees a
   different `x_t` distribution at every `t` than it was trained on,
   which would degrade the base prior the adapter is meant to leverage.

Both points above are *analysed estimates* from reading the code path and
upstream construction — they are not asserted by an experiment in this
repo (no ablation run is logged in `30_Knowledge/experiments/` at the time
of writing). Worth confirming with an A/B (`use_dynamic_rescale` on vs off,
same adapter, same data) before citing as fact in the thesis. → log as a
ticket if not already open.

## Gotchas worth flagging

- **No-op when off but always present.** `use_dynamic_rescale=False` makes
  `scale_x_start` an identity (`losses/diffusion.py:74–75`) and `scale_arr`
  is `None`. There is no other side-effect, so leaving the call in the
  trainer is safe.
- **Inference branch only fires for DDIM.** See the scope limit above.
  DDPM inference + `use_dynamic_rescale=True` is a silent mis-scale.
- **The schedule is fixed at constructor time.** `base_scale` and
  `turning_step` cannot be changed mid-run without rebuilding the
  objective. If you want to ablate the schedule shape (e.g. ramp duration,
  asymptotic `base_scale`), each variant is a separate run.
- **Buffer length is `turning_step + timesteps`, not `timesteps`.** Indexing
  past `timesteps - 1` is silently valid but never happens at training
  time. The trailing region exists to mirror the reference LVDM buffer; it
  is not used.
- **The loss is unchanged.** This is purely a data-side mechanism. No
  `min-SNR`-style loss weighting or SNR-aware reweighting is layered on
  here — if we ever want that, it's a separate change in
  `losses/diffusion.py`.
- **Adapters inherit the contract.** Because the trainer scales `x_0`
  before the forward pass, the adapter `Δ_φ` also sees the attenuated
  `x_t`. This is the correct behaviour for adapting a base model that was
  pretrained with the same shaping, but means adapter outputs are
  themselves living in scaled space — anything that compares adapter
  outputs to raw `x_0` directly (e.g. an output-space probe) needs to
  account for this.

## Defaults at a glance

| Param | Default | Source |
|---|---|---|
| `use_dynamic_rescale` | `False` (objective default); `True` for the DynamiCrafter checkpoint config | `losses/diffusion.py:27`; `models/base/dynamicrafter.py:192` |
| `base_scale` | `0.7` | `losses/diffusion.py:28`; `trainer.py:41` |
| `turning_step` | `400` | `losses/diffusion.py:29`; `trainer.py:42` |
| `timesteps` | `1000` | `losses/diffusion.py:21` |

## Related

- [[../../10_now/architecture]] — touches on the DynamiCrafter wrapping
  contract; should reference this note from the diffusion section.
- [[../related-work/_MOC]] — once a related-work note exists for
  DynamiCrafter / Zero-Terminal-SNR (Lin et al. 2024), link it here so the
  rationale chain is one click away.

## Open follow-ups

- [ ] Ablation: `use_dynamic_rescale={True, False}` × adapter family,
      measure adapted-rollout fidelity. _No run logged yet._
- [ ] Decide whether DDPM-inference users should hit an explicit error
      when `use_dynamic_rescale=True`. Currently silent. → consider opening
      `50_Decisions/open/ddpm-dynamic-rescale-guard.md`.
- [ ] Confirm `_dynamic_rescale_ddim_step` numerically matches LVDM
      `p_sample_ddim` on a synthetic case. _needs verification._
