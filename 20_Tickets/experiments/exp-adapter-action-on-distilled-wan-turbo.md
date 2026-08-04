---
type: exp
scope: adapter
status: open
priority: medium
created: 2026-08-03
updated: 2026-08-03
resolution:
resolution_note:
closed_at:
related: ["[[exp-shortcut-pdd-lora-distill-dc]]", "[[exp-shortcut-parallel-decoding-adapter-wan]]", "[[../feat-adapter-wan-per-frame-adaln]]"]
---

# Action adapter on an off-the-shelf distilled Wan (D2 × D4)

**DEFERRED** — parked while scope is distillation-only (Lukas, 2026-08-03). Logged
because it is the cheapest of the three routes and should not be lost.

## Idea (Lukas, 2026-08-03)

Skip training a distillation ourselves: take a **published few-step Wan** as the frozen
base and train the action adapter on it. Tests "does adaptation still work on a distilled
base?" for the cost of one adapter run and **zero distillation compute**.

## Repo correction

Lukas linked [`lightx2v/Wan2.2-Distill-Models`](https://huggingface.co/lightx2v/Wan2.2-Distill-Models) —
**wrong variant.** That repo is **A14B** (T2V/I2V, 14B, two-expert high/low-noise), full
DiT weights, 4-step. Our whole pipeline is **TI2V-5B**: different architecture, different
VAE, **16-channel latents vs our 48**. Every precomputed latent (`864x672`, 48-ch) would
be invalid, and 2×14B frozen experts is ~56 GB of weights before activations.
[`lightx2v/Wan2.2-Distill-Loras`](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)
is likewise A14B-only (rank-64 I2V LoRAs).

**The right artifact:**
[`quanhaol/Wan2.2-TI2V-5B-Turbo`](https://huggingface.co/quanhaol/Wan2.2-TI2V-5B-Turbo)
— our exact variant, 4-step, step- + CFG-distillation via Self-Forcing. Reported 121
frames @ 24 fps, 1280×704, in 4 steps. _Model-card claims, not verified by us._

## Why it is worth doing

The base handles speed, the adapter handles actions — the same decoupling as
[[exp-shortcut-pdd-lora-distill-dc]], but with the distillation already paid for by
someone else. If it works it is a direct D4 result: fast **and** action-conditioned.

## Honest caveat

This does **not** test the D2 failure. Our Wan action adapter measures temporal alignment
at chance, and the leading hypothesis is the **injection site** (cross-attention into the
residual stream vs per-frame AdaLN —
[[../../30_Knowledge/experiments/20260802-avid-wan-cleanroom-perframe-causal]],
[[../feat-adapter-wan-per-frame-adaln]]). Swapping the base for a distilled one does not
touch that. Expect this to answer "does the composition survive a distilled base?", not
"does the adapter follow actions now."

## DECIDED (Lukas, 2026-08-03): action conditioning only, no step-size conditioning

The adapter on this base runs **action-conditioned only**. `use_step_level_conditioning:
false`, no `shortcut_direction_weight`, no `multistep_consistency_weight`.

Rationale: `log2(d)` conditioning presumes a continuum of step sizes to interpolate over.
A step-distilled base has collapsed the step axis onto its own grid, so asking the adapter
to be step-size-aware on top of it is ill-posed. This makes the arm a **pure D2
measurement on a fast base**, with no D3 machinery to confound it.

## Timestep sampling — still changes, even without step-size conditioning

Dropping the step-size pathway does **not** remove the sampling problem. The adapter is
still trained at some `t`, and the base's output at that `t` is only meaningful on the
distilled grid — doubly so because `condition_on_base_outputs: true` feeds the base output
to the adapter as *input*.

Current sampling (`losses/flow_matching.py:53-86`): logit-normal `t = sigmoid(randn())`,
continuous over (0,1), plus resolution shift (`flow_shift_x1: 256`, `flow_shift_x2: 4096`)
and `sigma_shift: 5.0`. Smoke log `25166226`: *"median sigma 0.833; eval stays U(0,1)."*

**All three DECIDED (Lukas, 2026-08-03) — use the distilled grid and adjust the rest:**
1. **Sample `t` from the distilled grid** — uniform over `{t₀…t₃}`, replacing the
   logit-normal draw at `trainer.py:406`. New sampler, gated behind a config flag so the
   continuous path stays intact for every other arm.
2. **Use their shift, not ours.** `sigma_shift`/`inference_shift: 5.0` were tuned for the
   undistilled Wan. Read the value off the Turbo sampler config; do not guess.
3. **CFG.** CFG-distilled ⇒ no guidance pathway. Check `drop_condition_prob` / null-path
   plumbing, which assumes one.

No longer required (was only an issue under step-size conditioning):
4. ~~`eval_stepsize_blindness.py`'s dyadic ladder 1/128…1 is undefined on a 4-step model.~~
   Not used on this arm. **But** the action-sensitivity eval must be run *at the grid
   timesteps*, not at U(0,1).

Consequence to watch: the adapter sees only ~4 distinct noise levels (plus `t=0` on
observation frames under diffusion forcing). That matches deployment exactly, which is
good, but it is a far narrower training distribution than any arm so far — worth watching
for overfitting to the grid points.

## Other unknowns to check before committing

- Does the Turbo checkpoint keep Wan2.2 TI2V-5B's exact DiT layout (our vendored provider
  and the 48-ch VAE assume it)?
- Licence / redistribution terms for a thesis artefact.

## RESOLVED 2026-08-04: the checkpoint is DROP-IN COMPATIBLE

Downloaded and verified: `/scratch-shared/lbierling1/ckpts/Wan2.2-TI2V-5B-Turbo/model.pt`,
18.63 GiB, **825 tensors / 5.00B params**, loads clean.

(Correction: the HF API reports ~40 GB `usedStorage`, which counts dedup metadata,
not the file. 18.63 GiB IS the complete file — it had been done since 00:06 while
a hung `curl` process made it look like it was still downloading.)

**Key/shape diff against our vendored `ckpts/Wan2.2-TI2V-5B`:**

```
turbo tensors: 825      base tensors: 825
after rename -> missing: 0   extra: 0   shape mismatches: 0
VERDICT: DROP-IN COMPATIBLE
```

The checkpoint was saved from an **FSDP-wrapped** model. The entire "state-dict
remap" I had budgeted as major work is:

```python
k = k.removeprefix("model.").replace("._fsdp_wrapped_module.", ".")
```

**dtype is fp32** (hence 18.63 GiB for 5B params) — our provider loads `bf16`, so
cast on load.

### Quality gate BEFORE any training (Lukas, 2026-08-04)

Do not train an adapter on this base until the base itself is shown to produce
usable few-step video on our domain. Inference-only, no training:

1. Load with the rename + a shape assertion.
2. Generate at **4 steps** (its design point) on `configs/prompts/acwm_robotarm.yaml`.
3. Reference: undistilled Wan2.2 base at 50 steps.
4. Score with the configs' existing `psnr/ssim/lpips` + visual inspection.

`scripts/generate_shortcut_fewstep.py` exists and has never been run — this is its
first use.

**Risk this gate will surface:** Turbo is CFG-distilled, so it has no guidance
pathway. If our generation path assumes CFG it will error or silently degrade.
Better found here than after a training run.

## GATE PASSED — and this is the DECIDED training setup (Lukas, 2026-08-04)

Jobs `25192893` (prompt-free) and `25193375` (prompted), `scripts/gate_turbo_fewstep_quality.py`.
Both COMPLETED, no CFG failure, 1:11 wall for both models.

| run | mean | std | interframe_motion |
|---|---|---|---|
| Turbo @4, no prompt | −0.625 | 0.402 | 0.0326 |
| **Turbo @4, prompted** | **−0.704** | **0.313** | **0.0173** |
| Base @50, no prompt (guidance inert) | −0.631 | 0.325 | 0.0040 |
| Base @50, prompted + real CFG | −0.647 | 0.309 | 0.0117 |

**Lukas's verdict on the videos: the prompted distilled output "looks very nice" —
use exactly this setup for training.**

⚠️ Retracted: an earlier reading of "Turbo produces 8x more motion than the base"
compared against an UNPROMPTED, guidance-inert reference. Against the fair
(prompted + CFG) reference it is ~1.5x, not 8x. Quote the prompted row only.

### The setup to reproduce, exactly

```yaml
model:
  provider: wan2.2_turbo                      # Wan22TurboVideoModel; refuses guide_scale != 1.0
  pretrained_model_name_or_path: /scratch-shared/lbierling1/ckpts/Wan2.2-TI2V-5B-Turbo-hf
  freeze: true
```
- **prompt**: `configs/prompts/acwm_robotarm.contexts.pt`, key `__default__`
  (positive `[69,4096]`; `negative` `[126,4096]` is a plain shared tensor, NOT a dict —
  that asymmetry cost job `25193258`).
- **sampling_steps: 4**, **guide_scale: 1.0** (CFG-distilled — the provider raises otherwise).
- **shift: 5.0** — was flagged `_needs verification_` as inherited from the undistilled
  base. **Now empirically validated by output quality at 4 steps.** Still not the
  published value; if their sampler config ever surfaces, re-check.
- **geometry**: `frame_num 49`, `max_area 589824` -> 864x672, conditioning frame from
  `robot_arm/ind_test/episode_0.mp4`, seed 0.

### Still required for the TRAINING arm (not the gate)

1. **Grid timestep sampling** — draw `t` from `Wan22TurboVideoModel.timestep_grid()`
   (4 points at shift 5.0) instead of the continuous logit-normal at
   `losses/flow_matching.py:53`. Behind a config flag; every other arm keeps the
   continuous path.
2. `use_step_level_conditioning: false` — action conditioning ONLY (decided 2026-08-03).
3. Action-sensitivity eval must run **at the grid timesteps**, not U(0,1).

## DONE 2026-08-04: grid timestep sampling implemented

`trainer.py::_base_timestep_grid` + the flow sampling branch. Opt-in via
`training.extra.sample_timesteps_from_base_grid: true`; the base must expose
`timestep_grid()` (only distilled bases do). Overrides:
`base_grid_sampling_steps`, `base_grid_shift`.

- **Raises** if the flag is set on a base with no `timestep_grid()` — it must not
  silently fall back to continuous sampling, which would produce a plausible-looking
  but wrong run.
- **Caches** the grid (fixed per run) and **prints it once** at startup, so the log
  records which grid was trained on.
- Defaults off; 27 tests pass, every undistilled arm keeps the logit-normal draw.

## ⚠️ GOTCHA: the Turbo config is NOT a one-line edit of an existing action config

`provider: wan2.2` and `provider: wan2.2_turbo` are **different code paths**:

| | `wan2.2` | `wan2.2_turbo` |
|---|---|---|
| class | `Wan22DiTWrapper` (vendored) | `Wan22TurboVideoModel` -> `WanTI2VVideoModel` (upstream) |
| checkpoint knob | `model.extra.wan_config_path` | `model.pretrained_model_name_or_path` (ckpt DIR) |
| `extra` surface | `latent_channels`, `temporal_length`, `max_area`, … | `dtype`, `offload_model` |

So the action-only Turbo config must MERGE two sources:
- **action/adapter setup** from `configs/wan22/diffusion_wan22_avid_xattn_gatefix_acwm_robotarm.yaml`
  (already `use_step_level_conditioning: false` and `shortcut_direction_weight: 0.0`
  — i.e. already action-only, no D3 machinery)
- **external-provider surface** from `configs/wan22/diffusion_wan22_dcunet_output_metaworld.yaml`
  (the ONLY existing `wan2.2_external` config, and it is metaworld+dcunet, not robot-arm)

Not attempted — the merge needs the external provider's config surface verified
against a real build, not guessed. Do this first and smoke it before any training.

## STILL TODO
- Action-sensitivity eval at the GRID timesteps, not U(0,1)
  (`scripts/eval_action_sensitivity.py`). Otherwise it measures the adapter at noise
  levels the distilled base cannot operate at.
