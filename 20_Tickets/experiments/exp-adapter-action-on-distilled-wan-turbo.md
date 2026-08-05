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

> **⭐ Part of the efficiency axis** —
> [[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]] (decided
> 2026-08-04). This ticket is **L3 — free** (speed in the base itself).
> **Requires a matched conditioning-only control** (same adapter, base, data
> and depth, acceleration off) — without it the pre-registered comparison is
> unmeasurable. The predicted ordering is registered in that note **before**
> any level ran; do not restate it post hoc.


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

## UNBLOCKED 2026-08-04 — grid verified firing (job 25209235)

```
COMPLETED 00:30:29 0:0
SIGMA-GRID MARKER: 1
base=Wan22TurboVideoModel (external wan.WanTI2V)
[sigma-grid] drawing sigma from the base's distilled grid (4 points): [1.0, 0.9375, 0.83333, 0.625]
```
Grid concentrated at high sigma, as a shift-5 4-step schedule should be — a sanity
check, not just a marker.

### Ready to launch
`configs/wan22/diffusion_wan22turbo_action_robotarm.yaml` +
`jobs/experiments_cluster/acwm_phys/shortcut/submit_train_wan22turbo_action.sh`
(`--account=gusei17535`, `--mem=180G`).

### It took FIVE smokes — four silent no-ops, one per layer
Each returned a green job or died at startup; none of the flags errored on their own.
1. `25196512` COMPLETED — script hardcoded `config.model.provider = "wan2.2_external"`,
   clobbering `wan2.2_turbo`. Turbo WEIGHTS loaded, Turbo SEMANTICS did not.
2. `25205710` — grid flag set but base had no `timestep_grid()` (consequence of 1).
3. `25205782` — provider fixed; CFG guard refused `guide_scale=5.0`. Config had
   `inference_guide_scale: 5.0`; a CFG-distilled base needs 1.0.
4. `25205823` COMPLETED, marker 0 — grid code sat in `WanBatchPreprocessor.__call__`,
   but `Wan22DiffusionForcingPreprocessor` OVERRIDES `__call__` and draws its own sigma.
5. `25209183` — `NameError`: the import patch used `str.replace` with no match check and
   silently no-oped (script imports from the PACKAGE, not the module).

**Fix (Lukas's suggestion) was structural, not another flag:** a `_draw_sigma` seam on
`Wan22DiffusionForcingPreprocessor`, overridden by
`Wan22TurboDiffusionForcingPreprocessor`, selected **by provider** in the training
script — so distilled base and distilled sampling cannot drift apart.

`sigma_shift` is deliberately NOT applied on the grid path (the grid already encodes
its shift; both would move training off the grid). Eval keeps the continuous draw.

### ⚠️ Watch on the first real eval
- `eval_adapter_gate_mean = 0.00000`, `gate_std = 0.00000` — gate pinned. Same
  signature as `bug-adapter-gate-cap-equals-init-freezes-gate`. May be init at 12
  steps; if still exactly 0 after a few hundred, the adapter cannot blend.
- `eval_denoise_adapter_delta = -1.78` (base 0.148 -> adapted 1.93). Expected from a
  scratch-init adapter this early; the number to track.

### Lesson
Every failure above produced a no-op, not an error. Explicit markers (`[sigma-grid]`,
`base=`) were the only thing that caught them — the same role as the
`shortcut_direction_loss/N###` check for the PDD arms.

---

## 2026-08-05 — grid-consistent eval, VRAM sizing, SBU ledger

### Two defects found in the first full launch (`25218334`, cancelled at 10 min)
Both were config values **inherited from the undistilled Wan arm** and both put the
distilled base off its grid at eval time — the exact failure the `_draw_sigma` seam
fixed on the *training* side.

1. `eval_step_schedule: [{8, 0.125}, {50, 0.02}]` → the trainer's own guard fired:
   `trainer.py:1402 RuntimeWarning: sampling_steps=50 on a 4-step distilled base.`
   Now a single row, `{num_steps: 4, step_level: 0.25}`.
2. `quality_eval_num_steps: 10` → now `4`.

Cost of the defect, not just its wrongness: the 50-step row is ~75 s per rollout ×
2 models × 3 samples ≈ 7.5 min **per eval cycle**, plus the same at step 0. Job
`25218334` spent its entire 10-minute life in the step-0 eval and never reached a
training step. With both rows on-grid the eval is ~12× cheaper.

Generation and training now share one grid by construction: `inference_shift: 5.0`
== `base_grid_shift: 5.0`, and both sides call `get_sampling_sigmas(4, 5.0)` →
`[1.0, 0.9375, 0.83333, 0.625]`.

### Batch sizing is measured, not inherited — and the inherited number was wrong
`submit_train_wan22turbo_action.sh` carries a `bs=12 → 86.5 GiB` note (D3 shortcut
arm) and a `bs=12 → ~37 GiB` note for the action arm. Reasoning from the second one
— this arm is `shortcut_direction_weight: 0.0` and `multistep_consistency_weight:
0.0` (config.py default), so **one** adapter forward per step, therefore cheap — gave
"39 % of the card, lots of headroom". **That is false.** Measured:

| bs | outcome | allocated |
|---|---|---|
| 12 | **trains** | **76.9 GiB / 93 GiB = 82 %** |
| 16 | OOM | — |
| 20 | OOM | 92.2 GiB |
| 24 | OOM | 90.8 GiB |
| 28 | OOM | 92.6 GiB |
| 32 | OOM | 88.5 GiB (needed +5.2) |

All OOMs land in `rope_apply` (`wan/modules/model.py`), in the **frozen base's**
forward under `no_grad` — not in the adapter's backward graph. That is why "one
adapter forward per step" did not predict the cost: the dominant term is the base's
rotary embedding over 14 175 tokens/clip, which `no_grad` does not shrink. Note the
OOM totals barely move with batch size (90.8–92.6 GiB across bs 20–32) — the
allocator fills to the ceiling and dies wherever the next big block lands, so a
failed run's number is a *lower bound*, useless for extrapolation. Only a surviving
run's `peak_vram` is a measurement.

**Consequence: bs=12 is already the maximum this arm fits, at 82 % occupancy.**
There is no headroom to reclaim, and the usual "bigger batch vs. more optimizer
steps" trade does not arise here — the memory-maximising batch and the
convergence-maximising batch are the same one.

Added `peak_vram=X/YGiB(Z%)` to the trainer step line
(`trainer.py`, `log_every` branch) so batch sizing stops being read off a comment.

`submit_probe_turbo_vram.sh <bs>` sizes it empirically: it disables the step-0
baseline eval and sets `inference_every_n_steps: 6`, because the number that matters
is the gen_eval transient landing on a **warm** allocator pool — surviving the step-0
eval proves nothing, no training block is allocated yet.

### SBU ledger (billing = 192/h at `--mem=180G`, 1×H100)
Attributable to this workstream since 2026-08-03:

| Job | Name | State | Elapsed | SBU |
|---|---|---|---|---|
| 25187395–25187746 | dc-pdd-smoke 1–4 | FAILED/TIMEOUT | 42 min | 135 |
| 25188330 | dc-pdd-full | TIMEOUT | 10.0 h | 1921 |
| 25192893–25193375 | turbo-gate ×3 | mixed | 4.5 min | 14 |
| 25196512–25209235 | wan22turbo-action smokes ×6 | mixed | 99 min | 316 |
| 25217047 | turbo-shift-scan | COMPLETED | 2.5 min | 8 |
| 25218334 | wan22turbo-action-full | CANCELLED | 10 min | 32 |
| 25219073 | turbo-vram-probe (bs=32) | — | — | ~25 |

**≈ 2 450 SBU of the 20 000 cap.** `dc-pdd-full` alone is 78 % of that — a 10 h run
whose objective turned out to be endpoint inversion, not PDD.

Account-level (`accinfo`, budget EINF-17535/L1): 530 185 used / 469 815 remaining of
1 000 000. A further ~11 400 SBU on this account since 2026-08-03 belongs to
concurrently-running `ea-*` / `wan-actiononly*` jobs **not launched from this
workstream** — flagged so the two are not conflated in the cap.

### Does the gen_eval fit on top of an 82 %-full card? Yes — measured
The real risk of an 83 %-occupancy run is not the training step (that is steady) but
the periodic `gen_eval` landing on a **warm** allocator pool at step 200. Probe
`25219563` (bs=12) fired a gen_eval at step 6 and cleared it:

```
step 6  peak_vram=76.9/93GiB(82%)     <- training only
[eval-mem] rollout-start:      alloc=14.13G reserved=14.32G
[eval-mem] after-base-gen:     alloc=14.13G reserved=44.38G
[eval-mem] after-adapted-gen:  alloc=14.13G reserved=44.39G
step 7+ peak_vram=77.7/93GiB(83%)     <- +0.8 GiB, no OOM
```

`alloc` drops to 14.1 G during eval: the training blocks are freed and the eval's
~30 GiB of transients reuse them (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
So the eval transient is NOT additive on top of the training peak — the +0.8 GiB is
the whole cost. The probe's TIMEOUT is its own 25-min cap at step 12/14, not a
failure; loss fell 3.02 -> 1.72 over 12 steps.

### Overnight run
`25221298` — `--time=10:00:00`, `--mem=180G`, `--account=gusei17535`, `BATCH_SIZE=12`
(`submit_train_wan22turbo_action.sh` now honours `${BATCH_SIZE:-12}`; it previously
hardcoded 12 and silently ignored the environment).

Probe cost for the sizing sweep (bs 32/28/24/20/16/12): 6 jobs, **124 SBU**. That
bought a measured ceiling in place of a comment that was off by a factor of two.

Startup confirmed on `gcn147`: `base=Wan22TurboVideoModel`, `batch_size=12`,
`[sigma-grid] ... [1.0, 0.9375, 0.83333, 0.625]`, step-0 baseline eval completed, and
**zero** occurrences of `sampling_steps=50` in either stream — the off-grid eval is
gone. The whole step-0 eval + model load now takes ~14 min including wandb video
encoding; under the old 8/50-step schedule job `25218334` had not finished it in 10.

### SBU total for this workstream
| | SBU |
|---|---|
| spent so far (incl. probe sweep) | 2 596 |
| overnight run `25221298`, 10 h projected | 1 920 |
| **projected total** | **4 516** of the 20 000 cap |

Separately, 11 742 SBU since 2026-08-03 belongs to concurrent `ea-*` /
`wan-actiononly*` jobs on the same account, **not launched from this workstream** —
worth confirming who owns them before they are counted against the cap.
