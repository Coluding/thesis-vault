---
type: exp
scope: shortcut
status: open
priority: high
created: 2026-08-04
updated: 2026-08-04
resolution:
resolution_note:
closed_at:
related: ["[[exp-shortcut-parallel-decoding-adapter-wan]]", "[[exp-shortcut-pdd-lora-distill-dc]]"]
---

# Idea A on DC — parallel-decoding ADAPTER over a frozen DynamiCrafter base

The DC twin of [[exp-shortcut-parallel-decoding-adapter-wan]]. Base stays FROZEN;
the adapter carries N replicated heads and emits L corrected directions per pass.

**Not the paper's method.** PDD's student is the backbone with replaced heads
(Eq. 12→13) — that is [[exp-shortcut-pdd-lora-distill-dc]] (Idea B). This ticket
is our adapter transposition of the idea.

---

## Implementation spec (2026-08-04)

Lukas: "can we please just run proper pdd with dynamicrafter". What ran as
`dc-pdd` (job 25188330) is **not** PDD — it is the shortcut scaffolding with the
target swapped (denoise loss still on, `step_level` conditioning on,
`anchor_prob: 0.5`, off-policy, one velocity per pass). Useful as a clean
bootstrapped-vs-not comparison; not evidence about PDD.

### What proper PDD needs — four changes

**1. N heads on the adapter.** DC's UNet output stage
(`external_deps/lvdm/modules/networks/openaimodel3d.py:693`):
```python
self.out = nn.Sequential(normalization(ch), nn.SiLU(),
    zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)))
```
`model_channels: 32`, `out_channels: 4` -> final conv ~3.5k params.
**N=8 costs ~28k on an 11.2M adapter (0.25%).** Already `zero_module`-init, so
every head starts at the base velocity — PDD's teacher-init prior, for free.
Change: `params["out_channels"] *= N` in `DynamicCrafterOutputAdapter.__init__`
(note it is already multiplied by `output_channel_multiplier`), split in `forward`
to `[B, N, C, T, H, W]`.

**2. On-policy rollout.** PDD Algorithm 2, with `ddim_micro_step_v` as the step:
```
n   = block start on the timestep grid
u   = adapted_model(x_n, t_n)          # ONE pass -> N velocities
k   = randint(n, n+L_max)
x_k = x_n                               # roll on the ADAPTER's own outputs
for l in range(n, k): x_k = ddim_micro_step_v(x_k, u[l], t_l, t_{l+1}, ...).detach()
tgt = one FROZEN-BASE DDIM step at (x_k, t_k)   # 1 teacher call, not 4
loss = mse(u[k], tgt)
```

**3. Replace the primary loss.** This is the real work. `trainer.py:384` does
`loss = self.loss_fn(prediction, target)` and shortcut terms are *added*. PDD has
ONE term. Needs a dedicated `_pdd_forward_and_loss` dispatched from
`training_step`, NOT a config flag on the existing path.

**4. Config.** `use_step_level_conditioning: false` (head index replaces the
second time coordinate), `shortcut_anchor_prob: 0`, all shortcut/consistency
weights 0, no denoise loss.

### Deferred
Head fusion at inference (PDD Eq. 15) — training-time correctness first. Fusion
requires the composition stay linear in the adapter output, so the mask must be
**shared** across heads, not per-head.

### Cost
DC ~33 s/step at bs=24 (job 25141979); PDD needs 1 teacher call/step vs the
current 4, so expect FASTER than the 45 s/step job 25188330 is showing.

---

## 2026-08-06 — implemented and launched (job `25260518`)

All four changes are in. Config:
`configs/dynamicrafter/diffusion_dc_pddA_adapter_actionfree_robotarm.yaml`,
job script `jobs/experiments_cluster/acwm_phys/shortcut/submit_train_dc_pddA_adapter.sh`.
Launched 2026-08-06 00:54 CEST, `--time=08:00:00`, bs=56.

**The acceptance test passes on a real run.** Local RTX 3090 smoke (bs=1, 256×320,
4 steps): metrics carry `pdd_loss` / `pdd_loss_k###` and **no**
`shortcut_direction_loss/N###` key. The cluster run's step-0 eval reproduces this
(`eval_pdd_loss=0.38875`, `eval_pdd_num_k=8`).

### What shipped

1. **N heads** — `DynamicCrafterOutputAdapter(parallel_heads=N)`, widening the final
   conv rather than adding N modules. `_split_heads` → `[B, N, C, T, H, W]`; no-op at
   N=1 so every existing arm is byte-identical. `affine` + N>1 raises (affine already
   packs 2C; the split order would silently mis-pair scales with shifts). Gate is
   deliberately **not** widened.
   **Cost correction:** the estimate above said "N=8 costs ~28k on an 11.2M adapter
   (0.25%)". Measured: 11,203,876 → 11,211,968, i.e. **8,092 params, 0.072%**. The
   estimate assumed a 3-D conv; DC's output stage is 2-D (`32→4`, 3×3).
2. **On-policy rollout + 3. one-term loss** — `Trainer._pdd_forward_and_loss`,
   dispatched from `training_step` on `training.extra.objective: pdd`, calling the
   existing `pdd_objective`.
3. **Config** — as specified, plus `composition: add` (see below).

### Decisions taken while building

- **`add`, not `avid_mask_mix`.** With `output_mask: false` DC's final conv is
  `zero_module`-init, so every head emits 0 at step 0 and `u_j == v_base` for all j —
  PDD's teacher-initialised student (Eq. 12) for free. With `output_mask: true` only
  the *mask* is zero-init and the prediction head starts at noise. `add` also drops
  the gate and its saturation failure mode
  ([[../bug-adapter-gate-cap-equals-init-freezes-gate]]) for no loss of generality.
- **Head-aware composition.** `AdaptedModel._compose` now lifts `base_output` (and the
  gate) over the head axis. Without it `[B,C,T,H,W] + [B,N,C,T,H,W]` right-aligns and
  raises — except at `B == N`, where it silently returns a wrong tensor. Pinned in
  `tests/test_pdd_parallel_heads.py::test_batch_equal_to_heads_does_not_silently_broadcast`.
- **N must equal `pdd_grid_size`.** Head j supervises interval j by absolute index.
  `N < n_grid` indexes out of bounds; `N > n_grid` leaves tail heads never supervised
  and never used — silent, and would read as a capacity result. The trainer raises.
- **Eval differs from training on purpose.** Block start pinned to 0 and every
  interval swept. The teacher's mean-velocity magnitude varies ~8× across the
  schedule (measured: `pdd_loss_k002=0.042` vs `k006=0.330` in one cycle), so an
  eval_loss from random draws would track the draw, not training. n=0 is also the
  from-pure-noise regime few-step generation actually uses.

### Still deferred (unchanged, and now blocking the headline claim)

Head-indexed / fused rollout at inference (Eq. 15). The base's native DDIM loop calls
the adapter once per denoiser step and expects one velocity; under parallel decoding
it gets N. **Generative eval is therefore OFF on this arm**
(`inference_every_n_steps: 0`) — these runs test training-time correctness only. No
few-step video or timing number can come out of them until the rollout is wired.

### Tests
`tests/test_pdd_parallel_heads.py` — 15 tests, all passing. Covers composition
broadcast (incl. the `B == N` trap), the metric-key acceptance test, teacher-call
accounting (1 per supervised index + 1 student pass), the head/grid guard, the
`target`-vs-`x0` key, and frame-masked reduction.

### SBU ledger — PDD workstream, 2026-08-06 (billing 192/h at `--mem=180G`, 1×H100)

| Job | Name | State | SBU (budgeted) |
|---|---|---|---|
| 25260518 | dc-pddA-adapter | launched 00:54, `--time=08:00` | ≤ 1,536 |
| 25260768 | dc-pddB-lora | launched 01:05, `--time=08:00` | ≤ 1,536 |

Worst case for the pair: **3,072 SBU** of the 15,000 ceiling set 2026-08-05.
Zero SBU were spent on smoke-testing: both arms were smoke-tested on the local
RTX 3090 against the real `ckts/dynami512.ckpt` and the real robot-arm mp4s
(`ds/acwm-phys/kinematics/robot_arm/ind_train`, 2002 episodes), at bs=1 / 256×320.

### Sizing — corrected 2026-08-06, and the correction matters

The launch at bs=56 (`25260518`) was cancelled after producing **zero** training steps
in 25 min, while its step-0 eval had done 8 batches in 5 min at the same batch size.
Relaunched at bs=24 (`25260992`): trains at **0.03 steps/s (33 s/step), peak_vram
39.7/93 GiB (42%)**.

**The binding constraint on this arm is host RAM in the DataLoader, not VRAM.** Evidence:

- At bs=24 the GPU is at 42%; extrapolating (~12 GiB static + ~1.15 GiB/sample, from the
  sibling arm's bs=8→20.9 / bs=24→38.7) puts bs=56 at ~76 GiB, which **fits**. So the
  bs=56 failure was not GPU memory.
- `sstat` on the bs=56 job showed RSS climbing past 65 GB against the 180 G cgroup with
  only ~1 core busy — thrash, not compute.
- The process's main thread sat in `poll_schedule_timeout` (waiting on worker queues),
  and its 8 workers each carried ~17 GB RSS.

33 s/step at bs=24 exactly reproduces the sibling DC arm's documented rate (job
25141979) — an independent confirmation that PDD's step cost is *not* worse than the
path it replaces, as the ticket predicted (1 teacher call vs the shortcut arm's 4).

**Next run on this arm:** raise the batch AND cut `--num-workers` together (now wired as
`NUM_WORKERS` in the job script). Raising batch alone reproduces the stall; the GPU has
~55% headroom that host RAM is currently preventing us from using.

⚠️ **Do not tail these logs from the login node and conclude a job is stuck.** GPFS
caching made a healthy run look frozen for 40 minutes; `sacct`/`sstat` and an
`srun --overlap` into the node are the reliable views.

---

## 2026-08-06 03:20 — BOTH FIRST RUNS DISCARDED: the mean-velocity target was ~1000× too small

Jobs `25260992` (A) and `25261240` (B) were cancelled ~2 h in. They were optimising a
unit conversion, not parallel decoding. Relaunched as `25262886` (A) / `25262887` (B).

### What gave it away

Both arms started at **eval_loss ≈ 0.387** despite being teacher-initialised (Eq. 12) —
A via its `zero_module` head, B via replicated pretrained weights. A teacher-initialised
student should start at a *small* loss: over one short interval the mean velocity is
close to the instantaneous one. Starting at 0.387 for both, identically, is the
signature of a target that is ≈ 0 in the student's units, so that loss ≈ E[‖v‖²].

Then B fell to **0.00174 by step 200** while A was still at 0.234 — a 200× gap between
two arms solving the same problem. That is not a capability difference; B can rescale
its replicated conv directly, A must learn to emit ≈ −v_base.

### The bug

`_pdd_grid` builds the discretisation in **timestep units** (999 → 0), because that is
what the solver and the network need for schedule lookups. `_mean_velocity` then divided
the displacement by `dt` in those same units — `dt ≈ 125` for 8 intervals — while the
network predicts `v` with respect to **normalised time** `t ∈ [0,1]`.

Measured directly (`DiffusionTrainingObjective(1000)`, `ddim_micro_step_v`, 8 intervals):

| interval | displacement | target, dt in steps | target, dt normalised | teacher \|v\| |
|---|---|---|---|---|
| k=0 | 0.0538 | **0.00043** | 0.431 | 0.793 |
| k=3 | 0.1374 | **0.00110** | 1.100 | 0.793 |
| k=6 | 0.1801 | **0.00144** | 1.442 | 0.793 |

Three orders of magnitude. The dominant gradient was "output a much smaller number".

### The fix

`pdd_objective(..., time_scale=...)`, applied only to `dt` in `_mean_velocity`; the
trainer passes `num_train_timesteps`. Two tests, both verified non-vacuous by forcing
the old behaviour and watching them fail:

- `test_teacher_initialised_student_starts_near_zero_loss` — Eq. 12's guarantee stated
  as a bound against `E[‖v‖²]`. Fails at 0.238 vs the 0.064 threshold under the bug.
- `test_time_scale_puts_the_target_in_the_students_units` — with the correct scale the
  target reproduces the teacher's velocity to 1e-4 on a linear step function.

### Lesson for the arm

**Both arms starting at the same loss looked like confirmation of the Eq. 12 init and
was actually the bug's signature.** The value that init predicts is "small", not "equal
across arms" — and only the second was checked. Worth remembering when reading the
relaunched runs: if they again start at ~0.39, the target is still wrong.

### Confirmation that the fix is right — the loss now has PHYSICS in it

Idea B, step-0 eval, per-interval loss (`pdd_loss_k###`), before and after `time_scale`:

| interval | k000 | k001 | k002 | k003 | k004 | k005 | k006 | k007 | mean |
|---|---|---|---|---|---|---|---|---|---|
| **before** (`25261240`) | .38856 | .38830 | .38806 | .38793 | .38784 | .38737 | .38744 | .38773 | .3872 |
| **after** (`25262887`) | **.0689** | **.0214** | .1193 | .3888 | .8502 | 1.2900 | 1.3378 | **2.6658** | .8428 |

**Before: flat to three decimal places.** Eight intervals of a diffusion trajectory
cannot all be equally hard. That flatness *is* the signature of a target ≈ 0 in the
student's units — the loss was simply `E[‖u‖²]`, identical everywhere because `u` is
identical everywhere at init.

**After: monotone in the interval index, spanning 100×.** This is what PDD predicts. The
teacher-initialised student is nearly correct at early intervals (near pure noise, where
the mean velocity over the step ≈ the instantaneous one) and increasingly wrong at late
intervals, where the probability-flow arc curves hardest and one instantaneous evaluation
is a poor estimate of the mean. That gradient across the schedule is precisely the
quantity the N heads exist to absorb.

So Eq. 12's guarantee now shows up correctly: **small at short horizon (0.021–0.069)**,
not "equal across arms". The mean is higher than before (0.84 vs 0.39) purely because the
target is no longer ≈ 0 — the two numbers are in different units and must not be compared.

---

## RESULT (2026-08-06) — the rollout ran

**Full write-up:
[[../../30_Knowledge/experiments/20260806-pdd-parallel-decoding-works-on-the-lora-base-not-the-adapter]]**
(generation job `25284685`).

Headline: **Idea B decodes in parallel; Idea A does not.** One network call vs 8, same
noise/conditioning/grid — B emits a recognisable arm scene, A emits a featureless blur.

Two readings recorded earlier in this ticket rest on **PSNR-vs-ground-truth and are
retracted**: it ranks every comparison backwards here (in-run proof: the 50-step base
scores *below* the 8-step base on all three arms). Generation-from-noise is not
reconstruction, so blur near the conditional mean wins on PSNR.

Also retracted: the loss-curve reading. A's PDD loss fell across **all eight heads**
(k000 10.7× better) and still produced no usable rollout — low PDD loss did not transfer
to generation.

What stands: **1 call vs 8** (structural), and the matched-batch cost measurement
(A 20.3 GiB / 0.08 steps/s vs B 39.8 GiB / 0.05).
