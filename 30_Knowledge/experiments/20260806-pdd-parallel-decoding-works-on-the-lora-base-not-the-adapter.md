---
type: experiment
date: 2026-08-06
config: configs/dynamicrafter/{diffusion_dc_pddA_adapter_actionfree_robotarm, diffusion_dc_pddA_adapter_bs8lr5e5_control, diffusion_dc_pddB_lora_actionfree_robotarm}.yaml
commit: (⚠️ WORKING TREE, not a commit — ~135 uncommitted files at launch)
wandb_run_id: uusbz707 (A @146M), 75fikn1t (B @rank294); the 11.2M pair has none
ckpt_path: outputs/{dc-pddA-adapter-actionfree-robotarm, dc-pddA-adapter-bs8lr5e5-control, dc-pddB-lora-actionfree-robotarm}/checkpoints/best.pt
status: completed
deliverable: D3
metrics:
  train_jobs_11m: "25262886 (A bs24), 25262887 (B bs8), 25264058 (A control bs8)"
  train_jobs_146m: "25302987 (A), 25302988 (B) — capacity-matched, 0.02% apart"
  params_matched: "A 146,373,664 / B 146,405,408"
  eval_loss_146m: "A 0.128 / B ~0.64"
  latent_std_vs_gt_146m: "A 1.009-1.027 (no collapse) / B 2.21-2.24"
  generate_jobs: "25284685 (11.2M), 25313706 (146M, 16 clips), 25315512 (collapse diagnostic)"
  eval_loss_A_control_1100: 0.524
  eval_loss_B_1100: 0.495
  vram_matched_bs8_A_GiB: 20.3
  vram_matched_bs8_B_GiB: 39.8
  steps_per_s_matched_bs8: "A 0.08 / B 0.05"
  network_calls_parallel_vs_sequential: "1 vs 8"
notes: |
  Two formulations of Parallel Decoding Distillation (arXiv:2607.26004) on a frozen
  DynamiCrafter base, action-free, N = grid = 8. Idea A = N heads on a separate additive
  adapter; Idea B = LoRA on the backbone + its final conv replicated N times
  (paper-faithful). An earlier pair of runs was DISCARDED for a units bug (below).
---

# PDD: the paper-faithful form decodes in parallel; the adapter transposition does not

Tickets: [[../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-dc]] (A),
[[../../20_Tickets/experiments/exp-shortcut-pdd-lora-distill-dc]] (B).

**The claim under test.** One network call emits N mean velocities — one per interval of
the timestep grid — so an N-step rollout costs ONE forward pass instead of N.

## Result 1 (SOURCED, decisive): B decodes in parallel, A does not

Job `25284685`, 6 held-out clips per arm. Every decoder starts from the **same noise**
with the **same conditioning frame** and walks the **same grid**; the only difference is
how many forward passes it makes.

Visual inspection of the `[GT | parallel | seq-8 | seq-50]` panels
(`outputs/pdd-generations/*/clip00_gt-parallel-seq8-ref50.png`):

| arm | parallel decode (1 call) |
|---|---|
| **A** (adapter, frozen base) | **featureless brown-grey blur** — no arm, no box, no floor |
| **B** (LoRA'd base + N heads) | **recognisable robot arm on the box, floor lines visible**; over-dark and ghosted, but structured |

**B produces a scene in one network call. A produces mush.** For B the parallel row is
arguably more structured than its own 8-call baseline (which is hazy and washed out),
though its exposure is badly crushed. The 50-call reference is cleanest for both.

⚠️ **Qualitative, and read from one clip per arm.** A perceptual metric (FID/FVD, already
in the eval cycle) is needed before this is quotable as a number. But the A-vs-B
difference is not marginal — it is "scene" vs "no scene".

## Result 2 (SOURCED): PSNR-vs-ground-truth ranks every comparison BACKWARDS

| arm | parallel (1 call) | seq-8 | seq-50 |
|---|---|---|---|
| A bs24 | **18.37** | 17.72 | 16.88 |
| A control bs8 | **18.33** | 17.72 | 16.88 |
| B bs8 | 16.02 | 17.87 | 16.61 |

By PSNR: A's parallel decode is the best output produced, and A beats B by 2.35 dB. By
eye, both are false.

**The in-run proof is `seq-50 < seq-8` on every arm.** More sampling steps cannot mean
worse generation. These decoders start from pure noise with only a conditioning frame, so
they produce a *plausible continuation*, not a reconstruction: blur sits near the
conditional mean and scores well, sharp-but-different is penalised. A's parallel output is
the blurriest thing measured, so it wins on PSNR while being visibly the worst.

**Do not use PSNR-vs-GT to rank generative quality on this task.** It was used to write
two intermediate readings on 2026-08-06 that both had to be retracted.

## Result 3 (SOURCED): matched-batch training cost — A is half of B

At matched batch (8), learning rate (5e-5), objective, grid and data —
`25264058` vs `25262887`:

| | A control | B |
|---|---|---|
| peak VRAM | **20.3 GiB (22%)** | 39.8 GiB (43%) |
| throughput | **0.08 steps/s** | 0.05 steps/s |

**B costs ~2× the VRAM and runs ~1.6× slower.** Structural: B backprops through the whole
2.6B UNet (LoRA's weights live inside it) while A backprops through an 11.2M side network.
B also **OOMs at A's batch of 24** — the peak is the retained student graph *plus* a
concurrent frozen-base forward for the teacher, and PDD's target cannot be reordered
before the student pass because it is evaluated at states reached by rolling the student's
own outputs.

## Result 4 (SOURCED): training loss says the two arms are equal — and it is misleading

Matched (bs=8, lr=5e-5), `eval_loss` over steps 400–1100: **A mean 0.570, B mean 0.567**.
Indistinguishable. A declines monotonically (range 0.116); B swings over 0.456–1.562
(range 0.332, ~3×) and is still swinging at step 1100.

Set against Result 1, this is the substantive finding: **A's PDD loss fell across all
eight heads (k000 10.7× better, k007 1.3×) without producing a usable rollout.** Low PDD
loss did not transfer to generation — the degenerate-fixed-point risk flagged when the
objective was written, now observed.

## Discarded runs — the mean-velocity target was ~1000× too small

The first pair (`25260992`, `25261240`) was cancelled ~2 h in. `_pdd_grid` built the
discretisation in **timestep units** (dt ≈ 125 for 8 intervals) while the network predicts
`v` w.r.t. **normalised time**, so the target came out at 0.0004–0.0014 against a teacher
velocity of ~0.79. The dominant gradient was a unit conversion.

Measured (`DiffusionTrainingObjective(1000)`, `ddim_micro_step_v`, 8 intervals):

| interval | displacement | target, dt in steps | target, dt normalised | teacher \|v\| |
|---|---|---|---|---|
| k=0 | 0.0538 | **0.00043** | 0.431 | 0.793 |
| k=6 | 0.1801 | **0.00144** | 1.442 | 0.793 |

**How it hid:** both arms started at eval_loss ≈ 0.387 and that was read as *confirming*
the Eq. 12 teacher-init. Eq. 12 predicts a **small** initial loss, not an **equal** one;
0.387 was simply `E[‖v‖²]` because the target was ≈ 0. The real tell was in the log
throughout — `pdd_loss_k000..k007` flat to three decimal places. Eight intervals of a
diffusion trajectory cannot be equally hard.

After the fix the step-0 profile spans 100× monotonically (0.021 → 2.666), tracking
trajectory curvature — which is the gradient across the schedule the N heads exist to
absorb. Guarded by `test_teacher_initialised_student_starts_near_zero_loss` and
`test_time_scale_puts_the_target_in_the_students_units`, both verified non-vacuous by
forcing the old behaviour.

## Logging defect

**No wandb run exists for any of the three arms.** `video_logging: {enable: false}` was
set to suppress video panels (the N-head rollout would have shape-errored through the
base's DDIM loop), but that key is the legacy alias for the **entire** wandb block, so
scalar logging went off with it. Metrics survive in stdout and `metrics.jsonl`.
Fix: `training.extra.wandb: {enable: true, require_vae: false}`.

## Next

1. **FID/FVD on the generations** — the only way to turn Result 1 into a number.
2. **Head fusion (Eq. 15)**, still deferred — would let a rollout use fewer steps than heads.
3. A's failure is the interesting one: does more adapter capacity, or conditioning the
   adapter on the interval index rather than relying on separate heads, recover it?

---

# Addendum (2026-08-07): the capacity hypothesis, cleanly excluded

Jobs `25302987` (A @146M) / `25302988` (B @rank294), generation `25313706`, collapse
diagnostic `25315512`. wandb: `uusbz707` (A), `75fikn1t` (B).

**Why this ran.** Lukas: *"maybe the 11.2m adapter is not enough."* In the first pair,
formulation and trunk capacity were confounded: A's 8 heads read features from an 11.2M
network, B's from the full 1.4B backbone. So the original conclusion attributed to the
composition rule something that could have been a size effect.

**Design.** Trainable parameters matched to **0.02%**: A at `model_channels: 96`
(**146,373,664**, measured on the built model) vs B at LoRA **rank 294**
(**146,405,408** = 294 x 497,664 + 92,192 heads). Identical objective, N=8 grid, bs=8,
lr=5e-5, action-free, same data. Both start at the same eval_loss (0.885 / 0.846), which
is the Eq. 12 teacher init behaving correctly and independent of capacity.

## Result A1 (SOURCED): capacity transformed the LOSS

| | 11.2M / rank16 | 146M / rank294 |
|---|---|---|
| A best eval_loss | 0.524 | **0.128** |
| B best eval_loss | 0.456 | ~0.64 |

A improved **4x** and reached the same loss roughly 3x faster in steps. On the training
objective the capacity hypothesis is strongly supported.

## Result A2 (SOURCED, decisive): capacity did NOT fix the rollout

Generation over 16 held-out clips per arm, same noise / conditioning / grid:

| arm | parallel decode, 1 network call |
|---|---|
| **A @146M** | **still unusable**: mottled blue-black noise texture, no arm, no box, no floor |
| **B @rank294** | **recognisable robot arm on the box, floor grid visible**, sharper than at rank 16 |

So A fits the PDD targets **5x better than B** (0.128 vs 0.64) and cannot decode, while B
fits them 5x worse and produces the scene. **The capacity hypothesis is excluded**, and
the original conclusion survives a fair test.

## Result A3 (SOURCED, refutes the stated mechanism)

The prediction on record before this measurement was that A had found a **degenerate
fixed point**: PDD's target is evaluated at states the student itself reaches, so a
student that steers into a smooth low-curvature region matches the teacher there and
generates nothing. That predicts **low-variance** rollout latents.

Measured std of the parallel rollout latents relative to ground truth, 16 clips:

| arm | std vs GT | latent MSE vs GT |
|---|---|---|
| **A** | **1.009 – 1.027** (all 16 clips within 3% of 1.0) | 0.43 – 0.46 |
| **B** | **2.21 – 2.24** | ~2.0 |

**No collapse.** A's rollout carries almost exactly the data's spread. The
low-variance-region story is **refuted**. If anything B is the statistically odd one, at
2.2x over-dispersed, which matches its crushed high-contrast look.

## Result A4 (SOURCED): three latent-space metrics rank the arms BACKWARDS

| metric | A | B | which looks better | pixels say |
|---|---|---|---|---|
| PSNR vs GT | **17.85** | 16.33 | A | **B** |
| latent MSE vs GT | **0.44** | ~2.0 | A | **B** |
| implied latent correlation | **~0.78** | ~0 | A | **B** |

A is closer to the ground-truth latent by every scalar available and produces visibly
worse video. Adding to the earlier `seq-50 < seq-8` PSNR inversion, this is now four
independent quantitative signals pointing the wrong way.

**Working explanation (NOT established):** latent MSE is dominated by low-frequency
content while the VAE decoder needs correct local structure. A can be ~78% correlated at
coarse scale with wrong high-frequency detail everywhere, which is precisely what a
mottled noise texture is. B is globally off (exposure, 2.2x dispersion) but locally
structured. Untested; the check would be decoding A's rollout at each of the 8
intermediate steps to see whether it ever denoises.

**For the thesis, the safe statement is the negative one: no latent-space scalar we have
tried can rank these models. Only decoded perceptual comparison separates them.**

## What remains as the explanation for A's failure

Not capacity (A2), not variance collapse (A3). The untested asymmetry: **B's heads read
features from the 1.4B pretrained backbone that LoRA modulates; A's read from a network
trained from scratch**, with the frozen base contributing a single velocity evaluated
once at t_0 and identical across all eight heads. A's `Delta_j` must encode, from noise
alone, where the trajectory should be at eight different times.

So the finding is sharper than the original note's framing: it is not "additive vs
in-place composition", it is that **A's heads never see pretrained features**.

## Consequence for methodology

`best.pt` is selected on eval_loss. On arm A that metric rewards the failure: the
checkpoint chosen as best is not the one that generates best, and no loss-based criterion
would have caught it. **Any future PDD run needs a generation-based or perceptual
selection criterion.** This is a real limitation of the current training loop, not a
property of PDD.
