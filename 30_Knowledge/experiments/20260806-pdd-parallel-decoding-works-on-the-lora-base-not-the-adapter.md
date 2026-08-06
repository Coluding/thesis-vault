---
type: experiment
date: 2026-08-06
config: configs/dynamicrafter/{diffusion_dc_pddA_adapter_actionfree_robotarm, diffusion_dc_pddA_adapter_bs8lr5e5_control, diffusion_dc_pddB_lora_actionfree_robotarm}.yaml
commit: (⚠️ WORKING TREE, not a commit — ~135 uncommitted files at launch)
wandb_run_id: NONE — see "Logging defect"
ckpt_path: outputs/{dc-pddA-adapter-actionfree-robotarm, dc-pddA-adapter-bs8lr5e5-control, dc-pddB-lora-actionfree-robotarm}/checkpoints/best.pt
status: completed
deliverable: D3
metrics:
  train_jobs: "25262886 (A bs24), 25262887 (B bs8), 25264058 (A control bs8) — all TIMEOUT at their caps"
  generate_job: 25284685
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
