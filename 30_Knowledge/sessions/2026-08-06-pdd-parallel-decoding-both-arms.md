---
date: 2026-08-06
topic: pdd-parallel-decoding-both-arms
duration_minutes: 150
files_touched:
  - src/generative_flow_adapters/training/trainer.py
  - src/generative_flow_adapters/models/adapted_model.py
  - src/generative_flow_adapters/adapters/factory.py
  - src/generative_flow_adapters/adapters/output/dynamicrafter.py
  - src/generative_flow_adapters/adapters/low_rank/lora.py
  - src/generative_flow_adapters/adapters/low_rank/pdd_lora.py
  - configs/dynamicrafter/diffusion_dc_pddA_adapter_actionfree_robotarm.yaml
  - configs/dynamicrafter/diffusion_dc_pddB_lora_actionfree_robotarm.yaml
  - jobs/experiments_cluster/acwm_phys/shortcut/submit_train_dc_pddA_adapter.sh
  - jobs/experiments_cluster/acwm_phys/shortcut/submit_train_dc_pddB_lora.sh
  - tests/test_pdd_parallel_heads.py
  - tests/test_pdd_lora_student.py
tickets_created: []
---

# Session — both PDD arms built, tested and launched (overnight, unattended)

Lukas: *"you can build it overnight and run the two arms. I will go to sleep"*, under
the 15,000-SBU ceiling set the previous day.

## Launched

> ⚠️ **These are the FIRST launches and all four were superseded.** The jobs that
> actually matter are **`25262886`** (Idea A, bs=24) and **`25262887`** (Idea B, bs=8) —
> see the Addendum at the bottom. The table below is kept because the sizing and OOM
> measurements came from these attempts.

| Job | Arm | Config | bs | `--time` |
|---|---|---|---|---|
| `25260518` | **Idea A** — parallel-decoding *adapter*, frozen base | `diffusion_dc_pddA_adapter_actionfree_robotarm.yaml` | 56 | 8 h |
| `25260768` | **Idea B** — LoRA'd base + N replicated heads (paper-faithful) | `diffusion_dc_pddB_lora_actionfree_robotarm.yaml` | 40 | 8 h |

Both action-free, DC base, ACWM-Phys robot arm, N = `pdd_grid_size` = 8. Worst-case
**3,072 SBU** for the pair. Details in
[[../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-dc]] and
[[../../20_Tickets/experiments/exp-shortcut-pdd-lora-distill-dc]].

Also still running from earlier: `25259766`, the motion-correlation confirmation for
the Turbo action arm.

## The acceptance test the workstream was built around

From the Idea A ticket: *"If a run using this still logs `shortcut_direction_loss/N###`
keys, the old shortcut path is still active and it is not PDD — that was how job
25188330 came to carry a name it had not earned."*

Both arms now log `pdd_loss` / `pdd_loss_k###` and **no** shortcut key, verified on
local smoke runs and reproduced in each cluster run's step-0 eval. It is also pinned as
a test (`test_pdd_logs_only_pdd_keys`), so it cannot regress quietly.

## Everything was smoke-tested locally, for zero SBU

Both arms ran end-to-end on the local RTX 3090 against the real `ckts/dynami512.ckpt`
and the real robot-arm mp4s before either was submitted. That caught three defects
that would each have cost a cluster launch:

1. `parallel_heads` was never forwarded from config to the adapter — the factory
   branch didn't pass it, so the "N-head" run emitted 1 head.
2. Idea B died in backward with
   `CheckpointError: A different number of tensors was saved during the original
   forward and recomputation (70 vs 38)`.
3. The PDD step read the clean latent from `batch["x0"]`; DC's diffusion path names it
   `target`, so every DC batch would have raised.

## Two silent-failure classes found and fixed

**Composition under N heads.** `[B,C,T,H,W] + [B,N,C,T,H,W]` right-aligns and raises —
*except* at `B == N`, where it is a legal broadcast that mixes the batch axis into the
head axis and returns a plausibly-shaped, entirely wrong tensor. `AdaptedModel._compose`
now lifts the base output (and the shared gate) over the head axis, and the `B == N`
case is a test.

**Teacher purity in Idea B.** The student's trainable weights live *inside* the base,
so "call the teacher" and "call the student" are the same call distinguished by a
toggle that is deliberately left ON across forward+backward. The PDD step's teacher
closure called the base directly — which would have made the distillation target the
student's own output. Self-distillation wearing PDD's name, and invisible in a loss
curve. Fixed via the existing `clear_dynamic_parameters()` contract, plus a new
`restore_dynamic_parameters()` inverse (fixing the first broke checkpoint
recomputation — defect 2 above).

## Measured, not estimated

- **N=8 head cost on the Idea A adapter: 8,092 params (0.072%)** — 11,203,876 →
  11,211,968. The ticket had estimated ~28k / 0.25%, assuming a 3-D output conv; DC's
  is 2-D. Ticket corrected.
- **Idea B trainable: 8,054,816 of 2,617,183,821 (0.31%)** — LoRA r16 on
  `to_q/to_k/to_v/to_out` plus ~92k of heads.
- **Teacher mean-velocity magnitude varies ~8× across the schedule** — `pdd_loss_k002`
  = 0.042 vs `k006` = 0.330 within one eval cycle. This is why eval was made
  deterministic (block start pinned to 0, every interval swept) rather than left on the
  paper's single random draw: otherwise `eval_loss` would track the draw, not training.
- **Test suite: 342 passed**, up from 332, with the same 6 pre-existing environmental
  failures (a config path moved into `configs/wan22/`, two local-checkpoint tests, one
  GPU-only test). No regressions.

## The limitation to state before anything is claimed

**Generative eval is OFF on both arms** (`inference_every_n_steps: 0`). The base's
native DDIM loop calls the adapter once per denoiser step and expects one velocity back;
under parallel decoding it gets N. Rolling out correctly needs head-indexed stepping or
head fusion (PDD Eq. 15), which the Idea A ticket explicitly defers under
*"training-time correctness first"*.

So these two runs can support claims about **training-time correctness and the
adapter-vs-LoRA comparison, and nothing about few-step video quality or wall-clock
speedup** — which is the headline PDD claim. Wiring the N-head rollout is the next
piece of work, and it is not started.

## Open / needs Lukas

- The N-head rollout above — the gating item for any D3 headline number.
- The remote working tree still has ~135 uncommitted modified files; both runs used
  rsynced working-tree code, so exact reproduction needs those files. Same
  reproducibility caveat as
  [[../experiments/20260805-turbo-action-tokens-binned-to-latent-grid]]. **Commit.**
- Batch sizes are estimates, not measurements: A at 56 extrapolates from the sibling
  shortcut arm's bs=24 → 38.7 GiB; B at 40 is a conservative guess for backprop through
  the whole UNet. Read `peak_vram=` on the step lines and correct both job scripts.
- Whether a `60_Updates/` entry should be written for the weekly meeting — not
  auto-created.

---

## Addendum (03:20) — the first PDD runs were wrong, and the motion result flipped

### Both PDD arms relaunched: `25262886` (A, bs=24), `25262887` (B, bs=8)

The first pair (`25260992`, `25261240`) was discarded ~2 h in. The mean-velocity target
was **~1000× too small**: `_pdd_grid` builds the discretisation in timestep units
(dt ≈ 125 for 8 intervals) while the network predicts `v` w.r.t. normalised time. Target
came out 0.0004–0.0014 against a teacher velocity of ~0.79, so the dominant gradient was
"emit a smaller number". Full write-up + the before/after per-interval table in
[[../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-dc]].

**How it hid:** both arms started at eval_loss ≈ 0.387 and I read that as *confirming*
the Eq. 12 teacher-init. It confirmed nothing — Eq. 12 predicts a **small** initial loss,
not an **equal** one, and 0.387 was just `E[‖v‖²]` because the target was ≈ 0. The
giveaway was in the log the whole time: `pdd_loss_k000..k007` were flat to three decimal
places. Eight intervals of a diffusion trajectory cannot be equally hard. After the fix
they span 100× monotonically (0.021 → 2.666), which is the curvature structure the N
heads exist to absorb.

Guarded by two tests, both verified non-vacuous by forcing the old behaviour and watching
them fail.

### The motion confirmation landed and inverted the reading

Four draws (steps 400, 800; n=16). **adapted vs base is dead** — differences +0.13,
+0.10, +0.045, **−0.034**, sign-inconsistent; the frozen base tracks per-clip GT motion
about as well as the adapter. The hand-measured 0.66 gap does not survive. **adapted vs
shuffled holds** — positive in **4/4** (mean +0.143, sign test p≈0.06).

The reservation already written in the note was the right one and resolved against the
base: the frozen base differs from the adapter in *every* respect, not only action
access. The paired control was the instrument that mattered.

`Trainer._gain_ci` now bootstraps the gain directly (paired resampling), closing the
instrumentation gap — but it **cannot be applied retroactively**, since only summary
correlations were logged. Next run on that arm produces the first gain intervals.

### Three mistakes I made tonight, for the record

1. Read "both arms start at the same loss" as evidence *for* the teacher-init when it was
   the signature of the bug.
2. Cancelled A at bs=56 attributing it to GPU memory pressure; the GPU was at 42% and the
   real constraint was host RAM in the DataLoader. Cancelling was right, the reason wasn't.
3. Spent 40 minutes treating a healthy run as hung because a login-node `tail` showed a
   stale GPFS-cached view.
