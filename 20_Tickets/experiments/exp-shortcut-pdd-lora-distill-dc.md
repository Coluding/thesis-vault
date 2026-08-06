---
type: exp
scope: shortcut
status: open
priority: high
created: 2026-08-03
updated: 2026-08-03
resolution:
resolution_note:
closed_at:
related: ["[[exp-shortcut-parallel-decoding-adapter-wan]]", "[[exp-adapter-action-on-distilled-wan-turbo]]", "[[../../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]]"]
---

# PDD-distil the DynamiCrafter base with LoRA + N heads (D1 × D3)

> **⭐ Part of the efficiency axis** —
> [[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]] (decided
> 2026-08-04). This ticket is **L2 — separable** (speed in a second adapter on the base).
> **Requires a matched conditioning-only control** (same adapter, base, data
> and depth, acceleration off) — without it the pre-registered comparison is
> unmeasurable. The predicted ordering is registered in that note **before**
> any level ran; do not restate it post hoc.


## Idea (Lukas, 2026-08-03)

Rather than asking one small adapter to learn step-size **and** action conditioning at
once, split the jobs: **LoRA-distil the DC base into a few-step model** (PDD objective,
no actions), and leave action conditioning to the adapter later.

**Scope for now: distillation only.** The action-adapter-on-top stage is deferred.

## Why LoRA is the method, not a compromise

LoRA alone cannot do PDD — LoRA modifies existing weights, PDD needs the **final layer
replicated N times**. So the arm is:

```
DC UNet backbone ──── LoRA (rank r)        ← learns few-step behaviour
        └─ final conv ──── replicated N×    ← the N heads, trained normally
```

The paper full-finetunes a 14B model. Doing it parameter-efficiently is a **D1 result**
in its own right, and it finally gives LoRA a defined job in the thesis (open since the
LoRA-arm question was first raised).

LoRA is already implemented: `adapters/low_rank/lora.py:LoRAAdapter`,
`common.py:inject_lora_layers` with `target_modules` matching.

## Why DC and not Wan

On DC the action adapter is **proven** (the AVID replication), so the only new variable
is the distilled base. On Wan the action adapter is itself the open problem
([[../../30_Knowledge/experiments/20260802-avid-wan-cleanroom-perframe-causal]]), so
stacking would confound two failures.

## Why this is not blocked by the DC/diffusion D3 negative

[[../../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]] found no
step-size consistency learned on the DC cell — but that arm ran `endpoint_inversion`, a
**bootstrapped self-consistency** target. PDD's target is neither bootstrapped nor
curvature-sensitive: the mean velocity over a curved arc is well-defined and one RK step
estimates it. So PDD is a plausible explanation-and-fix for that negative, not a repeat
of it.

PDD explicitly covers diffusion — *"As we are interested only in deterministic processes,
for simplicity, we treat both as flows."* And the deterministic step already exists in
our code: `ddim_micro_step_v` (`shortcut_targets.py:289`) **is** the probability-flow ODE
step for v-prediction.

## Full finetune was considered and rejected — on time, not memory

- H100 here is **94 GB** (measured, `submit_train_dc_d3arm.sh` header).
- Full finetune of all 2,620,366,642 params: params 5.2 + grads 5.2 + AdamW fp32 states
  21.0 + fp32 master 10.5 ≈ **42 GB static**. Fits, and `use_checkpoint: True` is already
  set (`configs/base/dynamicrafter512.yaml:59`).
- **But**: DC already runs at **~33 s/step** (0.03 steps/s, batch 24, job `25141979`) with
  only an 11M trainable adapter. Backward through the full UNet is materially slower. Even
  at an optimistic 2×, 2000 steps ≈ 37 h ≈ **~7,000 SBU** — most of the remaining 10k, for
  the base alone, with no control.

DC is **compute-bound, not memory-bound**: bs=24 uses 38.7 GiB of 94, bs=8 uses 20.9 GiB.

## Blocking measurement

Step cost at a smaller batch is unknown and decides whether even the LoRA arm fits the
budget. 12-step smoke at `--batch-size 8`, `--mem=180G`. ~30 SBU.

## Note on billing

`submit_train_dc_d3arm.sh` carries `--mem=360G` → `billing=384`, double. Job `25141979`
billed 384. 180G is one GPU's fair share on a 4-GPU/720G node. 20 job scripts still carry
this line.

---

## SCOPE NOTE (2026-08-04)

This ticket is **Idea B — LoRA the DC base itself** (backbone LoRA + N replicated
output heads, no adapter). That is the **paper-faithful** PDD: the student in
Shaul et al. Eq. 12→13 *is* the backbone with its final layer replicated.

An Idea-A spec (N heads on the *adapter* over a FROZEN base) was mistakenly
appended here on 2026-08-04 and has been moved to
[[exp-shortcut-parallel-decoding-adapter-dc]].

---

## 2026-08-06 — implemented and launched (job `25260768`)

Config `configs/dynamicrafter/diffusion_dc_pddB_lora_actionfree_robotarm.yaml`,
job script `jobs/experiments_cluster/acwm_phys/shortcut/submit_train_dc_pddB_lora.sh`.
Launched 2026-08-06 01:05 CEST, `--time=08:00:00`, bs=40. Paired with Idea A
(`25260518`) — same objective, data, grid and N=8; the only difference is **where the
parallel decoder lives**.

**Trainable: 8,054,816 of 2,617,183,821 (0.31%)** — LoRA r16 on
`to_q/to_k/to_v/to_out` plus 8 replicated heads (~92k). Measured on the local smoke.

### What shipped

`adapters/low_rank/pdd_lora.py::PDDLoRAAdapter`, a `LoRAAdapter` subclass, reachable
as `adapter.type: pdd_lora`. It injects LoRA (inherited) and additionally replicates
the backbone's final conv N times, **initialised from the pretrained weights** — so at
step 0 every head emits exactly the teacher's velocity. That is Eq. 12's
teacher-initialised student, and it is asserted as an exact equality (zero non-zero
elements in the delta), not a tolerance.

`forward` returns `student − teacher` broadcast over the head axis, so
`composition: add` reconstructs the student outright. Returning a delta is what keeps
the frozen base reachable as the PDD teacher **without a second 2.6B model copy**.

### The subtle part: the switch

The student's trainables live *inside* the base, so "call the teacher" and "call the
student" are the same call distinguished only by a toggle. Two bugs here, both found
and both silent if missed:

1. **Teacher purity.** `_pdd_forward_and_loss`'s teacher closure called `base_model`
   directly. Under Idea B that runs the *student* — the distillation target becomes
   the student's own output, i.e. self-distillation wearing PDD's name. Now calls the
   adapter's existing `clear_dynamic_parameters()` contract first (a no-op for every
   adapter that sits outside the base).
2. **Checkpoint recomputation.** Fixing (1) then broke backward:
   `CheckpointError: A different number of tensors was saved during the original
   forward and recomputation. Forward: 70, recomputation: 38.` The teacher call lands
   *after* the student's forward but *before* backward, and gradient checkpointing
   recomputes that forward with the toggle now off. Same failure class as slurm
   25182655. Fixed with a new `restore_dynamic_parameters()` inverse, called at the
   end of the PDD step.

### Tests
`tests/test_pdd_lora_student.py` — 8 tests, all passing: exact teacher-init equality,
per-head independence (perturbing head 2 moves head 2 only), teacher output unchanged
after the student's weights move, the clear/restore round trip, gradient reaching LoRA
and heads but not the frozen conv, and N=1 rejected.

### Scope note
Action-free distillation only, as scoped above. Generative eval is OFF for the same
reason as Idea A — the N-head rollout (Eq. 15) is not wired, so **no few-step video or
timing number can come from this run**.

### VRAM — measured, 2026-08-06

| bs | outcome | note |
|---|---|---|
| 40 | **OOM** (job `25260768`, FAILED at 2:26) | 85.31 GiB in use, tried to allocate 9.38 GiB more of 93.34 GiB |
| 28 | job `25260856`, relaunched | |

The OOM is the expected shape of this arm and worth recording: Idea B backprops
through the **whole 2.6B UNet** (LoRA's weights live inside it) while Idea A backprops
through an 11.2M side network, so per-sample activation cost is far higher even with
the backbone's `use_checkpoint: True`. The local RTX 3090 comparison (12.2 vs 11.2 GiB
at bs=1) badly understated this — at bs=1 the shared static floor dominates and says
nothing about the marginal cost, which is the part that scales. **A failed run only
gives an upper bound; only a surviving run's `peak_vram=` is a measurement.**

Consequence for the pair: Idea A runs at bs=56 and Idea B cannot, so the two arms are
**depth-matched but not batch-matched**. Quote steps, not epochs, and re-run one of
them at the other's batch before making a like-for-like efficiency claim.

**Corrected 2026-08-06 (the table above records the launches, this records what they mean).**
bs=28 also OOM'd (job `25260945`): *86.66 GiB in use, tried to allocate 10.50 GiB* —
i.e. **worse** than bs=40's 85.31 / 9.38. Memory did not scale with batch, so the first
reading ("bs=40 is simply too big") was wrong.

The traceback locates it precisely: the OOM is in the **teacher call**
(`pdd_objective.py:113 → trainer.py:672 teacher → dynamicrafter_video.py:221 denoise`),
which by construction runs *after* the student's forward has built its backward graph.
So the peak is not "one forward at batch B", it is

> **the whole student graph, held, PLUS a concurrent full base forward.**

For Idea A that second term is cheap — the retained graph is an 11.2M side network. For
Idea B the retained graph spans the 2.6B UNet, so the two terms are the same order and
the teacher's transient allocation is what tips it over. **This is a structural property
of PDD-on-a-LoRA'd-base, not a batch-size accident**, and it is the real efficiency
argument against Idea B that the ticket's original "full finetune rejected on time, not
memory" analysis did not anticipate.

It also cannot be reordered away: PDD's target is evaluated at `x_k`, which is reached
by rolling on the *student's* outputs, so the teacher call must follow the student pass.
Options are smaller batch (taken: bs=8, job `25261240`), or recomputing the student
pass after the teacher instead of retaining it — untested.

**Two failed runs give only upper bounds.** No scaling curve can be fitted from them;
only a surviving run's `peak_vram=` is a measurement.

### Checkpoint hygiene after the relaunch (2026-08-06 04:21)

The relaunched arms reuse the discarded runs' `output_dir`, so checkpoints from the
units-bug runs sat in the same folder as valid ones, indistinguishable by name. Audited
against `sacct` start times (new jobs began 03:11:55):

- **Idea A** (`.../dc-pddA-adapter-.../checkpoints/`) — clean. Both files 04:19.
- **Idea B** — `step_00000200.pt` was written **02:42**, i.e. by the discarded run
  `25261240`. Moved (not deleted) to
  `outputs/dc-pddB-lora-actionfree-robotarm/discarded-units-bug-25261240/`.

It would have been silently overwritten when the new run reached step 200 anyway, so the
hazard was transient — but only if the run survives that far, and a checkpoint trained on
a 1000×-wrong target is not something to leave lying next to good ones.

**For future relaunches after a discarded run: change `output_dir` or clear the
checkpoint folder first.** `keep_last_checkpoints` rotation does not distinguish runs.

### First post-fix eval — Idea B gets WORSE, and the per-interval shape says how

`25262887`, eval_loss **0.843 (step 0) → 1.445 (step 100)**. Not a flat regression:

| k | 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 |
|---|---|---|---|---|---|---|---|---|
| step 0 | 0.069 | 0.021 | 0.119 | 0.389 | 0.850 | 1.290 | 1.338 | 2.666 |
| step 100 | **0.576** | **0.198** | 0.048 | 0.193 | 0.701 | **2.389** | **2.639** | **4.818** |
| ratio | 8.4× worse | 9.2× worse | 2.5× better | 2.0× better | 1.2× better | 1.9× worse | 2.0× worse | 1.8× worse |

**Only the middle improves.** The early intervals — where the teacher-initialised student
was already nearly correct (0.02–0.07) — degrade by ~9×, and the late ones degrade too.

Two mechanisms are consistent with this and they are separable by the next eval points:

1. **Shared-trunk contention (Idea-B-specific).** The N heads are independent, but LoRA
   modifies the *shared* backbone. Training samples k uniformly, so the average is
   dominated by the large-target late intervals; the trunk adapts toward them and
   sacrifices the early ones, where there was nothing to gain. If this is it, Idea A —
   whose adapter is also shared but which does not touch the base — should show a milder
   version, and the fix is loss weighting per interval (e.g. normalise by the target's
   scale) rather than uniform k.
2. **On-policy rollout compounding.** Eval pins block start n=0 and rolls the student's
   own outputs across all 8 intervals, so errors accumulate along the trajectory. As the
   student moves off the teacher, late `x_k` drift off-distribution and their targets get
   harder — which is exactly what k005–k007 show. This is intrinsic to PDD's on-policy
   design, and the n=0 eval is its worst case by construction.

Train loss is meanwhile *falling* (avg 2.02 @100 → 1.59 @200), so this is not a blown-up
run — it is train/eval disagreement of the kind (2) predicts.

**Not acted on.** Step 100 of ~1400, one point, and both candidate mechanisms are
research questions rather than bugs. The next eval points decide it; changing the LR or
the k-sampling now would destroy the evidence. ⚠️ The learning rates (A 1e-4, B 5e-5)
were chosen when the target was ~1000× smaller and have **not** been re-tuned for the
corrected scale — that is the first thing to reconsider if the rise continues.

### Step 200 discriminates the two mechanisms — and it favours shared-trunk contention

| | step 0 | step 100 | step 200 |
|---|---|---|---|
| **A** (`25262886`) — adapter, base untouched | 0.887 | **0.796** ↓ | — |
| **B** (`25262887`) — LoRA on the base | 0.843 | **1.445** ↑ | 1.269 ↓ |

At matched step 100, A improves **10%** while B degrades **71%**. That rules out
mechanism 2 as the sole explanation: on-policy rollout compounding is a property of the
*objective*, and both arms run the identical objective, the identical n=0 all-intervals
eval, and the identical grid. If compounding alone drove the step-100 rise, A would show
it too. A doesn't.

**The mechanism, stated precisely.** Both arms have a trunk shared across the N heads —
so trunk-sharing per se is not the difference. The difference is *what* the trunk is:

- **Idea A's** trunk is a separate 11.2M network whose output is **added** to the base,
  and it is `zero_module`-init, so at step 0 it contributes exactly nothing. It cannot
  damage the teacher-initialised prediction; it can only add to it.
- **Idea B's** trunk **is the pretrained base** — the very function the Eq. 12 teacher
  init relies on. Any LoRA update that helps the hard late intervals necessarily
  perturbs the function that was already nearly correct on the easy early ones. Hence
  k000/k001 degrading ~9× while k002–k004 improved.

B partially recovers by step 200 (1.445 → 1.269), consistent with the trunk finding a
compromise. Whether it returns below its 0.843 starting point is the open question.

⚠️ **Preliminary: two eval points for A, three for B, on one run each.** The direction is
clean and mechanistically explicable, but this is not yet a result. It is also confounded
by batch size (A=24, B=8) and learning rate (A=1e-4, B=5e-5), neither matched — see the
VRAM section above for why they could not be.

If it holds, it is a genuine argument for the adapter transposition over the
paper-faithful form on a *pretrained* base: **additive, zero-initialised capacity cannot
regress what the teacher already does well, while in-place capacity must trade against
it.** That is a D1 claim about the composition rule, not a D3 claim about few-step
generation.

### CORRECTION (06:20) — B's degradation was transient; the architecture claim does not hold

The step-200 reading above ("additive capacity cannot regress what the teacher does well,
in-place capacity must trade against it") was **premature and is retracted as stated**.
Four more eval points:

| step | 0 | 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|
| **A unmatched** (`25262886`, bs=24, lr=1e-4) | 0.887 | 0.796 | 0.663 | — | — |
| **A control** (`25264058`, bs=8, lr=5e-5) | 0.887 | 0.828 | 0.768 | — | — |
| **B** (`25262887`, bs=8, lr=5e-5) | 0.843 | 1.445 | 1.269 | 1.562 | **0.525** |

B's rise was a **transient**. By step 400 it reached 0.525 — the best value any arm has
posted, and below its own starting point. The "in-place capacity trades against the
teacher init" story explained the first two points and not the next two.

**What the data does support, and it is the thing the control was launched for:**

1. **B is far more volatile.** Both A runs descend smoothly and monotonically; B swings
   ±0.5 between consecutive evals (1.445 → 1.269 → 1.562 → 0.525). Consistent with a
   trunk being pulled between intervals, but that is now a description, not a mechanism
   — and volatility is not the same as regression.
2. **⭐ Matched-batch efficiency, the first clean number.** At bs=8, lr=5e-5, identical
   objective and grid: **Idea A peaks at 20.3 GiB (22%), Idea B at 39.8 GiB (43%)** —
   B costs **~2× the VRAM**. Throughput likewise **0.07 vs 0.05 steps/s**, A ~1.4×
   faster. This is the comparison that was impossible before the control existed, and it
   is a D1 claim about the composition rule that does not depend on the loss curves at all.

**Not yet comparable:** A-control and B are at different step counts (200 vs 500). Quote
them only at matched steps. A's advantage on cost is solid; any claim about which
converges *better* needs matched steps and more than one run.

### Matched-step comparison, 08:50 — A is monotone, B is volatile at the same level

Both at bs=8, lr=5e-5, identical objective/grid/data. `25264058` (A control) vs
`25262887` (B):

| step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|---|
| **A control** | 0.887 | 0.828 | 0.768 | 0.706 | 0.640 | 0.613 | 0.578 | 0.570 | **0.547** |
| **B** | 0.843 | 1.445 | 1.269 | 1.562 | **0.525** | 0.609 | 0.549 | **0.493** | 0.788 |

**A: nine consecutive improvements, monotone throughout.** **B: swinging over 0.49–1.56
and still swinging at step 800.** On level they are comparable — B is better at steps 400
and 700, worse at 100–300 and 800 — but B's value at any given eval is not predictive of
the next one.

For reference, A at bs=24 (`25262886`, not comparable) reaches 0.542 by step 500 — the
same level in fewer steps, as expected from 3× the batch.

**What can be said from this pair:** at matched batch, LR, objective and data, the
adapter transposition reaches the same loss level as the paper-faithful form while using
**half the VRAM (20.3 vs 39.8 GiB)**, running **~1.4× faster (0.07 vs 0.05 steps/s)**,
and converging monotonically rather than in ±0.5 swings.

**What cannot:** one run each, no seed replication, and the volatility means B's endpoint
depends on where you stop. Nothing here is about few-step generation quality — the N-head
rollout is still not wired.

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
