---
type: experiment
date: 2026-08-03
config: configs/wan22/diffusion_wan22_lora_acwm_robotarm.yaml vs ..._avid_xattn_tokennorm_nobase_acwm_robotarm.yaml
commit: uncommitted working tree @ 2026-08-03
wandb_run_id: acwm-robotarm-wan-LORA (slurm 25183721) vs vy9tcuco
ckpt_path: /scratch-shared/lbierling/outputs/acwm-robotarm-LORA-run/checkpoints/
status: running
deliverable: D1
metrics:
  lora_steps_per_hour: 66
  output_adapter_steps_per_hour: 205
  slowdown_factor: 3.1
  lora_trainable_params: 25965568
  output_adapter_trainable_params: 34885764
notes: "LoRA costs 3.1x the wall-clock per optimizer step of an output adapter on the same frozen 5B base, at MATCHED effective batch (LoRA batch 6 x grad_accum 2 = 12; output adapter batch 12). Structural, not incidental: LoRA's trainable weights live INSIDE the base, so its pass must be differentiable through all 30 blocks, and the resulting activation memory forces gradient checkpointing on top. Every other adapter family runs the base once under no_grad."
---

# LoRA costs 3.1× the wall-clock of an output adapter on the same base (D1)

## Result

| | steps/h | trainable | effective batch |
|---|---|---|---|
| output adapter (`vy9tcuco`) | **205** | 34.9 M | 12 |
| **LoRA** (`25183721`) | **66** | 26.0 M | 12 (6 × grad_accum 2) |

**3.1× slower per optimizer step**, on the same frozen Wan2.2-5B, same data, same
effective batch — so this is not a batching artifact. LoRA is also the *smaller*
adapter by parameter count, which makes the gap a property of **where** the
parameters sit, not how many there are.

## Why — and why it is a framework result, not an implementation detail

Every other adapter family in the taxonomy emits a delta from `(x_t, cond)`; its
trainables sit **outside** the frozen base, so the base is evaluated once under
`torch.no_grad()` and stores no activations. **LoRA's trainable weights live
INSIDE the base.** Two consequences follow directly:

1. **The base pass must be differentiable.** Gradients have to reach weights
   inside the frozen trunk, so the whole 30-block forward enters the graph.
2. **The resulting activation memory forces gradient checkpointing.** A
   differentiable pass OOMs a 93 GB H100 even at batch 6; checkpointing
   recomputes each block during backward — the standard ~1.5× compute trade — on
   top of (1).

   ⚠ **Correction 2026-08-03.** An earlier version of this note attributed the
   OOM to "14,400 tokens per sample (24×24 patches × 25 latent frames)". That was
   **wrong by 9×**: it derived the latent grid from `max_area` (768² → 48×48),
   but the config sets `latent_height: 16` / `latent_width: 16` explicitly, so
   with `patch_size 2` it is **8×8 × 25 latent frames = 1,600 tokens/sample**
   (19,200 at batch 12). The OOM is real and measured; the token-count
   explanation of its magnitude was not, and the true cause of the peak is
   **unattributed** — it was raised in self-attention `qkv_fn`, but 1,600 tokens
   does not obviously account for 93 GB, so something else dominates (plausibly
   the full-resolution generation eval, `inference_max_area: 589824`). _Needs
   verification before this note is cited on the memory point._

Analysed estimate of the split: checkpointing accounts for ~1.5× and the
differentiable base pass for the remaining ~2×. _Not separately measured_ — it
would need a run large enough to fit without checkpointing, which does not exist
at this sequence length.

## Why this matters for D1

The framework contribution argues adapters are the **cheap, composable** way to
add action conditioning to a frozen base. This measures that cost directly on
one axis and in the adapters' favour: a *smaller* LoRA is 3.1× more expensive to
train than an output adapter because of where its parameters live. It is also
the honest counterweight to the literature's frozen-vs-finetuned objection
(DriveVA finds full fine-tuning wins on this exact base) — the rebuttal is on
composability and cost, and this is the cost number.

## Consequence for the LoRA experiment itself

At 66 steps/h the 9 h run reaches only **~594 steps**, against `vy9tcuco`'s 3054.
Decision (user, 2026-08-03): **let it stop at ~600 steps** rather than move it to
the supervisor's account for ~46 h / ~8,900 SBU.

**So the LoRA action-following comparison is limited to step 400**, where
`vy9tcuco` read `effect_rel` **0.01234** — one early, noisy point. Any claim from
it must say so; the arms were still climbing well past step 1200. The wall-clock
result above is unaffected and is the durable finding from this run.

## Caveats

- n=1 per arm.
- The 3.1× is specific to this configuration and base. A setting that needed no
  checkpointing would narrow the gap toward the ~2× differentiable-pass term
  alone. (See the token-count correction above — the sequence is 1,600
  tokens/sample, not 14,400.)
- LoRA rank 16 on q/k/v/o (30 blocks, dim 3072) = 23.6 M, plus a 2.1 M action
  projector. A different rank changes the parameter count but not the structural
  argument.

## Related

- [[20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]
- [[../../10_now/compute-spend-ledger]] — the six LoRA integration failures, and
  why they were all the same root cause
