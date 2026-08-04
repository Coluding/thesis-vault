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
