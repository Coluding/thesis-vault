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
