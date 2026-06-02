---
type: feat
scope: training
status: open
priority: high
created: 2026-05-21
updated: 2026-05-21
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/decided/shortcut-anchor-schedule]]"
  - "[[../50_Decisions/decided/avid-adapter-init]]"
  - "[[risk-shortcut-self-consistency-collapse]]"
  - "[[feat-shortcut-add-d-zero-gate]]"
---

# Implement step warmup for the shortcut d=1 anchor in the training loop

## Motivation

Implementation of [[../50_Decisions/decided/shortcut-anchor-schedule]].

The training loop currently uses the shortcut-models paper's fixed
batch split between the d=1 data anchor and the d>0 self-consistency
loss throughout training. With [[../50_Decisions/decided/avid-adapter-init]]
resolved to status quo (step-0 prediction = `0.5·base + 0.5·random`),
the early-phase self-consistency teacher is corrupted by the adapter's
own random init. The warmup zeroes the self-consistency loss weight
for the first N optimizer steps, letting the adapter first learn
structured d=1 predictions on real data before the consistency
teacher comes online.

## Proposed change

**Config (`src/generative_flow_adapters/config.py:52-68`):** add

```python
shortcut_anchor_warmup_steps: int = 0
```

next to the existing `shortcut_direction_weight` and
`shortcut_target_method` fields. Default 0 keeps non-shortcut configs
bit-identical to today.

**YAML defaults:** set `training.shortcut_anchor_warmup_steps: 5000`
in both live shortcut configs:
- `configs/diffusion_avid_shortcut_metaworld.yaml`
- `configs/diffusion_hyperalign_shortcut_metaworld.yaml`

(`5000` is the analysed-estimate starting value from the decision —
sweep it as a follow-up ticket once a baseline run exists.)

**Trainer (`src/generative_flow_adapters/training/trainer.py`, exact
lines TBD):** gate the self-consistency loss weight on the global
step:

```python
if self.global_step < self.config.shortcut_anchor_warmup_steps:
    consistency_weight = 0.0
else:
    consistency_weight = self.config.shortcut_direction_weight
```

Sketch only — verify exact field/attribute names against the current
trainer when wiring this up.

**Logging:** emit two scalars to wandb so the warmup boundary is
visible in dashboards:
- `train/shortcut_consistency_active` — 0 during warmup, 1 after.
- `train/shortcut_warmup_remaining_steps` — countdown for diagnostic
  ease.

**Tests** (`tests/`): unit test that with `warmup_steps=N`, the
consistency loss term contributes zero to the total loss for all
`global_step < N`, and contributes its configured weight from
`global_step >= N`. Mock the schedule — no full training needed.

Rough sizing: ~20 lines of code + one test + two YAML edits.

## Acceptance criteria

1. With `shortcut_anchor_warmup_steps=0`, training is bit-identical
   to today (regression fixture).
2. With `shortcut_anchor_warmup_steps=N > 0`, the self-consistency
   loss term contributes zero for the first N optimizer steps, and
   its full configured weight from step N onward.
3. `train/shortcut_consistency_active` in the wandb dashboard shows
   a single 0→1 transition at the warmup boundary.
4. The d=1 anchor loss curve through the boundary is continuous (no
   discontinuity at step N from the warmup logic itself). Discontinuity
   in the *consistency* loss at the boundary is expected — that's the
   schedule activating.

## Out of scope (separate tickets)

- **Adaptive (EMA-loss-driven) warmup.** Defer until we know the
  shape of the d=1 anchor loss curve from a baseline fixed-N run.
- **Smooth decay schedule.** The decision explicitly chose step
  warmup over smooth decay — do not re-litigate.
- **Tuning N.** This ticket wires up the knob. The sweep
  ({1000, 5000, 20000}) is a follow-up experiment ticket once we
  have a baseline.
- **Loss-mode-conditional warmup** (active only under
  `shortcut_target_method: distillation`, since `two_step` targets
  are derived from the base rather than the adapter and may not
  need the warmup). Open as a follow-up once the fixed schedule is
  validated.

## Related

- [[../50_Decisions/decided/shortcut-anchor-schedule]] — decision.
- [[../50_Decisions/decided/avid-adapter-init]] — upstream init
  decision that makes this load-bearing.
- [[risk-shortcut-self-consistency-collapse]] — Option B mitigation
  that this implements, with the warmup wrinkle on top.
- [[feat-shortcut-add-d-zero-gate]] — Option A from the same risk
  (architectural zero-asymptote gate). Orthogonal to this schedule;
  both could coexist.
- Code anchors:
  - `src/generative_flow_adapters/config.py:52-68`
  - `src/generative_flow_adapters/training/trainer.py` (lines TBD)
  - `configs/diffusion_avid_shortcut_metaworld.yaml`
  - `configs/diffusion_hyperalign_shortcut_metaworld.yaml`
