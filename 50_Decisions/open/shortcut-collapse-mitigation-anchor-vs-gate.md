---
type: decision
status: open
created: 2026-06-02
decided_at:
updated: 2026-06-02
target_date:
scope: training
related:
  - "[[../../20_Tickets/done/risk-shortcut-self-consistency-collapse]]"
  - "[[../../20_Tickets/done/feat-shortcut-add-d-zero-gate]]"
  - "[[../../20_Tickets/done/feat-training-shortcut-anchor-step-warmup]]"
  - "[[../decided/shortcut-anchor-schedule]]"
  - "[[../decided/avid-adapter-init]]"
  - "[[../../30_Knowledge/related-work/shortcut-models]]"
---

# Decision: Shortcut collapse mitigation — diffusion-loss anchor + warmup as primary, d→0 gate deferred

## Status

**Open — captured 2026-06-02.** Primary mitigation is chosen: the data
anchor (in use) plus an anchor-step warmup for the early transient (chosen,
**implementation still pending** — see below). The open part is only whether
the architectural `g(d→0)` gate ever needs to be added on top. Not closing
until we have training evidence one way or the other.

This decision is now the single home for the shortcut-collapse-mitigation
plan; the warmup implementation ticket
([[../../20_Tickets/done/feat-training-shortcut-anchor-step-warmup]]) was
folded in here rather than tracked separately.

## Decision

**Primary defense against self-consistency collapse is the data anchor
through the standard diffusion/flow loss** — Option B of
[[../../20_Tickets/done/risk-shortcut-self-consistency-collapse]]. A large
fraction of each batch (`shortcut_anchor_prob: 0.75`, the paper's 3/4)
trains the composed prediction against empirical velocity at the smallest
step; the remaining fraction trains self-consistency on the dyadic ladder,
which bootstraps off the anchored bottom rung.

This is sufficient on its own to rule out the two named collapse modes **at
convergence**:

- **Cancellation collapse** (`Δ_φ ≡ −base ⇒ s ≡ 0`) — the anchor loss
  becomes `‖0 − v_target‖²`, large, so it is directly penalized.
- **Constant-field collapse** (`s` independent of `d`) — the anchor pins
  the smallest step to data; self-consistency then forces each higher rung
  to the true multi-horizon average via the intact bootstrap chain.

### Transient handling — anchor-step warmup (chosen; not yet implemented)

The anchor prevents collapse *at convergence*, but not in the early
transient: with the status-quo init
([[../decided/avid-adapter-init]], step-0 prediction = `0.5·base +
0.5·random`) a randomly-initialised `Δ_φ` corrupts the self-consistency
teacher before the anchor has taught it to back off. The chosen fix
(implementing [[../decided/shortcut-anchor-schedule]]) is a **step warmup**
that zeroes the self-consistency loss weight for the first N optimizer
steps, so the adapter first learns structured small-step predictions on
real data before the consistency teacher comes online. Step warmup was
chosen over a smooth decay schedule — do not re-litigate.

**Implementation sketch (preserved from the folded ticket; ~20 LoC + 1
test + 2 YAML edits):**

- `config.py` (~52-68): add `shortcut_anchor_warmup_steps: int = 0`
  (default 0 ⇒ non-shortcut configs bit-identical to today).
- `trainer.py`: gate the SC weight on the global step —
  `consistency_weight = 0.0 if global_step < shortcut_anchor_warmup_steps
  else shortcut_direction_weight` (verify exact attr names against the
  current trainer).
- YAML: set `training.shortcut_anchor_warmup_steps: 5000` in the live
  shortcut configs (analysed-estimate start; sweep `{1000, 5000, 20000}`
  as a follow-up once a baseline run exists).
- Logging: `train/shortcut_consistency_active` (0→1 at boundary) and
  `train/shortcut_warmup_remaining_steps`.
- Test: with `warmup_steps=N`, SC term contributes 0 for `global_step < N`
  and its full weight from `N` onward (mock the schedule; no full run).

Deferred warmup follow-ups (open as tickets only if/when needed): adaptive
EMA-loss-driven warmup, tuning N, loss-mode-conditional warmup (active only
under `shortcut_target_method: distillation`).

### The deferred gate

The architectural `g(d→0)=0` gate
([[../../20_Tickets/done/feat-shortcut-add-d-zero-gate]]) is **not** part of
the primary plan. It is deferred, not rejected.

## Why the gate is deferred, not adopted

With a 0.75 data anchor + intact dyadic bootstrap + sane loss weights, the
gate is largely **redundant for collapse-prevention**. Its only residual
value:

1. An *exact* `d→0` guarantee (the anchor grounds the smallest *trained*
   step `1/128`, not literally `0`) — mostly cosmetic since the frozen base
   is already correct there.
2. Smoothing the **early transient**, where a randomly-initialised `Δ_φ`
   corrupts the self-consistency teacher before the anchor has taught it to
   back off.

Point 2 overlaps heavily with the **anchor-step warmup above** (zero the SC
weight for the first N steps), which targets the same transient more
directly — which is why the warmup, not the gate, is the chosen transient
fix.

## Open question — when we would revisit the gate

Add the `g(d)` gate **only if** training evidence shows the anchor + warmup
are insufficient, e.g.:

- the collapse diagnostics (`‖Δ_φ(·,d=1)‖`, `‖s_composed − base‖` vs `d`)
  misbehave at small `d` even with the anchor on, or
- the early transient stays unstable after the warmup lands.

If revisited, the gate must target the **live** adapters
(`adapters/output/dynamicrafter.py`, `adapters/hidden_states/unicon.py` on
the `step_level` branch), not the deprecated `ShortcutDirectionOutputAdapter`
the original ticket pointed at. Evaluate the gate **together with** the
warmup, since they address the same failure window.

## Caveats carried forward

- The anchor prevents collapse *at convergence / in expectation*, not during
  the early transient — that is what the anchor-step warmup above handles.
- The grounding only holds if the bootstrap chain stays intact and the
  self-consistency weight does not swamp the anchor weight. Keep the anchor
  fraction large.
