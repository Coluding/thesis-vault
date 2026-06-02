---
date: 2026-05-28
category: progress
deliverable: D3
meeting:
sources:
  - "[[../../20_Tickets/bug-training-shortcut-target-timestep]]"
  - "code: src/generative_flow_adapters/training/shortcut_targets.py"
  - "code: src/generative_flow_adapters/training/trainer.py"
---

# Shortcut target computation consolidated into one paper-faithful module

## What
The shortcut training path had two implementations of the two-step
target: a stale public helper (`attach_shortcut_targets_from_base` in
`training/shortcut_targets.py`) carrying the bug from
`bug-training-shortcut-target-timestep` — second base call at the
**same** `t` instead of `t+d` (`# Use same t (simplified)`) — and a
correct in-trainer implementation (`_compute_two_step_target_v`)
that uses a proper DDIM micro-step and advances time. The stale module
was dead code (no script/test/config imported it) but was still
exported on `training/__init__.py` and used by the two
`examples/*_shortcut_training_test.py` smoke scripts, which therefore
silently bypassed the live path. Refactor: deleted the stale module
+ its `testing/` duplicate; moved the live target math into
`training/shortcut_targets.py` as pure free functions
(`compute_two_step_target_v`, `compute_self_consistency_target_v`,
`ddim_micro_step_v`); updated the trainer to call into them; updated
the examples to not pre-attach.

## Why it matters
- The single mathematical divergence from the paper-faithful target
  construction in the shortcut training path is closed. The headline
  D3 experiments will be argued against the paper's formulation, so
  removing this silent simplification protects the eventual claim.
- The dead public helper was exactly the kind of vestigial API that
  gets cargo-imported into a notebook later — removing it removes a
  future failure mode (the examples were already exercising it).
- Target math is now in one file, taking all dependencies (schedule
  tables, models) explicitly — testable without spinning up a trainer.

## Evidence / sources
- Ticket: [[../../20_Tickets/bug-training-shortcut-target-timestep]] —
  cites both lines (`shortcut_targets.py:122`, `trainer.py:327`) and
  the inline `# Use same t (simplified)` comment that confirmed the
  shortcut was deliberate at implementation time.
- Code (post-refactor): `training/shortcut_targets.py` now contains
  `compute_two_step_target_v`, `compute_self_consistency_target_v`,
  `ddim_micro_step_v`. `trainer.py:_maybe_prepare_shortcut` calls
  them; the trainer wrappers + module-level helpers were removed.
- _No metrics — refactor + bug-fix confirmation, not a run._

## Next
- Close the ticket via the standard close ritual once the user
  confirms (the live path was already correct; the fix here is
  "no remaining buggy path exists").
- Unrelated cleanup uncovered along the way: `_reshape_step_size_for_base`
  in trainer.py was unused; removed.
- Does **not** alter open D3 risk-shortcut-eval-steplevel-out-of-distribution
  or the d=0 gate work — those remain on the critical path.
