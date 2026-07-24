---
type: chore
scope: data
status: open
priority: medium
created: 2026-07-09
updated: 2026-07-09
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]"]
---

# chore: audit action↔frame alignment & normalisation on the new data

## Why

**Lower priority now.** The known changed variable in the 20260907 run is the
**base model** (random → real pretrained WAN), not the data — so the leading
hypothesis is a strong frozen base under-incentivising the adapter, not a data
bug (see [[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]).
Still worth ruling out cheaply: if the adapter *is* trying to use actions but they
carry no usable signal, the culprits below would explain it. Run after the
base-vs-adapted delta ([[feat-eval-base-vs-adapted-delta]]) and shuffle
([[experiments/exp-conditioning-action-shuffle-ablation]]) tests.

## Checks

1. **Temporal alignment.** Confirm `a_t` is the action that drives the transition
   the model is asked to predict (frame `x_t → x_{t+1}` conditioned on `a_t`), not
   off-by-one (`a_{t-1}` or `a_{t+1}`). Verify against the MetaWorld dataloader /
   batch preprocessor on the new episodes.
2. **Normalisation.** Are action-normalisation stats (mean/std or min/max)
   computed on the new data, or stale from the old set? Log the post-norm action
   distribution — degenerate/near-constant actions carry no signal.
3. **Diversity.** Do the new episodes actually contain varied actions, or are they
   near-static? A dataset where the arm barely moves gives the adapter nothing to
   condition on and `base_loss` would sit at the base's unconditional floor.
4. **Plumbing.** Confirm actions survive preprocessing → `cond` → adapter (not
   silently dropped or overwritten by `drop_condition_prob`).

## Output

A short finding note (or an update to the experiment note) stating which of the
above held. If any check fails, open a `bug-data-*` ticket for the fix.

## Notes

Do in parallel with [[experiments/exp-conditioning-action-shuffle-ablation]] — the shuffle
test says *whether* actions matter; this audit says *why* they might not.
