---
type: exp
scope: conditioning
status: done
priority: high
created: 2026-07-09
updated: 2026-07-21
resolution: shipped
resolution_note: >
  Executed 2026-07-21 (eval-only arms, per the Procedure) via the new
  --action-probe mode of scripts/generate_wan22_i2v_compare.py on the
  local replace checkpoint step_00001500.pt: true vs other-clip-shuffled vs
  zeroed actions, paired noise, at sigma in {0.1..0.99}. Result: decision
  rule branch 1 — NO change. Shuffled actions move the adapted loss by
  <1e-5, zeroed by <1e-4, at every noise level (nonzero zero-gaps prove
  the conditioning path is live). The adapter is completely action-blind;
  its ~5% deviation from base is action-independent domain adjustment.
  Escalation per the decision rule: data-alignment audit stays open
  (chore-data-action-frame-alignment-audit) and the action-informativeness
  question is now a dataset decision (ACWM-Phys candidate). Full table:
  30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe.md
closed_at: 2026-07-21
related: ["[[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]", "[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[../experiments/exp-adapter-adaln-gatelow-metaworld-run]]"]
---

# exp: action-shuffle / zero-action conditioning ablation

**The decisive test** for [[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]:
does the adapter actually use the action signal, or is it ignoring it?

## Hypothesis

On the 20260907 run (first with a **real pretrained WAN frozen base**), samples
are coherent but predicted arm poses diverge from GT — the adapter may be riding
the strong base prior and ignoring actions. If so, corrupting the actions will
leave the samples and the base-vs-adapted delta **unchanged**. (Note: with a real
base, `base_loss` alone is uninformative — pair this with
[[feat-eval-base-vs-adapted-delta]], not the base_loss trend.)

## Procedure

1. Take the 20260907 config (or an equivalent flow-shortcut action-conditioned
   config).
2. Two corruption arms (eval-only is enough to start; a short retrain arm is
   stronger):
   - **shuffle**: permute `a_t` across the batch (breaks action↔frame pairing,
     preserves marginal action distribution).
   - **zero**: set actions to zero / the null-condition embedding.
3. Compare vs the real-action baseline on: `base_loss` / `eval_base_loss`, and
   sample rollouts (does the predicted arm still diverge the same way?).

## Decision rule

- **No change in base_loss or samples** ⇒ adapter is not using actions →
  conditioning path is broken or actions uninformative. Escalate to the
  data-alignment audit ([[chore-data-action-frame-alignment-audit]]) and the
  adapter-magnitude check ([[feat-training-adapter-contribution-magnitude-logging]]).
- **Real actions clearly better** ⇒ the adapter *does* use actions; the flat
  base_loss is an optimisation / capacity / undertraining issue, not a broken
  signal. Redirect to LR / capacity / longer training.

## Notes

Cheapest test with the highest discriminating power — run before the others.
Source run: `data/results/20260907/`.

## Update (2026-07-14)

Two things from [[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]:

1. **Use `probe_denoise_delta`, not raw `base_loss`.** As of this session
   (`training/trainer.py` `_probe_eval`) the trainer logs a paired,
   low-variance base-vs-adapted denoising delta on a frozen probe batch —
   exactly the comparison this ticket needs, without the noisy-eval confound
   the ticket's own note (`base_loss` alone is uninformative) already flags.
2. **Sequence:** run this *after*
   [[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]] (gate-throttle
   fix) — otherwise a "no change" result is ambiguous between "adapter ignores
   actions" and "adapter's gradient is throttled ~50× regardless of action
   content." See the diagnosis note's do-now order (this is step 3, after the
   zero-GPU shape check and single-clip overfit, but the composition fix at
   step 6 should land first if sequencing allows — the diagnosis explicitly
   recommends running shuffle **both before and after** the fix since the
   pre/post delta in shuffle-sensitivity is itself diagnostic).

## Concrete config (2026-07-15)

The "after the fix" config now exists:
`configs/diffusion_wan22_avid_gatelow_metaworld.yaml`
([[../experiments/exp-adapter-adaln-gatelow-metaworld-run]]). For the "before" arm, use the
same config with actions shuffled/zeroed at the data level (eval-only
corruption is enough to start, per the Procedure above) — no need to fall back
to the old `gate_bias: 4.0` config for the "before" comparison, since the
shuffle-sensitivity delta is about presence/absence of gate saturation, not
about which specific historical run to reuse.
