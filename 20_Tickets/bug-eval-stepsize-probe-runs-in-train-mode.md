---
type: bug
scope: eval
status: done
priority: high
created: 2026-08-01
updated: 2026-08-01
resolution: shipped
resolution_note: "eval() + fork_rng added to _stepsize_sensitivity_eval; effect_rel now suppressed when the null control fires. Fixed and rsync'd to the cluster before the D3 arms started."
closed_at: 2026-08-01
related: ["[[experiments/exp-shortcut-d3-fewstep-vs-noshortcut-control]]", "[[bug-adapter-gate-cap-equals-init-freezes-gate]]", "[[../00_Inbox/2026-08-01-stepsize-null-violation-rootcause]]"]
---

# bug: the D3 step-size probe ran the frozen base in train mode

## Symptom

`eval_stepsize_base_null_violation` logged **0.028–0.039** on all 9 DC parity
runs. The frozen base cannot see `step_level` at all, so this control must be
**exactly 0** — and when it is not, `eval_stepsize_effect_rel` (the only metric
that can demonstrate a D3 adapter is genuinely step-size conditioned) is
measuring whatever *else* differs between the paired forwards.

## Root cause — verified

`_eval_epoch` restores train mode in its `finally` at
`training/trainer.py:617`, and `_stepsize_sensitivity_eval` is called at
**:626** — *after* the restore. Unlike the action probe, which delegates to
`run_action_sensitivity` and sets `model.eval()` at
`evaluation/action_sensitivity.py:314` plus `fork_rng`/`manual_seed` at
`:328,:354`, the step-size probe set **neither**.

So DC's frozen base re-armed its `ResBlock` `nn.Dropout(p=0.1)`
(`openaimodel3d.py:197`, `dynamicrafter512.yaml:54`) plus a hard-coded
`dropout=0.1` `TemporalConvBlock` (`:210`), and the two paired forwards
differed for reasons unrelated to `step_level`. `freeze()` calls `.eval()` at
construction (`interfaces.py:18-22`) but nothing overrides `train()`, so the
trainer recursively un-froze it.

**A `step_level` leak was ruled out by construction:** `_to_lvdm_cond`
(`dynamicrafter_video.py:303-315`) is a strict whitelist that drops
`step_level` before the UNet.

### Evidence

- **DC-only.** Of 10 runs across all 46 `coluding` projects that ever logged a
  stepsize key, 9 are DC (null 0.0278–0.0388) and `hcrnc9gf` is Wan — which has
  **zero** `Dropout` modules and logged the null as **exactly 0**. That exact
  zero also rules out bf16 accumulation noise: the harness demonstrably emits
  exact zeros.
- **Near-controlled A/B:** `eval_action_base_null_violation` = 0 on the *same*
  runs in the *same* eval cycle. The only difference is the two missing guards.
- **It drifts across evals** (0.0279 → 0.0449 → 0.0358) on a frozen base — a
  deterministic leak would be constant.
- Two runs with **different adapters** (`6oyu1inq` arm E, `tr0uovs5` arm 0) are
  bit-identical at every step while their `effect_rel` differs ⇒ purely a
  base-path property.
- CPU repro on the real vendored `ResBlock`: eval → `0.000000000` at every
  setting; train → 0.046–0.73.

## Blast radius

`eval_stepsize_effect_rel` on the 9 DC parity runs is **100% artifact** — those
configs have `use_step_level_conditioning: false`, so the ground truth is
exactly 0 while **0.049–0.061** was logged. Silver lining: that calibrates a
**DC noise floor of ~0.06** for this statistic on this data.

**No thesis prose is poisoned** — the curvature note's runs predate the probe
and log no stepsize keys. But its "Next" step would have produced a bad number.
Independent of the `gate_cap` freeze bug for D3 arms A/B (that config has no
`gate_cap`), though the two compound mechanically under `avid_mask_mix`.

## Fix — applied 2026-08-01

`training/trainer.py`:

1. `model.eval()` on entry, `model.train(was_training)` in a new `finally`.
2. Each probe level's forward wrapped in `torch.random.fork_rng(...)` +
   `manual_seed(0)`, mirroring the action probe, so residual stochasticity
   cancels between levels rather than masquerading as step-size sensitivity.
3. **Hard gate:** when `null_viol > 1e-3`, emit `eval_stepsize_probe_invalid`
   and **suppress** `eval_stepsize_effect_rel` entirely rather than publish a
   number the control says is invalid.

Verified: `py_compile`, import of `_rng_devices`, and all four markers present
in the source both locally and on the cluster. Pushed **before** D3 arms
25141979/25141980 started, so they will log a valid metric.

**Deliberately NOT fixed by caching the base and passing `base_output=`** —
that would make the null compare a tensor to itself, reporting 0 by
construction and destroying the control.

## Not done — follow-ups

- A `train()` override on frozen bases would kill this whole bug class
  (`interfaces.py`). Worth doing; not done here to keep the change minimal
  while jobs were queued.
- The 9 affected runs' `eval_stepsize_effect_rel` values should be treated as
  unreported, not as zeros. Recoverable offline from the retained checkpoints
  now that the probe is correct.
