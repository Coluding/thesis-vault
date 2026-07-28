---
type: exp
scope: shortcut
status: open
priority: high
created: 2026-07-26
updated: 2026-07-26
resolution:
resolution_note:
closed_at:
related:
  - "[[../../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]]"
  - "[[../../30_Knowledge/theory/shortcut-v-averaging-bias]]"
  - "[[../../30_Knowledge/writing/storyline-experiment-requirements]]"
  - "[[exp-shortcut-action-free-isolation]]"
  - "[[exp-shortcut-per-stepsize-loss-diagnosis]]"
  - "[[../bug-losses-shortcut-v-averaging-target]]"
---

# R7 — shortcut-target A/B (v_average vs endpoint_inversion), action-free

**Requirement R7** of [[../../30_Knowledge/writing/storyline-experiment-requirements]],
and the falsifiable test named in
[[../../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]]
§"How we will know it worked".

## The prediction being tested

[[../../30_Knowledge/theory/shortcut-v-averaging-bias]]: under `v_average`
(Frans eq. 4) the true velocity field is **not a fixed point** of the averaging
rule, so the coarse (few-step) rungs plateau and **cannot** be trained out.
Under `endpoint_inversion` the true field *is* a fixed point, so the same rungs
should converge.

This is the highest-value experiment in the thesis: it converts a derivation
into a **validated prediction**, and it holds whether or not any adapter run
succeeds.

## Why this is one run, not an implementation task

Audited 2026-07-26 — most of R7 already exists:

| Piece | State |
|---|---|
| Zero-model-error analysis (5.1 / 16.1 / 24.1 % vs 0.000000) | banked, theory note §4 |
| Per-rung logging (`shortcut_direction_loss/N{steps:03d}`) | shipped 2026-06-17 |
| "Before" arm: fine rungs converge, coarse plateau, ~50× spread | measured (June) |
| `invert_ddim_v` + `target_kind="endpoint_inversion"` | shipped, commit `279cdb7` (2026-06-24) |
| Regression test (inversion reproduces the 2-step landing; average does not) | `tests/test_shortcut_endpoint_inversion.py`, 4 passing |
| **A training run with the fix + per-rung curves** | **missing — this ticket** |

## Two design choices, both load-bearing

1. **Both arms at the same commit.** The June `v_average` curves predate
   `279cdb7` by ~2 months of unrelated changes, so comparing a new run against
   them would confound the target rule with everything else. `v_average` was
   deliberately kept as a config-selectable baseline arm for exactly this.
   Bonus: it repairs provenance —
   [[../../60_Updates/entries/2026-06-19-shortcut-v-averaging-bias-resolved]]
   records the June run's `wandb id / commit / ckpt` as `_needs verification_`,
   so that data is **not citable** under hard rule 8 as it stands.

2. **Action-free.** With actions in the loop, "does the shortcut work" and
   "does action conditioning work" are confounded — and the action side is the
   one currently failing (base-parity collapse). Stripping actions isolates the
   D3 question. See [[exp-shortcut-action-free-isolation]] for the full
   {action on/off} × {shortcut on/off} matrix.

   **Consequence for the storyline: D3 is not gated on D2.** The pure shortcut
   result is obtainable without action conditioning ever working —
   [[../../30_Knowledge/writing/thesis-storyline]] §6 is corrected accordingly.

## Tooling (built 2026-07-26)

- `configs/dynamicrafter/diffusion_avid_shortcut_actionfree_metaworld.yaml` —
  fork of the DC shortcut config with `conditions: []` (step-size conditioning
  only) and the full few-step `eval_step_schedule` restored.
- `--shortcut-consistency-target` CLI override on
  `scripts/train_avid_shortcut_metaworld.py` — the A/B switch.
- `jobs/experiments/exp_shortcut_target_ab_actionfree.sh` — runs both arms,
  writes `provenance.txt` (arm, commit, config, dataset, steps) per run dir, and
  warns when the working tree is dirty between arms.

## Run

```bash
STEPS=20000 jobs/experiments/exp_shortcut_target_ab_actionfree.sh
```

## Readout

Compare `shortcut_direction_loss/N001`, `/N002` (coarse) and `/N064`, `/N128`
(fine) across arms.

- **CONFIRMED** — coarse rungs plateau under `v_average`, converge under
  `endpoint_inversion`; fine rungs match across arms.
- **REFUTED** — coarse rungs behave the same in both arms. The plateau has
  another cause. **Still a real finding**: the derived bias would not be the
  binding constraint on few-step quality, and the D3 story needs rewriting
  around whatever is.
- **CONFOUNDED** — fine rungs differ between arms. Something other than the
  target rule changed; do not report. Fine rungs sit near the `d→0` limit where
  the bias vanishes, so they are the control.

Then the quality half: few-step rollout PSNR/SSIM/LPIPS/FVD at `s ∈ {1/2, 1/4}`,
adapted vs frozen base at the same NFE budget. Quantify, don't eyeball
(hard rule 8).

## Notes

- `displacement` (Option B) still raises `NotImplementedError` — a third arm is
  possible later but is not needed to test the prediction.
- The selector is a **no-op on flow matching** (κ=0, no bias), so this A/B is
  necessarily a diffusion/DynamiCrafter experiment. The flow-side action-free
  shortcut run is the separate [[exp-shortcut-action-free-isolation]].

## Outcome

_Not yet run._
