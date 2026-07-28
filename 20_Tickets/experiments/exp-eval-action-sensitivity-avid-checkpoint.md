---
type: exp
scope: eval
status: open
priority: high
created: 2026-07-26
updated: 2026-07-26
resolution:
resolution_note:
closed_at:
related:
  - "[[../../30_Knowledge/writing/storyline-experiment-requirements]]"
  - "[[../../30_Knowledge/writing/thesis-storyline]]"
  - "[[../../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]]"
  - "[[exp-adapter-avid-native-reference-run]]"
---

# R1 — action-sensitivity probe on the AVID/DynamiCrafter checkpoint

**Requirement R1** of [[../../30_Knowledge/writing/storyline-experiment-requirements]].
Tier 0: the storyline's node 1 ("AVID works") and node 2 (planning) both rest on
this, and it is the cheapest experiment on the list — an eval pass on an existing
checkpoint, no training.

## The question

Does perturbing the action change the adapter's prediction at all?

[[../../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]]
(wandb `pg3x72uc`) establishes clean convergence (~9.5× loss drop) and healthy
gate mechanics (mask mean 0.52 → 0.63). It does **not** establish that the model
uses actions. Those are different properties, and only the second one supports
planning: a planner needs the world model to *discriminate between candidate
action sequences*. If the prediction is action-invariant, every candidate rollout
scores identically and planning degenerates to random search.

## Tooling (built 2026-07-26, implementation repo)

- `src/generative_flow_adapters/evaluation/action_sensitivity.py` — measurement
  core, backbone-agnostic.
- `scripts/eval_action_sensitivity.py` — CLI, **dispatches on
  `config.model.provider`**: `dynamicrafter*` / `wan2.2*` / `wan2.1*` /
  `skyreels`. Unknown providers abort with the list of supported families
  rather than silently building the wrong preprocessor.
- `jobs/experiments/eval_action_sensitivity_dc_metaworld.sh`
- `jobs/experiments/eval_action_sensitivity_dc_acwm.sh`
- `jobs/experiments/eval_action_sensitivity_wan_acwm.sh` — the D2-headline
  readout (Wan is the backbone the ACWM dataset-axis runs are on).
- `tests/test_action_sensitivity.py` — 15 tests, passing.

**Conditioning keys.** `--action-keys` names the cond keys holding the action
(default `action,action_seq`). When passed explicitly, **every** named key must
be present in the batch or the run aborts. The probe also aborts when no action
key is found at all, and when the resolved keys differ across batches. All three
guard the same silent failure: an unperturbed action yields a confident
`ACTION-BLIND` verdict on a measurement of nothing.

**Metrics.** Primary is prediction-space: `rel_delta = |pred_true −
pred_perturbed| / |pred_true|` over predicted frames — it answers the question
directly and needs no target. Secondary: the loss gap (does the movement *help*),
`adapter_rel_contribution` (did the adapter move the output from the base at
all), and a bootstrap 95% CI over (batch × draw).

**Perturbations.** `shuffle` (another clip's actions — the primary, since the
actions stay on-distribution and only their correspondence breaks), `zero`,
`roll` (own actions, wrong temporal alignment — separates magnitude-use from
timing-use), `gauss`.

**Built-in null control.** The frozen base cannot see actions, so its loss must
be invariant across variants. A non-zero shift means the harness leaks actions
(e.g. `pass_cond_to_base`, or condition dropout still active) — the script exits
non-zero rather than reporting numbers that must not be trusted.

## Run

```bash
jobs/experiments/eval_action_sensitivity_dc_metaworld.sh <checkpoint.pt>
ENV=push_block jobs/experiments/eval_action_sensitivity_dc_acwm.sh <checkpoint.pt>
```

## Blocker to resolve first

`pg3x72uc` was trained with the **real upstream AVID code**
(`external_repos/avid/latent_diffusion/...`, config
`avid_11M_metaworld.yaml`), so its state dict does not match a
`generative_flow_adapters` model and `load_state_dict` will reject it. Two
options — **needs a decision**:

1. Point the probe at a **repo-side** DynamiCrafter/AVID checkpoint instead
   (works today, but then it is not the run cited in §6.1 of the outline).
2. Write an upstream→repo state-dict converter (more work; makes `pg3x72uc`
   itself citable).

## Interpretation guardrails

- **MetaWorld is the action-redundant dataset**
  ([[../../30_Knowledge/writing/ablation-axes]] Axis 1). A flat result there is
  confounded with the data and does **not** on its own show the adapter is
  broken. Re-run on ACWM Push Cube before concluding anything.
- **Sensitivity is necessary, not sufficient.** A non-zero `rel_delta` with a
  ~0 or negative loss gap means the model reacts to actions without using them
  usefully — still an action-blind world model for planning purposes.
- Whatever the outcome, log it to
  `30_Knowledge/experiments/_index.md` and write the run note; a flat result is
  a **result**, not a non-event, and it feeds §6.1 + the boundary map.

## Outcome

_Not yet run._
