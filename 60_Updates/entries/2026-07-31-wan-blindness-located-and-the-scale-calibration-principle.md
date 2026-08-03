---
date: 2026-07-31
category: finding
deliverable: D2
meeting:
sources: ["[[20260731-wan-action-trace-value-pathway-drowns]]", "[[20260730-dc-parity-arms-null-action-embedding-pedestal]]", "[[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]]", "[[../../20_Tickets/experiments/exp-conditioning-decouple-encoder-bias]]"]
---

# Wan's blindness located too — and the two mechanisms are mirror images

## What

A 23-depth propagation trace on the Wan adapter (paired true-vs-shuffled-action
forwards, hooks at every layer) located exactly where the action signal dies:
it **survives cross-attention intact** (the attention output is 44–56%
action-driven in all 10 blocks) and then **drowns at the residual add** — a
faithful contribution of RMS ~0.01 joins a stream of RMS 1.8–3.0, and the
action-driven fraction collapses 0.44 → 0.0085 in a single addition, exactly
matching the measured blindness (0.0096).

Root cause: qk-norm rescues the attention *logits*, but the **value pathway is
unnormalised** and inherits the raw action-token scale (RMS 0.006 vs the text
path's O(1)).

## Why it matters

Combined with the DC result, D2 now has a **unifying mechanism**:

| | DC | Wan |
|---|---|---|
| conditioning arrives | 14× too **loud**, 99.7% **constant** | 250× too **quiet**, ~97% **faithful** |
| failure | learned pedestal swamps the signal | stream swamps the signal |
| fix | subtract the constant (`condition_center`) | normalise the tokens (`action_token_norm`) |
| result | 0.004 → **0.092**, 3.1× the AVID reference, climbing | **6–10×** control, but eroding |

**The principle:** plug-and-play conditioning fails on *scale calibration at the
injection interface* — in either direction — and on *incentive*: wherever an
action-free shortcut to the loss exists (a copyable base output, a learnable
constant), the optimiser takes it. Wan's fix erodes because
`condition_on_base_outputs: true` keeps a 95%-correct answer inside the
adapter's input (`pred_base_cosine` 0.985); the arm removing that oracle
(token-norm + `condition_on_base_outputs: false`, job 25088945) is running.

## Evidence / sources

- Trace: job 25085110 on `ncztxyyo` step 1000; full profile in
  [[20260731-wan-action-trace-value-pathway-drowns]].
- Fix vs matched control (both live-gate): TOKENNORM 0.0113→0.0085 vs GATEFIX
  0.0012→0.0015 (jobs 25085598 / 25083978, in flight).
- DC arm E trajectory 0.026→0.092 (run `6oyu1inq`, in flight); controls flat.
- Hypotheses killed by measurement: train/eval σ-mismatch (flat 0.0084–0.0102
  across σ), positional-readout, attention-untrained.

## Caveats for the slide

- DC arm E and all Wan arms are **in flight** — trajectories, not endpoints.
- The `gate_cap` freeze bug (gate born on its clamp boundary, `gate_std ≡ 0`;
  6 configs fixed + code guard) **confounds the 2026-07-29 D3 flow-vs-diffusion
  comparison** — do not cite the 68× consistency-loss gap until re-run.
- Supervisor deck (interactive, all numbers run-sourced):
  claude.ai/code/artifact/197bc404-dee9-42df-a3d0-fc4b01b57aa1
