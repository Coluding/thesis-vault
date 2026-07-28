---
last_updated: 2026-07-28
status: living
---

# Experiment results ledger

Single reverse-chronological index of **every experiment result that actually
ran** (this folder's per-run notes). Purpose: reuse for the thesis at a glance —
what we have, with sources, and which section each result feeds.

**Conventions**
- One row per result note in this folder. Newest first (by the note's `date:`).
- **Every metric traces to a run** (wandb id + ckpt + commit) per CLAUDE.md hard
  rules 7–8. Headlines here are pointers; the full sourced numbers live in the
  linked note. `_nv_` = the note's own source is marked `_needs verification_`.
- **Thesis §** = where the result is *actually cited* in `70_Thesis/draft/`.
  `—` means **not yet used in the draft** (a reuse gap to close, not an error).
- Add a row whenever you promote a completed run into a note here (never for a
  planned run — those are tickets under `20_Tickets/experiments/`).

## Results

| Date | Run (wandb) | Base × Dataset | D | Headline (sourced in note) | Thesis § | Note |
|---|---|---|---|---|---|---|
| 2026-07-28 | `ncztxyyo` `c3pcewxk` `8zjjn7wl`; `423pjv8y`(AVID ref) | Wan/DC/SkyReels × ACWM Robot Arm | D2 | **all three adapters action-blind on Robot Arm** (effect_rel ~0, cos~1, null-violation 0) despite 8.7× residual — 3 distinct starvation signatures | — | [[20260728-acwm-robotarm-matrix-action-blind]] |
| 2026-07-25 | local diag (no wandb) | Wan5B × ACWM Arm/Cube | D2 | frozen-base residual **0.314 (Arm) vs 0.036 (Cube)**, ≈8.7× — base-parity is a flat-visuals problem | §5.1.y _(proposed, not yet written)_ | [[20260725-acwm-base-residual-diagnostic]] |
| 2026-07-24 | `rxzwh4ak` `o79ki0ul` `o9113j4h`(failed) `hvxlbfjx`; `uxrst2k5`(ref) | Wan5B × MetaWorld | D2 | base-parity persists after both optimization traps fixed (cos 0.86, denoise Δ≈0, FID within 0.1 of base) | §5.1.x | [[20260724-metaworld-cap-shift-triangle-base-parity]] |
| 2026-07-21 | `y1jrgxqp` `uxrst2k5` | Wan5B × MetaWorld | D2 | replace-noise root cause validated; total action-blindness of the xattn adapter (σ-sweep + action probe) | §5.1.x | [[20260721-replace-fix-validation-sigma-sweep-action-probe]] |
| 2026-07-16 | `bcipghvw` `uea10230` `5cxstyh4` | Wan5B × MetaWorld | D2 | three xattn runs: adapter converges to clone the base, not to use actions (wandb-API pulled) | — | [[20260716-wan-xattn-adapter-clones-base-not-actions]] |
| 2026-07-15 | `pg3x72uc` | AVID-native × MetaWorld | expl | AVID native reference run — gate healthy (positive control) | — | [[20260715-avid-metaworld-native-gate-healthy]] |
| 2026-07-12 | `xb76ptw2` | Wan5B × MetaWorld | D2 | xattn action injection worse than base on every eval metric (killed @2661) | — | [[20260712-wan-xattn-action-no-improvement]] |
| 2026-07-09 | _nv_ | Wan5B(pretrained) × MetaWorld | D4 | first genuinely-pretrained frozen base — coherent samples, weak action tracking; base_loss flat ~0.10–0.20 | — | [[20260907-flow-shortcut-weak-action-signal]] |
| 2026-06-29 | _nv_ | flow vs diffusion × MetaWorld | D3 | first post-pivot sample batch — loss ok, sample videos poor (blur/fog/colour drift) | — | [[20260629-flow-vs-diffusion-shortcut-samples]] |
| 2026-06-27 | _nv_ | Wan5B AVID no-shortcut × MetaWorld | D2 | action-conditioned AVID adapter without shortcut consistency (ablation) | — | [[wan22-avid-noshortcut-ablation]] |
| 2026-06-17 | _nv_ (proj `avid-shortcut-metaworld-0.45`) | AVID-shortcut × MetaWorld (large) | D3 | volatile shortcut loss diagnosed as a step-size-mixing artifact (resolved by per-step-size logging) | — | [[avid-shortcut-anchor045-volatile-loss]] |

## Not results (planning / protocol — kept here but not indexed as runs)

- [[protocol-param-matched-adapter-comparison]] — param-matched cross-family
  comparison **protocol** (D1/D2), not a run.

## Reuse gaps (rows with Thesis § = —)

These are completed results **not yet cited in the draft**. Candidates to pull
in when writing the relevant section:
- D2 base-parity build-up (`20260716`, `20260712`) → context for §5.1.x.
- D3/D4 shortcut + flow (`20260629`, `20260617`, `20260709`) → §5.2 / §5.3 (stubs).
- The **base-residual diagnostic** (`20260725`) → proposed §5.1.y (results) +
  §4.4 base-selection (SkyReels probes are qualitative, logged in
  [[../../60_Updates/entries/2026-07-25-base-parity-is-flat-visuals-robotarm-residual]]).
