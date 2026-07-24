---
type: chore
scope: writing
status: open
priority: low
created: 2026-07-15
updated: 2026-07-15
resolution:
resolution_note:
closed_at:
related: ["[[done/refactor-shortcut-deprecate-twostep-add-heun-smoothness]]"]
---

# chore: doc-polish leftover from the two_step deprecation / Heun-smoothness ship

Split out from
[[done/refactor-shortcut-deprecate-twostep-add-heun-smoothness]] when that
ticket closed 2026-07-15 — the code (all 3 patches) is fully shipped; these
are the remaining documentation/surface items only.

## What's left

1. Surface `heun_smoothness_weight` in at least one example config (the
   original ticket's acceptance criterion 4 — not yet done).
2. Vault documentation updates:
   - [[../30_Knowledge/tech/shortcut-training-modes]] — collapse to one active
     mode (`distillation`); move `two_step` to a "deprecated modes —
     historical" section, add a `heun_smoothness` section.
   - [[../30_Knowledge/theory/shortcut-training]] §4.2 — rewrite from "Two
     supported regimes" to "One shortcut regime (`distillation`) + an
     orthogonal Heun-smoothness regularizer."
   - [[../30_Knowledge/writing/explainer-shortcut-training]] and
     `figure-shortcut-training` — relabel or remove any `two_step` Heun
     construction shown as part of *shortcut training* specifically.
3. Optional: run the heavy DynamiCrafter shortcut configs ≥100 steps under
   `distillation` to confirm no blow-up (the dummy config path is already
   verified; this is the real-config confirmation, lower priority since the
   distillation path itself was untouched by the two_step deletion).

## Guardrails

Pure documentation/config-surfacing — no code changes. Low priority, no
urgency; the underlying functionality is already correct and tested.
