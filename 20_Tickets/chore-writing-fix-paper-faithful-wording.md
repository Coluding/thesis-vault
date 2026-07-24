---
type: chore
scope: writing
status: open
priority: medium
created: 2026-06-22
updated: 2026-06-24
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]]"
  - "[[bug-losses-shortcut-v-averaging-target]]"
---

# chore: correct the "paper-faithful" wording for the shortcut target

The docstring of `compute_self_consistency_target_v` and
[[../30_Knowledge/theory/shortcut-training]] §3 describe the `(v1+v2)/2` target
as "paper-faithful (Frans et al. 2024, eq. 4)." That is faithful only in the
`d→0` limit / flow-matching world. Under **v-prediction**, *faithful* means
**endpoint-consistent**, not **v-averaged** — see
[[../30_Knowledge/theory/shortcut-v-averaging-bias]] §6.

## Progress — 2026-06-24

- ✅ **Code docstring done.** `compute_self_consistency_target_v`'s module +
  function docstrings (`training/shortcut_targets.py`) no longer call
  `(v1+v2)/2` "paper-faithful" — they now state it is exact for flow matching
  but **biased for diffusion v-prediction**, and document the exact
  `endpoint_inversion` alternative. Landed with
  [[bug-losses-shortcut-v-averaging-target]].
- ⬜ **Remaining:** [[../30_Knowledge/theory/shortcut-training]] §3 still carries
  the old "paper-faithful" framing — correct it there too, then close.

## Done when

The wording is corrected in both places once the endpoint-consistent target
lands ([[bug-losses-shortcut-v-averaging-target]]), so the code comment no
longer asserts a biased rule is paper-faithful.
