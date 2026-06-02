---
date: 2026-05-28
category: decision
deliverable: D2
meeting:
sources:
  - "[[../../50_Decisions/decided/per-sample-frame-stride-sampling]]"
  - "[[../../30_Knowledge/tech/frame-stride-conditioning]]"
  - "[[../../20_Tickets/bug-data-metaworld-fs-anchor-default]]"
  - "[[2026-05-25-frame-stride-conditioning-audit]]"
---

# Decision: anchor MetaWorld `fs` at the AVID convention (no per-sample sampling)

## What
Resolved the open decision raised by last week's frame-stride audit. Read
the AVID team's MetaWorld data module
(`external_repos/avid/latent_diffusion/src/ldwma/lightning/data_modules/metaworld.py:107-108`)
and found that they write `fps=10` and `frame_stride=10`
unconditionally and read pixels **contiguously**
(`pixels[start:start+traj_len]`, no slice stride). Both numbers equal
the base DynamiCrafter UNet's pretrained `default_fs=10` anchor (config
`external_repos/avid/.../dynamicrafter_512.yaml:68`). Adopted the same
convention (**Option A**): the MetaWorld translator will write
`fps=frame_stride=10` and pin the slice stride at `1`. No per-sample
`fs` variation, no action subsampling.

## Why it matters
- Puts the frozen base's fps channel **in distribution by
  construction** — at its trained anchor — and sidesteps the
  `fps_condition_type` (`"fs"` vs `"fps"`) semantic ambiguity, since
  the base sees its anchor under either interpretation. The
  verification step that previously gated this decision is no longer
  on the critical path.
- Closes the action-dropping trap raised in the audit (`stride=k`
  silently drops `a₁..a_{k-1}` for an *action-conditioned* world model)
  by not subsampling at all.
- Closes the precedence bug
  (`batch_preprocessor.py:257`: `frame_stride` overrides `fps`) as a
  side effect — both keys now carry the same number, precedence stops
  mattering.
- Cost: the augmentation motivation that opened the decision (varied
  temporal resolution) is parked. Acceptable because D2's contribution
  surface is *action-conditioned dynamics*, not *temporal-resolution
  robustness*; the random clip **start index** in `dataset.py:95`
  already provides some temporal variation for free.

## Evidence / sources
- Decision (resolved):
  [[../../50_Decisions/decided/per-sample-frame-stride-sampling]] —
  full reasoning, shelved sub-resolutions (SUM action aggregation,
  per-sample uniform `{1..max_stride}`, mapped `fs(k)`), and the
  revisit triggers.
- Audit (last week's entry):
  [[2026-05-25-frame-stride-conditioning-audit]].
- Tech note (now resolved at the bottom):
  [[../../30_Knowledge/tech/frame-stride-conditioning]].
- AVID convention reference:
  `external_repos/avid/.../data_modules/metaworld.py:107-108`.
- Base anchor: `external_repos/avid/.../dynamicrafter_512.yaml:68`.
- _No metrics — design decision, not a run._

## Next
- Derived ticket: [[../../20_Tickets/bug-data-metaworld-fs-anchor-default]]
  (medium). Translator change is small; drops the `--frame-stride` CLI
  flag from the MetaWorld training scripts since it no longer changes
  anything.
- **Do not** open a separate ticket for the `_extract_fs` precedence
  in `batch_preprocessor.py` — it becomes a non-issue under (A) and
  any in-flight version of that ticket should be closed as obsolete.
- Future revisit triggers (recorded in the decided note): D2 model
  brittle to inference-time fps changes; evidence of constant-`fs`
  data masking an identifiable `Δ_φ` failure mode; or a base swap to
  one with fully-known fps-channel semantics.
