---
date: 2026-05-25
category: finding
deliverable: D2
meeting:
sources:
  - "[[../../30_Knowledge/tech/frame-stride-conditioning]]"
  - "[[../../50_Decisions/decided/per-sample-frame-stride-sampling]]"
  - "[[2026-05-28-frame-stride-decision-anchor-at-avid]]"
---

# Frame-stride conditioning audit

> **Update 2026-05-28 — resolved.** The decision that came out of this
> audit is closed: anchor MetaWorld `fs` at the AVID convention
> (`fps=frame_stride=10=default_fs`), no per-sample variation. See
> [[2026-05-28-frame-stride-decision-anchor-at-avid]] and
> [[../../50_Decisions/decided/per-sample-frame-stride-sampling]]. The
> "Next" block below records the in-flight state as of the audit date
> and is preserved for context — the gate it flagged (verify the
> DynamiCrafter `fps_condition_type`) became moot under the resolution.

## What
Audited how the frame-stride / fps signal (`fs`) flows through the pipeline.
Three findings: (1) `fs` reaches the **frozen base UNet** but is **not** an
explicit conditioning input to the trainable adapter `Δ_φ` (HyperAlign routes
it only into the frozen base's time embedding; the hypernetwork sees it only
indirectly via base activations). (2) The MetaWorld dataset adapter emits a
**constant** stride (default `1`) — no per-sample variation. (3) A precedence
bug feeds the *slice stride* into the base's fps channel instead of the
nominal fps.

## Why it matters
For the action-conditioned world model (D2), `fs` is currently neither a
learnable conditioning axis nor varied across training — so it carries no
information, and the base is fed an off-anchor, semantically-wrong constant.
This is a clean thing to fix before we trust any `fs`-related behaviour.

## Evidence / sources
- Mechanism + code citations: [[../../30_Knowledge/tech/frame-stride-conditioning]]
  (e.g. `hyperalign.py:676-682` + docstring 630-636; `batch_preprocessor.py:257`;
  base config `fs_condition: true`, `default_fs: 10`, `fps_condition_type: "fps"`).
- _No metrics — this is a code audit, not a run._

## Next
- Open decision worked through: [[../../50_Decisions/open/per-sample-frame-stride-sampling]].
  Resolved so far — `fs` is **augmentation/nuisance** (base-only, not `Δ_φ`);
  actions aggregate by **sum** under stride > 1; stride range is a swept
  hyperparameter. **Gate:** verify the pretrained DynamiCrafter fps-vs-fs
  convention before implementing (determines `5/k` vs raw `k`).
- The precedence bug is fixable independently of the sampler.
