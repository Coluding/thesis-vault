---
type: decision
status: decided
created: 2026-05-25
decided_at: 2026-05-28
updated: 2026-05-28
target_date:
scope: data
related:
  - "[[../../30_Knowledge/tech/frame-stride-conditioning]]"
  - "[[../../30_Knowledge/tech/structural-encoder]]"
  - "[[../../30_Knowledge/tech/shortcut-training-modes]]"
  - "[[../../20_Tickets/bug-data-metaworld-fs-anchor-default]]"
---

# Decision: MetaWorld frame-stride conditioning — anchor at AVID convention, do not vary per sample

## Status

**Decided 2026-05-28 — Option A (no per-sample sampling; anchor at
`fs=fps=10`).** The earlier in-progress resolutions (SUM action
aggregation, per-sample uniform draw, mapped fs) are **shelved, not
adopted** — see "Superseded resolutions" below.

> **SUPERSEDED 2026-06-04** by [[metaworld-frame-stride-load-time]]. Its revisit
> trigger fired: 16-frame *contiguous* clips cover only ~5% of the 300-frame
> episodes, so no action effect is visible in logging. The successor keeps `fs`
> base-only (this decision's surviving part) but **reads at a fixed stride k
> (default 4)** for a longer window, **adopts the shelved SUM action-
> aggregation**, and feeds a **constant `fs=1`** (not the anchor 10). The
> contiguous-reads choice below is no longer in force — read it for the design
> space, not the current behaviour.

## Decision

The MetaWorld dataset adapter writes `fps=10` and `frame_stride=10` into
every batch, regardless of the sliding-window slice or any user-set
stride flag. Frames are read contiguously (no slice-stride
sub-sampling). No per-sample variation of `fs`.

This matches the AVID team's MetaWorld data module
(`external_repos/avid/latent_diffusion/src/ldwma/lightning/data_modules/metaworld.py:107-108`),
which writes `fps=10, frame_stride=10` and reads pixels contiguously at
`pixels[start:start+traj_len]`. The number `10` is the base UNet's
`default_fs` from
`external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml:68`,
i.e. the frozen DynamiCrafter checkpoint's pretrained anchor for its
fps / fs conditioning channel.

## Why

The whole reason this was open was: what value should the base's fps
channel see, given the channel's semantics (`fs` vs `fps`) and the
pretrained range are both `_needs verification_`? Three honest options
existed (see the open-state body below for the full design space):

- (A) Don't vary anything. Anchor at the base's default.
- (B) Subsample for real, feed the anchor anyway (lie to the base).
- (C) Subsample for real, map `fs(k)` to a principled in-distribution
  value (requires verifying the base's training convention).

(A) is the only option that puts the base **in distribution by
construction** without needing the verification step, and without
introducing the action-dropping trap that gates (B)/(C). It matches
AVID's working pipeline exactly. The data-augmentation motivation that
opened this decision is real but not load-bearing for D2's contribution
(action-conditioned dynamics, not temporal-resolution robustness), and
the random clip *start index* already in `dataset.py:95` provides some
temporal variation for free.

## Consequences

- The MetaWorld translator's existing `frame_stride` parameter becomes a
  dead config knob — its only effect is to set the slice stride, which
  this decision says should always be `1`. Either remove it from the
  translator API or hard-pin its default to `1` and document that the
  recorded `frame_stride` value (10) is independent of the actual slice.
- The fps/frame\_stride precedence bug in
  `batch_preprocessor.py:257` (`_extract_fs` reads `frame_stride` before
  `fps`) is now a non-issue: both keys carry the same number (`10`), so
  precedence stops mattering. **Do not** open a separate bug ticket for
  the precedence — close any lingering one as obsolete.
- The action-aggregation discussion (SUM over `k` deltas), the
  aggregated-action OOD magnitude risk, and the `max_stride`
  hyperparameter all become moot for the baseline. They are recorded
  here in case (A) is revisited, but **must not be implemented
  speculatively**.
- No change to `Δ_φ` conditioning: `fs` continues to reach the frozen
  base only, never the adapter, consistent with the tech note.
- D2 baseline experiments train on a single, fixed temporal scale. If a
  later ablation shows the resulting model is brittle to inference-time
  fps changes, revisit by opening a fresh decision pointing here.

## Shelved sub-resolutions (do not implement)

These were reached during the 2026-05-25 live grilling under the
assumption that we were going to ship some form of (B) or (C). Option
(A) makes them irrelevant for the current baseline. Kept here so a
future revisit doesn't re-derive them from scratch:

- *Action handling under stride > 1 = SUM-aggregate per kept step.*
  Rationale was that all four MetaWorld action dims `(Δx, Δy, Δz,
  Δgripper)` are deltas, so summing the `k` intermediate actions gives
  the total displacement across the step. This is the right answer if
  (A) is ever reopened; for now, no code path needs it.
- *`fs` is base-only* (not into `Δ_φ`). This part survives — it's
  consistent with (A) too, since the adapter never had `fs` in its
  inputs under any of the options.
- *`fs` ⊥ shortcut `d`.* Also survives — orthogonality is by
  construction under (A) (one fixed `fs`, free `d`).
- *Range mechanism = per-sample uniform `{1..max_stride}` with
  `max_stride` as a swept hyperparameter.* Shelved; not used.
- *Base fps-channel value = mapped `fs(k)` rather than raw stride.*
  Shelved; the anchor `10` is fed unconditionally.

## Derived tickets

- [[../../20_Tickets/bug-data-metaworld-fs-anchor-default]] — change
  the MetaWorld translator defaults to `fps=10`, `frame_stride=10` and
  pin the slice stride at `1`; align with the AVID convention. Closes
  any latent precedence-bug concern as a side effect.

## Future-revisit trigger

Reopen if any of the following observed:

- A D2 model that generalises poorly across inference-time fps changes
  in a way the thesis story needs to address.
- Evidence that constant-`fs` data is masking an identifiable failure
  mode in `Δ_φ` (e.g. the adapter under-uses the base's motion prior).
- A move off DynamiCrafter to a base whose fps-channel semantics are
  fully known, in which case (C) becomes cheap.

## Context (original open-state body, preserved for reference)

The MetaWorld dataset adapter currently uses a **single fixed**
`frame_stride` (default `1`) for the entire dataset
(`data/dataset.py:35,50` → `load_clip(..., stride=self.frame_stride)` at
`dataset.py:101`). Only the clip **start index** is randomised per sample
(`dataset.py:95`). Consequences established in the tech note:

- `cond["fs"]` is **constant** across the dataset, so it carries no
  information — conditioning anything on it is unidentifiable.
- A precedence bug feeds the *slice stride* (1) into the base's
  fps channel instead of the nominal `fps` (5) (`batch_preprocessor.py:257`).
  Under (A) this bug evaporates (both values become `10`).

The original proposal was to sample `frame_stride` per sample
(e.g. uniformly from `{1, 2, 3, 4}`) so `fs` becomes a real, varied
axis.

### Why per-sample sampling was non-trivial

The strided slice is applied to **both** the pixels and the actions:

```python
sl = slice(start, start + span, stride)   # translators/metaworld.py:131
"video": _read(ep_group, "pixels", sl),   # line 134
"act":   _read(ep_group, "action", sl),    # line 135
```

`stride=k` keeps `a₀, aₖ, a₂ₖ…` and **drops the intermediate actions**.
For an action-conditioned world model (D2) this breaks the core premise:
the transition `0 → k` was caused by *all* of `a₀…a_{k-1}`. The
SUM-aggregation sub-resolution above was the intended fix; (A) sidesteps
it by not subsampling at all.

### The in-distribution argument

DynamiCrafter's `fs` channel encodes `seconds_per_step = stride /
source_fps`. AVID feeds the same number (`10`) as both `fps` and
`frame_stride`, anchoring the base at its trained `default_fs=10` and
making the `fps_condition_type` question (`"fps"` vs `"fs"`) moot — the
base sees its anchor either way. (A) inherits this.

## Related

- [[../../30_Knowledge/tech/frame-stride-conditioning]] — the audit
  that surfaced this; its "Open question" section now resolves here.
- [[../../30_Knowledge/tech/structural-encoder]] — `fs` would land here
  only under a future (C) revisit.
- Code: `src/generative_flow_adapters/data/translators/metaworld.py:15,131,134-138`
- Code: `src/generative_flow_adapters/data/dataset.py:35,50,95,101`
- Code: `src/generative_flow_adapters/data/batch_preprocessor.py:256-266`
- Reference: `external_repos/avid/latent_diffusion/src/ldwma/lightning/data_modules/metaworld.py:107-108`
- Reference: `external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml:35,68,69`
