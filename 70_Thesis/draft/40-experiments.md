---
section: experiments
status: drafting
deliverable: D2/D3
last_updated: 2026-05-28
sources:
  - "[[../../50_Decisions/decided/param-matched-adapter-comparison-definition]]"
  - "[[../../50_Decisions/decided/per-sample-frame-stride-sampling]]"
  - "[[../../30_Knowledge/tech/frame-stride-conditioning]]"
---

# 4. Experiments

> Setup and protocol only. Observed numbers live in ch. 5 and must cite
> runs. Planned runs are tickets, not results (Part 12).

## 4.1 Datasets and preprocessing

_Stub — MetaWorld; clip windowing, action handling, frame-stride. Sources:
`30_Knowledge/datasets/`,
[[../../50_Decisions/decided/per-sample-frame-stride-sampling]]._

### 4.1.x MetaWorld clip windowing and the `fs` anchor (drafting)

We train on a vendored MetaWorld HDF5 collection
(`<task>/<episode>/{pixels, action, ...}` with `pixels ∈ [T, H, W, 3]`
uint8 and `action ∈ [T, A]` float32). Each item is a sliding window of
`traj_len` consecutive frames sampled at a contiguous index range
`[start, start + traj_len)`; only the clip **start index** is
randomised per sample. We do **not** subsample frames within a clip
(no slice stride), and we do not vary the frame-stride / fps
conditioning per sample.

The frozen DynamiCrafter UNet exposes an `fs` scalar conditioning
channel, embedded via its own `fps_embedding` MLP and added to the
time embedding. Its pretrained anchor is `default_fs = 10` (config
`external_repos/avid/.../dynamicrafter_512.yaml`). We feed
`fs = fps = frame_stride = 10` unconditionally — matching the
convention used by the AVID team's MetaWorld data module, which writes
the same constants and reads frames contiguously. This places the
base's fps channel **at its trained anchor** by construction and
sidesteps the `fps_condition_type` (`"fs"` vs `"fps"`) semantic
ambiguity: under either interpretation, `10` is the value the base
was tuned to operate at.

This is a deliberate design choice rather than a fallback. We
considered (i) per-sample stride sampling with summed action
aggregation, and (ii) per-sample stride with a principled
in-distribution mapping `fs(k) = (source_fps_DC / fps_MetaWorld) · k`,
and rejected both for the baseline because D2's contribution surface
is *action-conditioned dynamics*, not *temporal-resolution
robustness*. The temporal variation that the random clip start index
provides is sufficient at this stage. Revisit triggers (e.g. observed
brittleness to inference-time fps changes; evidence that a constant
`fs` masks an identifiable `Δ_φ` failure mode) and the full reasoning
are recorded in
[[../../50_Decisions/decided/per-sample-frame-stride-sampling]].

A direct consequence of this choice is that the action interface
remains the simple per-step convention: each predicted transition
consumes the single action that produced it, so the `(B, T, A)`
action tensor consumed by the action-conditioning path of `Δ_φ` is
shape-stable across the dataset. No stride-induced action dropping or
aggregation is needed for the baseline.

The complementary design question — whether `Δ_φ` should ever
explicitly consume `fs` — is treated as a framework-level boundary in
§3.3 and is held outside the composition rule for the baseline.

## 4.2 Protocol: parameter / FLOPs-matched comparison
_Stub. Source:
[[../../50_Decisions/decided/param-matched-adapter-comparison-definition]]._

## 4.3 Metrics and baselines
_Stub — prediction accuracy, stability, inference cost; action-following._

## 4.4 Ablation design (the axes)

_Full design + rationale + run plan:
[[../../30_Knowledge/writing/ablation-axes]]._

The study is organised as a **dataset axis run in full** (the
action-informative-vs-redundant contrast — MetaWorld anchor + three ACWM-Phys
envs: Push Cube da=2, Robot Arm da=7, Reacher da=2) crossed with a **search
toolbox** of interventions pulled adaptively to find a configuration that
escapes base-parity: gate-cap, σ-shift, and an AVID-style pure-adapter warmup
(`pretrain_steps`), plus single mechanism-probe cells (base-output
conditioning off; AdaLN vs cross-attention). The adapter-family axis is held
to the output adapter empirically; the hidden-state and hypernetwork families
are treated as a computational-complexity discussion (D1 as a software +
cost-analysis contribution). Every cell reports the same five metrics
(action-sensitivity, prediction accuracy, generation quality, base-parity
cosine, cost), so the matrix is a result whether or not action-following
emerges (§5.1.x is the diagnostic anchor).
