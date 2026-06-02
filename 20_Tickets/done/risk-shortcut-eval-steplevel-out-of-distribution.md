---
type: risk
scope: shortcut
status: done
priority: high
created: 2026-05-25
updated: 2026-06-02
resolution: shipped
resolution_note: |
  B+C code fix (normalized `shortcut_step_schedule` + `log2`
  step_level_transform + schedule-driven eval grid) landed 2026-05-25
  and is verified in the tree; the AVID shortcut config trains the full
  paper-faithful log2 1/128…1 ladder. User-confirmed resolved on
  2026-06-02. NOTE: the newer `diffusion_unicon_shortcut_metaworld.yaml`
  and `diffusion_hyperalign_shortcut_metaworld.yaml` configs still use
  the legacy `shortcut_step_level_max: 4` raw-timestep path and were NOT
  migrated to `shortcut_step_schedule` — they reintroduce the same OOD
  mismatch. This is a per-config migration, not a code regression.
closed_at: 2026-06-02
related:
  - risk-shortcut-self-consistency-collapse.md
  - bug-training-shortcut-target-timestep.md
  - feat-training-separate-loss-logging-and-multistep-eval-grid.md
  - ../30_Knowledge/tech/shortcut-training-modes.md
  - ../30_Knowledge/related-work/shortcut-models.md
---

# Trained `step_level` range is far smaller than the step sizes few-step sampling needs

## Context

Surfaced while implementing the multi-step-size eval grid
([[feat-training-separate-loss-logging-and-multistep-eval-grid.md]]). To
sample a shortcut model at `N` steps you must tell the adapter which
multi-step horizon it is approximating, via `cond["step_level"]`. That
forced the question: **what `step_level` corresponds to an `N`-step
rollout, and was the model ever trained on it?**

## The unit mismatch

`step_level` is in **raw training timesteps**. In
`trainer._compute_self_consistency_target_v` the chained micro-step is
`prev_t = t - d` and the supervised value is `step_level = 2d` — so a
model conditioned on `step_level = k` is trained to jump `k` timesteps
out of `diffusion_timesteps` (`= 1000` in
`configs/diffusion_avid_shortcut_metaworld.yaml`).

**Trained range** (current config, `shortcut_step_level_max: 4`):
`_sample_dyadic_d` draws `d ∈ {1, 2}` (since `max_d = 4 // 2 = 2`), giving
supervised `step_level = 2d ∈ {2, 4}`, plus the anchor `step_level = 1`.
⇒ the adapter only ever sees `step_level ∈ {1, 2, 4}`.

**Inference range**: a uniform `N`-step DDIM rollout jumps `1000 / N`
timesteps per step (the scheduler's `prev_timestep = timestep -
num_train // num_inference`). So the physically-consistent `step_level`:

| N steps | step_level = 1000/N | trained? |
|--------:|--------------------:|:--------:|
| 1       | 1000                | no (250×)|
| 2       | 500                 | no (125×)|
| 4       | 250                 | no (62×) |
| 8       | 125                 | no (31×) |
| 25      | 40                  | no (10×) |

Even the 25-step "reference" rollout needs `step_level = 40`, **10× the
trained max of 4**. The shortcut conditioning is therefore essentially
never exercised in its trained regime during any realistic few-step
sampling — the eval grid will be sampling the adapter fully
out-of-distribution on the `d` axis.

## Grounding against the base and the paper (verified 2026-05-25)

- **Base sampling steps:** DynamiCrafter's full process is `timesteps:
  1000` (`configs/base/dynamicrafter_512.yaml:19`); default sampler is
  `ddim_steps: 50` (line 169), and our experiment configs use
  `inference_num_steps: 25`. So the base already samples fine in **25–50
  steps** — i.e. `step_level ≈ 20–40`.
- **Shortcut paper** (`docs/paper/shortcut_models.pdf`, lines 254–256, 401;
  see [[../30_Knowledge/related-work/shortcut-models]]): finest unit
  `M = 128`, giving 8 step sizes `d ∈ {1/128, …, 1/2, 1}` normalised to
  `[0,1]`; **max step size `d = 1` = one full-trajectory step (1-step
  generation)**; evaluated at **128 / 4 / 1 steps**. The flow-matching
  loss grounds the model at the *smallest* `d = 1/128`.
- **Mapping paper `d` → our `step_level`** (`step_level = d · 1000`): the
  paper's range is `step_level ≈ 8` (finest, 128 steps) up to `1000`
  (`d=1`, 1 step). **Our current max `step_level = 4` is BELOW the paper's
  finest step (≈8)** — we train on jumps of 1–4 timesteps, smaller than
  the smallest jump the paper ever uses.

### The sharper statement (this is the real issue)

`step_level = 4` ⟹ 250 steps — *correct arithmetic*, but 250 steps is
**finer than the base's own 25–50-step default**, so that operating point
buys no speedup. The regime where a shortcut is actually valuable (the
paper's 1/4/8-step regime) needs `step_level ∈ {125, 250, 500, 1000}` — all
untrained. So `max=4` is a reasonable *"does the self-consistency plumbing
run without collapsing"* setting, but **cannot demonstrate any few-step
benefit**, because it never touches the step sizes few-step sampling uses.
Poor few-step rows in the eval grid would be a training-coverage artifact,
not evidence about whether shortcuts work.

## Why this matters

The whole premise of the shortcut model is that conditioning on a larger
`d` makes the adapter predict the *averaged* direction over that horizon,
so few-step sampling stays accurate. If training never covers the `d`
values few-step sampling uses, that mechanism is untested and probably
non-functional: the few-step rows of the eval grid will likely look bad,
and not because the idea is wrong but because of this train/eval `d`
mismatch. We need to fix the regime before drawing any conclusion about
"does shortcut modeling help".

## Embedding asymmetry makes this worse than data coverage (verified 2026-05-25)

It is **not** just that few-step `step_level` values are unseen in training
— the conditioning path is *scaled wrong* for them:

- **Base `t`**: fed **unnormalized** (raw 0–999) into a **sinusoidal**
  `timestep_embedding` (`max_period=10000`) —
  `backbones/dynamicrafter/models/utils_diffusion.py:23`,
  `…/openaimodel3d.py:741`, wrapper passes `t` straight through
  (`models/base/dynamicrafter.py:152`, no rescale). Bounded + periodic →
  the base handles the whole 0–1000 range fine.
- **Adapter `step_level`**: same raw scale, but embedded by a plain
  **`Linear(1, hidden)`** fed the raw scalar
  (`adapters/output/dynamicrafter.py:69`,
  `conditioning/utils/dynamicrafter_conditioning.py:47`) — *linear,
  unbounded, not normalized*.

Trained on `step_level ∈ {1,2,4}` the linear weight is tuned for inputs
≈1–4; feeding 250–1000 gives a pre-activation ~60–250× larger than
anything seen → SiLU saturates, downstream embedding is garbage. So even
with more data at large `d` (Option C), a raw `Linear(1,·)` is a badly
conditioned way to encode a 1→1000 horizon. ⇒ strong argument for **B**,
and specifically for giving `step_level` the *same sinusoidal treatment as
`t`* (normalized `d/T` into a sinusoidal embed), not just rescaling the cap.

## Options (needs a decision)

- **A — Raise `shortcut_step_level_max` to cover the inference horizons.**
  To reach 1-step you need `step_level` up to ~1000, i.e. a dyadic ladder
  `{1, 2, 4, …, 512}` and self-consistency chaining a ~500-timestep DDIM
  micro-step. Simple knob, but large-`d` self-consistency targets are
  exactly where the collapse modes in
  [[risk-shortcut-self-consistency-collapse.md]] are most dangerous, and a
  single 500-step micro-step is a very coarse teacher.
- **B — Change `step_level` units/semantics** to something scale-shared
  between train and eval — e.g. normalized `d/T ∈ (0,1]`, or "number of
  inference steps this one step replaces". Removes the magnitude gap but
  touches the embedding input range, the target computation, and every
  shortcut config. Most principled; most invasive.
- **C — Train `d` across the full log/dyadic range.** `_sample_dyadic_d`
  is *already* logarithmic; raising `shortcut_step_level_max` (e.g. to
  1024) fills the ladder to `step_level ∈ {1,2,4,…,1024}` automatically —
  matching the paper's dyadic `d` and giving the bootstrapping rungs the
  current `{1,2,4}` lacks (the `2d` target chains two `d`-steps, so each
  rung needs the one below). Caveat: `step_level ≤ T = 1000`, and since
  1000 isn't a power of two the raw-timestep `2d` ladder can't land exactly
  on 1-step — either cap at 512 (~2 steps min) or move to normalized
  dyadic `d` (→ folds into B). Coverage-only; does **not** fix the
  `Linear(1,·)` scaling on its own — must be paired with a log/normalized
  embedding of `step_level`.

**Lean (updated after the embedding finding):** the `Linear(1,·)`
asymmetry above means **C alone is not enough** — adding large-`d` data
still feeds a badly scaled raw scalar into the adapter. Favour **B as the
foundation** (normalize `step_level → d/T` and embed it sinusoidally like
`t`, a small contained change to the step_level branch), then **C on top**
(train the horizons we actually sample). A is the blunt fallback and most
exposes the collapse modes. This is a research-design call, not a unilateral
code change — flagging for decision.

## Mitigation implemented 2026-05-25 (B + C)

Decision taken: go paper-faithful. Shipped in
[[feat-shortcut-configurable-paper-faithful-step-schedule.md]]:

- **B (units + embedding):** step sizes canonicalised to normalised
  `s ∈ (0,1]` via a configurable `training.extra.shortcut_step_schedule`;
  the injected `step_level` is now bounded, and a `step_level_transform:
  log2` on the adapter spreads a dyadic ladder into ~`[-7,0]` instead of
  feeding a raw 1→1000 scalar into `Linear(1,·)`. The base's `t` stays
  unnormalised/sinusoidal (unchanged); only the *adapter's* step_level
  branch was re-conditioned.
- **C (coverage):** the same schedule drives training sampling and the eval
  grid, so they share one source of truth. AVID config now trains the
  paper's full `log2 1/128…1` ladder instead of `{1,2,4}`.

Sinusoidal step_level embedding (matching `t` exactly) was scoped out —
`log2` already bounds + spreads the range; revisit only if `log2` underperforms.

## Remaining (empirical, not code)

- Run training with the full schedule and inspect the multi-step eval grid:
  does the adapter degrade gracefully at few steps vs. the base? This is the
  only open item; the OOD/units/embedding gaps are closed in code.
- Watch the collapse modes ([[risk-shortcut-self-consistency-collapse.md]])
  at large `s`, where the self-consistency teacher (two coarse `s/2` steps)
  is weakest.

## Related

- [[feat-training-separate-loss-logging-and-multistep-eval-grid.md]] — the
  feature that exposed this; the eval grid is the diagnostic.
- [[risk-shortcut-self-consistency-collapse.md]] — large-`d`
  self-consistency targets (Option A) intersect the collapse taxonomy.
- [[../30_Knowledge/tech/shortcut-training-modes.md]] — `distillation` vs
  `two_step`; the `d` sampling lives here.
- Code anchors: `training/trainer.py` `_sample_dyadic_d`,
  `_compute_self_consistency_target_v`, `_generate_step_size_grid`;
  `configs/diffusion_avid_shortcut_metaworld.yaml`
  (`shortcut_step_level_max`, `eval_step_schedule`).
