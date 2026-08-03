# DC structure-probe port — steering / temporal / spatial on DynamiCrafter

2026-08-01/02, overnight. Implementation only — **the Slurm job was written and
made executable but NOT submitted**, as instructed. Nothing here contains a
measured result; every number below is either a code constant, an analytic
chance level, or the output of a CPU unit test on stub models.

---

## 1. What was implemented and where

**Chosen script: `scripts/eval_action_sensitivity.py`** (the backbone-generic
one), not `generate_dynamicrafter_compare.py`. Reasons, in order of weight:

1. `eval_action_sensitivity.py` already builds everything the probes need for
   *both* families: the composed `AdaptedModel`, a `Trainer` (which owns the
   backbone's `x_t`/`t` construction), and preprocessed batches, dispatched on
   `config.model.provider` (`scripts/eval_action_sensitivity.py:90-105`,
   `:305-310`). `generate_dynamicrafter_compare.py` builds none of that — it has
   no `Trainer`, no preprocessor, and loads a single clip by `--clip-index`
   (`scripts/generate_dynamicrafter_compare.py:159-171`), so the batch-axis donor
   swap the probes are built on would have had to be invented from scratch.
2. It targets the **native** provider check `dynamicrafter_video` only
   (`generate_dynamicrafter_compare.py:112-116`), whereas
   `eval_action_sensitivity.py`'s `_family()` accepts any `dynamicrafter*`
   provider *and* Wan/SkyReels — so one implementation covers the DC-vs-Wan
   head-to-head.
3. Arm E / arm 0 configs use `provider: dynamicrafter_video`
   (`configs/dynamicrafter/diffusion_dc_acwm_robotarm_armE_center.yaml:66`), which
   `_family()` maps to the `dynamicrafter` stack — no new stack builder needed.

**New measurement core:
`src/generative_flow_adapters/evaluation/action_structure.py`** (709 lines,
backbone-agnostic, mirrors how `action_sensitivity.py` is factored):

| Symbol | Line | Role |
|---|---|---|
| `NULL_TOLERANCE = 1e-6` | `:92` | frozen-base leak threshold |
| `_targets_at_fixed_xt` | `:203` | probe A's target difference, diffusion **and** flow |
| `run_structure_probes` | `:285` | driver over batches × draws |
| `_accumulate` | `:371` | the three probes + all nulls, per (batch, draw) |
| `_alignment_score` | `:479` | probe B score + analytic chance (Wan's formula) |
| `_alignment_permutation_chance` | `:502` | probe B empirical chance |
| `_spatial_concentration` | `:515` | probe C score + realised mask fraction |
| `_spatial_permutation_chance` | `:532` | probe C empirical chance |
| `summarise` | `:549` | JSON summary, one chance level per probe |
| `format_structure_report` | `:632` | human-readable report |

Exported from `src/generative_flow_adapters/evaluation/__init__.py:17-23`.

**CLI wiring in `scripts/eval_action_sensitivity.py`:**

- `--action-analysis` (+ `--action-analysis-batches/-draws/-bins`): `:376-393`
- `_structure()` helper (incl. the batch-size ≥ 2 guard): `:714-731`
- runs inside the existing `--timesteps` stratification loop: `:767-768`
- aggregate run: `:773`
- stratified table + report printing: `:787-803`
- JSON keys `structure` / `structure_by_timestep`: `:831-836`
- non-zero exit (3) on a base-null violation: `:848-852`

**Behaviour without the flag is unchanged**: `structure` is `None`
(`:773`), no structure keys are added to the JSON (`:831` guarded), nothing new
prints, and the new exit path iterates an empty list.

**Slurm job:
`jobs/experiments_cluster/acwm_phys/dc/submit_probe_dc_arm_structure.sh`**
(305 lines, `chmod +x` applied, **not submitted**).

---

## 2. Every deviation from the Wan implementation

Reference: `scripts/generate_wan22_i2v_compare.py::_action_analysis`, lines
146-268.

### 2a. Deviations that DO NOT affect comparability

| # | Wan | Here | Why |
|---|---|---|---|
| 1 | action keys hardcoded `("action", "action_seq")` (`:143`, `:202-206`) | resolved from the caller's `--action-keys` via `_assert_actions_present` | DC emits the action under `act` (`configs/…armE_center.yaml:145`). Same operation, different key name. |
| 2 | `sigma` grid built by hand: `x_t = (1-fm5)*x0 + fm5*((1-σ)x0 + σ·noise)` (`:198`) | `x_t`/`t` come from `Trainer._forward_and_loss` (`trainer.py:332-342` diffusion, `:378-405` flow) | DC's `x_t` needs `scale_x_start` + `q_sample` against the beta schedule; hand-rolling it would have been a second, divergent copy. See 2b-(iii) for the consequence. |
| 3 | `frame_mask` always present | `_frame_mask_bool` returns `None` when absent (`:136-148`) | DC's preprocessor emits no `frame_mask` (`data/batch_preprocessor.py:153` returns only `{"target", "cond"}`), i.e. the whole clip is predicted. Wan's masked behaviour is byte-for-byte preserved when the mask exists. |
| 4 | donor = `torch.roll(v, 1, dims=0)` | identical (`action_structure.py:172-184`) | — |
| 5 | probe B binning `bins = min(8, t_lat)`, `px_per_bin = max(1, L//bins)` (`:221-222`) | identical, with `8` exposed as `--action-analysis-bins` | For DC: `L = t_lat = 16` → bins 8, per_bin 2, tolerance 2, chance 0.3125. |
| 6 | probe C `mean(dim=(1,2))`, motion from `x0`, `quantile(0.90)` (`:236-260`) | identical; motion read from `batch["x0"]` (flow) or `batch["target"]` (diffusion) | Both are the raw clean latent. |
| 7 | prints `10% = unconcentrated` | prints the **realised** mask fraction | `quantile(0.90)` ties can make the region larger than 10%; on the CPU test it came out 0.1111 and the random-region estimate agreed (0.1117). Strictly more honest, same statistic. |

### 2b. Deviations that DO change a number — read carefully

**(i) Probe A's target. This is the forced deviation, and it is the whole point.**

Wan computes `dv = roll(v,1,0) - v` where `v = noise - x0` is *each clip's own*
training target (`generate_wan22_i2v_compare.py:210-211`). Expanded:

```
dv_literal = (x0_A - x0_B)        <- the steering signal
           + (noise_B - noise_A)  <- independent noise, unrelated to the action
```

The second term is a per-clip independent draw. For **flow/velocity** it merely
*dilutes* the estimator (the signal survives with the right sign). For a
**noise-predicting diffusion model** the signal coefficient is exactly zero —
`dv_literal = ε_B − ε_A` is pure noise — so a verbatim port would have produced
a metric that is at chance *by construction*, for any model. That is the "wrong
probe would send the thesis in the wrong direction" failure the task warns about.

So the primary `steer_cos` here uses the **fixed-`x_t`** target difference: what
the regression target *would be* at this same `x_t`/`t` if the clean latent were
the donor's. Implemented by substitution, not by a formula (`action_structure.py:203-279`).

Both numbers are reported:

- `steer_cos` — primary, de-contaminated, defined for every parameterisation.
- `steer_cos_literal_wan_formula` — Wan's exact formula, computed through the
  model's own objective, so it bridges to the already-published Wan value.
- `literal_target_alignment` = `cos(dv_literal, dtarget_fixed)` — quantifies the
  dilution. For a perfectly-steering model, `steer_cos_literal` collapses to
  exactly this number (asserted in
  `tests/test_action_structure.py::test_wan_formula_variant_is_explained_by_its_target_alignment`).

**Comparability verdict:** `steer_cos_literal_wan_formula` is directly
comparable to the published Wan `steer_cos ≈ 0.00`. The primary `steer_cos` is a
*different, better* estimator — **to put it next to Wan, the Wan cell must be
re-run through `eval_action_sensitivity.py --action-analysis`, which is now
possible (the module is backbone-generic and the script already handles Wan).
Until that re-run exists, do not print a DC `steer_cos` next to the Wan 0.00 in
a table without a footnote.** This is the single most important caveat in this
note.

**(ii) Probe A gets two extra nulls, and the obvious one is wrong.**

My first implementation used the natural mismatched pairing — clip `i`'s `dpred`
against clip `i+1`'s steering direction — and the CPU harness returned **−0.43**
where chance should be 0. Cause: `dtarget_i` involves clips `(i, i−1)` and
`dtarget_{i+1}` involves `(i+1, i)`, so they share a term and correlate at ≈ −0.5.
Shipped instead:

- `steer_cos_random_null` — `dpred` vs an isotropic direction. Always available.
- `steer_cos_disjoint_null` — pairing skips to `i+2` so the two clip pairs are
  disjoint; needs batch ≥ 4 (the job runs batch 4).

Wan reports no empirical null at all, so this is an addition, not a change.

**(iii) Probe A is measured at the trainer's `t`, not at Wan's `--trace-sigmas`.**

Wan's published structure numbers were taken at σ ∈ {0.5, 0.83}
(`submit_probe_wan_action_trace.sh`). The DC job instead stratifies over AVID's
diffusion grid `t ∈ {100,300,500,700,900}` using the script's existing
`--timesteps` mechanism (`eval_action_sensitivity.py:733-770`), plus an
aggregate at the trainer's own sampler. **The noise levels are therefore not
matched between the DC and Wan runs.** Mapping σ to `t` across a rectified-flow
and a beta schedule is not a well-defined identity, and I did not want to invent
one. Flagged for human review (§7).

**(iv) One extra forward per draw.** `_forward_and_loss` is called for its
`x_t`/`t`/`cond` and its prediction is discarded, because the probe needs the
frozen base alongside the composed output and only a `return_base=True` forward
provides it. Pure cost, no effect on any number.

**(v) Degenerate results are refused rather than scored.** A zero-response
adapter row-normalises to a matrix of zeros whose `argmax` is 0 for every bin;
Wan's code would print an alignment score of ~0.25 for that (I reproduced
exactly this on the CPU harness). Here `alignment_score` and `concentration` are
emitted as `null` with `degenerate: true` when the raw response mass is 0
(`action_structure.py:549-630`). A Wan run through this code path would show the
same guard; it changes no non-degenerate number.

---

## 3. How diffusion vs flow is handled

Keyed off `model.model_type` / `model.prediction_type`, which `AdaptedModel`
delegates to the base (`models/adapted_model.py:113-119`) — never hardcoded.
The branch is `action_structure.py:203-279`.

**Diffusion path** (`model_type == "diffusion"`), `:231-253`:

1. `x0 = objective.scale_x_start(batch["target"], t)` — the same scaled space the
   trainer regresses in (`trainer.py:341`).
2. Recover the noise by inverting `q_sample`:
   `eps = (x_t − √ᾱ·x0) / √(1−ᾱ)` (exact inverse of `losses/diffusion.py:91-96`).
3. `target_self = objective.get_target(prediction_type, x_start=x0, x_t, t, eps)`.
4. Substitute the donor's latent at the **same** `x_t`, recompute its implied
   noise, and call `get_target` again → `target_donor`.
5. `dtarget = target_donor − target_self`.

Because step 4 runs the donor's `x_start` back through
`DiffusionTrainingObjective.get_target` (`losses/diffusion.py:98-106` — the
codebase's source of truth), the correct direction falls out for every
parameterisation the objective supports, **including the sign flip**: for
`noise`/`velocity` the difference is a positive multiple of `x0_self − x0_donor`;
for `sample`/`x0` it is `x0_donor − x0_self`, the opposite direction. No sign
table exists in my code, so there is nothing to get wrong.

Relevant to this campaign: `dynamicrafter_video` declares
`model_type="diffusion", prediction_type="velocity"`
(`models/base/dynamicrafter_video.py:81`), and both arm configs set
`prediction_type: velocity` (`…armE_center.yaml:67`). Derivation for that case:
`v(x0'|x_t) = (√ᾱ·x_t − x0')/√(1−ᾱ)`, so
`dtarget = (x0_self − x0_donor)/√(1−ᾱ)` — a **positive** multiple, same sign as
the flow case. (`v(x0')` is stated here as algebra, not as a measurement.)

**Flow path** (`:255-279`): the target is the rectified-flow velocity
`noise − x0` (`data/wan22_batch_preprocessor.py:145`). `prediction_type` is
asserted to be one of `{None, "velocity", "v"}` and the probe **raises** on
anything else rather than reporting a number with an unverified sign (`:255-261`);
`None` is accepted because `models/base/interfaces.py:57-64` infers `"velocity"`
for flow. σ is recovered per clip by projecting `x_t − x0` onto the target
(exact, since `x_t = x0 + σ·target`), giving
`dtarget = (x0_self − x0_donor)/σ`, σ > 0.

Verified by unit test on all three diffusion parameterisations plus flow (§5).

---

## 4. Null control and chance level, per probe

| Probe | Statistic | **Chance** | Frozen-base null | Empirical chance estimate |
|---|---|---|---|---|
| A steering | `steer_cos` | **0.0**, band ±2·(1/√D)/√n — both printed | `steer_cos_base_null`; hard gate `base_null_pred_rel_max ≤ 1e-6` | `steer_cos_random_null` (isotropic direction), `steer_cos_disjoint_null` (mismatched disjoint clip pair, needs batch ≥ 4) |
| B temporal | `alignment_score` | **`(2·tol+1)/T_latent`**, Wan's formula (`generate_wan22_i2v_compare.py:255`). For DC: `tol=2`, `T=16` → **0.3125** | response mass identically 0 → reported `base_null_degenerate: true`, not scored | `chance_permuted` — rows shuffled 400×, rescored |
| C spatial | `concentration` | **realised mask fraction** (≈0.10; 0.1111 on the CPU harness) | effect mass identically 0 → `base_null_degenerate: true` | `chance_permuted` — random same-size region, 400× |

The frozen-base null is run for **all three** probes: every forward is done with
`model(x_t, t, cond, return_base=True)` and the whole probe is recomputed on
`base_roll − base_true`. Because the base sees identical inputs across the action
swap, that delta is bit-identical zero, so all three statistics are at chance on
it **by construction** — which is what makes it a harness control (it detects
action leakage into the frozen base) rather than a spread estimate. That is why
each probe also carries the empirical chance estimate in the last column, printed
next to the analytic chance it is supposed to reproduce.

Failure handling: `base_null_pred_rel_max > 1e-6` ⇒ report prints
`HARNESS ERROR … VOID`, the script exits **3**, and the job script echoes what
that means. (Exit **2** remains the pre-existing base-*loss* null violation.)

---

## 5. Verification evidence

All on CPU, in `/home/lukas/projects/generative-flow-adapters`, `./.venv/bin/python`.

**(a) argparse interrogated directly** (not grepped — per the known false-positive
bug). Loaded the script by file path, monkeypatched `ArgumentParser.parse_args`,
and inspected `_actions`:

```
--action-analysis                present=True
--action-analysis-batches        present=True
--action-analysis-draws          present=True
--action-analysis-bins           present=True
dest: action_analysis | type: _StoreTrueAction | default: False
parsed action_analysis = True | batches 4 | draws 1 | bins 8
absent-by-default       = False
```

**(b) `python -m py_compile`** clean on `scripts/eval_action_sensitivity.py`,
`src/generative_flow_adapters/evaluation/action_structure.py`,
`.../evaluation/__init__.py`, `tests/test_action_structure.py`, and both python
heredocs extracted from the Slurm script. `bash -n` clean on the Slurm script.

**(c) CPU smoke tests — `tests/test_action_structure.py`, 18 tests, all pass**
(plus the 15 pre-existing `test_action_sensitivity.py` tests still pass: 33 total).
Method: build the clean latent *from* the action (`x0 = decode(act)`, a fixed
linear map) and give the stub model the exact target that implies, so
`pred(a_B) − pred(a_A)` is *identically* the fixed-`x_t` target difference and the
right answer is known in closed form.

- **Sign / parameterisation** (the decisive one): perfect steering scores
  **+1.0000** for `noise`, `velocity` **and** `sample`; the inverted model scores
  −1.0000 for all three; the flow oracle scores +1.0000.
- **Inversion guard**: a model that always moves toward the donor's *data*
  regardless of parameterisation reads **+1 on `noise` and −1 on `sample`** —
  i.e. the probe demonstrably keys off `prediction_type`.
- **Null control at chance**: the action-blind model gives
  `base_null_pred_rel_max == 0.0`, `steer_cos == chance` exactly,
  `steer_cos_base_null == 0.0`, and B/C reported `degenerate` with `None` scores
  (asserted, so a future regression that scores a zero matrix fails the suite).
- **Leaky base**: a stub whose *base* reads the action is reported
  `HARNESS ERROR … VOID`.
- **Chance levels validated against their own permutation estimates**:
  B analytic 0.375 vs permuted 0.354; C `chance == chance_permuted` within 0.05;
  A `random_null` inside the ±1/√D band.
- **Mutation test** (the tests have teeth): flipping `dtarget`'s sign in the
  implementation fails 8 of 18; restoring it passes all 18.

**(d) End-to-end harness rehearsal.** Ran the Slurm script's two embedded python
blocks standalone:
- the argparse guard exits **0** against the patched script and exits **1** with
  `missing flags: ['--action-analysis', …]` against `git show HEAD:` of the same
  file — so it will really stop a stale cluster copy;
- the side-by-side block was fed summaries generated by the CPU stubs (synthetic,
  clearly not checkpoints) and rendered correctly, including the missing-arm path
  (the failed arm's column reads `n/a` rather than repeating the other arm's
  number — a bug I found and fixed during this rehearsal) and the degenerate path
  (`n/a`, not a fabricated 0.25/0.0000).

**Not verified:** anything requiring a GPU, a real checkpoint, or the ACWM data
(no runs launched; none possible here).

---

## 6. Submit command and expected runtime

```bash
# on the cluster login node, from the repo root, AFTER rsyncing the repo
cd ~/generative-flow-adapters
mkdir -p logs/dc-avid-parity        # Slurm will NOT create --chdir's dir
sbatch jobs/experiments_cluster/acwm_phys/dc/submit_probe_dc_arm_structure.sh
```

No arguments; both arms run in the one job, sequentially, on one H100. Extra
flags are forwarded to both `eval_action_sensitivity.py` invocations via `"$@"`.

**Requested wall time: 4:00:00.** Estimate (analysed estimate, shown work — *not*
a measurement): per arm, 6 evaluation passes (5 stratified `t` + 1 aggregate) ×
[sensitivity: 8 batches × 3 draws × (1 ref + 2 variants) = 72 forwards] +
[structure: 4 batches × 1 draw × (2 swap + 8 bin) = 40 forwards] ≈ 670 composed
forwards per arm, ~1340 for both. The closest existing precedent is
`submit_probe_dc_parity_timesteps.sh`, which budgets 1:30:00 for one arm at
5 × 8 × 3 × 3 ≈ 360 forwards of the same model at the same geometry. Linear
scaling puts this near 2:45; 4:00 leaves margin for model build, VAE/CLIP load,
and the shortcut-target prep inside `_forward_and_loss`. **The per-forward cost
is inferred from that precedent, not measured — treat 4 h as a budget, not a
prediction.**

Outputs: `<run_dir>/eval_structure/action_sensitivity.json` per arm (keys
`structure`, `structure_by_timestep`) plus the side-by-side table in the job log.

---

## 7. Needs human review before submission

1. **σ ↔ `t` mismatch with the published Wan run (§2b-iii).** Wan's structure
   numbers were taken at σ ∈ {0.5, 0.83}; this job stratifies over AVID's
   `t` grid. For a strict head-to-head, decide the mapping (or re-run Wan through
   `eval_action_sensitivity.py --action-analysis` and compare aggregates).
2. **The Wan cell has not been re-run through the new code path.** Until it is,
   `steer_cos` (primary) is DC-only; only `steer_cos_literal_wan_formula` is
   comparable to the published 0.00. Do not table them together unfootnoted.
3. **`#SBATCH --chdir=/home/lbierling/...`** is copied verbatim from
   `submit_probe_wan_action_trace.sh` and hardcodes the username; `logs/` must
   exist or Slurm rejects the job at submit time.
4. **Checkpoint directory resolution.** The vault frontmatter says
   `/scratch-shared/lbierling/outputs/…`, but `submit_train_dc_avid_parity.sh`
   passed `--output-dir outputs/<run>` (repo-relative). The job tries the
   scratch-shared path first and falls back to `outputs/`; confirm which one is
   real (or whether `outputs` is a symlink) before trusting `ls -t | head -1` to
   pick the intended step-~3500 checkpoint. I could not check either path.
5. **`ind_test` availability.** The job requires `$ROOT/ind_test/metadata.pt`.
   Every existing DC job in the repo reads `ind_train`; I could not verify that
   `ind_test` has been downloaded on the cluster.
6. **Probe B's expected-peak geometry for DC** (`L = t_latent = 16`, bins 8,
   tolerance 2, chance 0.3125) is derived from the configs, not observed. If DC's
   latent frame count differs from 16 at run time, the chance level shifts —
   it is recomputed per run and printed, so read it from the log rather than
   from this note.
7. **`--action-analysis` forces `--batch-size ≥ 2`** and the disjoint null needs
   ≥ 4. The job uses 4. If VRAM forces it lower, `steer_cos_disjoint_null` will
   read `nan` (by design) — that is not a failure, but the table will show it.
8. **`pass_cond_to_base: true` on both DC arms** (`…armE_center.yaml:69`). The
   frozen base therefore *receives* the `act` tensor. If the vendored lvdm UNet
   consumes it in any way, the base null fires and the job exits 3 with every
   number void. That is the intended behaviour, but it is the most likely way
   this job comes back empty — worth expecting.

---

## Related

- [[../30_Knowledge/experiments/20260731-dc-condition-center-accelerates-escape]] —
  the cell under test; `ckpt_path` frontmatter is where the run dirs came from
- [[../30_Knowledge/experiments/20260729-shortcut-wan-vs-dc-curvature-signature]]
