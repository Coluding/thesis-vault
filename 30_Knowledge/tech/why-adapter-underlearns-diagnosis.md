---
type: theory
last_updated: 2026-07-15
sources:
  - "[[../experiments/20260907-flow-shortcut-weak-action-signal]]"
  - "[[../experiments/20260712-wan-xattn-action-no-improvement]]"
  - "[[../../50_Decisions/open/action-conditioning-injection-mechanism]]"
  - "[[../../20_Tickets/feat-adapter-dynamicrafter-output-on-wan-base]]"
---

# Why the action adapter under-learns — diagnosis (2026-07-14)

## Method note (read this first)

This is the synthesis of a 21-agent investigate→verify→synthesize workflow (9
hypothesis lenses, each independently code-grounded then adversarially
verified) plus **my own follow-up spot-checks** of the two highest-stakes
claims, done directly against the repo (not delegated) because they're
consequential enough to warrant first-hand confirmation before acting on them.

**Caveat on the workflow itself:** 2 of the 10 initial investigator calls
(`gating-suppression`, `test-design`) degenerated into placeholder/stub output
(literally `"hypothesis": "test"`) — a tool-use glitch, not a finding. The
verifier for `gating-suppression` caught this and investigated independently
rather than rubber-stamping a stub; its finding is folded in below. The
`test-design` lens's substance (the do-now experiment order) survived via the
synthesis agent's own reasoning. Net effect: the ranked-causes table below is
sound, but treat "8 of 9 lenses confirmed something" as roughly true, not an
exact tally.

**My own verification, done directly (not via the workflow):**
- Confirmed by direct grep: `configs/diffusion_wan22_avid_xattn_i2v_metaworld.yaml`
  never sets `action_seq_len`; `WanBatchPreprocessConfig.action_seq_len` defaults
  to `None` (raw unbinned passthrough). This independently confirms the
  workflow's token-misalignment finding for the xattn run.
- **New finding, not from the workflow:** confirmed via wandb run metadata
  (`run.metadata.program`/`args`) that the "broken base, learned from scratch"
  control run (`wan22-avid-i2v-metaworld34/l3p8kygb`) used
  `scripts/train_wan22_i2v_metaworld.py` (plain), while **both** xattn runs
  (`cq3e83pj` proper-base, `xb76ptw2` no-improvement) used
  `scripts/train_wan22_i2v_metaworld_external.py`. The plain script loads the
  base via `provider: wan2.2` → the vendored, hand-copied `Wan22DiTWrapper`; the
  external script loads it via `provider: wan2.2_external` → the real upstream
  `wan.WanTI2V`. This is very likely *why* the "broken base" run had no real
  prior. **Consequence:** the plain script is the wrong entry point for any
  real-weights run, and it has ZERO wiring for `action_per_frame`/
  `action_seq_len` (grepped — confirmed empty). `configs/diffusion_wan22_dcunet_output_metaworld.yaml`'s
  header pointed at the plain script; **fixed 2026-07-14** to point at
  `_external.py` explicitly, with the reasoning inlined in the config comment.
  **Every other `diffusion_wan22_*` config still points at the plain script**
  (`avid_i2v`, `avid_i2v_noshortcut`, `avid_xattn`) — not fixed yet, flagged
  as a follow-up (see Ranked causes below).

---

## The single most likely explanation

No single root cause. The evidence converges on **two independent, confirmed
bugs sitting on top of a genuinely-low-headroom regime** — not one silver
bullet, and not "nothing is wrong, it's just hard":

**Layer 0 (expected, not a bug).** MetaWorld clips are low-motion; the frozen
i2v base already nails the static background from the clean conditioning
frame, so the action-addressable residual is a small fraction of total pixels.
This alone predicts a loss curve that looks nearly flat even if the
addressable part is learning fine. `_frame_masked_mse`/`_flow_loss`
(`trainer.py`) and `flow_matching_loss` are flat, unweighted means over every
voxel — no spatial/motion weighting, no SNR/timestep weighting. **Medium
confidence as a magnitude claim** — verification corrected the framing from
"background gradients literally outvote foreground ones" (not how MSE
gradients work — each voxel's gradient scales with its own residual) to "the
*aggregate metric* can hide real foreground headroom, even if per-voxel
gradients are fine." Not yet empirically separated — see do-now step 5.

**Layer 1 (confirmed bug).** `composition: mask_mix` + `gate_bias: 4.0` in the
live Wan configs means `gate = σ(gate_logit + 4.0) ≈ 0.982` at init, never
annealed (no scheduler exists anywhere — grepped). Gradient into the adapter's
own prediction head scales by `(1 − gate) ≈ 0.018` — **~50× attenuation of the
adapter's only training signal, every step, for the whole run**
(`adapted_model.py` composition math, `adapters/output/wan.py` gate head,
`factory.py`'s `predict_full=is_mask_mix` routing). Orthogonal to headroom —
throttles whatever gradient exists, action-relevant or not. Confirmed
independently by two lenses (`base-residual` verified it cleanly; the stub
`gating-suppression` verifier re-derived it from scratch).

**Layer 2 (confirmed bug, independent of 0–1).** The training script actually
declared by the dcunet and xattn config headers (`train_wan22_i2v_metaworld.py`,
plain) never reads `action_per_frame`/`action_seq_len` — confirmed by direct
grep, zero matches near its `WanBatchPreprocessConfig(...)` call. Net effect:
- `action_per_frame: true` in the dcunet config was **silently inert** whenever
  run via the plain script (falls back to one pooled action vector broadcast
  to all 11 frames). *(Now moot for the dcunet config specifically — header
  fixed to require `_external.py`, which I personally validated wires it
  correctly this session: `action=per-frame[B,11,A]` printed and training ran.)*
- The xattn config's cross-attention operated on **unbinned raw-pixel-frame
  action tokens** against an **11-latent-frame** query grid, with **no
  temporal masking** — confirmed directly (`action_seq_len` unset,
  preprocessor default `None` = passthrough). This is exactly the failure
  mode [[../../50_Decisions/open/action-conditioning-injection-mechanism]]
  explicitly called out as **non-negotiable** to avoid *before* the run
  happened (decision doc, "Non-negotiable coupling" section, written
  2026-07-11 — one day before the `xb76ptw2` run on 2026-07-12): *"a lone KV
  token attended by all queries collapses back to a global bias → cross-attn
  ≈ AdaLN and the ablation is uninformative... the cross-attn arm MUST use a
  per-frame action-token sequence."* That requirement was not met.

**⇒ Correction to the standing record:** the "cross-attention did not improve
results" finding
([[../experiments/20260712-wan-xattn-action-no-improvement]],
[[../../20_Tickets/feat-adapter-wan-action-cross-attention]]) **should not be
treated as a clean test of localized/cross-attention action injection.** It
tested a broken, misaligned, unmasked variant that violated its own design's
explicit precondition. The eval numbers (adapter worse than base on all 6
metrics) still stand as *observed*, but the causal story ("cross-attention is
a step back from AdaLN") is not yet established — see do-now step 7.

**What does NOT get overturned:** established finding #2 (the corrupted-base
control run shows the trainer+adapter mechanism *can* learn given headroom)
still holds — it just now has an explanation (the vendored-wrapper base bug),
and it doesn't prove the gate/wiring throttles are harmless when headroom is
small, only that they're not a full block.

---

## Ranked causes

| Cause | Likelihood (post-verification) | Strongest fix | Effort |
|---|---|---|---|
| Gate saturation (`mask_mix`, `gate_bias=4.0`) throttles adapter gradient ~50× | **high — code-confirmed 2026-07-14, real-run-confirmed 2026-07-15**: the reference AVID repo's own `AVIDAdapter` uses the identical composition formula but `init_mask_bias: 0.0` (50/50 at init); a live run of that exact code **on our MetaWorld data** shows `mask_mean` climbing 0.52→0.63 and loss dropping ~9.5× over ~800 steps — [[../experiments/20260715-avid-metaworld-native-gate-healthy]] | Cheapest: `gate_bias: 4.0` → `0.0` to match the paper (now backed by a matching real run, not just a config comparison). Wan: `composition: mask_mix` → `gated_residual`. DynamiCrafter: needs code, not just a flag | low (Wan) / med (DC) |
| Cross-attn action tokens unbinned + unmasked (xattn run) | **high** (confirmed, my own spot-check) | Pin `action_seq_len = latent_frames` in the xattn config; re-run before drawing any injection-mechanism conclusion | low (config) |
| Wrong/legacy training script silently used for real-weights runs | **high, but downgraded to low priority (2026-07-14)** — real, dated bug (fixed in `ed6e42c`, 2026-07-02; `l3p8kygb` predates the fix by 3 days), but user confirmed they always use `_external.py`, so it's not a live risk to actual runs, just stale doc comments | Fix remaining config headers opportunistically; not urgent | low |
| Aggregate loss has near-zero *visible* headroom given low motion | **medium** (mechanism real, magnitude unconfirmed) | No-retrain motion/static split on the existing `_probe_batch` | low |
| Shortcut base-loss vs self-consistency-target overlap (`trainer.py`: the base flow loss fires unconditionally against the point-velocity target even on non-anchor/coarser steps) | **medium** (partially — real design characteristic; not yet established as harmful vs. a standard combined-loss pattern) | Complements the already-open [[../../20_Tickets/experiments/exp-shortcut-zero-weight-control-run]] — that ticket's weight=0 control run is the right test | med |
| Action embedding magnitude unbalanced (raw MetaWorld deltas, no normalization, naive add vs. O(1) sinusoidal time-emb) | **medium** (architecture confirmed; severity unmeasured — never logged) | Log `‖time_emb‖` vs `‖cond_proj(action_emb)‖` on a real batch before touching anything | low |
| Optimization SNR: batch_size=2, no grad accumulation, no EMA | **medium** (confirmed, secondary/amplifying) | Add `grad_accum_steps` | low |
| Action dropout / CFG diluting signal | **low — ruled out** | n/a (dropout=0.0 everywhere; eval CFG is text-only, structurally orthogonal to action) | n/a |

---

## Do-now experiment order (cheapest → most decisive)

Goal: separate "expected low headroom" from "real bug" before spending GPU
budget on architecture changes.

1. **Zero-GPU shape/config check.** Instantiate the preprocessor for each live
   config against one real batch; print `cond['action'].shape`,
   `cond['action_seq'].shape`, `z0.shape[2]` (latent-frame count). Confirms/
   refutes the wiring bugs above with no training run.
2. **Single-clip overfit test.** Train on one repeated clip. If the adapter
   (as-is, `mask_mix`/`gate_bias=4.0`) can't drive loss toward its floor on
   ONE clip in a few hundred/thousand steps ⇒ capacity/plumbing bug, not a
   dataset-level headroom problem. Most decisive single test — removes the
   headroom confound entirely.
3. **Action-shuffle counterfactual** (already an open ticket:
   `exp-conditioning-action-shuffle-ablation`). Permute action↔clip pairing;
   compare loss/`probe_denoise_delta` to true pairing. Indistinguishable ⇒
   action isn't being used, independent of composition. Worth running
   **both before and after** the Layer 1/2 fixes — the pre/post delta in
   shuffle-sensitivity is itself diagnostic.
4. **Gate/contribution-magnitude logging** — already an open ticket:
   [[../../20_Tickets/feat-training-adapter-contribution-magnitude-logging]].
   If `gate` is still ≈0.982 after thousands of steps, Layer 1 is confirmed
   *active and ongoing*, not just an init artifact. (The paired
   `denoise_adapter_delta` / `probe_denoise_delta` metrics added 2026-07-14
   are a complementary, already-implemented signal — see
   [[../../10_now/architecture]] and the DC-UNet ticket.)
5. **No-retrain motion/static loss-region split** on the existing
   `_probe_batch` (already captured every eval, per this session's changes):
   build a motion mask from ground-truth frame-to-frame latent differences,
   split `_flow_loss` into masked-motion vs masked-static sub-means. If
   `delta_motion ≫ delta_static`, Layer 0's "flat loss hides real headroom"
   reading is confirmed.
6. **Fix Layer 1** (Wan `mask_mix` → `gated_residual`), short re-run, compare
   `denoise_adapter_delta` trend against the un-fixed baseline at matched
   steps.
7. **Fix Layer 2** — wire `action_per_frame`/`action_seq_len` through the
   plain script (or drop it and standardize on `_external.py`), pin
   `action_seq_len = latent_frames` for the xattn config, **re-run the
   cross-attention experiment properly** before drawing any conclusion about
   injection mechanism vs. capacity.
8. **Only after 1–7:** the already-open shortcut isolation tickets
   ([[../../20_Tickets/experiments/exp-shortcut-zero-weight-control-run]],
   `exp-shortcut-action-free-isolation`) — real, distinct mechanism, but
   currently entangled with the same flat-`base_loss` symptom; don't chase it
   before ruling composition/wiring in or out.

---

## Early signal on do-now step 2 (2026-07-14, smoke-only)

User proposed a variant of the single-clip-overfit spirit: keep the base
prediction as adapter **input** (`condition_on_base_outputs: true`) but let the
adapter's output **fully replace** the base at composition
(`composition: replace`, an existing unused `_compose` branch — zero new code).
Isolates gradient-flow/optimization from the mask_mix gate throttle without
discarding the base's information. Config:
`diffusion_wan22_dcunet_replace_metaworld.yaml`. A ~15-step smoke run (not a
citable result — no wandb id captured) showed a much higher-SNR, briskly
descending loss (1.85→1.55 in 15 steps) vs. the mask_mix run's near-flat
descent over thousands of steps, and `probe_denoise_delta` shrinking fast
(−1.72→−1.51 over 10 eval steps). Consistent with — not proof of — gate
saturation being a real, significant contributor. Full detail:
[[../../20_Tickets/feat-adapter-dynamicrafter-output-on-wan-base]] §"Crazy
experiment." Needs a real, longer run to confirm.

## AVID-vs-ours structural comparison (2026-07-15)

A second 17-agent workflow (8 dimensions, investigate→verify) compared our
training setup directly against the **real AVID reference code** (not just its
composition/gate — the full loss, optimizer, conditioning-paradigm, VAE,
freezing, logging, and training-loop machinery), reading both codebases
file:line. All 8 dimensions survived verification (0 refuted) — two came back
clean/"nothing new" (freeze/checkpoint-loading mechanics; most of the
EMA/optimizer-schedule dimension), which is itself a useful negative result.
**I personally spot-checked the top two findings below against the code — both
confirmed exactly as stated.**

### New, real differences (not covered by gate_bias/action-binning/script-mismatch)

1. **The adapter's own AdaLN cannot express per-frame differences — architecturally, not just from the binning bug.**
   `adapters/output/wan.py:96-97`: `if t.dim() > 1: t = t.flatten(1).amax(dim=1)`
   — the Wan tiny-DiT adapter collapses WAN's per-latent-frame diffusion-forcing
   timestep down to ONE scalar before its FiLM modulation, which is then
   broadcast identically to every frame (`action_model.py`'s `unflatten(1,(6,dim))`).
   The **frozen base** gets genuine per-token AdaLN (`model2_2.py:462-472`).
   This is an existing, deliberately-commented simplification in the code (not
   introduced this session) — confirmed by direct read. Even with the
   `action_seq_len` binning bug fixed (per-frame action *tokens* correctly
   aligned for cross-attention), the AdaLN/FiLM conditioning path itself has no
   mechanism to say "predict frame 5 differently from frame 9." AVID cannot
   have this asymmetry — its whole paradigm is single-global-timestep, so base
   and adapter always see the identical scalar `t`. **This is the single
   biggest new lever found** — but also the costliest to fix (needs a
   per-frame FiLM broadcast, not a config flag).
2. **No gradient accumulation in our trainer at all** (`trainer.py` — confirmed
   zero accumulation code, `optimizer.step()` every physical batch). Our
   healthy-reference comparison run (`pg3x72uc`) used `batch_size=2,
   accumulate_grad_batches=4` (effective batch 8) — note this accumulation
   value was *my own choice* when building `avid_11M_metaworld.yaml` for local
   hardware, not part of AVID's original published methodology (their real
   `avid_11M.yaml` uses `accumulate_grad_batches=1` at `batch_size=16`). Still
   a real, current confound: any curve-shape comparison between our runs and
   `pg3x72uc` is affected by this averaging-per-step mismatch, independent of
   everything else. Cheap, clean fix.
3. **No LR warmup anywhere in our optimizer** (`builders.py:43-47`, flat AdamW
   from step 0) vs. AVID's active 250-step linear warmup
   (`ddpm3d.py:1407-1423`, live in the exact config chain behind `pg3x72uc`).
   Given `accumulate_grad_batches=4`, 250 steps = 1000 micro-batches ≈ 31% of
   the ~800-step window we've been citing as "healthy." Can't rule in/out
   without an ablation, but full LR from step 0 onto an already-adverse gate
   init compounds rather than independently causes the problem.
4. **A purpose-built boundary-sampling mitigation sits unused on the live
   path.** `FlowMatchingTrainingObjective.sample_timesteps`
   (`losses/flow_matching.py:53-106`, logit-normal + shift schedule) exists
   specifically to avoid flat-uniform sampling near the flow-matching
   boundary, but every `diffusion_wan22_*.yaml` sets
   `use_batch_timesteps_for_flow: true`, routing training through the
   preprocessor's flat-uniform `sigma = torch.rand(...)` instead
   (`wan22_batch_preprocessor.py:128-145`). **Caveat from verification:** the
   original "v-param degenerates to an easy target near t→0" physics
   justification does NOT hold (v-parameterization has constant variance
   across the schedule by construction) — what survives is narrower: an
   explicit, purpose-built mitigation for exactly this regime is silently
   switched off, worth testing on its own merits, not the disproven story.
5. **The shortcut self-distillation loss has no AVID analogue and isn't
   cleanly isolable yet.** `shortcut_direction_weight: 1.0`,
   `shortcut_anchor_prob: 0.6` is live on ~40% of steps in our comparison
   configs; AVID's `p_losses` is always a single denoising objective. The
   config meant to isolate this (`*_noshortcut.yaml`) doesn't set
   `composition`/`gate_bias` at all — defaults to `output_composition="add"`,
   not even the same composition formula being compared. Its independent
   contribution (beyond the already-confirmed gate throttle) is unmeasured.

### Ruled out — don't chase

Frame-masking density (91% vs 100%, numerically immaterial), loss-reduction
form (equivalent given current `cond_frames_dist`), autocast scope (symmetric,
correct both sides), grad-clip parameter set (effectively identical — frozen
params never populate `.grad` on either side), freeze/checkpoint-loading/
optimizer-param-group mechanics (clean on both sides — ours is if anything
more defensive), AVID's "denser" quality-metric logging (**false** — AVID logs
images/metrics every 1000 batches; we already log every 5 steps, we're
denser), AVID's gradient-norm logging (dead/vestigial code, never executes),
EMA absence (real but already captured in the existing optimization-SNR row
below).

### Logging gap, sharpened

The gate value is **structurally unreachable** downstream, not just unlogged:
`AdaptedModel._compose()` computes `gate = torch.sigmoid(...)` as a pure local
variable (`adapted_model.py:168,178`) with no return path — `forward()`'s
`return_base=True` mechanism (added this session for the paired-delta metric)
has no gate equivalent. AVID's `apply_model` returns `(combined_output, info)`
with `mask_mean/std/min/max` as a first-class part of the forward contract.
Concrete implementation detail for
[[../../20_Tickets/feat-training-adapter-contribution-magnitude-logging]]:
widen `AdaptedModel.forward`'s return contract (or cache `self._last_gate`)
the same way `return_base` was added, and thread it into the metrics dict
already flowing through `training_step`.

### Experiment plan (2026-07-15) — concrete tickets, supersedes both earlier do-now lists

Everything below has either shipped (code) or has a ready-to-run config.
Sequenced by dependency, not just cost — later items assume earlier ones have
landed or run.

**Shipped since the AVID-vs-ours comparison:**
- Gradient accumulation + LR warmup implemented and smoke-validated
  (mechanics confirmed via wandb: exact linear LR ramp, correct
  optimizer-step-gated cadence checks) —
  [[../../20_Tickets/feat-training-grad-accumulation-warmup]].
- Pre-training baseline eval (runs automatically before the first gradient
  update on any fresh run now — no config needed) — documented in
  [[../../10_now/training-hyperparameters]].
- `gate_bias: 0.0` applied to three configs: the AdaLN baseline, the DC-UNet
  capacity config, and (partially — see below) the xattn variants.

**Run order:**

1. [[../../20_Tickets/experiments/exp-training-single-clip-overfit]] — cheapest,
   highest-discriminating-power. Run before anything else; a failure here
   means stop and re-debug rather than run the rest of this plan.
2. [[../../20_Tickets/experiments/exp-adapter-adaln-gatelow-metaworld-run]] — the core
   validation run (`diffusion_wan22_avid_gatelow_metaworld.yaml`, all three
   confounds fixed). The load-bearing result this whole plan hinges on.
3. [[../../20_Tickets/experiments/exp-adapter-dcunet-gatelow-capacity-run]] — the capacity
   experiment, now unblocked, run alongside #2 for a
   capacity-vs-injection-mechanism read.
4. [[../../20_Tickets/experiments/exp-conditioning-action-shuffle-ablation]] — run on the
   #2 config, both to confirm the adapter uses actions at all and (compared
   against a pre-fix shuffle result if one exists) whether shuffle-sensitivity
   changed with the gate fix.
5. [[../../20_Tickets/experiments/exp-shortcut-zero-weight-control-run]] — new clean
   sibling config built (`diffusion_wan22_avid_gatelow_noshortcut_metaworld.yaml`),
   isolates the shortcut term against the #2 config.
6. [[../../20_Tickets/experiments/exp-adapter-xattn-gatelow-metaworld-run]] — **note:**
   as currently configured this isolates gate_bias only (the binning fix is
   off); a doubly-fixed sibling is still needed to actually resolve
   [[../../50_Decisions/open/action-conditioning-injection-mechanism]] — see
   the ticket.
7. [[../../20_Tickets/experiments/exp-adapter-wan-replace-metaworld-run]] — gradient-flow
   diagnostic for the Wan tiny-DiT adapter specifically (parallel to the
   already-smoke-tested DC-UNet replace run). Not yet smoke-tested even at
   that scope — do a short run before committing real time.
8. [[../../20_Tickets/bug-losses-flow-boundary-sampling-unused]] — A/B the
   dead boundary-sampling mitigation. Lower priority, independent of the
   above.
9. **Only after 1–8, if still weak:**
   [[../../20_Tickets/feat-adapter-wan-per-frame-adaln]] — the biggest single
   lever found, costliest to implement (real architecture change, not a
   config flag). Sequenced last so it isn't conflated with the cheaper items,
   and because it wouldn't fix anything on the cross-attention path anyway
   (see the ticket).

**Still blocking clean interpretation of all of the above:**
[[../../20_Tickets/feat-training-adapter-contribution-magnitude-logging]]
(gate/mask value logging) is not yet implemented — every run above can only
be read via loss curves and the `denoise_adapter_delta`/`probe_denoise_*`
diagnostics, not the gate trajectory directly. Worth landing before or
alongside run #2 if feasible, since it's what let the AVID comparison run
show `mask_mean` moving 0.52→0.63 so clearly.

## Update (2026-07-16) — gate_bias fix confirmed necessary but not sufficient

Steps 6 and 7 of the plan above ran:
[[../../20_Tickets/experiments/exp-adapter-xattn-gatelow-metaworld-run]] (`bcipghvw`,
gate_bias=0.0) and [[../../20_Tickets/experiments/exp-adapter-wan-replace-metaworld-run]]
(`5cxstyh4`, replace/no-gate). A third run intended as the single-clip
overfit test (step 1, `uea10230`) turned out — verified via the wandb API,
not memory — to have run on the **unfixed** `gate_bias: 4.0` config despite
the plan's guardrail.

Full analysis:
[[../experiments/20260716-wan-xattn-adapter-clones-base-not-actions]].
Headline: in **all three** runs, `denoise_adapter_delta` collapses to ~0
within 60-150 steps regardless of gate_bias (0.0 and 4.0 give nearly
identical initial transients — inconsistent with `mask_mix`'s own
identity-at-init design intent for `gate_bias: 4.0`). The adapter's own
prediction is converging to **clone the frozen base** rather than diverge
using the action signal. Under mask_mix this shows up as "composed ≈ base,
video looks like a copy"; under replace it shows up as "training loss looks
fine but decoded video is catastrophic" (probe-batch delta and eval-video
quality metrics reveal a real gap that `train/loss` hides).

**Consequence for the plan:** step 2 (the AdaLN gatelow validation run,
[[../../20_Tickets/experiments/exp-adapter-adaln-gatelow-metaworld-run]]) is still worth
reading closely if/when it lands, but should not be expected to be a clean
"gate_bias was the fix" result — expect the same clone-base pattern unless
it's specifically checked for. **Priority moves up** for step 4
([[../../20_Tickets/experiments/exp-conditioning-action-shuffle-ablation]]) and
[[../../20_Tickets/experiments/exp-shortcut-action-free-isolation]] (not yet in the
numbered plan — add it) — these are the two experiments that can actually
tell "the adapter ignores the action" apart from "the adapter found a
degenerate optimum regardless of action," which is now the central open
question. Continuing to sweep gate_bias / composition mode without also
running one of these is unlikely to be diagnostic on its own.

## What to explicitly NOT do

- **Don't treat `xb76ptw2` as a refutation of local/cross-attention action
  injection.** It violated its own design decision's non-negotiable
  precondition. Re-run with `action_seq_len` pinned before concluding
  anything about injection mechanism vs. capacity.
- **Don't touch `frame_stride`.** Deliberate, correctly-reasoned decision
  (striding pushes motion outside the frozen base's native distribution) —
  not a live lever here.
- **Don't expect the vault-decided anchor-step warmup to fix the
  shortcut-interference mechanism.** As specified it only zeroes
  `shortcut_direction_weight` — it doesn't touch the base-flow-loss-vs-target
  overlap. Different problem.
- **Don't invest in DynamiCrafter's `add_act_time_emb=False` magnitude-balanced
  restructuring yet.** Real architecture debt, but higher effort and
  unmotivated until the cheap embedding-norm logging (do-now-adjacent, low
  effort) shows the mismatch is actually severe.
- **Don't chase action dropout/CFG.** Conclusively ruled out.
- **Don't conclude "no headroom, nothing to fix" from the flat aggregate loss
  alone.** Two confirmed bugs (gate saturation, dead/misaligned per-frame
  wiring) are independently sufficient to produce a flat-looking curve even
  if real headroom exists. Run do-now step 5 before accepting that reading.

## Related

- [[../../20_Tickets/feat-adapter-wan-action-cross-attention]] — outcome
  section corrected 2026-07-14 to flag the invalid xattn run
- [[../../50_Decisions/open/action-conditioning-injection-mechanism]] — the
  decision this run was supposed to resolve; not resolved
- [[../../20_Tickets/feat-training-adapter-contribution-magnitude-logging]] —
  do-now step 4
- [[../../20_Tickets/experiments/exp-conditioning-action-shuffle-ablation]] — do-now step 3
- [[../../20_Tickets/experiments/exp-shortcut-zero-weight-control-run]] — do-now step 8
- [[../../20_Tickets/feat-adapter-dynamicrafter-output-on-wan-base]] — the
  capacity-lever ticket this diagnosis feeds into
