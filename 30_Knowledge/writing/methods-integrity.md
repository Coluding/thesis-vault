---
type: writing
status: living
last_updated: 2026-08-03
sources:
  - "[[rubric/03-experimental-evaluation]]"
  - "[[rubric/05-reflection]]"
  - "[[../experiments/_index]]"
  - "[[../experiments/20260801-wan-rt1-indistribution-plateau]]"
  - "[[../experiments/20260802-shortcut-works-on-flow-not-diffusion]]"
  - "[[../../00_Inbox/2026-08-01-effect-rel-is-a-gain-metric]]"
  - "[[../../00_Inbox/2026-08-01-rt1-heldout-split]]"
  - "[[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]]"
  - "[[../../20_Tickets/bug-data-silent-window-filter-drops-episodes]]"
---

# Methodological integrity — what went wrong, how we found it, what it cost

> **Direct source for Ch4 §4.6.** Every invalidation in the campaign was
> **self-detected**. That fact is the difference between the rubric's
> *"judge the setup of an existing experiment and include modifications if
> needed"* (band 8) and *"errors are made in the process, invalidating
> part of the experiment"* (the failing band) — the same events, read
> through whether the author found them
> ([[rubric/03-experimental-evaluation]]).
>
> **Write this section. Do not bury these.** A reader who sees we caught
> our own in-sample evaluation stops hunting for the one we missed.
> Concealment is the only path from here to the failing band.

Each entry uses the five-part negative-result shape
([[thesis-style-guide]] §5): expectation → detection → damage → correction
→ what now stands.

---

## I1 — Evaluation was in-sample

**Expectation.** Held-out evaluation, as configured; the trainer exposes a
held-out branch.

**Detection.** Read-through of the trainer's evaluation path while auditing
an anomalous result. The RT-1 and OpenVid jobs pass the *same directory* to
`--data-dir` and `--eval-data-dir`, and the trainer's held-out branch
(line 419) never splits it. ACWM is clean — it uses explicit
`ind_train`/`ind_test`.

**Damage.** **All RT-1 and OpenVid numbers.** Since generative models
memorise, an in-sample perceptual metric is not evidence about
generalisation at all.

**Correction.** Quarantined: nothing RT-1 enters the thesis until the
held-out re-eval lands ([[../../00_Inbox/2026-08-01-rt1-heldout-split]]).
The ACWM cells, which carry the headline claims, are unaffected.

**What stands.** Every ACWM and clean-room result. The quarantine is
narrower than it first appears, and saying so precisely is the point.

---

## I2 — A silent window filter dropped 98.5% of the data

**Expectation.** `dataset_size = 5000` on the SkyReels × RT-1 cell.

**Detection.** Config audit prompted by a result that was *too good*
(the arm appeared to reach 91% of the AVID reference). Actual
`dataset_size` was **76**: `temporal_length: 97` against RT-1 episodes of
~22–115 frames, and the window filter at `dataset.py:71` silently dropped
everything shorter.

**Damage.** The entire SkyReels cell. Specifically **retracted**: the
0.0450 peak, the "91% of the reference" claim, the **35× data-axis
figure**, and the SkyReels quality numbers. Compounded by I1 (the same cell
was also evaluated in-sample).

**Correction.** Cell voided in the ledger rather than quietly dropped. A
re-run needs `temporal_length: 17` plus a held-out split. Ticket:
[[../../20_Tickets/bug-data-silent-window-filter-drops-episodes]].

**What stands.** Nothing from that cell. **The two-factor "data sets the
level" law that rested on the 35× is downgraded** — it is no longer a
thesis claim.

**Transferable lesson:** a filter that drops data should never be able to
drop *most* of it silently. Log the retained fraction, and fail loudly
below a threshold.

---

## I3 — `gate_cap` equal to the gate's init froze the gate

**Expectation.** `gate_cap` clamps σ(gate) so the adapter branch retains
gradient.

**Detection.** Gate telemetry — the gate held exactly at its cap for the
whole run.

**Damage.** The **D3 curvature comparison**. The headline 68×
flow-vs-diffusion consistency-loss ratio was measured under a frozen gate
and cannot be attributed to the objective.

**Correction.** Bug ticketed
([[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]]) and
the comparison re-run with live gates. The clean re-measurement
([[../experiments/20260802-shortcut-works-on-flow-not-diffusion]])
**supersedes** the 68×: depth-matched, Wan treated `consistency_cos` 0.302
[0.251, 0.356] vs control 0.034 [0.026, 0.042] — 9×, non-overlapping CIs.

**What stands.** The re-run. The superseded number is reported *as
superseded*, not deleted — that is what makes the re-run credible.

---

## I4 — The primary metric could not distinguish gain from information

**Expectation.** `action_effect_rel` measures how much action *information*
the adapter uses.

**Detection.** Our own analysis: the metric is **monotone in
action-pathway gain**, so any intervention that scales that pathway raises
it regardless of information content. The vault's own gate control moved it
4.8× with the action path untouched
([[../../00_Inbox/2026-08-01-effect-rel-is-a-gain-metric]]).

**Damage.** Both "mechanism fix" claims — `condition_center` (DC) and
`action_token_norm` (Wan) — are *themselves gain increases*, so the metric
could not establish that they repaired a mechanism rather than turning up a
volume knob.

**Correction.** A **temporal control probe** whose null is a *shape* rather
than a magnitude. On the clean-room A/B the per-frame arm's diagonal
concentration was 0.390 vs the pooled arm's 0.199 (chance 0.200), and the
pooled arm's per-frame response rows were **bit-identical** — a gain
increase cannot manufacture per-frame differentiation. See
[[../tech/probe-suite]] §6.

**What stands.** The **pathway** claim, which the temporal control settles.
The two scale-calibration interventions remain characterised as *gain*
changes with measured effects, not as mechanism repairs — the honest
reading.

**Transferable lesson, and the thesis's methodological core:** a scalar
sensitivity metric on a conditioned model cannot separate *more signal*
from *louder signal*. Only a probe whose null is a shape can.

---

## I5 — An unresolved design confound (still open)

**Expectation.** The flow-vs-diffusion shortcut comparison isolates base
geometry.

**Detection.** Design review before writing the result up.

**Damage.** The two arms differ in **consistency target *and* depth**
(`v_average` on the Wan/flow arm, `endpoint_inversion` on the DC/diffusion
arm; 800 vs 400 steps). The **cross-base** reading is therefore confounded.

**Status: not resolved.** The **within-arm** results stand — each arm is
compared against its own matched control at matched depth, and both
comparisons are internally valid. A second independent signal supports the
same direction: the gain profile is flat O(1) for Wan-treated versus a
collapse for the Wan control and a 4e4 blow-up for DC.

**Writing rule:** state the confound **in the same paragraph as the 9×**.
An unflagged number here takes the whole D3 claim with it when a committee
member finds it — and they will, because it is in our own experiment note.

---

---

## I6 — A null that fired, and was *not* the obvious cause

**Expectation.** `eval_stepsize_base_null_violation` = 0, as it is
everywhere else. The frozen base cannot see `step_level`, so its output must
be invariant.

**Detection.** The null control did its job — it fired on DC while returning
exactly 0 on Wan, and exactly 0 for the *action* probe on the very same DC
runs.

**The obvious inference was wrong.** A non-zero null normally means a
conditioning leak, which would have voided every DC step-size number. It was
instead **nondeterminism**: the frozen base was running in `train()` mode, so
its `nn.Dropout(p=0.1)` layers drew a fresh mask on every forward.
`step_level` is *structurally incapable* of reaching the DC base.

**Damage.** None to the results, because nothing had been quoted from the
affected probe. The cost was diagnostic time.

**What stands.** The null control's value is demonstrated twice over here:
it caught a real harness defect, *and* the discipline of root-causing rather
than assuming saved a set of numbers that a leak-assumption would have
discarded ([[../../00_Inbox/2026-08-01-stepsize-null-violation-rootcause]]).

**Transferable lesson:** a control that fires tells you *something* is
wrong, not *what*. Diagnosing before discarding is part of the instrument.

---

## I7 — A test suite that passed vacuously

**Expectation.** Five unit tests covering the new channel-concat action
injection, all green.

**Detection.** Review of what the assertions actually compared.

**The defect.** With the default `predict_full=False` the output head is
zero-initialised, so the model emits exactly `0.0` and every assertion
compared zeros to zeros. The suite could not have failed regardless of the
feature's correctness.

**Damage.** None to any reported result — the defect was found before the
arm was interpreted. The cost is the false assurance it provided while the
run was being set up.

**Correction.** The suite now asserts a non-zero output *first*, so a
degenerate model fails the precondition rather than passing every check.

**Transferable lesson, and a standing check:** a green test suite over a
zero-initialised path proves nothing. Any test of a component whose output
can legitimately be zero must first establish that the output is non-zero.
This is the unit-test analogue of the degenerate-input problem the structure
probes already guard against ([[../tech/probe-suite]] §5g) — the same
failure mode at a different layer.

---

## I8 — A base that rendered noise while every automated check passed

**Expectation.** A newly integrated frozen base (EasyAnimate) producing
valid video, as every shape, finiteness and file-existence check reported.

**Detection.** **A human looked at one frame.** No automated check failed.

**The defect — three stacked.** A zero text context was passed instead of a
real embedding (absmax 21); the diffusers pipeline silently dropped the
entire T5 stream; and classifier-free guidance was forced to 1.0. Each is
individually survivable; together the base was generating noise.

**Damage.** **Every `effect_rel` number logged on that base before
2026-08-05 is void.** The cost is recorded in
[[../../10_now/compute-spend-ledger]].

**Correction.** Base fixed first, then re-measured; the numbers that survive
([[../experiments/20260806-objective-governs-action-specificity-not-adapter-capacity]])
are all post-fix.

**Transferable lesson — the sharpest in this section.** Shape checks,
finiteness checks and file-existence checks are all *type* checks: they
verify that an object of the right kind was produced, never that it is the
right object. A generative pipeline can satisfy all of them while producing
noise. **The only check that caught it was rendering one frame and looking
at it**, and that check costs seconds. It is now standing practice before
any new base is trusted.

This is the same failure mode as the vacuous test suite (I7) and the
degenerate-response matrix ([[../tech/probe-suite]] §5g), at a third layer:
*a well-formed output is not a correct one.* Three independent instances in
one campaign is itself the finding.

---

## I9 — A run that cannot be exactly reproduced

**Expectation.** Runs are reproducible from the recorded commit.

**Detection.** Launch-time audit of the remote working tree.

**The defect.** 135 uncommitted modified files at launch; the run executed
rsynced working-tree code, not the recorded commit `75721b7`.

**Damage.** None to the result's validity — config, adapter config and the
resolved `action_seq_len` are captured in the note and the startup log — but
**exact re-creation requires files that were never committed.**

**Correction.** Commit before launch, adopted as standing practice.

**Why it is reported rather than quietly fixed.** The provenance convention
in this thesis is *run id + checkpoint + commit*
([[thesis-formal-rules]] §5). Where the commit does not fully determine the
code, saying so is the only honest form of the receipt — and a reader who
sees one such disclosure can trust the others.

---

## I10 — The control was the wrong control, and it manufactured a result

**Expectation.** Comparing the adapted rollout against the **frozen base**
isolates the effect of action conditioning: the base never sees an action,
so a gap between them is action-driven.

**Detection.** The reservation was written into the preliminary note
*before* the confirmation run — *"the frozen base is not a sufficient
control; it differs from the adapter in every respect, not only action
access"* — and the instrumented run tested it.

**The defect.** That objection was not academic; **it was the entire
effect.** A hand-measured preliminary comparison had the adapted rollout
tracking ground-truth motion at r ≈ +0.75 against the base's r ≈ +0.09 — a
gap of **0.66**, and a striking result. Under logging, `adapted − base`
across four draws is **+0.13, +0.10, +0.045, −0.034**: an order of magnitude
smaller and **sign-inconsistent**. The frozen base, which cannot see
actions, tracks per-clip motion about as well as the adapter does.

**What the right control shows.** A **paired shuffled-action** control —
identical weights, identical conditioning frame, identical seed, only
another clip's actions — differs from the adapted rollout in exactly one
respect. Its gain is positive in **4/4 draws** (+0.069, +0.168, +0.122,
+0.213; mean **+0.143**; sign test p ≈ 0.06 one-tailed). So there *is* a
real effect, roughly a fifth the size of the illusory one.

**Damage.** None reported — the preliminary claim was flagged *"do not cite
until the instrumented run reports"* and never entered the draft. Had the
base remained the control, this would have been recorded as a strong
positive result resting on a comparison confounded by capacity, conditioning
pathway and training all at once.

**Transferable lesson — the design rule for every action claim in this
thesis:** *a control must differ from the treatment in exactly one respect.*
The frozen base is convenient and available and differs in everything; the
paired shuffled-action control is the one that isolates the variable.
**The paired control is what turned a false positive into a small true
one.**

This is why the efficiency-axis protocol requires a *matched
conditioning-only control per level*
([[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]]) rather than
comparing each level to the frozen base — the same error, one level up.

## What the section should conclude

Not an apology. The closing move is the transferable one:

> Four of the five invalidations above were found by an audit rather than
> by a failing test, and three of them produced *plausible* numbers —
> a metric that moved, an arm that beat its reference, a ratio that
> matched a theoretical prediction. **Plausibility is not a check.** The
> practices that actually caught them were: verifying a null instead of
> assuming it, logging the retained fraction after every filter, reading
> the evaluation path rather than trusting the flag, and asking of every
> metric what *else* could move it.

## Practices adopted, and what we would do differently

For §6.2/§8.2 — the rubric's "insights for improvement":

- **Enforce held-out splits at the data layer**, not per job script.
- **Validate the instrument before trusting it** — ask of each metric what
  else could move it, and build a probe whose null is a shape.
- **Structure probes before quality metrics.** Quality improved on the very
  cell that carried no action information.
- **Pre-register the decision rule before the run**, not after seeing it
  (adopted for [[../../20_Tickets/experiments/exp-adapter-lora-vs-output-comparison]]).
- **Log the retained fraction after every filter**; fail loudly below a
  threshold.
- **Report superseded numbers as superseded** rather than deleting them.

## Related

- [[rubric/03-experimental-evaluation]] · [[rubric/05-reflection]]
- [[../tech/probe-suite]] — the instrument and its validation
- [[../experiments/_index]] — the ledger, where retractions are recorded in place
- [[../../70_Thesis/outline]] §4.6
