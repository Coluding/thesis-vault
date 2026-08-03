2026-08-01 overnight — **the red team's most dangerous finding: `effect_rel` may be a gain metric, and both our "fixes" are gain knobs**

From the red-team audit ([[2026-08-01-red-team-audit]]). Flagging separately
because if it holds it undercuts the two headline D2 results, and it needs a
decision before any of Ch5 is written.

## The argument

`effect_rel = ||pred_true − pred_shuffled|| / ||pred_true||` is monotone in the
**gain** of the action pathway. Both interventions the campaign presents as
mechanism fixes are, mechanically, gain increases:

| "fix" | what it does |
|---|---|
| `condition_center` (DC arm E) | `BatchNorm1d(affine=False)`; the in-code comment says it "rescales to unit variance" |
| `action_token_norm` (Wan) | raises action-token RMS 0.004 → 0.757 |

And the vault already contains the control that shows the metric responds to
gain alone: **changing only the gate (0.5 → 0.899) moved `effect_rel` 4.8×
with the action path untouched.**

The audit reports that on a gain-normalised version of the metric arm E is
**0.74× the AVID reference — i.e. worse**, not 3.9× better. _Needs
verification: I have not independently reproduced the normalised figure or
confirmed which normalisation was used._

## Why this is not paranoia

It converges with two independent observations from tonight:

1. **`effect_rel` is anti-correlated with fit quality across 8 runs** with no
   overlap between the groups ([[2026-08-01-quality-vs-sensitivity-inverse]]).
2. The **structure triad has never been run on any DC cell** — and on the one
   cell where it *was* run (Wan, after the analogous gain fix) steering,
   temporal alignment and spatial concentration all came back **at chance**
   ([[../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]]).

So the pattern to rule out is: *we raised the gain on a pathway that carries no
usable action structure, and the metric rewarded us for it.*

## What settles it

Running the structure triad on **DC arm E vs arm 0** — the treated and
untreated cells of the spine. It is decisive in both directions:

- **structure above chance on arm E, at chance on arm 0** ⇒ `condition_center`
  bought real action *control*, the metric was reading something real, and D2
  has a positive result.
- **at chance on both** ⇒ the DC fix raised gain without buying control, D2's
  positive claim collapses to "sensitivity without control" on *both*
  backbones, and the honest thesis result is the domain-correction
  characterisation instead.

The triad did not exist for DC (only in `generate_wan22_i2v_compare.py`, which
is Wan-specific); a port into the backbone-generic
`scripts/eval_action_sensitivity.py` plus an arm-E-vs-arm-0 job was built
overnight → `00_Inbox/2026-08-01-dc-structure-probe-port.md`. **~1 GPU-hour,
and it is the highest-value experiment in the queue.** Job written but NOT
submitted — needs review first.

## Note on what this does NOT touch

The methodological contribution is independent of it: "loss, gate norm, FID and
sample quality are all blind to action-blindness, so purpose-built probes were
required" stands whether or not `effect_rel` is an information measure — and
tonight's quality/sensitivity table is *additional* evidence for it.
