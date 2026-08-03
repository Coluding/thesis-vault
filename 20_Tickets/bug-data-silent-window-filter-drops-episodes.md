---
type: bug
scope: data
status: open
priority: high
created: 2026-08-02
updated: 2026-08-02
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]]", "[[../00_Inbox/2026-08-01-rt1-heldout-split]]", "[[bug-eval-stepsize-probe-runs-in-train-mode]]"]
---

# bug: `dataset.py:71` silently drops short episodes — cost us a whole cell

## What happened

Both SkyReels × RT-1 runs trained on **76 episodes instead of 5000**. From the
job logs (`logs/skyreels/acwm-robotarm-skyreels-skyreels-rt1-{oracle-25133625,tn-nobase-25112302}.out`):

```
dataset_size=76 steps=5000000 batch_size=6 eval=on gen_eval=on
```

Both configs carried `temporal_length: 97` with the comment "128-frame
episodes" — a stale copy from ACWM Robot Arm. RT-1 episodes are ~22–115 frames
(the Wan RT-1 config says exactly this and uses 17). `data/dataset.py:71` drops
every episode shorter than the window **silently**: no error, no warning, no
log line beyond a `dataset_size` the reader has to notice is wrong.

**Cost:** the entire SkyReels × RT-1 cell, including the "35× data axis" and
"91% of the AVID reference" headlines, plus ~30 GPU-hours.

## Fixed 2026-08-02

- `configs/skyreels/diffusion_skyreels_rt1_tokennorm_{nobase,oracle}.yaml`:
  `temporal_length: 97 → 17`, with a comment explaining why so it does not get
  copied back.
- `jobs/experiments_cluster/rt1/submit_train_skyreels_rt1_{oracle,tokennorm_nobase}.sh`:
  eval repointed from `rt1/val_long97` (400 → only 8 usable at width 97) to
  **`rt1/val`** (400 episodes, **400/400** usable at width 17).
- Both synced to the cluster and verified there.

**Note on the first attempt:** the initial fix pointed eval at a *long* held-out
split to accommodate the 97-frame window. That accommodated the symptom and
would have left training on 76 episodes. The window itself was the bug.

## STILL OPEN — the actual defect

**The silent filter is still silent.** Nothing prevents this recurring on the
next dataset whose episodes are shorter than a copied window.

Proposed guard (not implemented — wants a decision on the threshold):

- **Hard error** when the filter removes more than some fraction (50%?) of
  episodes, with an escape hatch flag, mirroring the `--allow-insample-eval`
  guard added alongside.
- At minimum, a **loud warning** naming both counts: `"window=97 dropped
  4924/5000 episodes (98.5%) — check temporal_length against this dataset's
  episode lengths"`.

The same class of bug as [[bug-eval-stepsize-probe-runs-in-train-mode]] and the
in-sample-eval footgun: **a silent default that produces a plausible-looking
number.** Three in one night suggests the codebase should fail loud on
"suspiciously degenerate configuration" as a general policy.

## Related audit finding

`train_skyreels_acwm.py` prints `eval dataset: ACWM-Phys split …` regardless of
`--dataset`. The **builder dispatch is correct** (`:207`, `:221` select
`build_rt1_clip_dataset` for `--dataset rt1`) — only the log string is wrong,
but it cost real time during diagnosis. Worth fixing while nearby.
