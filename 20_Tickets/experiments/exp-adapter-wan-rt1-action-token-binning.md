---
type: exp
scope: adapter
status: in-progress
priority: high
created: 2026-08-02
updated: 2026-08-02
resolution:
resolution_note: PROPOSED (awaiting confirmation per CLAUDE.md hard rule 2) — Hypothesis NOT supported — binning action tokens onto the latent grid gives no benefit at any depth to 3200 steps (mean 0.0250 vs 0.0270 unbinned). Fixes retained (they are correct regardless); the cause is the injection site, not the token layout.
closed_at:
related: ["[[../../30_Knowledge/experiments/20260802-avid-wan-cleanroom-perframe-causal]]", "[[../../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]]", "[[../../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]]", "[[../feat-adapter-wan-per-frame-adaln]]"]
---

# Does binning action tokens onto the latent grid fix our Wan adapter? (RT-1)

## Hypothesis

The clean-room showed per-frame **addressability** is causal (2.49x,
[[../../30_Knowledge/experiments/20260802-avid-wan-cleanroom-perframe-causal]]).
Our own RT-1 run `5w72bo01` had per-frame action *tokens* available via
cross-attention, yet measured temporal alignment **0.25 = chance**. Proposed
reason: **17 per-pixel-frame tokens were handed to a DiT operating on 5 latent
frames**, so token `j` corresponded to no latent frame and the correspondence had
to be discovered through permutation-invariant attention. It never was.

## The bug (found 2026-08-02)

The binning code existed and computed the right count, but was gated:

```python
action_seq_len=(latent_frames if action_per_frame else None)
```

`action_per_frame` governs the **encoder input**, not the cross-attention tokens
— which are consumed regardless. With `action_per_frame: false` (the RT-1 run)
this passed `None` → passthrough. Two unrelated concerns on one switch.

Second defect: `_action_sequence` binned with uniform `linspace` edges. For
17→5 that gives boundaries [0,3,7,10,14,17] — the right **count**, the wrong
**alignment**. The Wan2.2 VAE layout is frame 0 alone then groups of 4:
[0:1],[1:5],[5:9],[9:13],[13:17]. Fixed to use the exact layout when
`T == 4*(T'-1)+1`.

Fixes: `data/wan_batch_preprocessor.py`, `scripts/train_wan22_i2v_metaworld_external.py`
(`_resolve_action_seq_len`, legacy behaviour preserved), config
`configs/wan22/diffusion_wan22_action_rt1_binned.yaml` (`action_seq_len: latent`).

## Design

Identical to `5w72bo01` in every respect — same config lineage, `action_token_norm`,
NOBASE, data, geometry — **except** the action tokens are binned 17→5 onto the
VAE's exact latent grid. Run `25154241`, wandb `wan-rt1-BINNED`.

## Result — NO benefit through step 2800; hypothesis NOT supported

Like-for-like against the old run (`25107236`, same eval cadence):

| step | OLD `effect_rel` | NEW | OLD share | NEW share |
|---|---|---|---|---|
| 400 | 0.02336 | 0.02256 | 26.6% | 25.7% |
| 800 | 0.02640 | 0.02586 | 30.2% | 30.0% |
| 1200 | 0.02744 | 0.02760 | 31.6% | 32.1% |
| 1600 | 0.02848 | 0.02766 | 30.6% | 30.6% |
| 2000 | 0.02986 | 0.02590 | 36.2% | 32.3% |
| 2400 | 0.02381 | 0.02559 | 26.4% | 29.2% |
| 2800 | **0.03315** (old peak) | **0.02217** | 32.9% | 24.1% |
| 3200 | 0.02325 | 0.02238 | 30.1% | 29.9% |

**Run complete (walltime, step 3200).** Mean over all eight matched evals:
**old 0.02697 vs new 0.02497** — the binned run finishes **~7% below** the
unbinned one.

**No benefit at any point, and the binned run declines monotonically from step
1600** (0.0277 → 0.0259 → 0.0256 → 0.0222; share 30.6% → 24.1%) while the old
run was still oscillating up to its peak. Windowed means over 2000–2800:
**new 0.0246 vs old 0.0290** — the binned run runs ~15% *below* the unbinned one.
`base_null_violation` is exactly 0 at every eval, so the measurement is sound.

### Verdict: the 17→5 token misalignment was NOT the cause

The hypothesis predicted a lift; there is none, at any depth measured. Binning
the cross-attention action tokens onto the latent grid does not recover
action-following in our framework.

**What this leaves.** The clean-room showed addressability is causal (2.49x) —
but its faithful arm injects the action through **per-frame AdaLN concat**, not
cross-attention. Our path supplies per-frame *tokens* and still measures
alignment at chance, binned or not. So the evidence now points at the
**injection site**, not the token layout: addressability appears to come from
the modulation pathway (a per-frame scale/shift on normalised activations),
which cross-attention into the residual stream does not provide — consistent
with the 07-31 trace showing the xattn contribution drowning at the residual add.

➜ Next candidate: [[../feat-adapter-wan-per-frame-adaln]], for which
`external_repos/avid/wan_diffusion/src/wdwma/models/action_wan.py` is now a
working reference implementation. That is a real code change, not a config flag.

## Old-run trajectory (for reference)

The old run's erosion begins at ~2800–3200 and settles by 5600:

```
2800: 0.0332 / 32.9%  (peak)   4000: 0.0201 / 26.6%
3200: 0.0233 / 30.1%           4800: 0.0184 / 24.0%
3600: 0.0249 / 27.7%           5600: 0.0168 / 21.0%
```

Retention was the stated discriminator. The binned run began declining *earlier*
than the old one, so the discriminator has been answered in the negative.

## Honest note

The early curve was expected to lift if the hypothesis were right, and it did
not. That is evidence against the hypothesis even before the retention window,
and should be recorded as such rather than deferred entirely to the later
comparison.
