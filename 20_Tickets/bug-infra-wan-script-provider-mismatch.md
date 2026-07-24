---
type: bug
scope: infra
status: open
priority: low
created: 2026-07-14
updated: 2026-07-14
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[feat-adapter-dynamicrafter-output-on-wan-base]]", "[[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]"]
---

# bug: most `diffusion_wan22_*` config headers point at the wrong training script

## What

There are two WAN2.2 training entry points with genuinely different base loading:

- `scripts/train_wan22_i2v_metaworld.py` (plain) — `provider: wan2.2` → the
  **vendored, hand-copied `Wan22DiTWrapper`**.
- `scripts/train_wan22_i2v_metaworld_external.py` — `provider: wan2.2_external` →
  the **real upstream `wan.WanTI2V`** from `external_repos/Wan2.2`, with genuine
  pretrained weights.

Confirmed via wandb run metadata (`run.metadata.program`/`args`): the "broken
base, adapter learned from scratch" control run
(`coluding/wan22-avid-i2v-metaworld34/l3p8kygb`) used the **plain** script and
almost certainly got a base with no real prior from it — this is very likely *why*
that base was broken. Both xattn runs (`cq3e83pj` proper-base, `xb76ptw2`
no-improvement) used `_external.py` and had a working prior.

**Every `diffusion_wan22_*` config's header except `gated_i2v` currently says**
`# Run with scripts/train_wan22_i2v_metaworld.py` **(the plain, wrong-base
script):** `diffusion_wan22_avid_i2v_metaworld.yaml`,
`diffusion_wan22_avid_i2v_metaworld_noshortcut.yaml`,
`diffusion_wan22_avid_xattn_i2v_metaworld.yaml`. (`diffusion_wan22_dcunet_output_metaworld.yaml`
was fixed 2026-07-14, see below.)

**Compounding issue:** the plain script also has **zero wiring** for
`training.extra.action_per_frame`/`action_seq_len` (grepped — no matches near its
`WanBatchPreprocessConfig(...)` call), unlike `_external.py` (fixed 2026-07-14).
So even where a config sets `action_per_frame: true`, running it via the plain
script silently ignores the flag.

## Why it matters

Anyone following a config's own documented "Run with" instruction for
`avid_i2v`/`avid_i2v_noshortcut`/`avid_xattn` gets the vendored base (unknown
prior quality — the one confirmed instance was fully broken) instead of the real
pretrained WAN. This is a landmine for any future run, and retroactively casts
some doubt on which base the historical `avid_i2v`/`avid_i2v_noshortcut` runs
actually used unless their wandb metadata is checked the same way.

## Fix

1. **Audit every historical run's `metadata.program`** for the affected configs
   (`avid_i2v`, `avid_i2v_noshortcut`) to know which base they actually ran
   against — don't assume from the config header.
2. **Fix the config headers** (`avid_i2v`, `avid_i2v_noshortcut`, `avid_xattn`) to
   say `_external.py` explicitly, mirroring the fix already applied to
   `diffusion_wan22_dcunet_output_metaworld.yaml`.
3. **Consider deprecating the plain script outright** (or adding a loud runtime
   warning) rather than leaving two entry points with silently different base
   quality and silently different flag support — the asymmetry is the actual
   hazard, not just the stale comments.

## Guardrails

- Don't just fix the comments and call it done — the plain script itself is the
  hazard as long as it exists un-flagged. Prefer removing/redirecting it over
  patching docs around it.

## Update (2026-07-14) — downgraded to low priority; root cause narrowed and dated

Re-verified with the user. Cross-referenced git history: the hard-fail-unless-
`--allow-random-base` check in `train_wan22_i2v_metaworld.py` (which would loudly
error rather than silently proceed on a missing checkpoint) was added in commit
`ed6e42c` (**2026-07-02**, "fixed flow base model generation"). `l3p8kygb` was
created **2026-06-29** — three days *before* that fix — so it very likely hit the
older, more permissive checkpoint-resolution path this commit patched. This is
strong (dated, code-verified) support for the *original* "random/broken base"
characterization of that run, not a new finding — just confirmation with harder
evidence than the earlier script-identity inference alone gave.

**Downgraded because:** the user confirmed they always use `_external.py` in
practice, so the plain script isn't a live risk to their actual workflow — the
config-header staleness is a documentation issue, not an active hazard. Config
headers are still worth fixing opportunistically, but this is no longer a
blocker for any real run.
