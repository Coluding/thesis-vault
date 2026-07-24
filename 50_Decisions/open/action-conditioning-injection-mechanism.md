---
type: decision
status: open
created: 2026-07-11
decided_at:
updated: 2026-07-14
target_date:
scope: architecture
related:
  - "[[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]"
  - "[[../../20_Tickets/feat-adapter-wan-action-cross-attention]]"
  - "[[../../20_Tickets/experiments/exp-conditioning-action-shuffle-ablation]]"
  - "[[../../20_Tickets/feat-adapter-dynamicrafter-output-on-wan-base]]"
  - "[[../../10_now/architecture]]"
  - "[[../../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]"
---

# Decision: WAN action-conditioning injection — AdaLN broadcast vs cross-attention

## Status

**Still open — resolution attempt on 2026-07-12 was invalid, must be re-run
(2026-07-14).** The build ticket ran (`xb76ptw2`,
[[../../20_Tickets/feat-adapter-wan-action-cross-attention]]) but the config
never set `action_seq_len`, so it fed unbinned raw-per-frame action tokens
against the latent grid with no temporal masking — exactly the failure mode
the "Non-negotiable coupling" section below warned about *in advance*. The run
is not evidence for or against option B. Re-run with `action_seq_len` pinned
to the latent frame count before treating this decision as resolved. Full
diagnosis: [[../../30_Knowledge/tech/why-adapter-underlearns-diagnosis]].

**Open — captured 2026-07-11.** Motivated by the 20260907 finding
([[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]):
the WAN output adapter produces coherent video and *some* task-directed behaviour,
but the action effect is weak ("adapter helps but not enough"). Hypothesis: the
current conditioning mechanism is structurally too weak. Nothing here is a result
yet — the ablation ticket resolves it.

**Scope: the WAN output adapter only** (`adapters/output/wan.py` →
`ActionWanModel`). Not about HyperAlign or other families.

## The question

How does the **action** enter the WAN output adapter?

The real axis is **global vs localized conditioning**:

- **AdaLN broadcast (current).** `backbones/wan/modules/action_model.py:5-7`:
  the action is *summed into the timestep embedding* and drives every block's
  AdaLN modulation — `e = time_embed(t) + cond_proj(c) + step_embed(d)`. One
  (scale, shift) per layer, **identical across all spatial/temporal tokens**. Can
  only move global feature statistics; structurally cannot say "move the arm, not
  the table." Fine for genuinely global scalars (timestep, step-size) — wrong
  inductive bias for localized action-driven motion.
- **Cross-attention context (proposed).** Each latent token attends to action
  tokens independently → **localized, content-dependent** control. The mechanism
  DynamiCrafter uses for image context; the right bias for action dynamics.

**Key enabling fact:** the adapter's blocks are already full `WanAttentionBlock`s
with a `t2v_cross_attn`, and a `use_text_context`/`text_embedding`→`context` path
(`action_model.py:54,61,109,115,118`). It's currently fed a **null/zero token**
(`use_text_context=False` → *"cross-attn becomes a no-op shift"*, line 218). So
the cross-attention slot **already exists and is unused** — this is a wiring
change, not a new architecture.

## Non-negotiable coupling — action representation

Cross-attention only beats AdaLN if there is something to attend *to*. Today the
action reaching the adapter is a single `[B,4]` summed vector; a lone KV token
attended by all queries **collapses back to a global bias** → cross-attn ≈ AdaLN
and the ablation is uninformative. So the cross-attn arm **must** use a
**per-frame action-token sequence** `[B, L, 4]` (`L` = latent temporal length,
Wan2.2 `L=(frame_num−1)/4+1`), obtained by binning the raw per-step deltas onto
the latent grid and summing within each bin. The current single vector is the
degenerate `L=1` case.

## Recommended cross-attn design (the arm to build)

Drop an `action_embedding` into `ActionWanModel`, mirroring `text_embedding`, and
feed its tokens as `context`:

```
[B, L, 4] → Linear(4→dim) → GELU → Linear(dim→dim)
          → + temporal position embedding (learned or sinusoidal)   # NOT optional
          → context tokens [B, L, dim] → WanAttentionBlock t2v_cross_attn
```

- **One token per latent frame** (`L`) so latent frame *t* can attend action *t*.
- **Temporal position embedding required** — cross-attn is permutation-invariant
  over KV; without it you lose trajectory ordering.
- **Per-frame delta** as the raw signal (cumulative optional as extra channels).
- **Zero-init the last Linear** for a clean identity start (belt-and-suspenders;
  the adapter's final projection is already zero-init).
- **`step_level` + timestep stay in AdaLN** (correct — global scalars).

## Options

| Option | Action path | Notes |
|---|---|---|
| **A — AdaLN (status quo)** | summed `[B,4]` into time-embed → AdaLN | global only; current weak baseline |
| **B — cross-attention** | `[B,L,4]` tokens → `t2v_cross_attn` context | localized + temporal; the bet |
| **C — both** | AdaLN global summary **+** cross-attn tokens | often best in practice; muddies the clean ablation → keep as a 3rd config |

## How this resolves

A clean ablation on the **same** WAN adapter / base / data, one flag
`action_injection: adaln | cross_attention | both`:
- Cross-attn (B) clearly beats AdaLN (A) on action-following (base-vs-adapted
  delta + the shuffle counterfactual + NFE-row grid) ⇒ decide B (or C).
- No difference ⇒ conditioning mechanism was **not** the binding constraint;
  redirect to capacity ([[../../20_Tickets/feat-adapter-dynamicrafter-output-on-wan-base]])
  / under-incentivisation / data. (Run the cheap shuffle + `‖g·Δ‖` probes first to
  know which world we're in.)

This is also a **thesis-worthy D1/D2 result** on adapter *conditioning mechanism*,
not just a fix.

## Consequences / coupled changes

- **Preprocessor** `data/wan_batch_preprocessor.py:_aggregate_action` needs a
  `[B,T,4]→[B,L,4]` binning mode (keep the sum, per-bin).
- `cond["action"]` shape becomes `[B,L,4]` on the cross-attn path.
- New config(s) `*_wan22_*_xattn_metaworld.yaml` with the `action_injection` flag.

## Derived tickets

- [[../../20_Tickets/feat-adapter-wan-action-cross-attention]] — build B (+ flag, + C).

## Related

- [[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]
- [[../../10_now/architecture]] §"Unified output adapter"
- [[../../20_Tickets/feat-adapter-dynamicrafter-output-on-wan-base]] — the *capacity*
  lever (a different answer to the same weak-action finding; DynamiCrafter's
  strength is itself cross-attention conditioning).
