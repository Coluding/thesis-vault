---
type: decision
status: open
created: 2026-06-09
decided_at:
updated: 2026-06-09
target_date:
scope: data
related:
  - "[[../decided/metaworld-frame-stride-load-time]]"
  - "[[../decided/per-sample-frame-stride-sampling]]"
  - "[[../../20_Tickets/exp-conditioning-add-actions-to-shortcut-adapter]]"
  - "[[../../30_Knowledge/tech/frame-stride-conditioning]]"
  - "[[../../60_Updates/entries/2026-06-05-anchor-baseline-confirms-shortcut-fewstep-gain]]"
---

# Decision: Increase the load-time frame stride so actions visibly drive the dynamics

## Status

**Open, captured 2026-06-09.** This revisits — does not overturn — the
mechanism decided in [[../decided/metaworld-frame-stride-load-time]] (load-time
frame stride `k` with action SUM, constant `fs` to the base). That note set the
*default* at `k = 4`. The open question here is purely the **value of `k`**:
how far to raise it (and whether to sweep it) so a single conditioning action
produces a visible change in the predicted next frame. Not closing until a run
shows action-following emerging at the chosen stride.

## Trigger

On the smallest AVID adapter (11M params) the local shortcut runs predict
dynamics that look *somewhat random / not action-following*
([[../../60_Updates/entries/2026-06-05-anchor-baseline-confirms-shortcut-fewstep-gain]],
[[../../20_Tickets/exp-conditioning-add-actions-to-shortcut-adapter]]). The
action conditioning is confirmed *wired* (the adapter UNet is natively
`action_conditioned: True` and `a_t` reaches its action head), yet ineffective.
A leading structural suspect: at the current stride, consecutive kept frames
barely move, so the per-step (delta) action attached to a frame is **too
fine-grained to displace the image enough to be learnable / observable**. More
inter-frame motion gives each action more displacement to account for, which
should make the action signal both stronger to learn from and visible to
evaluate.

## Decision (proposed, not yet taken)

Raise the load-time stride `k` (`data.frame_stride`) above the current default
of 4 for the action-conditioned runs, and **sweep it** rather than guess a
single value. The existing mechanism already supports this with no code change:
`k` is a config knob, dropped delta-actions are SUM-ed into the kept frame's
action (so the conditioning stays the net control over the wider window), and a
constant `fs` is fed to the base regardless of `k`.

- **Candidate sweep:** `k ∈ {4, 8, 12}` (analysed-estimate starting grid).
- **Hard constraint:** the action window binds at `length · k ≤ 300`
  (MetaWorld episodes are 300 frames, 16-frame clips), so `k ≲ 18`. Stay well
  inside that.
- **Selection signal:** the smallest `k` at which action-following becomes
  visible (frame-delta responds to action-delta) *without* the clip skipping so
  much that motion becomes discontinuous / the base motion prior breaks down.

## Why this is a decision, not just a knob-tweak

- It changes the **headline data regime** for every D2/D4 action result, so the
  reported action-following numbers are stride-dependent and must be pinned and
  cited. Crossing it silently would make runs incomparable.
- It interacts with the base's motion prior: the decided note feeds `fs = 1`
  irrespective of `k` on the argument that MetaWorld's per-frame disparity makes
  the base fps-prior a poor fit anyway. A much larger `k` stresses that
  argument — worth confirming the adapter still absorbs the mismatch rather than
  fighting it.
- It is entangled with the action-effectiveness diagnosis: if a larger stride
  fixes the "random dynamics", that *confirms* the fine-grained-action
  hypothesis over the null-action / undertraining hypotheses, which redirects
  the rest of [[../../20_Tickets/exp-conditioning-add-actions-to-shortcut-adapter]].

## Open questions / what closes this

- The chosen `k` (or per-config `k`) once a run shows action-following emerging.
- Whether `k` should differ between the shortcut-robustness arm and the
  action-conditioned arm, or be held common for comparability.
- Whether to retain `fs = 1` to the base at large `k`, or revisit the constant-
  `fs` choice from [[../decided/metaworld-frame-stride-load-time]].

## Caveats carried forward

- This must be tested at a real scale, not just the 11M local model — the
  current evidence is qualitative and undertrained, so a larger stride and a
  larger model / more data move together (see the Snellius run next-step).
- Raising `k` shortens the covered fraction *per kept frame gap* trade-off:
  more motion per step but fewer distinct windows per episode; watch effective
  dataset size.
