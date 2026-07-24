---
type: feat
scope: eval
status: open
priority: high
created: 2026-07-09
updated: 2026-07-09
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]", "[[experiments/exp-conditioning-action-shuffle-ablation]]", "[[feat-eval-base-vs-adapted-delta]]"]
---

# feat: interactive action-conditioning debug UI

## Idea

An interactive UI (likely **Gradio**, in-process with the model) to probe a
trained checkpoint's action-following behaviour by hand:

1. **Load** a model checkpoint (frozen base + adapter).
2. **Seed** from a real MetaWorld episode (initial frame + its GT action chunk).
3. **Edit the action chunk** — zero it, shuffle it, scale it, swap in a different
   episode's actions, or hand-tweak values.
4. **Generate** and view the video, **side by side**: `GT | base(frozen) |
   adapted(base+adapter)`, and `real-action | perturbed-action`.

## Why it matters (ties to the open diagnosis)

This is the **manual, visual form of the action-shuffle test**
([[experiments/exp-conditioning-action-shuffle-ablation]]) and the base-vs-adapted delta
([[feat-eval-base-vs-adapted-delta]]) for the 20260907 finding
([[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]): if
editing the actions doesn't change the video, the adapter ignores actions; if it
does, you can *see* how much — and localise "adapter helps but not enough."

Also a genuinely demoable thesis artifact (D2/D4): interactively steering a
pretrained world model with actions.

> **Quantitative metrics still required** (hard rule 8) — this tool is for
> qualitative probing/debugging, not for producing headline numbers. It
> complements, does not replace, the base-vs-adapted delta and shuffle ablation.

## Design decisions (user-confirmed 2026-07-09)

- **Framework: Gradio** — `gr.Blocks`, in-process with the torch model, native
  video widgets, side-by-side panels, shareable localhost link.
- **Action input: BOTH co-equal** — (a) seed-from-real-episode + one-click
  perturbation ops (zero / shuffle-order / scale ×k / swap-in-another-episode),
  and (b) a full manual per-timestep × per-dim editor. Not one-or-the-other.
- **Output layout:** `GT | base(frozen) | adapted` (3 panels) and a
  `real-action vs perturbed-action` compare (2 panels).
- **Base vs adapted:** needs a clean "adapter off" path (zero the gate / call base
  directly) — verify it exists in the scout.
- **Perf:** generation is GPU + multi-step; expect seconds-to-minutes per clip →
  progress bar, cache decoded frames, optional low-NFE "preview" mode.

## Location

Lives in the impl repo (`/home/lukas/projects/generative-flow-adapters/`), likely
`scripts/action_debug_ui.py` or a small `tools/` module — **not** in the vault.

## Status

Scoping the inference/checkpoint/conditioning/decode API in the impl repo first;
design + first Gradio cut to follow.
