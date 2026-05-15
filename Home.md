# Thesis Vault — Home

Second brain for the thesis **"Adapting Pretrained Generative Models into
Action-Conditioned World Models via Plug-and-Play Adapters."**

Sits next to the implementation at
`/home/lukas/projects/generative-flow-adapters/`. Read [[CLAUDE]] for the
operating spec before doing anything substantive in here.

## Thesis at a glance

Core composition rule:

```
f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)
```

`f_base` is a frozen pretrained diffusion or flow-matching model. `Δ_φ` is
the trainable adapter. `a_t` is the action; `d` is the step-size for
shortcut generation. Four deliverables — see
[[10_now/positioning|🎯 Thesis framing]].

## Now (living docs)

- [[10_now/architecture|🏛 Codebase architecture]] — what's actually built in `src/generative_flow_adapters/`
- [[10_now/product-state|📦 Experiment state]] — which runs have happened, which adapters work
- [[10_now/positioning|🎯 Thesis framing]] — contributions, deliverables, what makes this a thesis
- [[10_now/setup-status|🧭 Vault setup]] — what the second brain has and is missing

## Active work

- `20_Tickets/` — open experiments, bugs, sweeps, write-up TODOs
- `00_Inbox/` — fleeting thoughts before they're filed
- `50_Decisions/open/` — 🟡 unresolved design choices (which backbone, which adapter family for D2, etc.)
- `50_Decisions/decided/` — ✅ resolved (with derived experiments)

## Knowledge — by domain

### Research
- [[30_Knowledge/related-work/_MOC|📚 Related work]] — one file per paper that shapes the thesis
- `30_Knowledge/theory/` — diffusion / flow-matching / shortcut math notes
- `30_Knowledge/experiments/` — per-experiment write-ups (configs, metrics, plots)
- `30_Knowledge/sessions/` — work-session logs

### Codebase
- `30_Knowledge/tech/` — implementation notes on backbones, adapters, training loop, data pipeline
- `30_Knowledge/datasets/` — MetaWorld, video datasets, conditioning sources

### Writing
- `30_Knowledge/writing/` — thesis outline, chapter drafts, figures, citations
- `30_Knowledge/advisor/` — meeting notes, feedback, action items

## Generative queries (ask Claude these)

- "What experiment should I run next?"
- "What do we know about [HyperAlign / shortcut / UniCon / AVID]?"
- "What's the current story for chapter 3?"
- "What am I ignoring?" — stale tickets, orphan notes, unresolved decisions
- "Draft the related-work section for [topic]"

Claude reads the relevant slice of the vault, identifies gaps, grills you
on the gaps if needed (one question at a time), then produces the artifact.

## Vault meta

- [[CLAUDE|🤖 CLAUDE.md — operating spec]]
- `90_Meta/scripts/snapshot.sh` — vault git snapshot helper
