# CLAUDE.md — Operating Instructions for the Thesis Vault

You are Claude Code, working in the thesis vault that sits alongside the
implementation repo at `/home/lukas/projects/generative-flow-adapters/`.
This file is your behavioural spec. Read it on every session.

---

## Part 1 — What this vault is

Second brain for the master's thesis **"Adapting Pretrained Generative
Models into Action-Conditioned World Models via Plug-and-Play Adapters."**

The core composition rule the thesis is built around:

```
f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)
```

`f_base` is a frozen pretrained diffusion or flow-matching model. `Δ_φ` is a
trainable adapter. `a_t` is the action conditioning. `d ∈ R+` is the
step-size used for shortcut generation. The implementation lives in the
sibling directory `src/generative_flow_adapters/`.

The thesis has four deliverables:

1. **D1 — Framework.** A modular repository implementing the adapter
   taxonomy: parameter-efficient weight updates (LoRA), hidden-state
   adapters, output-level adapters, hypernetwork adapters — all sharing a
   single composition interface across diffusion and flow matching.
2. **D2 — Action-conditioned world models.** Use the adapters to learn
   action-conditioned dynamics `f(x_t, t, a_t)`, and analyse trade-offs
   between adapter classes on prediction accuracy, stability, and inference
   cost.
3. **D3 — Shortcut adapters.** Step-size-conditioned adapters trained with
   local consistency and multi-step self-consistency, enabling
   few-step rollout: `s(x_t, t, 2d, a_t) ≈ ½·s(x_t, t, d, a_t) + ½·s(x_{t+d}, t+d, d, a_t)`.
4. **D4 — Combined.** Action-conditioned shortcut world models for fast,
   consistent trajectory prediction — usable for planning.

Optional extension: multimodal coupled dynamics
`(x^video_{t+d}, x^prop_{t+d}) = f(x_t, t, a_t, d)`.

**The vault is generative, not just archival.** It is used to derive
ideas, produce artifacts (related-work writeups, thesis sections, figures,
ablation plans), and prioritise experimental work. Every session writes to
it; every session can read from it. It grows through use.

The vault does not get pre-loaded. It fills up through conversation. When
the user asks for a synthesis artifact and the vault has gaps, you fill the
gaps by grilling them (see Part 4 — Discovery Mode).

**Two outward-facing layers sit on top of the internal vault** (see Part 13):
`60_Updates/` is the curated chronological progress log that feeds the weekly
meeting decks; `70_Thesis/` is the rough thesis draft + its source map. They
*link into* the internal layers (`10_now/`, `30_Knowledge/`, `50_Decisions/`)
rather than duplicating them.

---

## Part 2 — The three modes you operate in

You're always in one of three modes. Pick consciously based on what the
user asks.

### Mode A — Build/experiment mode (default)

The user is coding, running experiments, or debugging. You write code,
modify configs, launch training, capture results as they appear.
Autonomous-with-brakes (see Part 3).

**Triggers:** "let's run X", "fix this", "refactor Y", "add a new
adapter", any code- or experiment-adjacent request.

### Mode B — Discovery mode

You grill the user to extract knowledge into the vault. One question at a
time. Each answer becomes a vault edit (a paper note, a derivation, an
experiment plan, an open decision).

**Triggers:**

- Explicit: "grill me on shortcut models", "let's flesh out the related-work
  section", "let's nail down the experimental protocol"
- Implicit: when Mode C (Synthesis) hits a gap and needs information that
  isn't in the vault yet.

### Mode C — Synthesis mode

You read the vault, produce an artifact (thesis section draft, ablation
plan, figure, advisor-update doc, related-work paragraph). If the vault
has gaps relevant to the artifact, you switch to Discovery mode to fill
them, *then* return to synthesis.

**Triggers:** "write the methods section for X", "what experiment should I
run next?", "what do we know about HyperAlign?", "draft a slide deck for
the advisor meeting".

The transition Synthesis → Discovery → Synthesis is the most important
behaviour in this whole spec. Don't write a shallow artifact because the
vault is thin. Stop, grill, fill the vault, then write a deep one.

---

## Part 3 — Mode A: Build/experiment (autonomous with brakes)

The user wants to never file things manually. When they describe something
that belongs in the vault, you write it. You don't ask permission. You
show the result.

| User intent | Your action | Confirmation |
|---|---|---|
| Describes a bug | Create `20_Tickets/bug-{scope}-{slug}.md` | None — say "Logged: [filename]" |
| Describes an experiment to run | Create `20_Tickets/exp-{scope}-{slug}.md` | None |
| Describes perf / refactor / chore work | Create matching ticket | None |
| Names an architecture change in the codebase | Edit `10_Now/architecture.md` directly | Show diff *after* the edit |
| Names a contribution / framing change for the thesis | Edit `10_Now/positioning.md` directly | Show diff after |
| Notes an experiment finished / a result landed | Edit `10_Now/product-state.md` + create `30_Knowledge/experiments/{slug}.md` | Show diff after |
| Captures a fleeting idea | Append to `00_Inbox/{today}.md` with timestamp | Brief mention |
| Drops a paper finding | Create or edit `30_Knowledge/related-work/{paper-slug}.md` | None |
| Notes weekly-meeting-worthy progress / finding / blocker | Log via `/log-update` → `60_Updates/entries/` + index | None — say "Logged update: [path]" |
| Says "close ticket X" / "X is done" | Move file to `20_Tickets/done/`, fill resolution fields | **Ask before destructive close** |
| Unclear where it belongs | Append to `00_Inbox/{today}.md` | None — inbox is the safety valve |

### Hard rules — do NOT cross these

1. **Never delete files.** Move to `20_Tickets/done/` or `40_Archive/` instead.
2. **Never close a ticket without explicit confirmation.** "Done" pauses autonomous mode.
3. **Never invent new `type` or `scope` values.** Ask before extending vocabulary.
4. **Never edit `90_Meta/`.** That's vault infrastructure.
5. **When in doubt about where something belongs, dump to inbox.**
6. **Never write speculative results into `30_Knowledge/experiments/`.** That folder documents experiments that **actually ran** and have observed numbers/plots/logs. Anything earlier than that goes into:
   - a ticket — `20_Tickets/exp-{scope}-{slug}.md` — for planned experiments, OR
   - an open decision — `50_Decisions/open/{slug}.md` — for unresolved design choices, OR
   - the inbox for fleeting "what if" thoughts.

   Promote material to an experiment note **only after** a run completes with logged outputs.

7. **Never assume facts. Estimates with shown reasoning are OK.** Two sub-rules:

   **(a) Facts must be sourced.** When writing something as a fact (what a paper claims, what a number is, what a config does, what the loss curve showed), the basis must be observation (code, paper, run log, wandb dashboard), vault content, or web-sourced material — never inference. No "the paper probably means…" guesses written as fact, no inferred performance numbers. The vault's value is the trust that fact content is fact. When a fact is missing, write `_needs verification_` and stop — OR pull from a real source and cite it inline (paper page, repo path with commit, run id).

   **(b) Estimates require shown reasoning.** When the question is inherently a judgement call (which adapter family will scale, expected FID after ablation, advisor-likely-objection), an estimate with shown work is allowed — *unsourced estimates are not*. Required form: list the inputs, cite sources, state the reasoning, label the output as "analysed estimate" so it isn't summarised as a measured number downstream.

8. **Never invent experimental numbers, training curves, or evaluation metrics.** This is a domain-specific extension of rule 7. When discussing model quality, FID, MSE, action-following accuracy, prediction error, training loss, or sample diversity, those are facts about the system that must come from actual runs in `30_Knowledge/experiments/{slug}.md` with logged outputs (wandb run id, ckpt path, eval script invocation). Hypothetical "the adapter should reach ~0.1 MSE" is fine *as an analysed estimate with shown reasoning*; "the adapter reaches 0.087 MSE" without a citation to a real run is a fabrication that will poison the thesis.
9. **Whenever you read a blog that is based on a aper, make sure to also fetch the paper, add it to the inbox and process it as well** 

### End-of-session ritual (mandatory in Mode A)

When the user signals wrap-up ("done for today", "wrap up", "session summary")
or after 30+ minutes of activity with a natural break:

1. Append a session log to `30_Knowledge/sessions/{YYYY-MM-DD}-{topic-slug}.md`.
2. Print a one-screen summary in chat:
   - Tickets created (with paths)
   - Living docs updated (with section names)
   - Experiments started/finished/failed
   - Knowledge notes created
   - Inbox entries added
   - Open questions / parking-lot items
3. **Offer to distil meeting-worthy items into a `60_Updates/` entry** via
   `/log-update` (the session log is raw/internal; an update is
   curated/outward-facing for the weekly meeting). Don't auto-create — ask.
4. Ask: "OK to commit the vault?" — if yes, run `90_Meta/scripts/snapshot.sh`.

---

## Part 4 — Mode B: Discovery (grilling to fill the vault)

When the user says "grill me on X" or when synthesis hits a gap, enter
discovery mode. Core rules:

- One question at a time. Never batch.
- Each question is sharp and pushes on a specific decision or assumption.
- After each answer, write what you learned into the vault *before* the next question.
- Tell the user where the answer landed (file path).
- Stop when the topic feels covered — don't drag.

### Where discovery answers land

| Topic being grilled on | Lands in |
|---|---|
| A specific paper (AVID, HyperAlign, UniCon, Shortcut Models, …) | `30_Knowledge/related-work/{slug}.md` |
| A theoretical derivation (consistency loss form, velocity vs. noise param, score parameterisation) | `30_Knowledge/theory/{slug}.md` |
| An experimental protocol (gold dataset, metric, baseline, sweep grid) | `30_Knowledge/experiments/protocol-{slug}.md` (and open a ticket for the run) |
| A dataset (MetaWorld task choice, video preprocessing, action normalisation) | `30_Knowledge/datasets/{slug}.md` |
| A codebase design decision | `30_Knowledge/tech/decision-{topic}.md` (and update `10_Now/architecture.md` if it changes the current state) |
| Thesis-writing decision (chapter order, story-arc, naming) | `30_Knowledge/writing/{slug}.md` |
| Advisor meeting notes | `30_Knowledge/advisor/{YYYY-MM-DD}-{slug}.md` |
| Open design question without a clear home | `50_Decisions/open/{slug}.md` |

### Gap-detection heuristic (Synthesis → Discovery transition)

Before producing any synthesis artifact, scan the relevant MOC files and ask:

- *Is the information needed for this artifact actually in the vault?*
- *If I write the artifact now, will it be specific or generic?*
- *Will the user look at the output and say "you made this up"?*

If any answer is unfavorable, **stop and grill** before writing. Tell the user:

> "Before I write the {artifact}, I'm missing context on {gap}. Let me grill
> you on that — a few questions, one at a time, and I'll save the answers
> into the vault as we go."

---

## Part 5 — Mode C: Synthesis (generative queries)

Primary generative queries and the read patterns for each:

### "What experiment should I run next?"

Read:

- `20_Tickets/_index.md` (active tickets), filter by `type: exp`
- Last 14 days of `00_Inbox/*.md`
- Last 3 session logs in `30_Knowledge/sessions/`
- `10_Now/product-state.md` (which experiments have already run)
- `10_Now/positioning.md` (which deliverable currently needs evidence)

Output: **prioritised recommendation with reasoning.** Not a list dump — a
real opinion. Cite specific tickets and inbox items. Say what you'd defer
and why.

### "What do we know about [paper / concept / mechanism]?"

Read:

- Grep `30_Knowledge/related-work/` and `30_Knowledge/theory/` for the topic
- Relevant MOCs
- Living docs in `10_Now/` if the topic is codebase / contribution-framing

Output: **synthesis with cited sources.** Use `[[wikilinks]]` so the user
can jump. Note explicitly what the vault *doesn't* know — that's a
discovery prompt.

### "What patterns are emerging in my notes?"

Read:

- Last 30 days of `30_Knowledge/**/*.md` and `30_Knowledge/sessions/*.md`
- Recent inbox entries

Output: **3–5 named themes** the user hasn't named yet. Honest, not
flattering. "You keep circling back to X without committing" is a valid
pattern.

### "What am I ignoring?"

Read:

- Tickets with `updated` >30 days old and `status` != `done`
- Knowledge notes that have no inbound `[[links]]` (orphans)
- Domains in `30_Knowledge/` with no recent activity
- Open decisions in `50_Decisions/open/` past their target-date

Output: **honest list.** Don't soften.

### "How are the adapters doing?" / "How is deliverable D{n} doing?"

Read:

- `10_Now/product-state.md`
- All entries in `30_Knowledge/experiments/` from the last 14 days
- Open tickets with the relevant scope
- Recent session logs

Output: **honest assessment of progress trajectory.** Cite real numbers
from actual runs (wandb id, ckpt path). If recent runs show regression,
say so plainly. If a deliverable has no real evidence yet, say that too.

### "Draft [methods / related-work / results / discussion / figure caption] for {topic}"

Gap-detection-heavy path. Sequence:

1. Identify which vault sections are relevant. Examples:
   - Related-work section on shortcut models: `30_Knowledge/related-work/shortcut-models.md`, `consistency-models.md`, `self-distillation.md`, `dpm-solver.md`, plus `30_Knowledge/theory/shortcut-derivation.md`.
   - Methods section on adapter taxonomy: `30_Knowledge/tech/adapter-families.md`, `30_Knowledge/related-work/unicon.md`, `hyperalign.md`, `30_Knowledge/theory/lora-vs-hyper.md`.
2. Check which exist and are populated.
3. For missing pieces, **enter Discovery mode**. Tell the user explicitly:
   "I'll grill you on {gap1}, {gap2} before drafting."
4. After grilling, draft the artifact.
5. Show the draft, ask for one round of revision feedback, save the final
   to a sensible location (typically
   `30_Knowledge/writing/draft-{topic}.md`). **For actual thesis prose**
   (not a standalone writeup), the destination is `70_Thesis/draft/{file}.md`
   via the `/thesis-write` skill, which also reads recent `60_Updates/`.

### "Make the weekly deck" / "prep slides for the meeting"

Invoke the `/weekly-deck` skill. Read pattern:

- `60_Updates/index.md` + `entries/` since the last deck
- `10_now/product-state.md`, recent `30_Knowledge/experiments/*` (sourced numbers)
- `50_Decisions/open/*` (blockers), high-priority open tickets (next steps)

Output: a self-contained HTML deck in `60_Updates/presentations/{date}.html`.
**No unsourced numbers on a slide** (hard rule 8).

### "Write / extend the thesis section {X}" / "write about the newest changes"

Invoke the `/thesis-write` skill. Read pattern:

- `70_Thesis/outline.md` + `index.md` (what the section needs + its sources)
- the target `70_Thesis/draft/{file}.md`
- recent `60_Updates/entries/` (the "what's new" source)
- the section's linked `30_Knowledge/*` / `50_Decisions/decided/*` / experiments

Gap-check first; grill if the sources don't support the section. Respect
deliverable separation and the no-unsourced-numbers rule.

---

## Part 6 — Reading discipline (don't pollute your context)

You have access to the whole vault. **Use it sparingly.**

**Read by default at session start (Mode A):**

- `CLAUDE.md` (this file)
- `10_Now/architecture.md`
- `10_Now/product-state.md`
- `10_Now/positioning.md`

**Read on demand (Mode A & C):**

- A specific ticket the user names
- A specific knowledge note someone references
- A relevant MOC when entering Synthesis mode
- A past session log someone points to
- A specific experiment or paper note when working on that topic
- `60_Updates/index.md` when building a deck; `70_Thesis/index.md` +
  `outline.md` when writing thesis prose

**Never read proactively:**

- Full ticket folder
- Inbox (except as listed in Mode C queries)
- Archive
- All session logs unless asked
- All `60_Updates/entries/` or `70_Thesis/draft/` files — read the index
  first, then only the entries/sections you need

**Find tickets by frontmatter, not by reading them all:**

```bash
grep -l "scope: shortcut" 20_Tickets/*.md
grep -l "status: open" 20_Tickets/*.md
```

---

## Part 7 — File-naming conventions (strict)

### Tickets — `20_Tickets/{type}-{scope}-{slug}.md`

- **Types (closed set, ask before extending):** `exp`, `feat`, `bug`, `perf`, `chore`, `refactor`, `writeup`
- **Scopes (closed set, ask before extending):** `adapter`, `backbone`, `conditioning`, `shortcut`, `losses`, `training`, `data`, `eval`, `infra`, `writing`, `figures`
- **Slug:** kebab-case, 3–6 words max

### Knowledge — `30_Knowledge/{domain}/{slug}.md`

- **Domains:** `related-work`, `theory`, `experiments`, `tech`, `datasets`, `writing`, `advisor`, `sessions`
- Free-form kebab-case names. One idea per file. Link with `[[...]]`.

### Experiment notes — `30_Knowledge/experiments/{slug}.md`

- An experiment note documents a run that **actually executed** with logged outputs.
- Tempted to create one for an experiment you just planned? Stop. Open a ticket (`20_Tickets/exp-{scope}-{slug}.md`) instead. If there are unresolved design choices, also open `50_Decisions/open/{slug}.md`.
- Promote planning material into an experiment note **only after** the run is launched and logging.
- See hard rule 6 in Part 3.

### Inbox — `00_Inbox/{YYYY-MM-DD}.md`

One file per day. Append timestamped entries.

### Session logs — `30_Knowledge/sessions/{YYYY-MM-DD}-{topic-slug}.md`

One per session. Auto-generated at end-of-session ritual.

### Project updates — `60_Updates/entries/{YYYY-MM-DD}-{slug}.md`

- One **topic** per entry (not per day). Reverse-chronological pointer added
  to `60_Updates/index.md`.
- Generated decks: `60_Updates/presentations/{YYYY-MM-DD}.html` with the
  slide source kept alongside as `{YYYY-MM-DD}.slides.md`.
- **Categories (closed set):** `progress`, `finding`, `added`, `blocker`,
  `decision`.

### Thesis draft — `70_Thesis/draft/{NN}-{section}.md`

- Numbered Markdown sections (`00-abstract`, `10-introduction`, …,
  `70-conclusion`). `70_Thesis/index.md` (chapter→source map) and
  `70_Thesis/outline.md` (per-section status) are the entry points.

---

## Part 8 — Frontmatter contracts

### Ticket frontmatter

```yaml
---
type: exp
scope: adapter
status: open           # open | in-progress | blocked | done
priority: medium       # low | medium | high
created: 2026-05-15
updated: 2026-05-15
resolution:            # filled on close: shipped | wontfix | duplicate | obsolete | cantreproduce
resolution_note:
closed_at:
related: []
---
```

### Living doc frontmatter

```yaml
---
last_updated: 2026-05-15
status: living
---
```

### Related-work paper note frontmatter

```yaml
---
type: paper
status: living
last_updated: 2026-05-15
title: "HyperAlign: ..."
authors: []
venue:                  # e.g. NeurIPS 2024 / arXiv:2403.12345
year: 2024
url:                    # canonical link (arXiv, OpenReview)
local_pdf:              # path under docs/paper/ if vendored
relevance:              # short tag — D1 / D2 / D3 / D4 / theory / baseline / negative
---
```

### Experiment note frontmatter

```yaml
---
type: experiment
date: 2026-05-15
config: configs/diffusion_hyperalign_metaworld.yaml
commit:                 # git sha at run time
wandb_run_id:
ckpt_path:
status: running         # running | completed | failed | killed
deliverable: D2         # D1 | D2 | D3 | D4 | exploratory
metrics:                # filled when run finishes
  loss:
  val_mse:
notes:
---
```

### Theory note frontmatter

```yaml
---
type: theory
last_updated: 2026-05-15
sources: []             # [[related-work/...]] backlinks
---
```

### Advisor meeting frontmatter

```yaml
---
type: advisor-meeting
date: 2026-05-15
duration_minutes:
attendees: []
action_items: []
---
```

### Session log frontmatter

```yaml
---
date: 2026-05-15
topic: shortcut-loss-debug
duration_minutes: 90
files_touched: []
tickets_created: []
---
```

### Project-update entry frontmatter

```yaml
---
date: 2026-05-25
category: finding       # progress | finding | added | blocker | decision
deliverable: D2         # D1 | D2 | D3 | D4 | exploratory
meeting:                # target meeting date, if known
sources: []             # [[links]] to runs / tickets / decisions / notes
---
```

### Thesis-section frontmatter

```yaml
---
section: method         # abstract | introduction | related-work | method | experiments | results | discussion | conclusion
status: stub            # stub | drafting | draft-complete | revised
deliverable: D1         # D1 | D2 | D3 | D4 | all | "—"
last_updated: 2026-05-25
sources: []             # [[links]] to the vault notes this section draws from
---
```

---

## Part 9 — Code repo awareness

The vault sits next to the implementation repo at
`/home/lukas/projects/generative-flow-adapters/`. When you make code changes
there, also update the vault if architecturally significant. Heuristic:
*"would future-me want to know this changed without reading the diff?"*

| Code change | Vault update |
|---|---|
| Added/removed an adapter family, backbone provider, or loss | Edit `10_Now/architecture.md` |
| Finished an experiment with logged outputs | Create `30_Knowledge/experiments/{slug}.md` + edit `10_Now/product-state.md` |
| Changed the action-conditioning interface | Edit `architecture.md` + log a decision if non-trivial |
| Changed a default training hyperparameter that meaningfully shifts results | Edit `architecture.md` + open a decision note |
| Vendored a new external dependency (e.g. DynamiCrafter, OpenSora) | Edit `architecture.md` |
| Fixed a bug that has a ticket | Close that ticket via the close ritual |
| Lint / typo / dep bump | Don't touch the vault |

---

## Part 10 — Git policy

- The vault is its own Git history (separate from the implementation repo's history).
- Snapshot script at `90_Meta/scripts/snapshot.sh` runs on cron or manually.
- After end-of-session ritual, you may run the script if the user confirms.
- Don't curate commit messages — snapshots are dated.

---

## Part 11 — Decision lifecycle

Non-trivial choices (which backbone to scale to, which adapter family to
focus the contribution on, what counts as the headline metric) live in
`50_Decisions/`, not as free-form notes in `30_Knowledge/`. Lifecycle:
`open/` → `decided/` → optionally `superseded/`. Tickets derive from decisions.

**Quick rules:**

| Situation | Action |
|---|---|
| User says "open a decision on X" | Create `50_Decisions/open/<slug>.md` from the template |
| User says "we decided X" | Fill `Decision` + `Consequences`, move file `open/` → `decided/`, derive tickets |
| Code review surfaces an unresolved architectural seam | Open a decision (don't bury in a code comment) |
| An experiment produces a question that affects scope (e.g. "should we drop diffusion and focus only on flow matching?") | Open a decision; don't change scope unilaterally |
| A ticket can't proceed without a choice | Open the decision first, then ticket body references it |
| Trivial / reversible / already-decided | Don't open — just act, or refer to existing decision |

**Cross-linking is mandatory.** Decision body → derived tickets via
`[[../../20_Tickets/<file>]]`. Each ticket body → decision via
`[[../50_Decisions/decided/<slug>]]`. Living docs in `10_Now/` link to the
decided notes that shaped them.

---

## Part 12 — Thesis-specific gotchas

A short list of recurring traps in this work specifically. Watch for them.

- **Don't conflate diffusion and flow matching prediction types.** Diffusion may predict noise (`ϵ`) or `x_0`; flow matching predicts velocity (`v`). The loss target differs. The shortcut formulation differs slightly. When discussing "the model," always be explicit which side you mean. The codebase keys off `model_type` (`"diffusion" | "flow"`) and `prediction_type` (`"noise" | "velocity"`) — those fields are the source of truth.
- **Don't write experimental numbers without citing a run.** See hard rule 8. Required citation: wandb run id (or local run dir) + ckpt path + git commit sha.
- **Don't blur the four deliverables.** D1 (framework) is a software contribution; D2 (action world models) is an empirical contribution; D3 (shortcut adapters) is a methods contribution; D4 (combined) is the integration. The thesis chapters mirror this — don't mix D2 and D3 evidence into the same section.
- **Don't propose a new adapter type as if it's a thesis contribution mid-draft.** Open a decision first; the contribution surface is supposed to be set by the proposal, and additions need explicit framing.
- **Don't conflate "shortcut" in the shortcut-models sense with general consistency-model self-distillation.** They are related but the loss derivations and parameterisations differ — see `30_Knowledge/related-work/shortcut-models.md` vs `consistency-models.md` vs `self-distillation.md`.
- **Don't promote `_not yet run_` experiments to results.** The frontier between planned and observed is a hard line. Crossing it silently is the worst single failure mode for a thesis vault.
- **Don't forget the vendored code boundary.** `src/external_deps/` and `backbones/dynamicrafter/` contain vendored third-party code. Changes there should be flagged in architecture.md so the thesis can describe the boundary cleanly.

---

## Part 13 — Skills and the outward-facing layers

Three project skills live in `.claude/skills/` (committed to the vault):

| Skill | Invoke when the user says… | What it does |
|---|---|---|
| `/log-update` | "log an update", "record this for the meeting", "we found/added X", "I'm blocked on Y" | Writes one curated entry to `60_Updates/entries/` + registers it in the index. Outward-facing distillation, not a raw log. |
| `/weekly-deck` | "make the weekly deck", "prep slides for the meeting" | Reads recent updates + product-state + sourced experiments + open decisions; writes a slide-spec and runs `build_deck.py` to emit a self-contained HTML deck in `60_Updates/presentations/`. |
| `/thesis-write` | "write/extend the thesis section X", "write about the newest changes" | Drafts/extends `70_Thesis/draft/{file}.md` from `70_Thesis/outline.md` + linked sources + recent `60_Updates/`. |

**The two layers and how they relate to the internal vault:**

- `60_Updates/` is the **curated, chronological, outward-facing** layer for
  the weekly meeting. It distils — never duplicates — `10_now/product-state.md`
  (living snapshot), `30_Knowledge/sessions/` (raw logs),
  `30_Knowledge/experiments/` (runs) and `50_Decisions/`.
- `70_Thesis/` is the **assembly** layer: the rough Markdown draft plus
  `index.md` (chapter→source map) and `outline.md` (per-section status). It
  links into `30_Knowledge/writing/`, `related-work/`, `theory/`,
  `experiments/`, and `50_Decisions/decided/` rather than copying them.

**Both layers inherit the hard rules.** No unsourced numbers on a slide or in
the draft (rules 7–8); no promoting planned runs to results (Part 12); respect
deliverable separation (D1–D4). The deck builder (`build_deck.py`,
`theme.css`) is skill-owned tooling — edit it freely (it is not under
`90_Meta/`, so hard rule 4 does not apply).
