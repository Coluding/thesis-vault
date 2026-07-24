---
name: blog
description: Start or continue a blog post in 80_Blog/ — the outward-facing layer for things learned through the thesis, shaped into standalone posts. Use when the user says "blog about X", "start a blog on X", "let's write a post on X", "brainstorm a blog topic", or wants to continue/draft an existing post. Scaffolds a per-topic subdir (brainstorm.md + draft.md) linked to existing vault notes, then brainstorms one-question-at-a-time and co-writes the draft.
---

# blog

Create or continue a blog post under `80_Blog/{slug}/`. This is the
**outward-facing writing layer** — posts that preserve what was learned
through the thesis, written for a general technical reader, not the advisor
(that's `60_Updates/`) and not the thesis itself (that's `70_Thesis/`).

Each post lives in its own subdir with two files:
- `brainstorm.md` — the working doc (angle, audience, hook, key ideas,
  outline, what-I-learned). Where the grilling/brainstorming happens.
- `draft.md` — the post.

## When to use
- "blog about {topic}", "start a post on {topic}", "let's write up {topic}".
- "brainstorm a blog on {topic}" → scaffold + go straight into brainstorm mode.
- "continue the {slug} blog", "draft the post now" → open the existing subdir.

## Two entry paths

### A. New post (topic not yet a subdir)
1. **Slugify** the topic (kebab-case, 3–6 words). Confirm the slug if ambiguous.
2. **Find related vault notes — do not invent links.** Grep the internal
   layers for the topic and pick the genuinely relevant ones:
   ```bash
   grep -ril "{topic-terms}" 30_Knowledge/related-work 30_Knowledge/theory \
     30_Knowledge/experiments 30_Knowledge/writing 50_Decisions
   ```
   These become the `sources:` and inline `[[links]]`. If nothing matches,
   say so — a blog with no vault grounding is a flag to brainstorm first.
3. **Create** `80_Blog/{slug}/brainstorm.md` and `80_Blog/{slug}/draft.md`
   from the templates below.
4. **Register** in `80_Blog/index.md`: prepend a row to the "Posts (newest
   first)" table:
   `| {created} | {status} | {title or topic} | [[{slug}/draft]] |`
5. **Go to brainstorm mode** (below).

### B. Continue an existing post
1. Read that post's `brainstorm.md` (and `draft.md` if drafting).
2. Resume where it left off — more brainstorming, or move to drafting.
3. Update `status` + `last_updated` in both files and the index row.

## Brainstorm mode (the core of this skill)

This is Discovery mode (CLAUDE.md Part 4) pointed at a blog. **One sharp
question at a time**, each pushing on the post's angle, hook, or a technical
claim. After each answer, **write it into `brainstorm.md` before the next
question**. Drive toward:

- **Angle / thesis of the post** — the one idea a reader leaves with.
- **Audience + assumed background** — sets how much to explain.
- **Hook** — why a reader cares in the first paragraph.
- **Key beats** — 3–6 points, each tied to a vault `[[source]]` where possible.
- **What I learned** — the personal/insight angle that makes it worth reading.
- **Figures** — reuse from `30_Knowledge/writing/figure-*` if relevant.
- **Outline** — ordered section list, ready to draft.

Stop when the outline is solid — don't drag. Then offer to draft.

## Drafting mode

Write `draft.md` from the brainstorm outline, section by section. Show the
draft, take one round of revision feedback, save. Set `status: draft-complete`
when the user is happy; `published` (+ `published_url`) once posted.

## Templates

`brainstorm.md`:
```yaml
---
title: ""
slug: {slug}
type: blog-brainstorm
status: brainstorming      # idea | brainstorming | drafting | draft-complete | published
created: {YYYY-MM-DD}
last_updated: {YYYY-MM-DD}
deliverable:               # D1 | D2 | D3 | D4 | exploratory | "—"  (thesis area it draws from)
sources: []                # [[links]] to 30_Knowledge / 50_Decisions notes
---

# Brainstorm — {topic}

**Angle:** _the one idea the reader leaves with_
**Audience:** _who + assumed background_
**Hook:** _why they read past line one_

## Key beats
- _point — [[source]]_

## What I learned
_the insight that makes this worth writing_

## Figures
- _[[../../30_Knowledge/writing/figure-...]] if reused_

## Outline
1. _section_
```

`draft.md`:
```yaml
---
title: ""
slug: {slug}
type: blog-draft
status: drafting           # idea | brainstorming | drafting | draft-complete | published
created: {YYYY-MM-DD}
last_updated: {YYYY-MM-DD}
deliverable:
sources: []                # [[links]] this post draws from
published_url:
---

# {title}

_draft — written from brainstorm.md_
```

## Guardrails
- **No unsourced technical claims, no fabricated numbers** (CLAUDE.md hard
  rules 7–8). A blog can be informal in tone, but any metric/result must trace
  to a real run (wandb id + ckpt + commit) and any paper claim to the paper.
  When a fact is missing, write `_needs verification_` — never guess as fact.
- **Link, don't duplicate.** Pull ideas from `30_Knowledge/` via `[[links]]`;
  don't copy whole notes into the post.
- **One topic per subdir.** A second angle is a second post.
- **Don't fabricate vault links.** Only `[[link]]` notes that actually exist
  and are relevant (step 2 grep). A topic with no vault grounding → brainstorm
  before drafting.
