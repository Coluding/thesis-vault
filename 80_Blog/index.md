---
last_updated: 2026-06-17
status: living
---

# 80_Blog — Blog Index

The **outward-facing writing layer** for things learned through the thesis,
shaped into standalone blog posts. Parallel to `60_Updates/` (meeting log) and
`70_Thesis/` (thesis draft); it *links into* the internal knowledge layers
(`30_Knowledge/`, `50_Decisions/`, `30_Knowledge/experiments/`) rather than
duplicating them.

Each post is a **subdir** `80_Blog/{slug}/` containing:

- `brainstorm.md` — the working doc: angle, audience, hook, key ideas,
  outline, what-I-learned. This is where we grill/brainstorm before drafting.
- `draft.md` — the post itself.

Create one with the `/blog` skill. The skill scaffolds the subdir, finds and
links related vault notes, then brainstorms with you and co-writes the draft.

**Hard rules still apply** (CLAUDE.md Part 3): no unsourced technical claims or
fabricated numbers — a blog can be informal, but any metric/result must trace
to a real run, and any paper claim to the paper. When unsure, write
`_needs verification_`.

## Status lifecycle

`idea` → `brainstorming` → `drafting` → `draft-complete` → `published`

## Posts (newest first)

| Created | Status | Topic | Post |
|---|---|---|---|
| 2026-06-24 | drafting | To condition my video model on a frame, I set its noise level to zero | [[diffusion-forcing/draft]] |
| 2026-06-17 | drafting | When increasing batch size crashes your training, but memory isn't the issue | [[cuda-grid-limits-in-pytorch/draft]] |
