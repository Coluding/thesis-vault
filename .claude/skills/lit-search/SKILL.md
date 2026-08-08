---
name: lit-search
description: Find, verify and file literature for the thesis. Use when the user says "find papers on X", "what's the literature on X", "verify this citation", "we need citations for X", or when a section is blocked on missing related-work notes. Enforces the peer-reviewed-first rule, writes the vault note, and adds the verified entry to refs.bib.
---

# lit-search

Find literature, **verify it**, and file it so it can be cited. The output
is not a reading list; it is vault notes plus `refs.bib` entries that the
thesis can cite without further checking.

## The overriding rule

> **Cite the peer-reviewed version wherever one exists.** Only where no
> peer-reviewed publication exists may the arXiv preprint be the citation of
> record, and the entry must say so.

This is not a preference. A thesis bibliography full of preprints when
proceedings versions exist reads as a literature search that stopped at the
first search result. Many papers appear on arXiv a year before their venue,
so the arXiv listing alone is never sufficient evidence of status.

## Steps

1. **Search**, then **fetch the primary source**. A search snippet is not
   verification. Fetch the arXiv abstract page, the proceedings entry, or
   the publisher page.

2. **Establish venue status for every paper.** Report one of:
   - a peer-reviewed venue with year, and oral/spotlight if applicable
     (`ICLR 2023`, `NeurIPS 2024 (Oral)`, `Nature 640:647--653, 2025`)
   - `NO PEER-REVIEWED VENUE FOUND`, stated explicitly
   Check the arXiv comments field, the DOI, DBLP, and the venue's own
   proceedings site. arXiv often carries the venue in `comments`.

3. **Capture the full author list.** Complete and correctly spelled: it goes
   into a `.bib` file. Truncated lists must be marked
   `note = {Author list incomplete}`.

4. **Write the vault note** at `30_Knowledge/related-work/{slug}.md` using
   the paper-note frontmatter contract (CLAUDE.md Part 8). The note carries
   what the paper *claims*; the bib entry carries only metadata.

5. **Add to `70_Thesis/latex/refs.bib`**, in the block for the section that
   will cite it. Any field you could not confirm gets a `note`.

6. **Report** what was found, what could not be verified, and explicitly
   whether anything found **weakens or pre-empts a claim the thesis makes**.
   That last one matters more than coverage.

## Hard rules

- **Never report a paper you could not verify exists.** No guessed titles,
  venues, years or author lists.
- **`unverified` per field** rather than filling it in. An unverified field
  that is silently guessed is worse than a gap.
- **Separate what the paper claims from your inference.** Hard rule 7a
  applies to citations with full force: no "the paper probably means".
- **Say when we are scooped.** If a paper already makes a claim the thesis
  treats as novel, say so plainly and early. Finding it now is cheap;
  finding it at a viva is not.
- **Accuracy over coverage.** Eight verified entries beat twenty shaky ones.

## Parallelism

For a broad sweep, launch one agent per domain rather than one agent for
everything, and give each the thesis context it needs to judge relevance.
Each agent must be told the peer-reviewed-first rule explicitly.

## Where results land

| Output | Destination |
|---|---|
| what a paper claims, and its delta to us | `30_Knowledge/related-work/{slug}.md` |
| metadata only | `70_Thesis/latex/refs.bib` |
| a claim of ours that is now narrowed | the relevant `rubric/` note **and** the storyline |
| a landscape or tension across several papers | a synthesis note in `related-work/` |
