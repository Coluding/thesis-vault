# Teaching notes — PDD workspace

## Workspace location

The `/teach` skill says "treat the current directory as the workspace". The current
directory is the thesis vault, whose root is a strictly-numbered structure
(`00_Inbox` … `90_Meta`), so dropping seven loose files there would have been
vandalism. The workspace lives at `85_Learning/` instead — between `80_Blog`
(public writing) and `90_Meta` (infrastructure). All paths inside are relative, so
it is movable if Lukas prefers otherwise.

**This directory is NOT governed by the vault's CLAUDE.md content rules** in the
sense that it is a learning scaffold, not a knowledge claim — but the *sourcing*
discipline (hard rules 7–8) still applies and is applied: every equation in Lesson 1
and the cheat sheet was transcribed from `pdftotext` of the v1 PDF, not recalled.

## Observed preferences (inferred — confirm or correct)

- Wants precision about **what a result does and does not support**. Has repeatedly
  pushed back on over-claiming, and the vault is littered with self-issued
  corrections. Lessons should therefore be explicit about claim discipline, not just
  mechanism.
- Compute-constrained and deadline-constrained. Lessons that don't change a decision
  are not worth his time right now.
- Prefers being told the divergence bluntly over being reassured.

## Open teaching threads

1. **Mission unconfirmed.** `MISSION.md` was drafted from session context, not an
   interview. Confirm before Lesson 2.
2. **Lesson 2 candidate:** on-policy vs off-policy target distributions — why PDD
   grades at X̄_k, what exposure bias costs in a doubling tower, and whether the
   frozen-base adapter can be made on-policy cheaply. Directly decides what to do
   with the remaining SBUs.
3. **Lesson 3 candidate:** the parametrisation question — head-indexing (PDD) vs.
   step-size conditioning (shortcut models, ours). This is the one an advisor will
   ask about, and it is cheap to test on an *output-level* adapter, where the "final
   linear layer" is already tiny.
4. **Unread:** PDD Appendix C (connection to flow maps) and D (Prop. 1 proof).
   Worth reading if Lesson 3 goes ahead.
5. **Glossary not started.** Per GLOSSARY-FORMAT, terms go in only once Lukas can use
   them correctly — wait for evidence from the Lesson 1 quiz/recall.
