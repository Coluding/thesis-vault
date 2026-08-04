# Mission: The conditional-variance view of conditioning

## Why

The thesis' central empirical finding is that a plug-and-play adapter on a frozen
video model becomes an excellent **domain corrector** and a poor **action
conditioner** — and every attempt to fix it by moving where the action enters
(cross-attention, concat, adaLN, gain normalisation) landed in the same place.
The conditional-variance decomposition is the tool that turns that pile of null
results into a single structural explanation, and it has to be defensible in a
viva in ~11 days.

## Success looks like

- Derive, from scratch, why L2 regression can only ever learn `E[Y | X]` — and
  therefore why the residual is `Var[Y | X]`
- State and apply the law of total variance to split "how much room is there" from
  "how much of that room actions explain"
- Explain, unprompted, why this makes the action ceiling a property of the **data
  and representation** rather than of the architecture — and why that predicts the
  injection-site experiments were doomed
- Explain **explaining-away**: why conditioning on `x_t` shrinks what actions can
  contribute, and why the action-only (blind) arm therefore has a *higher* ceiling
- Convert the campaign's 0.45%-of-loss number into `ΔR²` and defend the
  normalisation choice under questioning

## Constraints

- ~11 days to thesis submission; lessons must be short and immediately usable
- Strong ML background: comfortable with diffusion/flow objectives, expectations,
  gradients. Do not re-teach basics.
- Every claim must be citable — the vault forbids unsourced facts, and so does the
  viva

## Out of scope

- Measure-theoretic probability. Intuition + the algebra that survives scrutiny.
- Estimating conditional variance non-parametrically; we use the model as the
  conditional-mean estimator and the loss as the residual.
- The SDE/ODE sampling question — related, but a separate thread.
