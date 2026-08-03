# Prior knowledge: flow matching, shortcut models, and adapter composition are already fluent

Lukas arrives at PDD with a working command of rectified flow / flow matching, the
diffusion-vs-flow prediction-type distinction (`noise` / `velocity`), Frans et al.
shortcut models and their self-consistency target, and adapter composition
(`f_base + g(d)·Δ_φ`). He has already implemented a mean-velocity teacher-rollout
target in `shortcut_targets.py`, validated it on CPU, and queued a Wan run — i.e.
he can read this literature and write against it without scaffolding.

**Implication for teaching:** never re-teach the flow-matching or shortcut basics.
Lessons should start at the level of *what is different about this paper* and should
be decision-oriented — the value is in distinguishing PDD from what he already built,
not in explaining ODEs. The zone of proximal development is at the level of
on-policy vs off-policy target distributions, architectural parametrisation of the
jump size, and how these translate to a *frozen-base adapter* setting the paper
never considers.

**Depth claimed:** implementation-level for shortcut/flow; first exposure for PDD
specifically (paper is six days old).
