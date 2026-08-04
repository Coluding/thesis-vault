# The conditional-variance decomposition is the frame for the whole D2 story

Established 2026-08-03. The user arrived at the decomposition themselves — first
by hypothesising that flow matching's determinism leaves an adapter no room to
steer, then by sharpening it to "noising gives many trajectories per data point."
That is the law of total variance in disguise, so the teaching job is not to
motivate the idea but to make the algebra precise and defensible under questioning.

Prior knowledge confirmed: comfortable with diffusion vs flow objectives,
prediction types, expectations. Do not re-teach ML basics.

Two corrections already delivered that future sessions should not re-litigate:
(1) both diffusion and flow matching have random pairing, so "one-to-many vs
one-to-one" is not the distinction — the real axis is how tightly `x_t` pins the
target; (2) sampling determinism (DDIM is an ODE too) is separable from the
training-time question.

Implications for what to teach next: the derivation is done (Lesson 1). The open
gap is the **normalisation** — turning `action_loss_gap` into a dimensionless ΔR²
so DC and Wan sit on one axis. That is both the next lesson and a real
deliverable, since no current metric permits the cross-backbone comparison the
thesis wants to make.
