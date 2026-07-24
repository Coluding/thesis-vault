---
type: feat
scope: adapter
status: open
priority: high
created: 2026-05-22
updated: 2026-05-22
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/decided/param-matched-adapter-comparison-definition]]"
  - "[[experiments/exp-adapter-param-matched-comparison]]"
---

# Per-family `flops_per_step(input_shape)` estimator

Add a FLOPs-per-training-step estimator to each adapter family in the
framework so the FLOPs-matched comparison
([[../50_Decisions/decided/param-matched-adapter-comparison-definition]])
can pre-register its sweep grid.

## Why

The decision picked **training-time FLOPs as the primary matched axis**
for the cross-family comparison. Without a per-family FLOPs estimator,
the sweep cells (3 budgets x 4-5 families x 3 seeds) are not
computable and the protocol is not pre-registrable. This ticket is the
gate on the run-ticket
[[experiments/exp-adapter-param-matched-comparison]] moving from `blocked` →
`open`.

## What

Each adapter family must expose:

```python
def flops_per_step(self, input_shape: tuple) -> int:
    """Forward + backward FLOPs added by this adapter, per training step.

    Does NOT include the frozen-base forward FLOPs (which are constant
    across families). Returns only the family-specific delta so the
    matched axis is family-attributable cost.
    """
```

Coverage required:

- AVID-style output residual: extra MLP/projection + residual add.
- LoRA: rank-r matmul deltas summed across injected layers.
- HyperAlign-style hypernetwork: separate hypernet forward + the
  generated-weight matmul deltas it produces.
- UniCon-style hidden-state: per-injection-point projection cost.
- Full-FT subset: forward + backward FLOPs of the unfrozen subset.

Inference variant (`flops_per_inference_step`) for the reported
secondary column in the comparison table — same families.

## Where

`src/generative_flow_adapters/adapters/<family>/*.py` — one method per
family. Likely a shared mixin in
`src/generative_flow_adapters/adapters/base.py` for the boilerplate.

## Definition of done

- Method implemented on all five rows of the comparison.
- Unit tests in `tests/` that pin numerical estimates against a known
  small config (catches drift if someone changes the adapter shape).
- A short script `scripts/compute_flops_budget.py` that takes a config
  and prints the per-family FLOPs at that config — used to pin the
  three sweep budgets.
- Numbers entered into the protocol at
  [[../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]].

## Out of scope

- Peak-memory estimator — separate concern, the comparison only
  *reports* memory, doesn't match on it.
- Hardware-specific FLOPs (sparsity, kernel fusion). Use dense
  theoretical FLOPs; the comparison is a methodology axis, not a
  benchmark.

## Related

- Decision: [[../50_Decisions/decided/param-matched-adapter-comparison-definition]]
- Run ticket (blocked on this): [[experiments/exp-adapter-param-matched-comparison]]
