# PDD / Few-Step Distillation Resources

## Knowledge

- [Paper: _Parallel Decoding Distillation for Fast Image and Video Generation_ — Shaul, Liu, Vahdat, Berner (arXiv:2607.26004, 28 Jul 2026)](https://arxiv.org/abs/2607.26004)
  The primary source. NVIDIA/Meta-FAIR authors. Use for: the PD loss (Eq. 11), the
  parallel-decoder architecture (Eq. 13), layer fusion (Eq. 15), Algorithms 1–3,
  and the Table 1 comparison against flow maps and Pi-Flow.
  HTML full text: <https://arxiv.org/html/2607.26004v1>. Local copy pulled to scratchpad
  (45 MB — the video figures; `pdftotext -f 1 -l 9` gives the whole method).

- [Paper: _One Step Diffusion via Shortcut Models_ — Frans, Hafner, Levine, Abbeel (arXiv:2410.12557)](https://arxiv.org/abs/2410.12557)
  Already the thesis's D3 backbone. Use for: the contrast case — self-consistency
  bootstrapping and step-size conditioning, which PDD deliberately does *not* use.
  Vault note: `30_Knowledge/related-work/shortcut-models.md`.

- [Paper: _Pi-Flow_ (ref [7] in PDD)](https://arxiv.org/abs/2510.14974)
  _needs verification — arXiv id inferred from the PDD bibliography, not yet checked._
  The closest prior work: same "the block is determined by its initial state"
  observation, but a Gaussian-mixture policy head and fixed NFE. Use for: sharpening
  what is actually novel in PDD.

- [Paper: _Flow Matching Guide and Code_ — Lipman et al.](https://arxiv.org/abs/2412.06264)
  Lipman/Shaul are shared authors with PDD; this is the notation PDD assumes.
  Use for: mean velocity, interpolants, and the `u` vs `v` distinction.

## Wisdom (Communities)

- No community preference stated yet. Candidates worth considering when a question
  turns out to need practitioner judgement rather than paper-reading:
  - The paper authors are reachable — Berner and Shaul both respond on arXiv/X threads
    about their own work; a precise question about on-policy vs off-policy targets
    for a *frozen-base adapter* is exactly the kind they answer.
  - `r/MachineLearning` is too noisy for this; the EleutherAI / Latent Space Discords
    have active diffusion-distillation channels.
  _Ask Lukas whether he wants any of this before proposing further._

## Gaps

- No independent reproduction or third-party critique of PDD yet — the paper is
  six days old. Every claim below is the authors' own.
- No source in the vault on **what "diversity" is measured as** in PDD's Table 4;
  relevant if the thesis wants to borrow the diversity argument.
- Appendices C (connection to flow maps) and D (Proposition 1 proof) not yet read.
