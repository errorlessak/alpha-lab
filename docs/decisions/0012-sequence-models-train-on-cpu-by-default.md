# ADR 0012 — Sequence models train on CPU unless a device is named

- **Status:** Accepted
- **Date:** 2026-07-28
- **Phase:** 13

## Context

§10.3 lists "MPS/CUDA if available" among the things the sequence layer should
support. §4 (I8) requires that a run be reproducible: the same config and seed
must produce byte-identical artifacts, and `alpha doctor` reports whether the
machine can deliver that.

These two pull in opposite directions. Floating-point reductions on an
accelerator are not deterministic in the way a single-threaded CPU reduction is:
kernel selection varies with driver and hardware, cuDNN's RNN kernels are
explicitly non-deterministic unless forced, and MPS makes no reproducibility
guarantee at all across macOS versions. A run that silently picks a GPU because
one happens to be present produces artifacts nobody else can reproduce, and the
manifest would still claim I8 held.

The same trade-off is already settled elsewhere in this codebase: every GBM runs
with `n_jobs=1` (`alphacore.models.trees.N_JOBS`) because multi-threaded
histogram building sums gradients in a nondeterministic order. Choosing a
different answer here would be inconsistent.

## Decision

`device` defaults to `cpu`. `cuda`, `mps` and `auto` are accepted and honoured;
`auto` picks an accelerator when one is available. Torch runs single-threaded
(`torch.set_num_threads(1)`), the network is seeded immediately before
initialisation, and batch order comes from `deterministic_rng(seed, "sequence",
name, "batch-order")` rather than a torch DataLoader shuffle.

Asking for a non-CPU device logs a warning stating that the run's artifacts
satisfy I8 only on that machine. Asking for one that is not present raises
`ModelError` rather than falling back — a silent fallback would make the same
config mean different things on two machines.

## Consequences

**Benefit.** The default run is reproducible, and the spec's accelerators remain
one config key away for anyone doing exploratory work where that matters more.

**Benefit.** The warning puts the reproducibility claim in the log next to the
run that weakened it.

**Cost.** The default is slower than it could be on a machine with a GPU. The
networks in §10.3 are small — tens of thousands of parameters over a few thousand
windows — so on the configs in this repository CPU training is seconds, not
minutes. This would be the wrong default for a larger model.

**Cost.** The `auto` path is not exercised by CI, which has no accelerator. It is
kept small for that reason: resolve a device, warn, and hand it to the same
training loop.
