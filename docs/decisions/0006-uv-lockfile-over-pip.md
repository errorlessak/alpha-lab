# ADR 0006 — `uv` + `uv.lock` instead of `pip` + `requirements.txt`

- **Status:** Accepted
- **Date:** 2026-07-27
- **Phase:** 0

## Context

Invariant I8 says that given `(config_hash, data_hash, seed)`, every artifact is
byte-identical across runs. That claim is false if the package set can move
underneath it. A LightGBM minor release can change a tie-break in tree splitting;
a NumPy release can change a reduction order. Neither shows up in the config
hash.

`requirements.txt` without hashes pins direct dependencies and lets transitive
ones float. That is precisely the gap this invariant cannot tolerate.

## Decision

`uv` with a workspace `pyproject.toml` and a committed `uv.lock` covering the
full transitive closure. Two packages, `alphacore` and `alphaservice`, as
workspace members.

The manifest additionally records the installed versions of every package that
can change results (`_TRACKED_PACKAGES` in `runtime/manifest.py`). So when two
runs with identical config hashes disagree, the manifest says which package
moved rather than leaving the difference unexplained.

`just export-requirements` emits a `requirements.txt` via `uv export` for anyone
who wants one. It is a derived artifact, never the source of truth.

## Consequences

**Benefit.** Reproducibility is enforced by the resolver rather than requested
in a README. `uv sync` is also fast enough that CI can do a clean install on
every push instead of caching a partially-stale environment.

**Cost.** A tool most reviewers have not used, and a lockfile that produces
large diffs on upgrade. Both are acceptable: the install path is one command,
and a large diff on a dependency upgrade is information, not noise.

**Deviation from §17.** The `setup` recipe runs `uv sync --all-packages` without
`--all-extras`. `--all-extras` pulls PyTorch, a multi-gigabyte download that
nothing before Phase 13 needs — and §10.3 explicitly requires the base install
to work without it. `just setup-dl` is the opt-in.
