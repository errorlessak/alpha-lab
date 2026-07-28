# ADR 0010 — Optuna lives behind a `tune` extra and is imported lazily

- **Status:** Accepted
- **Date:** 2026-07-28
- **Phase:** 10

## Context

§10.4 specifies Optuna with a TPE sampler and a median pruner for
hyperparameter search. §10.3 separately requires that `import alphacore` work
without torch, and §2 establishes the pattern already used for the boosting
libraries: anything heavy that a run can legitimately not need lives in an
optional extra.

Optuna is not heavy in the way torch is, but it is not free either. Installing
it pulls SQLAlchemy, alembic, Mako and colorlog — a database migration stack, in
service of a study that this project never persists to a database. None of it is
touched by a run with `models.tuning.enabled: false`, which includes the smoke
config that `just verify` gates on.

The alternative was making optuna a hard dependency of `alphacore`. That would
have put four packages into every install and every container image so that one
optional stage could import one module.

## Decision

Optuna is an optional extra, `alphacore[tune]`, and
`alphacore.pipelines.tune` imports it inside `_import_optuna()` rather than at
module scope.

Two consequences are handled explicitly rather than left to a traceback:

1. **A run that asks for tuning without the extra gets a typed error naming the
   fix**, not an `ImportError`. `PipelineError` says which extra to install and
   that disabling tuning is the other way forward.
2. **The tuning tests skip rather than fail when the extra is absent**, via
   `pytest.importorskip`. `just setup` does not install extras, so a fresh clone
   still gets a green `just verify`; `uv sync --all-packages --extra gbm --extra
   tune` gets the full suite.

## Consequences

**Benefit.** `import alphacore` stays light, and the dependency footprint of a
run reflects what the run actually does.

**Cost.** A test that only runs when an extra is installed is a test that can
rot unnoticed. The mitigation is that the tuning tests are not the only thing
guarding §10.4: `TrainingResult.n_trials_total` and the `n_trials` column on
every leaderboard row are asserted in the always-installed integration suite, so
a search whose trial count stopped reaching the deflated Sharpe would fail the
gate whether or not optuna is present.

**Cost.** CI must install the extras deliberately. The `just setup` recipe
covers the base install; `just setup-dl` and an explicit `--extra` are the
documented way to get the rest.
