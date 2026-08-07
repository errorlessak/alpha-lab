# Contributing

Thanks for looking. This document is short on ceremony and long on the two or
three things that are genuinely load-bearing here — a change that gets those
wrong produces numbers that look fine and are not.

## The one-paragraph orientation

`ARCHITECTURE.md` is the build specification and the contract. `HANDOFF.md` is
the shortest useful path into the codebase: environment traps, conventions that
look arbitrary until you know why, and the current list of known gaps. Read
`HANDOFF.md` first if you intend to change anything.

## Setting up

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
just setup
```

Optional extras are deliberately excluded — `import alphacore` must work
without any of them. `just setup-dl` adds all three (LightGBM/XGBoost/CatBoost,
PyTorch, Optuna) if you are working on models or tuning.

## The gate

```bash
just verify
```

Format, lint, both type checkers, the test suite with an 85% coverage floor,
the leakage suite, an end-to-end smoke run, the Playwright golden path, and the
bundle budget. **A change is not done until it is green.** CI runs the same
things, split across jobs, plus a Docker build and a check that the generated
files are current.

Individual recipes exist for the fast loop: `just test`, `just test-leak`,
`just types`, `just lint`, `just smoke`.

## What a good change looks like

**Tests arrive in the same commit as the code.** This repository was built
test-first throughout; a pull request whose tests were written afterwards to
match the implementation is easy to spot and no use to anybody.

**Every canary needs a control.** A test that would pass against a broken
implementation is worse than no test. The tree already holds several worked
examples: the future-poison harness has a deliberately leaky centred-window
control; the same-bar-execution canary has a perfect-foresight inverse; the
window-purge test has the obvious positional windower, which straddles the gap
and fails the same assertion. If you add a test that proves an invariant holds,
add the thing that proves the test can fail.

**Invariants are enforced, not documented.** If you find yourself writing "must
not" in a docstring, ask whether it can raise instead. A feature naming a label
column cannot be registered. `execution_lag_bars: 0` raises. Quantile label
edges cannot be fitted without a training mask. None of these warn.

**Nothing returns `NaN`.** A metric that cannot be computed raises (§0.7). A
silent NaN in a Sharpe ratio is a critical bug, not a missing value.

**Every error is a typed exception from `alphacore.errors`.** No bare
`ValueError`.

**Every annualisation is explicit.** `periods_per_year` is a required argument,
derived from the calendar and recorded in the manifest. 252 is never hardcoded.

**Determinism is a feature (§4, I8).** Canonical JSON, sorted iteration, seeded
RNGs through `deterministic_rng(seed, *stream_labels)`, `n_jobs=1` on every
gradient booster. The same config run twice must produce a byte-identical
tearsheet. `uv run alpha doctor` tells you whether your machine can.

**Docstrings carry `Args`, `Returns` and `Raises`**, and the formula for
anything financial. ruff's `D` and `ANN` rules enforce the shape; the formula
is on you.

## Deviating from the specification

`ARCHITECTURE.md` wins when it and instinct disagree. When it is silent, or
when following it would be wrong, write an ADR in `docs/decisions/` — one page:
context, decision, consequences, and the trade-off you accepted. There are
twelve already; `docs/decisions/README.md` explains the format.

## Commits and pull requests

Conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
`chore:`). Messages explain *why*; the diff already shows what.

On Windows, pass multi-line messages through a file — `git commit -F <file>`.
PowerShell here-strings break argument splitting and git receives the message as
dozens of pathspecs.

Pull requests should say what changed, why, and what you ran. The template asks
for exactly that.

## Reporting a problem

Bugs and feature requests go to
[GitHub issues](https://github.com/errorlessak/alpha-lab/issues); the templates
ask for the config, the run id and the `alpha doctor` output, which is usually
enough to reproduce. Security issues do **not** go there — see
[SECURITY.md](SECURITY.md).

## What this project will not accept

- **Trading, order routing or broker connectivity.** The system reports
  evidence and stops. That is the whole design, not a missing feature.
- **A backtest shortcut.** Same-bar execution, fitting before splitting, global
  quantile labels, zero costs, or reporting a best-of-N Sharpe without a
  haircut. Each of those has a test standing in front of it.
- **A dependency that only saves a few lines.** Every one is a supply-chain
  surface and a version to resolve. `docs/decisions/0001-in-house-indicator-library.md`
  is the precedent.
