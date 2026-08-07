# What and why

<!-- What changed, and the reason. The diff already shows the what; the why is
     the part that cannot be recovered later. -->

## Evidence

<!-- Paste the tail of `just verify`. A phase is not done until it is green,
     and neither is a pull request. -->

```

```

## Checklist

- [ ] `just verify` is green, and the output is above.
- [ ] Tests are in this commit, not a later one, and I watched them fail for
      the right reason first.
- [ ] Any new canary test has a control that proves it can fail.
- [ ] New invariants raise rather than warn.
- [ ] No new `NaN` return path; a metric that cannot be computed raises.
- [ ] Errors are typed exceptions from `alphacore.errors`.
- [ ] Any annualisation takes `periods_per_year` explicitly.
- [ ] Docstrings carry `Args`, `Returns`, `Raises`, and the formula for
      anything financial.
- [ ] If this deviates from `ARCHITECTURE.md`, there is an ADR in
      `docs/decisions/`.
- [ ] If this changes the API, `just client` was run and both generated files
      are committed.
- [ ] If this changes the feature registry, `just docs` was run.
- [ ] `CHANGELOG.md` has an entry under Unreleased.

## Does this change a number?

<!-- If a run's tearsheet would come out different after this change, say so
     explicitly and say why the new number is the right one. Determinism (§4,
     I8) means an unexplained change to a result is a bug report about this
     pull request. -->
