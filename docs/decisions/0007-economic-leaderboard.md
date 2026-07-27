# ADR 0007 — Out-of-fold economic leaderboard instead of an accuracy leaderboard

- **Status:** Accepted
- **Date:** 2026-07-27
- **Phase:** 0 (decision), 5 and 8 (implementation)

## Context

The conventional presentation ranks models by classification accuracy on price
direction. It is close to meaningless here, for three separate reasons:

1. **Accuracy ignores magnitude.** A model that is right 52% of the time on
   small moves and wrong on large ones loses money while topping the table.
2. **Accuracy ignores cost.** Turnover is invisible to it. Two models with
   identical accuracy can differ by a Sharpe point once spread and impact are
   charged.
3. **Accuracy invites the wrong tuning objective.** Optimising in-fold log-loss
   is the single most common way a tutorial produces a fantasy result, because
   the quantity being optimised is not the quantity being claimed.

## Decision

Rank by **out-of-fold net-of-cost Sharpe with a deflated-Sharpe haircut**.
Accuracy, AUC and Brier remain on the leaderboard as secondary columns, because
they are diagnostic — they just are not the ranking key.

Every leaderboard row carries the evidence needed to distrust it:

- `sharpe_ci` — block-bootstrap interval, not just a point estimate
- `dsr` and `psr` — the multiple-testing correction and its input trial count
- `break_even_bps` — the round-trip cost at which net Sharpe reaches zero,
  reported next to the headline Sharpe, always
- `ic` and `ic_tstat` — rank correlation with the forward return
- `n_trials` — collected automatically from the Optuna study, because a manually
  entered trial count is self-deception

Baselines are mandatory rows, including buy-and-hold and — the honest one — the
same model trained on shuffled labels. If the shuffled-label Sharpe distribution
overlaps the real model's, the UI says so prominently rather than burying it.

## Consequences

**Benefit.** The ranking key is the quantity the project actually claims to
measure. A model cannot win the leaderboard by being accurate in a way that does
not survive costs.

**Cost.** Producing one leaderboard row means running a full backtest, not
evaluating a metric on a prediction vector. The tuning objective inherits this:
each Optuna trial runs an inner walk-forward and a costed backtest, which is
much slower than scoring log-loss.

**Consequence accepted.** Ranking by a noisy estimator means the top row is
partly luck. That is not a flaw in the ranking, it is the fact DSR and PBO exist
to quantify — and reporting it is the point of the project.
