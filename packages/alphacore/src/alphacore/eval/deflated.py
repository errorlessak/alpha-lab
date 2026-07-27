"""Probabilistic and deflated Sharpe ratios, and minimum track record length.

These are the metrics that make the project credible (§11.2), because they are
the ones that take a number away rather than adding one.

**PSR** — the probability the true Sharpe exceeds a benchmark, adjusting for
skew and kurtosis:

```
PSR(SR*) = Φ( (SR − SR*)·√(n−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) )
```

Negative skew and fat tails both *lower* it, which is right: a strategy that
usually makes a little and occasionally loses a lot has a Sharpe that overstates
it.

**DSR** — PSR evaluated at the Sharpe you would expect from the *best of N
tries* on pure noise:

```
SR* = √Var[SR_n] · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ],  γ = 0.5772…
```

This is the correction for having looked at many strategies. Search sixty
configurations and the best of them has a good Sharpe *by construction*; DSR
asks whether it is good enough to survive that fact. The `N` comes from the
Optuna study automatically (§10.4) — a hand-entered trial count is
self-deception.

**minTRL** — how many observations you would need for a Sharpe to be
significant at 95%. Usually a longer history than anyone has.
"""

from __future__ import annotations

import math
from typing import Final

from scipy import stats

from alphacore.errors import EvaluationError

__all__ = [
    "EULER_MASCHERONI",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
]

EULER_MASCHERONI: Final[float] = 0.5772156649015329
NORMAL_KURTOSIS: Final[float] = 3.0
_MIN_OBSERVATIONS: Final[int] = 2
_DEFAULT_CONFIDENCE: Final[float] = 0.95


def _sharpe_denominator(sharpe: float, skew: float, kurtosis: float) -> float:
    """The variance term in the PSR formula.

    Args:
        sharpe: Observed Sharpe ratio, per observation.
        skew: Return skewness.
        kurtosis: Return kurtosis, not excess.

    Returns:
        The square root of the adjusted variance.

    Raises:
        EvaluationError: If the adjustment drives the variance non-positive,
            which means the moments are inconsistent with each other.
    """
    variance = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance <= 0:
        msg = (
            f"The skew/kurtosis adjustment gives a non-positive variance ({variance:.4f}) "
            f"for Sharpe {sharpe:.4f}, skew {skew:.4f}, kurtosis {kurtosis:.4f}. "
            "These moments cannot come from the same sample."
        )
        raise EvaluationError(msg)
    return math.sqrt(variance)


def probabilistic_sharpe_ratio(
    *,
    sharpe: float,
    n_observations: int,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
) -> float:
    """Probability the true Sharpe exceeds a benchmark (§11.2).

    Args:
        sharpe: Observed Sharpe per observation, on the same scale as
            `benchmark`.
        n_observations: Number of return observations.
        benchmark: Sharpe to beat. Zero asks "is this better than nothing?".
        skew: Return skewness.
        kurtosis: Return kurtosis, not excess. Normal is 3.

    Returns:
        A probability in `[0, 1]`.

    Raises:
        EvaluationError: If the sample is too short, or the moments are
            mutually inconsistent.
    """
    if n_observations < _MIN_OBSERVATIONS:
        msg = f"PSR needs at least {_MIN_OBSERVATIONS} observations, got {n_observations}."
        raise EvaluationError(msg)
    denominator = _sharpe_denominator(sharpe, skew, kurtosis)
    z = (sharpe - benchmark) * math.sqrt(n_observations - 1) / denominator
    return float(stats.norm.cdf(z))


def expected_max_sharpe(*, n_trials: int, variance: float) -> float:
    """The Sharpe you would expect from the best of `n_trials` on pure noise.

    ```
    SR* = √Var[SR] · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
    ```

    This is the number a leaderboard's top row has to beat before it means
    anything. Searching widely raises it, which is the entire point.

    Args:
        n_trials: Configurations evaluated.
        variance: Variance of the Sharpe ratios observed across trials.

    Returns:
        The expected maximum Sharpe. Zero for a single trial — nothing was
        selected, so there is nothing to correct for.

    Raises:
        EvaluationError: If `n_trials` is below 1 or the variance is negative.
    """
    if n_trials < 1:
        msg = f"n_trials must be at least 1, got {n_trials}."
        raise EvaluationError(msg)
    if variance < 0:
        msg = f"Sharpe variance cannot be negative, got {variance}."
        raise EvaluationError(msg)
    if n_trials == 1:
        return 0.0

    gamma = EULER_MASCHERONI
    first = stats.norm.ppf(1.0 - 1.0 / n_trials)
    second = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(variance) * ((1.0 - gamma) * first + gamma * second))


def deflated_sharpe_ratio(
    *,
    sharpe: float,
    n_observations: int,
    n_trials: int,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
) -> float:
    """PSR evaluated against the expected maximum Sharpe from `n_trials`.

    Args:
        sharpe: Observed Sharpe per observation.
        n_observations: Number of return observations.
        n_trials: Configurations evaluated in the search.
        sharpe_variance: Variance of Sharpe ratios across those trials.
        skew: Return skewness.
        kurtosis: Return kurtosis, not excess.

    Returns:
        A probability in `[0, 1]`. Read it as: given that you looked this hard,
        how likely is it the edge is real?

    Raises:
        EvaluationError: If any input is out of range.
    """
    benchmark = expected_max_sharpe(n_trials=n_trials, variance=sharpe_variance)
    return probabilistic_sharpe_ratio(
        sharpe=sharpe,
        n_observations=n_observations,
        benchmark=benchmark,
        skew=skew,
        kurtosis=kurtosis,
    )


def min_track_record_length(
    *,
    sharpe: float,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
    confidence: float = _DEFAULT_CONFIDENCE,
) -> float:
    """Observations needed for a Sharpe to be significant (§11.2).

    ```
    minTRL = 1 + (1 − γ₃·SR + ((γ₄−1)/4)·SR²) · (Φ⁻¹(confidence) / (SR − SR*))²
    ```

    Frequently the most sobering number in the tearsheet: a Sharpe of 0.4 needs
    roughly forty years of daily data to clear 95% confidence.

    Args:
        sharpe: Observed Sharpe per observation.
        benchmark: Sharpe to beat.
        skew: Return skewness.
        kurtosis: Return kurtosis, not excess.
        confidence: Required confidence level.

    Returns:
        The required number of observations.

    Raises:
        EvaluationError: If the Sharpe does not exceed the benchmark, in which
            case no amount of data would establish significance.
    """
    edge = sharpe - benchmark
    if edge <= 0:
        msg = (
            f"A Sharpe of {sharpe:.4f} does not exceed the benchmark {benchmark:.4f}, so no "
            "track record length would ever make it significant."
        )
        raise EvaluationError(msg)
    variance = _sharpe_denominator(sharpe, skew, kurtosis) ** 2
    z = stats.norm.ppf(confidence)
    return float(1.0 + variance * (z / edge) ** 2)
