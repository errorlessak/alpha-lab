"""The backtest and evaluation stage (§11, §12, Phase 10).

This is the stage that turns a leaderboard of prediction-quality metrics into an
economic one. For one model's out-of-fold probabilities it generates a signal,
sizes it, runs the costed event-driven engine over the same bars the predictions
cover, and computes the metric set §11.2 asks for — including the honesty
metrics, which are the reason any of the rest is worth reading.

Two rules this stage exists to keep:

* **The backtest sees only out-of-fold probabilities.** They arrive already
  stacked from `fit_predict_oof`, so there is no path here that could score a
  bar with a model that trained on it.
* **The break-even cost travels with the Sharpe.** §11.2 requires it next to
  the headline figure, always, and `economic_columns` therefore emits the two
  together or not at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from alphacore.backtest import cost_sensitivity, round_trip_bps, run_backtest
from alphacore.errors import EvaluationError, PipelineError
from alphacore.eval import (
    block_bootstrap_ci,
    break_even_cost_bps,
    deflated_sharpe_ratio,
    max_drawdown,
    min_track_record_length,
    net_sharpe,
    probabilistic_sharpe_ratio,
    return_metrics,
    rolling_sharpe,
    sharpe_ratio,
    trade_metrics,
)
from alphacore.features.primitives import realised_vol
from alphacore.runtime.logging import logger
from alphacore.signal import generate_signal, size_positions

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from alphacore.backtest import BacktestResult
    from alphacore.config.schema import AlphaConfig
    from alphacore.models.oof import OOFPredictions

__all__ = [
    "ROLLING_SHARPE_WINDOWS",
    "BacktestOutcome",
    "backtest_predictions",
    "deflate_outcomes",
    "economic_columns",
    "sharpe_variance",
]

ROLLING_SHARPE_WINDOWS: Final[tuple[int, ...]] = (63, 252)
"""§11.2 asks for rolling 63- and 252-bar Sharpe ratios: one quarter and one
year on a daily calendar."""

_VOL_WINDOW: Final[int] = 20
"""Window for the realised-volatility series that drives `vol_target` sizing.
Short enough to react, long enough not to be noise."""

_MIN_BOOTSTRAP_OBSERVATIONS: Final[int] = 30


@dataclass(frozen=True)
class BacktestOutcome:
    """Everything the evaluation stage computed for one model.

    Attributes:
        model: Model name.
        result: The raw engine output.
        returns: Per-bar simple returns of the equity curve.
        metrics: The §11.2 return-based metric set.
        trades: The §11.2 trade-based metric set.
        cost_sensitivity: The cost sweep, one point per level in
            `backtest.cost_sensitivity_bps`.
        break_even_bps: Round-trip cost at which net Sharpe reaches zero.
        round_trip_bps: The round-trip cost this run actually assumed.
        sharpe: Annualised net-of-cost Sharpe.
        sharpe_ci: Bootstrap confidence interval for it, or None when the
            sample is too short to resample honestly.
        psr: Probabilistic Sharpe ratio against a zero benchmark.
        dsr: Deflated Sharpe ratio, haircut for the number of trials.
        min_track_record_length: Bars needed for the Sharpe to be significant.
        n_trials: Trials the deflation was computed against.
        rolling: Rolling Sharpe series by window length.
        periods_per_year: Annualisation factor used throughout (§11.1).
    """

    model: str
    result: BacktestResult
    returns: pd.Series
    metrics: dict[str, float]
    trades: dict[str, float]
    cost_sensitivity: list[dict[str, float]]
    break_even_bps: float
    round_trip_bps: float
    sharpe: float
    sharpe_ci: tuple[float, float] | None
    psr: float | None
    dsr: float | None
    min_track_record_length: float | None
    n_trials: int
    rolling: dict[int, pd.Series]
    periods_per_year: float

    @property
    def warnings(self) -> tuple[str, ...]:
        """Flags this outcome earns, on top of the engine's own.

        Returns:
            Sorted flags. `SHORT_SAMPLE` means the sample is shorter than the
            minimum track record length, so the interval around the Sharpe is
            wide whatever the point estimate says.
        """
        flags = list(self.result.warnings)
        required = self.min_track_record_length
        if required is not None and required > len(self.returns):
            flags.append("SHORT_SAMPLE")
        return tuple(sorted(set(flags)))


def economic_columns(outcome: BacktestOutcome) -> dict[str, Any]:
    """Render an outcome as the leaderboard's economic columns (ADR 0007).

    Args:
        outcome: A completed backtest.

    Returns:
        A mapping keyed by `alphacore.models.ECONOMIC_COLUMNS`, ready for
        `attach_economics`.
    """
    return {
        "oos_net_sharpe": outcome.sharpe,
        "sharpe_ci": outcome.sharpe_ci,
        "dsr": outcome.dsr,
        "psr": outcome.psr,
        "break_even_bps": outcome.break_even_bps,
        "max_drawdown": outcome.metrics.get("max_drawdown"),
    }


def sharpe_variance(sharpes: list[float]) -> float:
    """Variance of the Sharpe ratios a search produced (§11.2).

    This is the `Var[SR_n]` in the deflated Sharpe's expected-maximum term. It
    is estimated from the configurations actually evaluated in this run rather
    than assumed, because assuming it is how a deflated Sharpe stops deflating
    anything.

    Args:
        sharpes: One Sharpe per configuration evaluated.

    Returns:
        The sample variance, or a small positive floor when fewer than two
        configurations were evaluated or they all landed on the same number.
        A zero variance would make the expected maximum zero and the deflation
        a no-op, which would silently overstate every result.
    """
    floor = 1e-6
    if len(sharpes) < 2:  # noqa: PLR2004
        return floor
    return max(float(np.var(np.asarray(sharpes, dtype=float), ddof=1)), floor)


def _signal_and_weights(
    proba: NDArray[np.float64], bars: pd.DataFrame, config: AlphaConfig, *, periods_per_year: float
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Turn probabilities into a signal and per-bar target weights.

    Args:
        proba: Calibrated probability of the positive class, per bar.
        bars: The bars being traded.
        config: The resolved configuration.
        periods_per_year: Annualisation factor for the volatility series.

    Returns:
        The signal in `{-1, 0, +1}` and the signed target weights.

    Raises:
        BacktestError: If sizing is missing a required input.
    """
    signal = generate_signal(proba, config.signal)
    volatility = None
    if config.signal.sizing.method == "vol_target":
        close = bars["close"].to_numpy(dtype=float)
        raw = realised_vol(close, _VOL_WINDOW, periods_per_year=periods_per_year)
        warm_up = ~np.isfinite(raw)
        if warm_up.any():
            # Those bars have no volatility estimate yet, so they are held flat.
            # Back-filling the first computed value would look ahead: the
            # volatility of bars 1..20 is not knowable at bar 1, and using it to
            # size a position there is a leak, however small (§4, I1).
            signal = np.where(warm_up, 0, signal)
            logger.bind(phase="backtest").debug(
                "held flat for {} bars while the volatility window warmed up", int(warm_up.sum())
            )
        # A finite placeholder keeps the sizing arithmetic defined on those
        # bars. Their weight is zero either way, because the signal is.
        first_finite = raw[~warm_up][0] if (~warm_up).any() else 1.0
        volatility = np.where(warm_up, first_finite, raw)
    weights = size_positions(signal, config.signal.sizing, volatility=volatility, proba=proba)
    return signal, weights


def backtest_predictions(
    oof: OOFPredictions,
    bars: pd.DataFrame,
    config: AlphaConfig,
    *,
    periods_per_year: float,
    seed: int,
) -> BacktestOutcome:
    """Run the costed backtest for one model's out-of-fold predictions.

    The deflated Sharpe is deliberately **not** computed here. Deflation needs
    the spread of Sharpe ratios across every configuration the run evaluated,
    which is not known until they have all been backtested; `deflate_outcomes`
    fills it in afterwards. Computing it per model would mean either running the
    engine twice or deflating against an assumed variance, and an assumed
    variance is a decoration rather than a haircut (§11.2).

    Args:
        oof: Stacked out-of-fold predictions.
        bars: Validated bars covering at least the prediction index.
        config: The resolved configuration.
        periods_per_year: Bars per year (§11.1).
        seed: Run seed, for the bootstrap's own random stream.

    Returns:
        The completed outcome, with `dsr` still unset.

    Raises:
        PipelineError: If the predictions and bars cannot be aligned.
        BacktestError: If the engine rejects the signal or the sizing inputs.
        EvaluationError: If the equity curve admits no Sharpe ratio.
    """
    frame = oof.frame
    missing = frame.index.difference(bars.index)
    if len(missing):
        msg = (
            f"{len(missing)} predicted timestamps are absent from the bars, so those "
            "predictions cannot be traded. Features, labels and bars must come from the "
            "same aligned feature set."
        )
        raise PipelineError(msg)

    traded = bars.loc[frame.index]
    proba = frame["proba"].to_numpy(dtype=float)
    signal, weights = _signal_and_weights(proba, traded, config, periods_per_year=periods_per_year)

    result = run_backtest(
        traded,
        signal=signal,
        config=config.backtest,
        risk=config.risk,
        weights=weights,
        periods_per_year=periods_per_year,
    )

    equity = result.equity["equity"]
    returns = equity.pct_change().dropna()
    if returns.empty:
        msg = (
            f"The backtest of {oof.model!r} produced no return observations, so no metric "
            "can be computed from it."
        )
        raise PipelineError(msg)

    metrics = return_metrics(
        returns.to_numpy(dtype=float), equity, periods_per_year=periods_per_year
    )
    metrics.setdefault("max_drawdown", max_drawdown(equity))
    sharpe = net_sharpe(returns.to_numpy(dtype=float), periods_per_year=periods_per_year)
    metrics["sharpe"] = sharpe

    sweep = cost_sensitivity(
        traded,
        signal=signal,
        config=config.backtest,
        risk=config.risk,
        weights=weights,
        periods_per_year=periods_per_year,
    )
    outcome = BacktestOutcome(
        model=oof.model,
        result=result,
        returns=returns,
        metrics=metrics,
        trades=trade_metrics(result.trades),
        cost_sensitivity=sweep,
        break_even_bps=break_even_cost_bps(sweep),
        round_trip_bps=round_trip_bps(config.backtest.costs),
        sharpe=sharpe,
        sharpe_ci=_bootstrap_ci(
            returns, seed=seed, periods_per_year=periods_per_year, config=config
        ),
        psr=_psr(sharpe, returns, metrics, periods_per_year=periods_per_year),
        dsr=None,
        min_track_record_length=_mtrl(sharpe, metrics, periods_per_year=periods_per_year),
        n_trials=1,
        rolling=_rolling(returns, periods_per_year=periods_per_year),
        periods_per_year=periods_per_year,
    )
    logger.bind(phase="backtest").info(
        "{}: net Sharpe {:.3f}, break-even {:.1f} bps against {:.1f} assumed",
        oof.model,
        outcome.sharpe,
        outcome.break_even_bps,
        outcome.round_trip_bps,
    )
    return outcome


def deflate_outcomes(
    outcomes: Mapping[str, BacktestOutcome],
    *,
    n_trials_by_model: Mapping[str, int] | None = None,
    extra_sharpes: Sequence[float] = (),
) -> dict[str, BacktestOutcome]:
    """Fill in each outcome's deflated Sharpe once every trial is known (§11.2).

    `SR* = sqrt(Var[SR_n]) * [ (1-y)*Phi^-1(1 - 1/N) + y*Phi^-1(1 - 1/(N*e)) ]`,
    with `Var[SR_n]` estimated from the Sharpe ratios this run actually
    produced — every leaderboard model, plus every tuning trial. `N` is the
    number of configurations evaluated, which is collected rather than
    configured (§10.4).

    Args:
        outcomes: Backtest outcomes by model name.
        n_trials_by_model: Trials behind each model, from a tuning study.
            Models absent from it are counted as one configuration each.
        extra_sharpes: Sharpe ratios from configurations that were evaluated
            but are not on the leaderboard — the discarded tuning trials. They
            widen the observed spread, which is exactly what they should do.

    Returns:
        New outcomes carrying `dsr` and the trial count it was computed against.
    """
    trials = dict(n_trials_by_model or {})
    observed = [outcome.sharpe for outcome in outcomes.values()] + list(extra_sharpes)
    variance = sharpe_variance(observed)
    total = max(sum(trials.get(name, 1) for name in outcomes) + len(extra_sharpes), len(outcomes))

    deflated: dict[str, BacktestOutcome] = {}
    for name, outcome in outcomes.items():
        per_observation_variance = variance / float(outcome.periods_per_year)
        deflated[name] = replace(
            outcome,
            n_trials=total,
            dsr=_dsr(
                outcome.sharpe,
                outcome.returns,
                outcome.metrics,
                periods_per_year=outcome.periods_per_year,
                n_trials=total,
                variance=per_observation_variance,
            ),
        )
    logger.bind(phase="metrics").info(
        "deflated {} Sharpe ratios against {} configurations, Var[SR] {:.6f}",
        len(deflated),
        total,
        variance,
    )
    return deflated


def _per_observation(sharpe: float, *, periods_per_year: float) -> float:
    """Convert an annualised Sharpe to a per-observation one.

    PSR and DSR are defined on the per-observation scale; feeding them an
    annualised figure inflates the probability towards 1.0 and makes the whole
    honesty section read as good news.

    Args:
        sharpe: Annualised Sharpe.
        periods_per_year: Annualisation factor.

    Returns:
        The per-observation Sharpe.
    """
    return sharpe / float(np.sqrt(periods_per_year))


def _psr(
    sharpe: float, returns: pd.Series, metrics: dict[str, float], *, periods_per_year: float
) -> float | None:
    """Probabilistic Sharpe ratio, or None when the sample cannot support one.

    Args:
        sharpe: Annualised Sharpe.
        returns: The return series.
        metrics: The return metric set, for skew and kurtosis.
        periods_per_year: Annualisation factor.

    Returns:
        The PSR, or None if the sample is too short.
    """
    try:
        return probabilistic_sharpe_ratio(
            sharpe=_per_observation(sharpe, periods_per_year=periods_per_year),
            n_observations=len(returns),
            skew=metrics.get("skew", 0.0),
            kurtosis=metrics.get("kurtosis", 3.0),
        )
    except EvaluationError as exc:
        logger.bind(phase="metrics").warning("PSR not computed: {}", exc)
        return None


def _dsr(
    sharpe: float,
    returns: pd.Series,
    metrics: dict[str, float],
    *,
    periods_per_year: float,
    n_trials: int,
    variance: float,
) -> float | None:
    """Deflated Sharpe ratio, or None when the sample cannot support one.

    Args:
        sharpe: Annualised Sharpe.
        returns: The return series.
        metrics: The return metric set, for skew and kurtosis.
        periods_per_year: Annualisation factor.
        n_trials: Configurations evaluated.
        variance: Variance of Sharpe across those configurations, already on
            the per-observation scale the deflation works in.

    Returns:
        The DSR, or None if the sample is too short.
    """
    try:
        return deflated_sharpe_ratio(
            sharpe=_per_observation(sharpe, periods_per_year=periods_per_year),
            n_observations=len(returns),
            n_trials=max(int(n_trials), 1),
            sharpe_variance=variance,
            skew=metrics.get("skew", 0.0),
            kurtosis=metrics.get("kurtosis", 3.0),
        )
    except EvaluationError as exc:
        logger.bind(phase="metrics").warning("DSR not computed: {}", exc)
        return None


def _mtrl(sharpe: float, metrics: dict[str, float], *, periods_per_year: float) -> float | None:
    """Minimum track record length, or None when it is undefined.

    Args:
        sharpe: Annualised Sharpe.
        metrics: The return metric set, for skew and kurtosis.
        periods_per_year: Annualisation factor.

    Returns:
        Observations required, or None for a non-positive Sharpe where the
        question does not arise.
    """
    try:
        return min_track_record_length(
            sharpe=_per_observation(sharpe, periods_per_year=periods_per_year),
            skew=metrics.get("skew", 0.0),
            kurtosis=metrics.get("kurtosis", 3.0),
        )
    except EvaluationError as exc:
        logger.bind(phase="metrics").debug("minimum track record length not computed: {}", exc)
        return None


def _bootstrap_ci(
    returns: pd.Series, *, seed: int, periods_per_year: float, config: AlphaConfig
) -> tuple[float, float] | None:
    """Bootstrap a confidence interval for the Sharpe (§11.2).

    Args:
        returns: The return series.
        seed: Run seed.
        periods_per_year: Annualisation factor.
        config: The resolved configuration, for `reporting.bootstrap_samples`.

    Returns:
        The interval, or None when the sample is too short to resample.
    """
    if len(returns) < _MIN_BOOTSTRAP_OBSERVATIONS:
        return None
    try:
        return block_bootstrap_ci(
            returns.to_numpy(dtype=float),
            lambda sample: sharpe_ratio(sample, periods_per_year=periods_per_year),
            seed=seed,
            n_samples=config.reporting.bootstrap_samples,
        )
    except EvaluationError as exc:
        logger.bind(phase="metrics").warning("Sharpe interval not bootstrapped: {}", exc)
        return None


def _rolling(returns: pd.Series, *, periods_per_year: float) -> dict[int, pd.Series]:
    """Compute the rolling Sharpe series §11.2 asks for.

    Args:
        returns: The return series.
        periods_per_year: Annualisation factor.

    Returns:
        A mapping of window length to series. Windows longer than the sample,
        or degenerate throughout it, are absent rather than empty.
    """
    series: dict[int, pd.Series] = {}
    for window in ROLLING_SHARPE_WINDOWS:
        try:
            series[window] = rolling_sharpe(
                returns, window=window, periods_per_year=periods_per_year
            )
        except EvaluationError as exc:
            logger.bind(phase="metrics").debug("rolling {}-bar Sharpe skipped: {}", window, exc)
    return series
