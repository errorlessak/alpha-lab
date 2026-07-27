"""The configuration schema — the single source of truth for a run (§5).

YAML is deserialised into these models. Unknown keys are a hard error: a typo in
`comission_bps` must not silently leave the default in place and produce a
plausible-looking backtest. Every model therefore sets
`extra="forbid"`.

The fully resolved config is serialised into every run manifest, and
`config_hash = blake2b(canonical_json(config))`. That is what makes two runs
comparable, so nothing that affects a result may live outside this schema.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AlphaConfig",
    "BacktestConfig",
    "CVConfig",
    "CostsConfig",
    "DataConfig",
    "FeaturesConfig",
    "Interval",
    "LabelsConfig",
    "MetaConfig",
    "ModelsConfig",
    "ReportingConfig",
    "RiskConfig",
    "SignalConfig",
]

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class Base(BaseModel):
    """Base for every config model: unknown keys are rejected, values frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class Interval(StrEnum):
    """Bar interval.

    The set is closed deliberately. An arbitrary interval string would have no
    defined session grid (§6.2) and no defined annualisation factor (§11.1),
    which makes every downstream metric unfalsifiable.
    """

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        """Length of one bar in minutes."""
        return _INTERVAL_MINUTES[self]

    @property
    def is_intraday(self) -> bool:
        """Whether this interval subdivides a trading session."""
        return self is not Interval.D1

    @property
    def pandas_freq(self) -> str:
        """Pandas offset alias for this interval, e.g. `'15min'`."""
        return f"{self.minutes}min"


_INTERVAL_MINUTES: dict[Interval, int] = {
    Interval.M1: 1,
    Interval.M5: 5,
    Interval.M15: 15,
    Interval.H1: 60,
    Interval.H4: 240,
    Interval.D1: 1440,
}


class Adjustment(StrEnum):
    """How split and dividend adjustment is handled (§4, I7).

    `unadjusted_plus_factor` is the default and the recommended choice: the
    backtester trades the prices that actually printed, and the adjustment
    factor travels alongside as its own column rather than being baked in.
    """

    UNADJUSTED_PLUS_FACTOR = "unadjusted_plus_factor"
    ADJUSTED = "adjusted"
    RAW = "raw"


class MetaConfig(Base):
    """Identity of the experiment."""

    name: str = Field(min_length=1, description="Experiment name; becomes part of the run id.")
    description: str = Field(default="", description="Free-text note carried into the tearsheet.")
    seed: int = Field(ge=0, le=2**32 - 1, description="Run seed. Fixes every RNG (§4, I8).")


class QualityThresholds(Base):
    """Thresholds that turn a quality report into PASS / WARN / FAIL (§6.4)."""

    max_missing_bar_pct: Probability | float = Field(
        default=2.0, ge=0.0, le=100.0, description="Percent of expected bars allowed to be absent."
    )
    max_zero_volume_pct: float = Field(
        default=1.0, ge=0.0, le=100.0, description="Percent of bars allowed to have zero volume."
    )
    fail_on_ohlc_violation: bool = Field(
        default=True,
        description="Treat any bar violating low <= min(o,c) <= max(o,c) <= high as fatal.",
    )
    max_duplicate_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Percent of duplicate timestamps tolerated."
    )
    extreme_return_z: float = Field(
        default=10.0, gt=0.0, description="Absolute z-score above which a return is flagged."
    )


class DataConfig(Base):
    """Where the bars come from and what counts as usable (§6)."""

    provider: Literal["yahoo", "binance", "alphavantage", "csv"] = "yahoo"
    symbols: tuple[str, ...] = Field(min_length=1, description="Point-in-time symbol list (I6).")
    benchmark: str | None = Field(
        default=None, description="Symbol used for the information ratio and the equity overlay."
    )
    interval: Interval = Interval.D1
    start: dt.date
    end: dt.date
    calendar: str | None = Field(
        default=None,
        description=(
            "exchange_calendars code, e.g. 'XNYS'. None means infer: a 24/7 grid for crypto "
            "providers, 'XNYS' otherwise."
        ),
    )
    adjustment: Adjustment = Adjustment.UNADJUSTED_PLUS_FACTOR
    csv_dir: str | None = Field(
        default=None, description="Source directory for the `csv` provider."
    )
    quality: QualityThresholds = Field(default_factory=QualityThresholds)

    @model_validator(mode="after")
    def _check_span_and_provider(self) -> Self:
        """Reject an empty date span or a csv provider with nowhere to read from.

        Returns:
            The validated config.

        Raises:
            ValueError: If `end` is not after `start`, or the csv provider has
                no `csv_dir`.
        """
        if self.end <= self.start:
            msg = f"data.end ({self.end}) must be after data.start ({self.start})."
            raise ValueError(msg)
        if self.provider == "csv" and not self.csv_dir:
            msg = "data.csv_dir is required when data.provider is 'csv'."
            raise ValueError(msg)
        return self


class FeatureSpecConfig(Base):
    """One entry in the feature spec list.

    `params` values may be scalars or lists; a list is expanded into the
    cartesian product at pipeline build time (§7.1).
    """

    name: str = Field(min_length=1, description="Name resolved against the feature registry.")
    params: dict[str, Any] = Field(default_factory=dict)


class PruneConfig(Base):
    """Correlation pruning settings (§7.3)."""

    method: Literal["none", "correlation", "clustered_correlation"] = "clustered_correlation"
    threshold: Probability = 0.95


class FeaturesConfig(Base):
    """Which features to build and how to thin them out (§7)."""

    specs: tuple[FeatureSpecConfig, ...] = Field(default_factory=tuple)
    lags: tuple[int, ...] = Field(
        default=(), description="Additional lags applied to every column."
    )
    prune: PruneConfig = Field(default_factory=PruneConfig)
    winsorise_quantile: Probability = Field(
        default=0.001,
        description=(
            "Tail fraction clipped. Fitted inside the sklearn pipeline, on train rows only (I3)."
        ),
    )

    @model_validator(mode="after")
    def _check_lags(self) -> Self:
        """Reject non-positive lags.

        Returns:
            The validated config.

        Raises:
            ValueError: If any lag is less than 1. A lag of 0 is the unlagged
                column; a negative lag is a look-ahead (I1).
        """
        if any(lag < 1 for lag in self.lags):
            msg = f"features.lags must all be >= 1, got {self.lags}. A negative lag is a leak."
            raise ValueError(msg)
        return self


class TripleBarrierConfig(Base):
    """Triple-barrier labelling parameters (§8.1)."""

    profit_mult: PositiveFloat = Field(default=2.0, description="Upper barrier in ATR multiples.")
    loss_mult: PositiveFloat = Field(default=1.0, description="Lower barrier in ATR multiples.")
    vertical_bars: PositiveInt = Field(default=10, description="Vertical barrier, in bars.")
    atr_window: PositiveInt = 14
    min_return: NonNegativeFloat = Field(
        default=0.0, description="Dead zone; labels inside it are dropped rather than called flat."
    )


class SampleWeightsConfig(Base):
    """Sample weighting for overlapping labels (§8.2)."""

    uniqueness: bool = Field(
        default=True, description="Weight by average inverse label concurrency."
    )
    time_decay: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Weight of the oldest observation. 1.0 disables decay.",
    )


class LabelsConfig(Base):
    """What the model is asked to predict (§8)."""

    method: Literal["horizon_sign", "horizon_return", "horizon_quantile", "triple_barrier"] = (
        "triple_barrier"
    )
    horizon_bars: PositiveInt = 10
    triple_barrier: TripleBarrierConfig = Field(default_factory=TripleBarrierConfig)
    meta_labelling: bool = False
    sample_weights: SampleWeightsConfig = Field(default_factory=SampleWeightsConfig)
    n_quantile_bins: PositiveInt = Field(
        default=5, description="Bucket count for horizon_quantile. Edges are fitted per fold."
    )


class WalkForwardConfig(Base):
    """Walk-forward split geometry (§9)."""

    train_bars: PositiveInt = 1260
    test_bars: PositiveInt = 252
    anchored: bool = Field(default=False, description="True expands the train window each fold.")
    step_bars: PositiveInt = 252


class CPCVConfig(Base):
    """Combinatorial purged cross-validation geometry (§9)."""

    n_groups: PositiveInt = 6
    n_test_groups: PositiveInt = 2

    @model_validator(mode="after")
    def _check_groups(self) -> Self:
        """Reject a held-out group count that cannot leave a train set.

        Returns:
            The validated config.

        Raises:
            ValueError: If `n_test_groups` is not strictly less than `n_groups`.
        """
        if self.n_test_groups >= self.n_groups:
            msg = (
                f"cv.cpcv.n_test_groups ({self.n_test_groups}) must be less than "
                f"n_groups ({self.n_groups}); otherwise no training data remains."
            )
            raise ValueError(msg)
        return self


class CVConfig(Base):
    """Cross-validation scheme (§9). Purge and embargo are never optional."""

    scheme: Literal["purged_kfold", "walkforward", "cpcv"] = "walkforward"
    n_splits: PositiveInt = Field(default=5, description="Fold count for purged_kfold.")
    walkforward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    cpcv: CPCVConfig = Field(default_factory=CPCVConfig)
    embargo_pct: Probability = Field(
        default=0.01, description="Fraction of bars after each test window dropped from training."
    )
    min_train_bars: PositiveInt = 500


class TuningConfig(Base):
    """Hyperparameter search (§10.4). The objective is out-of-fold and economic."""

    enabled: bool = True
    n_trials: PositiveInt = 60
    timeout_s: PositiveInt = 1800
    objective: Literal["oos_net_sharpe", "oos_logloss", "oos_information_coefficient"] = (
        "oos_net_sharpe"
    )
    pruner: Literal["none", "median", "hyperband"] = "median"


class ModelsConfig(Base):
    """Candidate models and how their probabilities are calibrated (§10)."""

    candidates: tuple[str, ...] = Field(
        min_length=1,
        description="Model names resolved against the model registry. Baselines are always added.",
    )
    calibrate: Literal["none", "sigmoid", "isotonic"] = "isotonic"
    tuning: TuningConfig = Field(default_factory=TuningConfig)


class SizingConfig(Base):
    """Position sizing (§signal)."""

    method: Literal["fixed_fraction", "vol_target", "kelly_capped"] = "vol_target"
    fixed_fraction: Probability = 0.1
    target_annual_vol: PositiveFloat = 0.10
    max_leverage: PositiveFloat = 1.0
    kelly_cap: Probability = 0.25


class SignalConfig(Base):
    """Turning calibrated probabilities into positions."""

    long_threshold: Probability = 0.56
    short_threshold: Probability = 0.44
    allow_short: bool = True
    hysteresis: Probability = Field(
        default=0.02, description="Extra margin required to flip an existing position."
    )
    sizing: SizingConfig = Field(default_factory=SizingConfig)

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        """Reject thresholds that overlap.

        Returns:
            The validated config.

        Raises:
            ValueError: If the short threshold is not below the long threshold,
                which would make the two signals ambiguous.
        """
        if self.short_threshold >= self.long_threshold:
            msg = (
                f"signal.short_threshold ({self.short_threshold}) must be below "
                f"signal.long_threshold ({self.long_threshold})."
            )
            raise ValueError(msg)
        return self


class RiskConfig(Base):
    """Stops, limits, and the drawdown kill switch (§risk)."""

    stop_loss_atr: PositiveFloat | None = 2.0
    take_profit_atr: PositiveFloat | None = 4.0
    trailing_stop_atr: PositiveFloat | None = None
    max_drawdown_kill: Probability = 0.25
    max_daily_loss: Probability = 0.03
    max_concurrent_positions: PositiveInt = 5


class CostsConfig(Base):
    """The cost model (§12.2). Zero costs are permitted but never silent (I10)."""

    commission_bps: NonNegativeFloat = 0.5
    min_commission: NonNegativeFloat = 1.0
    half_spread_bps: NonNegativeFloat = 1.0
    impact_model: Literal["none", "linear", "sqrt"] = "sqrt"
    impact_coefficient: NonNegativeFloat = 0.1
    max_participation: Probability = Field(
        default=0.05,
        description="Cap on a fill as a fraction of the bar's volume. Excess is partially filled.",
    )
    borrow_bps_annual: NonNegativeFloat = 50.0
    financing_bps_annual: NonNegativeFloat = 300.0

    @property
    def is_frictionless(self) -> bool:
        """Whether this cost model charges nothing at all.

        A frictionless backtest is a legitimate diagnostic — it is what the
        reference-engine cross-check in §12.5 runs against — but presenting one
        as a result is not. A run with this true is flagged `UNCOSTED`.
        """
        return (
            self.commission_bps == 0.0
            and self.half_spread_bps == 0.0
            and self.min_commission == 0.0
            and (self.impact_model == "none" or self.impact_coefficient == 0.0)
        )


class BacktestConfig(Base):
    """Execution assumptions (§12.1)."""

    initial_capital: PositiveFloat = 1_000_000.0
    execution_lag_bars: int = Field(
        default=1,
        ge=1,
        description=(
            "Bars between signal and fill. Minimum 1: same-bar execution is banned at the "
            "engine level (§4, I2)."
        ),
    )
    fill_price: Literal["next_open", "next_vwap"] = "next_open"
    costs: CostsConfig = Field(default_factory=CostsConfig)
    cost_sensitivity_bps: tuple[float, ...] = Field(
        default=(0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
        description="Round-trip cost levels re-run to produce the Sharpe sensitivity curve.",
    )


class ReportingConfig(Base):
    """What gets written out (§11.4)."""

    bootstrap_samples: PositiveInt = 1000
    n_trials_for_dsr: Literal["auto"] | int = Field(
        default="auto",
        description=(
            "Trial count for the deflated Sharpe haircut. 'auto' counts the configurations "
            "actually evaluated; setting it low by hand is self-deception (§10.4)."
        ),
    )
    export: tuple[str, ...] = (
        "tearsheet_json",
        "metrics_csv",
        "trades_csv",
        "predictions_parquet",
        "feature_importance_csv",
        "research_md",
    )


class AlphaConfig(Base):
    """A fully resolved experiment configuration.

    Built by `alphacore.config.loader.load_config`, which layers
    `defaults.yaml`, the experiment YAML, environment overrides, and CLI
    overrides in that order.
    """

    meta: MetaConfig
    data: DataConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    models: ModelsConfig
    signal: SignalConfig = Field(default_factory=SignalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Flags this configuration earns before a single bar is read.

        Returns:
            Warning codes, sorted. `UNCOSTED` means the cost model charges
            nothing (I10); `NO_SHORT_THRESHOLD_USE` means shorting is disabled
            while a short threshold is configured, which is usually a mistake.
        """
        flags: list[str] = []
        if self.backtest.costs.is_frictionless:
            flags.append("UNCOSTED")
        if not self.signal.allow_short and self.signal.short_threshold > 0.0:
            flags.append("SHORT_THRESHOLD_UNUSED")
        return tuple(sorted(flags))
