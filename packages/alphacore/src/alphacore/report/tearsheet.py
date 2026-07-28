"""The canonical result artifact (§11.4).

`runs/{run_id}/tearsheet.json` is the **only** thing the API and the UI read for
results. Everything else in a run directory is either an input to it or a large
series it points at.

That single-source rule is what makes the Provenance Stamp in §13.2 truthful:
if a number is on a chart, it came from this file, and this file came from a
run whose config and data hashes are recorded in the same object. A UI that
computed anything itself could show a figure that exists in no artifact.

Large series — equity, drawdown, trades, predictions — live beside this as
parquet and are referenced by name. Inlining a hundred thousand equity points
into a JSON document would make it unreadable and unloadable at once.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from alphacore.errors import EvaluationError
from alphacore.models.leaderboard import LeaderboardRow

__all__ = [
    "TEARSHEET_FILENAME",
    "TEARSHEET_SCHEMA_VERSION",
    "WORSE_THAN_CHANCE_PBO",
    "DatasetSection",
    "FeaturesSection",
    "LabelsSection",
    "OverfittingSection",
    "Tearsheet",
    "validate_tearsheet",
]

TEARSHEET_FILENAME: Final[str] = "tearsheet.json"
TEARSHEET_SCHEMA_VERSION: Final[str] = "1.0"

DISCLAIMER: Final[str] = "Research output. Not investment advice."

WORSE_THAN_CHANCE_PBO: Final[float] = 0.5
"""§11.3: above this, the selection process performs worse than random, and the
UI must say so in plain language rather than print the number and move on."""

_UNRELIABLE_PBO: Final[float] = 0.25


class Base(BaseModel):
    """Base for tearsheet sections: unknown keys rejected, values frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ManifestSection(Base):
    """Provenance: what produced this result."""

    config_hash: str
    data_hash: str | None = None
    code_version: str | None = None
    seed: int
    python: str
    packages: dict[str, str] = Field(default_factory=dict)
    duration_s: float | None = None
    host: str | None = None


class DatasetSection(Base):
    """What data the run was computed on."""

    symbols: tuple[str, ...]
    interval: str
    calendar: str
    span: tuple[dt.date, dt.date]
    n_bars: int = Field(ge=0)
    quality: str
    quality_report_ref: str | None = None
    periods_per_year: float = Field(
        gt=0,
        description=(
            "Bars per year used for every annualised metric. Recorded because a Sharpe "
            "with an undocumented annualisation is unfalsifiable (§11.1)."
        ),
    )


class FeaturesSection(Base):
    """How many features were built and how many survived pruning."""

    n_built: int = Field(ge=0)
    n_after_prune: int = Field(ge=0)
    manifest_ref: str | None = None
    max_lookback: int = Field(ge=0)
    families: dict[str, int] = Field(default_factory=dict)


class LabelsSection(Base):
    """What the model was asked to predict."""

    method: str
    horizon_bars: int = Field(ge=1)
    class_balance: dict[str, float] = Field(default_factory=dict)
    mean_uniqueness: float | None = None
    binarised: str | None = None


class FoldSection(Base):
    """One cross-validation fold, as the Fold Strip draws it (§13.2)."""

    fold: int
    train_span: tuple[dt.datetime, dt.datetime]
    test_span: tuple[dt.datetime, dt.datetime]
    n_train: int = Field(ge=0)
    n_test: int = Field(ge=0)
    purged: int = Field(ge=0)
    embargoed: int = Field(ge=0)


class CVSection(Base):
    """The validation scheme and its folds."""

    scheme: str
    embargo_pct: float = Field(ge=0.0)
    folds: tuple[FoldSection, ...] = ()


class BacktestSection(Base):
    """Backtest results and the cost sensitivity curve."""

    metrics: dict[str, float] = Field(default_factory=dict)
    trade_metrics: dict[str, float] = Field(default_factory=dict)
    cost_sensitivity: tuple[dict[str, float], ...] = ()
    break_even_bps: float | None = None
    round_trip_bps: float | None = None
    kill_switch: dict[str, Any] | None = None
    series_refs: dict[str, str] = Field(default_factory=dict)


class OverfittingSection(Base):
    """The honesty section (§11.3).

    `pbo > 0.5` means the selection process performs worse than chance, and
    §11.3 requires that to be stated in plain language rather than left as a
    number for the reader to interpret.
    """

    pbo: float | None = Field(default=None, ge=0.0, le=1.0)
    median_logit: float | None = None
    n_configs_evaluated: int = Field(default=0, ge=0)
    n_splits: int = Field(default=0, ge=0)
    shuffled_label_sharpe: float | None = None
    shuffled_label_ic: float | None = None

    @property
    def verdict(self) -> str:
        """A plain-language reading of the PBO, for the margin (§13.2)."""
        if self.pbo is None:
            return "Probability of backtest overfitting was not computed for this run."
        if self.pbo > WORSE_THAN_CHANCE_PBO:
            return (
                f"PBO {self.pbo:.2f} — the selection process performs worse than chance. "
                "The configuration chosen here is more likely than not to underperform "
                "the median out of sample."
            )
        if self.pbo > _UNRELIABLE_PBO:
            return (
                f"PBO {self.pbo:.2f} — selection is better than chance, but a quarter to a "
                "half of the time the chosen configuration lands below the median."
            )
        return f"PBO {self.pbo:.2f} — the selection process holds up out of sample."


class ExplainSection(Base):
    """Pointers to the SHAP artifacts (§9)."""

    shap_summary_ref: str | None = None
    top_features: tuple[str, ...] = ()
    n_explained: int = Field(default=0, ge=0)


class Tearsheet(Base):
    """The canonical run result (§11.4)."""

    schema_version: str = TEARSHEET_SCHEMA_VERSION
    run_id: str
    created_at: dt.datetime
    manifest: ManifestSection
    dataset: DatasetSection
    features: FeaturesSection | None = None
    labels: LabelsSection | None = None
    cv: CVSection | None = None
    leaderboard: tuple[LeaderboardRow, ...] = ()
    selected_model: str | None = None
    backtest: BacktestSection | None = None
    overfitting: OverfittingSection = Field(default_factory=OverfittingSection)
    explain: ExplainSection | None = None
    warnings: tuple[str, ...] = ()
    disclaimer: str = DISCLAIMER

    @property
    def headline(self) -> dict[str, Any]:
        """The figures a reviewer reads first, and the ones that qualify them.

        Deliberately returns the Sharpe *and* its haircut, its interval, and
        the cost at which it disappears. §11.2 requires the break-even cost to
        sit next to the headline Sharpe, always.
        """
        top = self.leaderboard[0] if self.leaderboard else None
        return {
            "model": top.model if top else None,
            "oos_net_sharpe": top.oos_net_sharpe if top else None,
            "sharpe_ci": list(top.sharpe_ci) if top and top.sharpe_ci else None,
            "dsr": top.dsr if top else None,
            "psr": top.psr if top else None,
            "break_even_bps": self.backtest.break_even_bps if self.backtest else None,
            "assumed_round_trip_bps": self.backtest.round_trip_bps if self.backtest else None,
            "pbo": self.overfitting.pbo,
            "warnings": list(self.warnings),
        }


def validate_tearsheet(payload: dict[str, Any]) -> Tearsheet:
    """Validate a tearsheet document against the schema.

    Args:
        payload: A parsed `tearsheet.json`.

    Returns:
        The validated tearsheet.

    Raises:
        EvaluationError: If the document does not match the schema. Raised as
            an `EvaluationError` rather than a pydantic error so the API can
            report it as a malformed artifact rather than a server fault.
    """
    try:
        return Tearsheet.model_validate(payload)
    except Exception as exc:
        msg = f"tearsheet.json does not match schema version {TEARSHEET_SCHEMA_VERSION}: {exc}"
        raise EvaluationError(msg) from exc
