"""The feature pipeline: spec list to feature frame (§7.1).

Steps, in order:

1. expand list-valued parameters into their cartesian product
2. resolve each combination against the registry, injecting dataset-level
   context such as `periods_per_year`
3. compute every feature
4. append lags
5. drop `max(lookback)` warm-up rows — dropped, never filled
6. return the frame plus a `FeatureManifest` recording where every column came
   from

Pruning is deliberately **not** step 7. It is a fitted decision and must see
only training rows (§4, I3), and the pipeline has no folds yet. Callers apply
`alphacore.features.selection.prune_correlated` inside the fold, with an
explicit mask.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from alphacore.data.calendars import DEFAULT_EQUITY_CALENDAR, periods_per_year
from alphacore.errors import FeatureError
from alphacore.features import registry as reg
from alphacore.features.transforms import add_lags, drop_warmup, lag_column_name
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from alphacore.config.schema import FeaturesConfig, Interval

__all__ = [
    "ColumnProvenance",
    "FeatureManifest",
    "build_features",
    "column_name",
    "expand_params",
]


class ColumnProvenance(BaseModel):
    """Where one feature column came from.

    Every entry here answers a reviewer's question about a column on a chart:
    what is it, what parameters produced it, how many bars of history does it
    need, and is it lagged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str = Field(description="Registry name, e.g. 'rsi'.")
    params: dict[str, Any] = Field(description="Concrete parameters used.")
    output: str = Field(description="Which output of a multi-output feature, e.g. 'upper'.")
    family: str
    lookback: int = Field(ge=0, description="Bars of warm-up, before any lag.")
    lag: int = Field(default=0, ge=0, description="Bars shifted; 0 for an unlagged column.")
    description: str


class FeatureManifest(BaseModel):
    """The full provenance record for a feature frame, written to `features.json`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: dict[str, ColumnProvenance]
    max_lookback: int = Field(ge=0, description="Rows dropped from the head of the frame.")
    n_columns: int = Field(ge=0)
    interval: str
    calendar: str
    periods_per_year: float = Field(
        gt=0,
        description=(
            "Bars per year for this interval and calendar. Recorded because a metric "
            "annualised by an undocumented factor is unfalsifiable (§11.1)."
        ),
    )

    @property
    def families(self) -> dict[str, int]:
        """Column count by feature family."""
        counts: dict[str, int] = {}
        for entry in self.columns.values():
            counts[entry.family] = counts.get(entry.family, 0) + 1
        return dict(sorted(counts.items()))


def expand_params(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand list-valued parameters into their cartesian product.

    Keys are sorted before expanding, so `{"a": [1, 2], "b": [3]}` and
    `{"b": [3], "a": [1, 2]}` produce identical column names in identical order
    (§4, I8).

    Args:
        params: Parameters, values either scalars or lists.

    Returns:
        One dict per combination. `[{}]` for empty input — a feature with no
        parameters is still computed once.
    """
    if not params:
        return [{}]
    keys = sorted(params)
    values = [v if isinstance(v, list | tuple) else [v] for v in (params[k] for k in keys)]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def column_name(feature: str, params: Mapping[str, Any], output: str) -> str:
    """Build a deterministic, self-describing column name.

    Args:
        feature: Registry name.
        params: Concrete parameters, excluding injected runtime values.
        output: Output name from the compute function.

    Returns:
        e.g. `'sma_window10'`, `'bollinger_k2.0_window20_upper'`. Parameters are
        sorted by name; a single output called `'value'` contributes no suffix.
    """
    parts = [feature]
    parts += [f"{key}{params[key]}" for key in sorted(params)]
    if output != "value":
        parts.append(output)
    return "_".join(parts)


def build_features(
    bars: pd.DataFrame,
    config: FeaturesConfig,
    *,
    interval: Interval,
    calendar: str = DEFAULT_EQUITY_CALENDAR,
) -> tuple[pd.DataFrame, FeatureManifest]:
    """Compute the configured feature frame from validated bars.

    Args:
        bars: Validated bars for one symbol.
        config: The `features` section.
        interval: Bar interval, used for the annualisation factor.
        calendar: Calendar code, used for the annualisation factor.

    Returns:
        The feature frame, warm-up dropped, and its manifest.

    Raises:
        FeatureError: If a spec names an unknown feature, a parameter is not
            accepted, or the warm-up consumes every row.
        DataError: If the calendar is unknown.
    """
    annualisation = periods_per_year(interval, calendar)
    context = {"periods_per_year": annualisation}

    frames: list[pd.DataFrame] = []
    provenance: dict[str, ColumnProvenance] = {}

    for spec_config in config.specs:
        for params in expand_params(spec_config.params):
            spec = reg.resolve(spec_config.name, params, context=context)
            computed = spec.fn(bars, spec.call_params)
            renamed: dict[str, str] = {}
            for output in computed.columns:
                name = column_name(spec.name, spec.params, str(output))
                if name in provenance:
                    msg = (
                        f"Two specs produced the same column {name!r}. Feature names and "
                        "parameters together must be unique."
                    )
                    raise FeatureError(msg)
                renamed[str(output)] = name
                provenance[name] = ColumnProvenance(
                    feature=spec.name,
                    params=dict(spec.params),
                    output=str(output),
                    family=spec.family,
                    lookback=spec.lookback,
                    description=spec.description,
                )
            frames.append(computed.rename(columns=renamed).astype("float64"))

    if not frames:
        msg = "features.specs is empty; there is nothing to build."
        raise FeatureError(msg)

    frame = pd.concat(frames, axis=1)
    frame = frame.loc[:, sorted(frame.columns)]

    base_lookback = max(entry.lookback for entry in provenance.values())
    if config.lags:
        frame = add_lags(frame, config.lags)
        for lag in sorted(set(config.lags)):
            for base in sorted(provenance):
                if provenance[base].lag:
                    continue
                provenance[lag_column_name(base, lag)] = provenance[base].model_copy(
                    update={"lag": lag}
                )
        max_lookback = base_lookback + max(config.lags)
    else:
        max_lookback = base_lookback

    if max_lookback >= len(frame):
        msg = (
            f"The warm-up of {max_lookback} bars leaves no rows in a {len(frame)}-bar "
            "series. Shorten the longest lookback, drop the feature, or ingest more history."
        )
        raise FeatureError(msg)

    frame = drop_warmup(frame, max_lookback)
    frame = frame.loc[:, sorted(frame.columns)]

    remaining_na = frame.columns[frame.isna().any()].tolist()
    if remaining_na:
        msg = (
            f"Columns {remaining_na[:5]} still contain NaN after dropping {max_lookback} "
            "warm-up bars, which means a declared lookback is too short. Fix the "
            "declaration rather than dropping the rows."
        )
        raise FeatureError(msg)

    manifest = FeatureManifest(
        columns={name: provenance[name] for name in sorted(frame.columns)},
        max_lookback=max_lookback,
        n_columns=len(frame.columns),
        interval=interval.value,
        calendar=calendar,
        periods_per_year=annualisation,
    )
    logger.bind(phase="features").info(
        "built {} columns, dropped {} warm-up bars, {} rows remain",
        manifest.n_columns,
        max_lookback,
        len(frame),
    )
    return frame, manifest
