"""The canonical bar schema (§6.2) — the type system for market data.

A dataframe that passes `validate_bars` is guaranteed to satisfy every
assumption the rest of the library makes about price history:

* the index is a unique, monotonically increasing, timezone-aware UTC timestamp
* prices are finite and strictly positive
* `low <= min(open, close) <= max(open, close) <= high` on every bar
* volume is non-negative and the adjustment factor is strictly positive

The OHLC ordering check is the one that matters most in practice. A vendor bar
whose `high` is below its `close` will produce a triple-barrier label that can
never be reproduced, and a stop that fires on a price that never traded.

Note on pandas 3: the default resolution for a `DatetimeIndex` is microseconds.
§6.2 specifies nanoseconds, so the schema coerces explicitly rather than
inheriting whatever the pandas default happens to be this release.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from alphacore.errors import SchemaValidationError

__all__ = [
    "BAR_COLUMNS",
    "OHLC_COLUMNS",
    "RAW_BARS_SCHEMA",
    "STRUCTURAL_BARS_SCHEMA",
    "TS_DTYPE",
    "TS_INDEX_NAME",
    "empty_bars",
    "normalise_bars",
    "ohlc_violations",
    "validate_bars",
]

TS_INDEX_NAME: Final[str] = "ts"
TS_DTYPE: Final[pd.DatetimeTZDtype] = pd.DatetimeTZDtype(tz="UTC", unit="ns")

OHLC_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
BAR_COLUMNS: Final[tuple[str, ...]] = (*OHLC_COLUMNS, "volume", "adj_factor", "symbol")

_MAX_REPORTED_FAILURES: Final[int] = 5


def _ohlc_ordering(df: pd.DataFrame) -> pd.Series:
    """Check `low <= min(open, close) <= max(open, close) <= high` per bar.

    Args:
        df: A frame carrying the four OHLC columns.

    Returns:
        A boolean Series, True where the bar is internally consistent.
    """
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    return (df["low"] <= body_low) & (body_high <= df["high"])


def _monotonic_index(df: pd.DataFrame) -> bool:
    """Check the index strictly increases.

    Args:
        df: The frame to check.

    Returns:
        True if timestamps are strictly increasing.
    """
    return bool(df.index.is_monotonic_increasing) and bool(df.index.is_unique)


def _build_schema(*, check_ohlc_ordering: bool, name: str) -> pa.DataFrameSchema:
    """Assemble the bar schema, with OHLC ordering optional.

    §6.2 lists OHLC ordering as a schema check while §6.4 exposes
    `fail_on_ohlc_violation` as a config threshold. Both readings are honoured:
    the strict schema enforces it, so clean data is the default path, and the
    structural schema omits it so `--allow-dirty-data` can let a known-dirty
    vendor series through to a quality report that counts the violations.

    Args:
        check_ohlc_ordering: Whether to include the intra-bar ordering check.
        name: Schema name, surfaced in pandera's error output.

    Returns:
        The assembled schema.
    """
    # The `error` text, not the check name, is what pandera surfaces in
    # `failure_cases`, so the name is repeated inside it.
    checks = [
        pa.Check(
            _monotonic_index,
            name="monotonic_ts",
            error="monotonic_ts: timestamps are not strictly increasing",
        )
    ]
    if check_ohlc_ordering:
        checks.insert(
            0,
            pa.Check(
                _ohlc_ordering,
                name="ohlc_ordering",
                error="ohlc_ordering: low <= min(open, close) <= max(open, close) <= high violated",
            ),
        )
    return pa.DataFrameSchema(
        columns={
            "open": pa.Column(float, checks=[pa.Check.gt(0)], nullable=False),
            "high": pa.Column(float, checks=[pa.Check.gt(0)], nullable=False),
            "low": pa.Column(float, checks=[pa.Check.gt(0)], nullable=False),
            "close": pa.Column(float, checks=[pa.Check.gt(0)], nullable=False),
            "volume": pa.Column(float, checks=[pa.Check.ge(0)], nullable=False),
            "adj_factor": pa.Column(float, checks=[pa.Check.gt(0)], nullable=False),
            "symbol": pa.Column("category", nullable=False),
        },
        index=pa.Index(TS_DTYPE, name=TS_INDEX_NAME, unique=True, coerce=True),
        checks=checks,
        strict=True,
        ordered=True,
        coerce=True,
        name=name,
    )


RAW_BARS_SCHEMA: Final[pa.DataFrameSchema] = _build_schema(check_ohlc_ordering=True, name="RawBars")
STRUCTURAL_BARS_SCHEMA: Final[pa.DataFrameSchema] = _build_schema(
    check_ohlc_ordering=False, name="RawBarsStructural"
)


def ohlc_violations(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Find bars whose high/low do not bracket their open/close.

    A bar whose `high` is below its `close` describes a price that never
    traded. Any stop or barrier evaluated against it produces a fill that could
    not have happened.

    Args:
        df: A frame carrying the four OHLC columns.

    Returns:
        Timestamps of the offending bars, in order.
    """
    if df.empty:
        return pd.DatetimeIndex([], dtype=TS_DTYPE, name=TS_INDEX_NAME)
    bad = ~_ohlc_ordering(df)
    return pd.DatetimeIndex(df.index[bad], name=TS_INDEX_NAME)


def empty_bars() -> pd.DataFrame:
    """Build an empty frame with the canonical bar dtypes.

    Useful as the zero value for a provider that legitimately returns nothing
    for a span, so callers never have to special-case `None`.

    Returns:
        An empty, schema-shaped frame.
    """
    frame = pd.DataFrame(
        {
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "adj_factor": pd.Series(dtype="float64"),
            "symbol": pd.Series(dtype="category"),
        }
    )
    # Built with the dtype up front: pandas refuses to `astype` a tz-naive
    # index to a tz-aware one, even when it is empty.
    frame.index = pd.DatetimeIndex([], dtype=TS_DTYPE, name=TS_INDEX_NAME)
    return frame


def normalise_bars(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    """Coerce a provider's frame into canonical shape, before validation.

    Does the mechanical work a provider should not each reinvent: lowercase
    column names, a UTC nanosecond index named `ts`, a default `adj_factor` of
    1.0, sorted timestamps, and duplicate timestamps resolved by keeping the
    last observation.

    Keeping the *last* duplicate is the point-in-time choice: when a vendor
    sends a bar twice, the second is the correction.

    Args:
        df: The provider's frame. Must carry a `ts` column or a DatetimeIndex.
        symbol: Value for the `symbol` column when the frame does not have one.

    Returns:
        A new frame in canonical column order. Not yet validated.

    Raises:
        SchemaValidationError: If there is no usable timestamp, or a required
            OHLCV column is missing, or `symbol` can be neither found nor
            supplied.
    """
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    if TS_INDEX_NAME in out.columns:
        out = out.set_index(TS_INDEX_NAME)
    elif not isinstance(out.index, pd.DatetimeIndex):
        msg = (
            "Bars need a 'ts' column or a DatetimeIndex; got index of type "
            f"{type(out.index).__name__} and columns {list(out.columns)}."
        )
        raise SchemaValidationError(msg)

    index = pd.DatetimeIndex(out.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    out.index = index.astype(TS_DTYPE)
    out.index.name = TS_INDEX_NAME

    missing = [c for c in (*OHLC_COLUMNS, "volume") if c not in out.columns]
    if missing:
        msg = f"Bars are missing required column(s): {missing}. Present: {list(out.columns)}."
        raise SchemaValidationError(msg)

    if "adj_factor" not in out.columns:
        out["adj_factor"] = 1.0
    if "symbol" not in out.columns:
        if symbol is None:
            msg = "Bars have no 'symbol' column and no symbol was supplied to normalise_bars."
            raise SchemaValidationError(msg)
        out["symbol"] = symbol

    out = out.loc[:, list(BAR_COLUMNS)]
    for column in BAR_COLUMNS[:-1]:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    out["symbol"] = out["symbol"].astype("category")

    out = out.sort_index(kind="stable")
    return out[~out.index.duplicated(keep="last")]


def _format_failures(exc: SchemaErrors) -> str:
    """Render pandera's failure cases into a message a human can act on.

    Args:
        exc: The lazily-collected pandera failure set.

    Returns:
        An indented summary naming each failed check, the count, and the first
        few offending timestamps.
    """
    cases = exc.failure_cases
    lines: list[str] = []
    for check, group in cases.groupby("check", observed=True, sort=True):
        # A dataframe-wide check reports one failure case per column of the
        # offending row, so the same timestamp appears several times. Report
        # distinct rows: "7 failing rows" for one bad bar is worse than useless.
        rows = group["index"].dropna().drop_duplicates()
        indices = [str(value) for value in rows.head(_MAX_REPORTED_FAILURES)]
        where = f" first at {', '.join(indices)}" if indices else ""
        lines.append(f"  {check}: {len(rows)} failing row(s){where}")
    return "\n".join(lines)


def validate_bars(df: pd.DataFrame, *, check_ohlc_ordering: bool = True) -> pd.DataFrame:
    """Validate a frame against the canonical bar schema.

    Args:
        df: A frame already through `normalise_bars`.
        check_ohlc_ordering: Set False to skip the intra-bar ordering check, so
            a known-dirty series can reach the quality report instead of being
            rejected outright. Only `--allow-dirty-data` should pass False, and
            the run manifest is stamped when it does.

    Returns:
        The validated frame, with dtypes coerced.

    Raises:
        SchemaValidationError: If any check fails. The message names every
            failed check with a count and example timestamps, because "schema
            error" alone tells a researcher nothing about which vendor bar to
            go and look at.
    """
    schema = RAW_BARS_SCHEMA if check_ohlc_ordering else STRUCTURAL_BARS_SCHEMA
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        msg = f"Bars failed the RawBars schema (§6.2):\n{_format_failures(exc)}"
        raise SchemaValidationError(msg) from exc
    except SchemaError as exc:  # non-lazy paths, e.g. index coercion failure
        msg = f"Bars failed the RawBars schema (§6.2): {exc}"
        raise SchemaValidationError(msg) from exc
