"""Future-poison: no feature may depend on data it cannot have seen (§12.3.1).

The method: compute a feature on a series; replace everything after index `k`
with garbage; recompute; assert rows `0..k` are unchanged to `1e-12`.

Any dependence on a future bar — a centred window, a `bfill`, a global mean, a
scaler fitted on the whole series — changes those early rows and the test
fails. This is invariant I1 made executable, and it runs over the entire
registry rather than a hand-picked sample.

`test_the_poison_harness_detects_a_real_leak` is the control. Without it, a
harness that silently compared nothing would pass every feature forever, and
this file would be a comfortable lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import Interval
from alphacore.data.calendars import periods_per_year
from alphacore.features import registry as reg
from alphacore.schemas.bars import normalise_bars, validate_bars

TOL = 1e-12
CALENDAR = "XNYS"
N_BARS = 900
POISON_AT = 600
"""Index after which the series is replaced. Chosen so that even the longest
declared lookback (fractional differencing, 282 bars at the default threshold)
has fully warmed up before it."""


def make_bars(seed: int, n: int = N_BARS) -> pd.DataFrame:
    """Build a validated bar frame from a seeded random walk."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2015-01-02", periods=n, freq="D", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    open_ = np.concatenate(([close[0]], close[:-1]))
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    frame = pd.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": body_high * (1.0 + np.abs(rng.standard_normal(n)) * 0.004),
            "low": body_low * (1.0 - np.abs(rng.standard_normal(n)) * 0.004),
            "close": close,
            "volume": 1e6 * (1.0 + np.abs(rng.standard_normal(n)) * 0.3),
            "adj_factor": 1.0,
            "symbol": "SYNTH",
        }
    )
    return validate_bars(normalise_bars(frame))


def poison(bars: pd.DataFrame, at: int = POISON_AT) -> pd.DataFrame:
    """Replace every bar after `at` with a wildly different series.

    Not NaN: a NaN would propagate visibly and could mask a subtler dependence.
    A different but *valid* price path is the harder test — the feature has no
    excuse to notice.
    """
    poisoned = bars.copy()
    rng = np.random.default_rng(999)
    tail = len(bars) - at - 1
    close = 5_000.0 * np.exp(np.cumsum(rng.standard_normal(tail) * 0.05))
    poisoned.iloc[at + 1 :, poisoned.columns.get_loc("close")] = close
    poisoned.iloc[at + 1 :, poisoned.columns.get_loc("open")] = close * 0.99
    poisoned.iloc[at + 1 :, poisoned.columns.get_loc("high")] = close * 1.02
    poisoned.iloc[at + 1 :, poisoned.columns.get_loc("low")] = close * 0.97
    poisoned.iloc[at + 1 :, poisoned.columns.get_loc("volume")] = 9.9e8
    return poisoned


def compute(name: str, bars: pd.DataFrame) -> pd.DataFrame:
    """Compute one registered feature with its default parameters."""
    spec = reg.resolve(
        name, {}, context={"periods_per_year": periods_per_year(Interval.D1, CALENDAR)}
    )
    return spec.fn(bars, spec.call_params).astype("float64")


@pytest.mark.parametrize("name", reg.registered_names())
def test_future_poison(name: str):
    """Corrupting the future must not change a single past value (§4, I1)."""
    clean = make_bars(seed=20260727)
    before = compute(name, clean)
    after = compute(name, poison(clean))

    head_before = before.iloc[: POISON_AT + 1]
    head_after = after.iloc[: POISON_AT + 1]

    assert list(head_before.columns) == list(head_after.columns)
    for column in head_before.columns:
        left = head_before[column].to_numpy(dtype=float)
        right = head_after[column].to_numpy(dtype=float)
        np.testing.assert_allclose(
            left,
            right,
            rtol=0,
            atol=TOL,
            equal_nan=True,
            err_msg=(
                f"{name}.{column} changed in the past when the future was replaced. "
                "The feature can see data it could not have had (§4, I1)."
            ),
        )


def test_the_poisoned_future_really_is_different():
    """The poison must actually change the tail, or the test proves nothing."""
    clean = make_bars(seed=1)
    dirty = poison(clean)
    assert not np.allclose(
        clean["close"].to_numpy()[POISON_AT + 1 :],
        dirty["close"].to_numpy()[POISON_AT + 1 :],
    )
    # ...and must leave the past untouched, or every feature would fail.
    np.testing.assert_array_equal(
        clean["close"].to_numpy()[: POISON_AT + 1],
        dirty["close"].to_numpy()[: POISON_AT + 1],
    )


def test_the_poison_harness_detects_a_real_leak():
    """The control: a knowingly non-causal feature must fail this harness.

    A centred rolling mean averages `window // 2` bars on either side, so its
    value at `t` depends on `t + 1`. If the harness cannot catch that, it
    cannot catch anything, and every passing feature above means nothing.
    """
    clean = make_bars(seed=7)

    def leaky(bars: pd.DataFrame) -> pd.DataFrame:
        centred = bars["close"].rolling(21, center=True, min_periods=21).mean()
        return pd.DataFrame({"value": centred}, index=bars.index)

    before = leaky(clean).iloc[: POISON_AT + 1]["value"].to_numpy(dtype=float)
    after = leaky(poison(clean)).iloc[: POISON_AT + 1]["value"].to_numpy(dtype=float)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(before, after, rtol=0, atol=TOL, equal_nan=True)


def test_every_registered_feature_is_covered():
    """The parametrisation must span the registry, not a subset of it."""
    assert len(reg.registered_names()) >= 25
