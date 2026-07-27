"""Regenerate `reference.csv`, the frozen indicator table (§7.2).

Expected values come from the **naive oracle** in
`tests/golden/reference_indicators.py`, not from `alphacore.features.
primitives`. Freezing the library's own output would only prove it is
self-consistent; freezing an independent transcription of the published formula
means a regression in either implementation shows up as a diff here.

Run deliberately, and only when you can explain the diff:

    uv run python tests/golden/indicators/make_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Runnable as a plain script from anywhere in the repo, without PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.golden.reference_indicators import (
    ref_atr,
    ref_bollinger,
    ref_ema,
    ref_macd,
    ref_rsi,
    ref_sma,
    ref_wma,
)

SEED = 20260727
N_BARS = 60
OUTPUT = Path(__file__).parent / "reference.csv"


def build() -> pd.DataFrame:
    """Build the input series and every frozen expected column."""
    rng = np.random.default_rng(SEED)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(N_BARS) * 0.01))
    open_ = np.concatenate(([close[0]], close[:-1]))
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    high = body_high * (1.0 + np.abs(rng.standard_normal(N_BARS)) * 0.004)
    low = body_low * (1.0 - np.abs(rng.standard_normal(N_BARS)) * 0.004)
    volume = np.round(1e6 * (1.0 + np.abs(rng.standard_normal(N_BARS)) * 0.3))

    closes = close.tolist()
    macd_line, macd_signal, macd_hist = ref_macd(closes, 12, 26, 9)
    bb_upper, bb_lower, bb_pct_b, bb_bandwidth = ref_bollinger(closes, 20, 2.0)

    return pd.DataFrame(
        {
            "bar": np.arange(N_BARS),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "sma_10": ref_sma(closes, 10),
            "ema_12": ref_ema(closes, 12),
            "wma_10": ref_wma(closes, 10),
            "rsi_14": ref_rsi(closes, 14),
            "atr_14": ref_atr(high.tolist(), low.tolist(), closes, 14),
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_pct_b": bb_pct_b,
            "bb_bandwidth": bb_bandwidth,
        }
    )


def main() -> None:
    """Write the frozen table with enough precision to assert at 1e-9."""
    OUTPUT.write_text(
        build().to_csv(index=False, float_format="%.15g", lineterminator="\n"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
