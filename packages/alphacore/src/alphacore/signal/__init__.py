"""Signal generation and position sizing (§signal)."""

from __future__ import annotations

from alphacore.signal.sizing import (
    fixed_fraction_weight,
    kelly_capped,
    size_positions,
    vol_target_weight,
)
from alphacore.signal.thresholds import (
    FLAT,
    LONG,
    SHORT,
    apply_hysteresis,
    count_flips,
    generate_signal,
    threshold_signal,
)

__all__ = [
    "FLAT",
    "LONG",
    "SHORT",
    "apply_hysteresis",
    "count_flips",
    "fixed_fraction_weight",
    "generate_signal",
    "kelly_capped",
    "size_positions",
    "threshold_signal",
    "vol_target_weight",
]
