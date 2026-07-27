"""The label frame contract (§8).

Every labelling method returns the same shape:

| column | meaning |
|---|---|
| index `ts` | `t_start` — the bar the observation is made on |
| `t_end` | the bar the label resolves on |
| `label` | the target |
| `realised_return` | log return from `t_start` close to `t_end` close |
| `weight` | sample weight, attached by `build_labels` |

`t_end` is the load-bearing column. It is what the CV purger consumes to decide
which training samples overlap a test window (§4, I4), and §8.2 is explicit
that nothing else is acceptable. A label frame without it cannot be purged, and
a model trained on unpurged overlapping labels reports an out-of-sample score
that is nothing of the kind.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from alphacore.errors import LabelError

__all__ = ["LABEL_COLUMNS", "TS_START_NAME", "validate_labels"]

TS_START_NAME: Final[str] = "ts"
LABEL_COLUMNS: Final[tuple[str, ...]] = ("t_end", "label", "realised_return", "weight")


def validate_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Check a label frame satisfies the contract.

    Args:
        labels: The frame to check.

    Returns:
        The frame, unchanged.

    Raises:
        LabelError: If a required column is missing, the index is not a sorted
            unique UTC timestamp, `t_end` precedes `t_start`, or a weight is
            not finite and positive.
    """
    missing = [c for c in LABEL_COLUMNS if c not in labels.columns]
    if missing:
        msg = (
            f"Label frame is missing {missing}. `t_end` in particular is what the CV "
            "purger consumes; without it a fold cannot be purged (§8.2)."
        )
        raise LabelError(msg)

    if not isinstance(labels.index, pd.DatetimeIndex):
        msg = f"Label frame index must be a DatetimeIndex, got {type(labels.index).__name__}."
        raise LabelError(msg)
    if labels.index.tz is None:
        msg = "Label frame index must be timezone-aware UTC."
        raise LabelError(msg)
    if not labels.index.is_monotonic_increasing:
        msg = "Label frame index must be sorted."
        raise LabelError(msg)
    if not labels.index.is_unique:
        msg = "Label frame index must be unique; one observation per bar."
        raise LabelError(msg)

    if labels.empty:
        return labels

    if (labels["t_end"] < labels.index).any():
        offenders = labels.index[labels["t_end"] < labels.index][:3].tolist()
        msg = (
            f"t_end precedes t_start at {offenders}. A label that resolves before it is "
            "observed is a look-ahead (§4, I1)."
        )
        raise LabelError(msg)

    if not labels["weight"].between(0.0, float("inf"), inclusive="neither").all():
        msg = "Every sample weight must be finite and strictly positive."
        raise LabelError(msg)

    return labels
