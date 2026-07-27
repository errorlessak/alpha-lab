"""Reducing a three-class label frame to a binary target (§8.1, §10.1).

Triple-barrier labelling emits `{-1, 0, +1}`: loss barrier, vertical expiry,
profit barrier. Every classifier in `alphacore.models` is binary. Something has
to give, and the choice matters enough that it is an explicit call rather than
a silent coercion inside the model layer.

Two reductions, and they answer different questions:

* **`direction`** — keep `{-1, +1}`, drop the expiries. The question is "which
  way did it move?". Dropping the neutral class removes the observations where
  nothing happened, which is right for a directional model and wrong if you
  care how often nothing happens.
* **`touch`** — map `+1` to 1 and both `-1` and `0` to 0, keeping everything.
  The question is "did the profit barrier get hit?". This is the reduction
  meta-labelling wants, because "no trade" and "losing trade" are the same
  outcome to a take/skip decision.

Neither is a default the library picks for you. A pipeline that silently threw
away a third of its observations, or silently merged losses with expiries,
would change what the reported accuracy means without saying so.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from alphacore.errors import LabelError

__all__ = ["BinariseMethod", "is_binary", "to_binary"]

BinariseMethod = Literal["direction", "touch"]

_BINARY_CLASSES = 2


def is_binary(labels: pd.DataFrame) -> bool:
    """Report whether a label frame already has exactly two classes.

    Args:
        labels: A label frame.

    Returns:
        True if `label` takes exactly two distinct values.
    """
    return len(pd.unique(labels["label"])) == _BINARY_CLASSES


def to_binary(labels: pd.DataFrame, *, method: str = "direction") -> pd.DataFrame:
    """Reduce a multiclass label frame to `{0, 1}`.

    Args:
        labels: A label frame, normally from `triple_barrier`.
        method: One of `BinariseMethod`. Typed as `str` rather than the Literal
            because it is reachable from a config file, so the value has to be
            validated at runtime rather than assumed by the type checker.

    Returns:
        A new label frame whose `label` is 0 or 1. Sample weights are carried
        through unchanged — they describe label *overlap*, which dropping rows
        does not alter.

    Raises:
        LabelError: If the method is unknown, or the reduction leaves fewer
            than two classes, which would make every fold degenerate.
    """
    if method == "direction":
        kept = labels.loc[labels["label"] != 0].copy()
        kept["label"] = (kept["label"].to_numpy() > 0).astype(int)
    elif method == "touch":
        kept = labels.copy()
        kept["label"] = (kept["label"].to_numpy() > 0).astype(int)
    else:
        msg = f"Unknown binarise method {method!r}. Known: 'direction', 'touch'."
        raise LabelError(msg)

    if kept.empty:
        msg = f"Binarising with method={method!r} removed every observation."
        raise LabelError(msg)
    if len(np.unique(kept["label"])) < _BINARY_CLASSES:
        msg = (
            f"Binarising with method={method!r} left a single class. The barriers are "
            "too wide or too narrow for this series; widen the vertical barrier or "
            "loosen the multiples."
        )
        raise LabelError(msg)

    kept.attrs = {**labels.attrs, "binarised": method}
    counts = kept["label"].value_counts(normalize=True).sort_index()
    kept.attrs["class_balance"] = {str(k): float(v) for k, v in counts.items()}
    return kept
