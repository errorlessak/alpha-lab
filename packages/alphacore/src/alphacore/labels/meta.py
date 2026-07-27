"""Meta-labelling (§8.1).

A primary rule — a momentum crossover, say — decides the *side*. The model then
answers only one question: **should this trade be taken at all?**

Splitting the problem this way is worth the extra machinery. A directional
model has to be right about direction and about magnitude and about timing; a
meta-model only has to recognise when the primary rule is in conditions where it
works. That is an easier problem, it produces a probability that maps directly
onto position sizing, and it raises precision without touching the primary
rule's recall.

Bars the primary rule stayed out of are dropped. A trade that was never taken
is not evidence about whether to take it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacore.errors import LabelError

__all__ = ["meta_label"]


def meta_label(labels: pd.DataFrame, side: pd.Series) -> pd.DataFrame:
    """Convert directional labels into take/skip labels for a primary rule.

    `label = 1` when the primary side would have profited, `0` otherwise.

    Args:
        labels: A label frame carrying `realised_return`, normally from
            `triple_barrier`.
        side: The primary rule's side per bar: `+1` long, `-1` short, `0` flat.
            Must share the label frame's index.

    Returns:
        A label frame with a binary `label` and the primary `side` retained as
        a column — the model needs it as a feature, and the backtester needs it
        to know which way to trade.

    Raises:
        LabelError: If the indexes do not match, or the side contains values
            outside `{-1, 0, 1}`.
    """
    if not labels.index.equals(side.index):
        msg = (
            "meta_label needs the primary side on exactly the label frame's index; "
            f"got {len(side)} side values for {len(labels)} labels."
        )
        raise LabelError(msg)

    sides = side.to_numpy()
    if not np.isin(sides, (-1, 0, 1)).all():
        msg = f"Primary side must be in {{-1, 0, 1}}, got {sorted(set(sides.tolist()))}."
        raise LabelError(msg)

    taken = sides != 0
    if not taken.any():
        msg = "The primary rule never takes a position, so there is nothing to meta-label."
        raise LabelError(msg)

    out = labels.loc[taken].copy()
    signed = sides[taken] * out["realised_return"].to_numpy(dtype=float)
    out["label"] = (signed > 0).astype(int)
    out["side"] = sides[taken].astype(int)
    return out
