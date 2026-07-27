"""The `Split` record and the purge/embargo machinery (§9).

Every scheme in this package produces the same object, and every scheme applies
the same two removals before handing a fold back:

**Purge.** A label spanning `[t_start, t_end]` that overlaps the test window is
dropped from training. Without this, a model trains on a label whose outcome is
determined by bars it is about to be tested on — the single most effective way
to manufacture out-of-sample performance that does not exist. With a ten-bar
horizon, ten training samples on each side of every test block are contaminated.

**Embargo.** A further `embargo_pct` of bars immediately *after* the test window
is dropped from training. Purging handles labels that overlap; the embargo
handles serial correlation that outlives the label. Features are built from
trailing windows, so a training sample starting one bar after the test block
still shares most of its inputs with it.

Both are mandatory. §4 (I4) is explicit that they are not optional flags, and
neither scheme exposes a way to turn purging off.

`purged_idx` and `embargoed_idx` are carried alongside the counts so the UI can
draw the exact gaps that were used (§13.2) rather than inferring them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alphacore.errors import CVError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "FOLDS_FILENAME",
    "Split",
    "apply_purge_and_embargo",
    "splits_from_json",
    "splits_to_json",
]

FOLDS_FILENAME = "folds.json"


@dataclass(frozen=True)
class Split:
    """One fold: which rows train, which test, and what was removed between.

    Attributes:
        fold: Zero-based fold number, in the order the scheme yields them.
        train_idx: Positional indices into the label frame, after purge and
            embargo.
        test_idx: Positional indices into the label frame.
        train_span: First and last training timestamp.
        test_span: First and last test timestamp.
        purged: Count of training samples removed because their label span
            overlapped the test window.
        embargoed: Count removed by the embargo.
        purged_idx: The purged positions, so the UI can draw the gap.
        embargoed_idx: The embargoed positions.
    """

    fold: int
    train_idx: NDArray[np.int64]
    test_idx: NDArray[np.int64]
    train_span: tuple[pd.Timestamp, pd.Timestamp]
    test_span: tuple[pd.Timestamp, pd.Timestamp]
    purged: int
    embargoed: int
    purged_idx: NDArray[np.int64]
    embargoed_idx: NDArray[np.int64]

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-compatible mapping.

        Returns:
            A mapping with index arrays as lists and timestamps as ISO strings.
        """
        return {
            "fold": self.fold,
            "train_idx": [int(i) for i in self.train_idx],
            "test_idx": [int(i) for i in self.test_idx],
            "train_span": [self.train_span[0].isoformat(), self.train_span[1].isoformat()],
            "test_span": [self.test_span[0].isoformat(), self.test_span[1].isoformat()],
            "purged": self.purged,
            "embargoed": self.embargoed,
            "purged_idx": [int(i) for i in self.purged_idx],
            "embargoed_idx": [int(i) for i in self.embargoed_idx],
            "n_train": int(self.train_idx.size),
            "n_test": int(self.test_idx.size),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Split:
        """Rebuild a split from its serialised form.

        Args:
            payload: A mapping produced by `to_dict`.

        Returns:
            The reconstructed split.

        Raises:
            CVError: If a required key is missing.
        """
        required = ("fold", "train_idx", "test_idx", "train_span", "test_span")
        missing = [key for key in required if key not in payload]
        if missing:
            msg = f"Serialised split is missing {missing}."
            raise CVError(msg)
        return cls(
            fold=int(payload["fold"]),
            train_idx=np.asarray(payload["train_idx"], dtype=np.int64),
            test_idx=np.asarray(payload["test_idx"], dtype=np.int64),
            train_span=(
                pd.Timestamp(payload["train_span"][0]),
                pd.Timestamp(payload["train_span"][1]),
            ),
            test_span=(
                pd.Timestamp(payload["test_span"][0]),
                pd.Timestamp(payload["test_span"][1]),
            ),
            purged=int(payload.get("purged", 0)),
            embargoed=int(payload.get("embargoed", 0)),
            purged_idx=np.asarray(payload.get("purged_idx", []), dtype=np.int64),
            embargoed_idx=np.asarray(payload.get("embargoed_idx", []), dtype=np.int64),
        )


def splits_to_json(splits: Sequence[Split]) -> str:
    """Serialise splits to canonical JSON for `folds.json`.

    Args:
        splits: The folds to serialise.

    Returns:
        Canonical JSON — sorted keys, no insignificant whitespace — so two runs
        with the same configuration produce byte-identical fold artifacts
        (§4, I8).
    """
    return json.dumps(
        [split.to_dict() for split in splits],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def splits_from_json(payload: Sequence[dict[str, Any]] | str) -> list[Split]:
    """Rebuild splits from `folds.json`.

    Args:
        payload: The parsed list, or the raw JSON string.

    Returns:
        The reconstructed folds, in file order.

    Raises:
        CVError: If the payload is not a list of split mappings.
    """
    parsed = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(parsed, list):
        msg = f"folds.json must contain a list of splits, got {type(parsed).__name__}."
        raise CVError(msg)
    return [Split.from_dict(item) for item in parsed]


def _check_labels(labels: pd.DataFrame) -> None:
    """Reject a label frame that cannot be split.

    Args:
        labels: The frame to check.

    Raises:
        CVError: If it is empty or lacks `t_end`.
    """
    if labels.empty:
        msg = "Cannot build folds from an empty label frame."
        raise CVError(msg)
    if "t_end" not in labels.columns:
        msg = (
            "Label frame has no `t_end` column, so its spans cannot be purged. "
            "§8.2 requires it on every label frame."
        )
        raise CVError(msg)


def apply_purge_and_embargo(
    labels: pd.DataFrame,
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    *,
    embargo_pct: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Remove training samples that overlap or immediately follow the test set.

    Args:
        labels: The label frame, carrying `t_end`.
        train_idx: Candidate training positions.
        test_idx: Test positions.
        embargo_pct: Fraction of the sample embargoed after the test window.

    Returns:
        `(kept, purged, embargoed)` positional index arrays. The three are
        disjoint and together reproduce `train_idx`.

    Raises:
        CVError: If `embargo_pct` is outside `[0, 1)`.
    """
    if not 0.0 <= embargo_pct < 1.0:
        msg = f"embargo_pct must be in [0, 1), got {embargo_pct}."
        raise CVError(msg)
    if train_idx.size == 0 or test_idx.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return train_idx.astype(np.int64), empty, empty

    index = pd.DatetimeIndex(labels.index)
    t_end = pd.DatetimeIndex(labels["t_end"])

    test_start = index[test_idx].min()
    test_end = max(index[test_idx].max(), t_end[test_idx].max())

    train_start = index[train_idx]
    train_end = t_end[train_idx]

    # Overlap is the negation of "ends before it starts, or starts after it
    # ends" — the standard interval test, written that way to avoid getting the
    # boundary cases wrong.
    overlaps = ~((train_end < test_start) | (train_start > test_end))

    embargo_bars = math.ceil(embargo_pct * len(labels))
    if embargo_bars > 0:
        # The embargo window runs from the end of the test span forward, in
        # bars of the label index.
        last_position = int(index.searchsorted(test_end, side="right"))
        embargo_stop = min(last_position + embargo_bars, len(index)) - 1
        embargo_end = index[embargo_stop] if embargo_stop >= 0 else test_end
        embargoed = (~overlaps) & (train_start > test_end) & (train_start <= embargo_end)
    else:
        embargoed = np.zeros(train_idx.size, dtype=bool)

    kept = ~(overlaps | embargoed)
    return (
        train_idx[kept].astype(np.int64),
        train_idx[overlaps].astype(np.int64),
        train_idx[embargoed].astype(np.int64),
    )


def build_split(
    fold: int,
    labels: pd.DataFrame,
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    *,
    embargo_pct: float,
) -> Split:
    """Purge, embargo, and package one fold.

    Args:
        fold: Fold number.
        labels: The label frame.
        train_idx: Candidate training positions.
        test_idx: Test positions.
        embargo_pct: Fraction of the sample embargoed after the test window.

    Returns:
        The completed split.

    Raises:
        CVError: If the fold has no test samples, or `embargo_pct` is invalid.
    """
    _check_labels(labels)
    if test_idx.size == 0:
        msg = f"Fold {fold} has no test samples."
        raise CVError(msg)

    kept, purged, embargoed = apply_purge_and_embargo(
        labels, np.sort(train_idx), np.sort(test_idx), embargo_pct=embargo_pct
    )
    index = pd.DatetimeIndex(labels.index)
    train_span = (
        (index[kept].min(), index[kept].max())
        if kept.size
        else (index[test_idx].min(), index[test_idx].min())
    )
    return Split(
        fold=fold,
        train_idx=kept,
        test_idx=np.sort(test_idx).astype(np.int64),
        train_span=train_span,
        test_span=(index[test_idx].min(), index[test_idx].max()),
        purged=int(purged.size),
        embargoed=int(embargoed.size),
        purged_idx=purged,
        embargoed_idx=embargoed,
    )
