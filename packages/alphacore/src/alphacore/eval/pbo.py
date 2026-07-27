"""Probability of backtest overfitting, via CSCV (§11.3).

The question PBO answers is not "is this strategy good?" but **"is my selection
process better than picking at random?"**

The method: split the out-of-sample return matrix into `S` contiguous chunks.
For each of the `C(S, S/2)` ways of splitting those chunks into a training half
and a testing half, pick the configuration with the best in-sample Sharpe, then
find where it ranks out of sample. PBO is the fraction of splits where the
in-sample winner lands below the out-of-sample median.

`PBO > 0.5` means the selection procedure is *worse than a coin flip*, and §11.3
requires the UI to say so in plain language rather than printing a number and
moving on.

Contiguous chunks, not random ones: the whole point is to preserve the time
ordering that makes a financial backtest a backtest.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from alphacore.errors import EvaluationError

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["DEFAULT_CHUNKS", "cscv_pbo"]

DEFAULT_CHUNKS: Final[int] = 16
_MIN_CONFIGS: Final[int] = 2
_MIN_CHUNKS: Final[int] = 2


def _sharpe_columns(block: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-configuration Sharpe over a block of returns.

    Annualisation is deliberately omitted: PBO compares *ranks* within a split,
    and a constant factor cannot change a rank.

    Args:
        block: `(n_observations, n_configs)` returns.

    Returns:
        One Sharpe per configuration; zero where a column has no dispersion.
    """
    deviation = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(deviation > 0, block.mean(axis=0) / deviation, 0.0)


def cscv_pbo(returns: NDArray[np.float64], *, n_chunks: int = DEFAULT_CHUNKS) -> dict[str, Any]:
    """Compute the probability of backtest overfitting (§11.3).

    Args:
        returns: An `(n_observations, n_configs)` matrix of out-of-sample
            returns, one column per configuration evaluated.
        n_chunks: Number of contiguous chunks, `S`. Must be even, since each
            split takes exactly half.

    Returns:
        A mapping with `pbo`, `n_configs`, `n_chunks`, `n_splits`, the median
        logit, and the fraction of splits where the in-sample winner beat the
        out-of-sample median.

    Raises:
        EvaluationError: If there are fewer than two configurations, the chunk
            count is odd or too small, or the sample cannot be split.
    """
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:  # noqa: PLR2004
        msg = f"CSCV needs a 2-D (observations, configs) matrix, got shape {matrix.shape}."
        raise EvaluationError(msg)

    n_obs, n_configs = matrix.shape
    if n_configs < _MIN_CONFIGS:
        msg = (
            f"PBO compares configurations against each other and needs at least "
            f"{_MIN_CONFIGS}, got {n_configs}. With one configuration there is no "
            "selection process to evaluate."
        )
        raise EvaluationError(msg)
    if n_chunks < _MIN_CHUNKS or n_chunks % 2 != 0:
        msg = f"n_chunks must be an even number of at least {_MIN_CHUNKS}, got {n_chunks}."
        raise EvaluationError(msg)
    if n_obs < n_chunks * _MIN_CHUNKS:
        msg = f"Need at least {n_chunks * _MIN_CHUNKS} observations for {n_chunks} chunks."
        raise EvaluationError(msg)

    bounds = np.linspace(0, n_obs, n_chunks + 1).astype(int)
    chunks = [matrix[bounds[i] : bounds[i + 1]] for i in range(n_chunks)]

    half = n_chunks // 2
    logits: list[float] = []
    beat_median = 0

    for train_ids in combinations(range(n_chunks), half):
        test_ids = [i for i in range(n_chunks) if i not in train_ids]
        in_sample = np.vstack([chunks[i] for i in train_ids])
        out_sample = np.vstack([chunks[i] for i in test_ids])

        best = int(np.argmax(_sharpe_columns(in_sample)))
        out_sharpes = _sharpe_columns(out_sample)

        # Relative rank of the in-sample winner, in (0, 1).
        rank = float((out_sharpes <= out_sharpes[best]).sum()) / (n_configs + 1)
        rank = float(np.clip(rank, 1e-9, 1.0 - 1e-9))
        logits.append(float(np.log(rank / (1.0 - rank))))
        if rank > 0.5:  # noqa: PLR2004
            beat_median += 1

    n_splits = len(logits)
    return {
        "pbo": float(sum(1 for value in logits if value <= 0) / n_splits),
        "median_logit": float(np.median(logits)),
        "n_configs": int(n_configs),
        "n_chunks": int(n_chunks),
        "n_splits": int(n_splits),
        "fraction_beating_median": float(beat_median / n_splits),
    }
