"""The out-of-fold leaderboard (§10.1, ADR 0007).

Ranked by **out-of-fold net-of-cost Sharpe with a deflated-Sharpe haircut**,
not by accuracy. Accuracy on price direction is close to meaningless: it
ignores magnitude, it ignores turnover, and optimising it produces models that
are right about small moves and wrong about large ones.

The economic columns need a cost model and a backtester, which arrive in
Phase 7. Until then they are **`None`**, and every row carries a `NO_BACKTEST`
flag. Filling them with a placeholder number would be worse than leaving them
empty — someone would read it.

Two rows are always present regardless of what the config asked for:
`baseline_buyhold`, because a strategy that cannot beat it is not a strategy,
and `baseline_shuffled`, because it is the only row that says what this
pipeline scores on pure noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from alphacore.errors import ModelError
from alphacore.eval.classification import classification_metrics
from alphacore.models.baselines import MANDATORY_BASELINES
from alphacore.models.oof import fit_predict_oof
from alphacore.models.registry import build_model
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from alphacore.config.schema import ModelsConfig, PruneConfig
    from alphacore.cv import Split
    from alphacore.models.oof import OOFPredictions

__all__ = [
    "ECONOMIC_COLUMNS",
    "WEAK_DSR",
    "LeaderboardRow",
    "attach_economics",
    "build_leaderboard",
    "leaderboard_row",
    "score_models",
    "validate_leaderboard",
]

ECONOMIC_COLUMNS: Final[tuple[str, ...]] = (
    "oos_net_sharpe",
    "sharpe_ci",
    "dsr",
    "psr",
    "break_even_bps",
    "max_drawdown",
)
"""Columns the backtester fills in from Phase 7. `None` until then, and the
`NO_BACKTEST` flag says so on every row."""

_SUSPICIOUS_ACCURACY: Final[float] = 0.55
"""§19: accuracy above roughly this on next-bar direction almost certainly
means leakage. Real short-horizon edges live at 0.51 to 0.54."""

_WEAK_IC_TSTAT: Final[float] = 2.0

WEAK_DSR: Final[float] = 0.5
"""§11.2: below this, the deflated Sharpe says the result is not distinguishable
from the best of a set of coin flips, however good the raw Sharpe looks.

Public because the run-level `LOW_DSR` warning in the tearsheet has to be
raised on exactly the same threshold as the row-level flag. Two copies would
drift, and the tearsheet would then contradict its own leaderboard."""


class LeaderboardRow(BaseModel):
    """One leaderboard entry, as it appears in `tearsheet.json` (§11.4).

    Defined here rather than in `report/` because the leaderboard is built
    before the tearsheet exists. Phase 8 embeds this model in the tearsheet
    schema rather than restating it, so the two cannot drift.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    rank: int = Field(ge=1)
    n_predictions: int = Field(ge=0)
    n_folds: int = Field(ge=0)
    n_trials: int = Field(
        ge=1,
        description=(
            "Configurations evaluated for this row. The N in the deflated Sharpe "
            "haircut (§11.2); collected automatically because a hand-entered trial "
            "count is self-deception (§10.4)."
        ),
    )
    flags: list[str] = Field(default_factory=list)
    error: str | None = None

    accuracy: float | None = None
    balanced_accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    auc: float | None = None
    pr_auc: float | None = None
    brier: float | None = None
    log_loss: float | None = None
    ic: float | None = None
    ic_tstat: float | None = None

    oos_net_sharpe: float | None = None
    sharpe_ci: tuple[float, float] | None = None
    dsr: float | None = None
    psr: float | None = None
    break_even_bps: float | None = None
    max_drawdown: float | None = None


def validate_leaderboard(rows: Sequence[dict[str, Any]]) -> list[LeaderboardRow]:
    """Validate leaderboard rows against the schema.

    Args:
        rows: Rows from `build_leaderboard`.

    Returns:
        The parsed rows.

    Raises:
        ModelError: If any row does not match the schema. Naming the offending
            row matters: a leaderboard that fails validation at the tearsheet
            stage would otherwise surface as an unattributed pydantic error at
            the end of a long run.
    """
    parsed: list[LeaderboardRow] = []
    for row in rows:
        try:
            parsed.append(LeaderboardRow.model_validate(row))
        except ValidationError as exc:
            msg = f"Leaderboard row for {row.get('model')!r} does not match the schema: {exc}"
            raise ModelError(msg) from exc
    return parsed


def leaderboard_row(oof: OOFPredictions, *, n_trials: int) -> dict[str, Any]:
    """Score one model's out-of-fold predictions.

    Args:
        oof: Stacked out-of-fold predictions.
        n_trials: Configurations evaluated to produce this row. Feeds the
            deflated Sharpe haircut in Phase 8; recorded now so it cannot be
            back-filled by hand later (§10.4).

    Returns:
        A JSON-serialisable row.

    Raises:
        ModelError: If the predictions are degenerate — one class, or a
            constant score, both of which make the metrics undefined.
    """
    frame = oof.frame
    y = frame["label"].to_numpy()
    proba = frame["proba"].to_numpy(dtype=float)
    forward = frame["realised_return"].to_numpy(dtype=float)

    flags: list[str] = ["NO_BACKTEST"]
    if oof.model == "baseline_shuffled":
        flags.append("SHUFFLED_LABELS")

    if np.ptp(proba) == 0:
        # A constant score is what every deterministic baseline produces. It is
        # a legitimate row, but rank metrics are undefined on it, so the honest
        # move is to record that rather than invent a number.
        metrics: dict[str, float | None] = dict.fromkeys(
            (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "pr_auc",
                "brier",
                "log_loss",
                "ic",
                "ic_tstat",
            ),
            None,
        )
        metrics["accuracy"] = float(np.mean((proba >= 0.5).astype(int) == y))  # noqa: PLR2004
        flags.append("CONSTANT_PREDICTION")
    else:
        metrics = dict(classification_metrics(y, proba, forward_return=forward))
        if metrics["accuracy"] is not None and metrics["accuracy"] > _SUSPICIOUS_ACCURACY:
            flags.append("SUSPICIOUS_ACCURACY")
        if metrics["ic_tstat"] is not None and abs(metrics["ic_tstat"]) < _WEAK_IC_TSTAT:
            flags.append("WEAK_IC")

    row: dict[str, Any] = {
        "model": oof.model,
        "n_predictions": len(frame),
        "n_folds": int(oof.n_folds),
        "n_trials": int(n_trials),
        "rank": 0,
        "flags": sorted(set(flags)),
        **metrics,
    }
    row.update(dict.fromkeys(ECONOMIC_COLUMNS, None))
    return row


def attach_economics(
    rows: Sequence[dict[str, Any]],
    economics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fill the economic columns from costed backtests and re-rank (ADR 0007).

    This is the step that turns the leaderboard from a prediction-quality table
    into an economic one. Ranking switches with it: `_sort_key` already prefers
    net Sharpe over the information coefficient's t-statistic, so a row that
    gains an `oos_net_sharpe` here rises or falls on that instead.

    A model with no entry in `economics` keeps its `NO_BACKTEST` flag and its
    empty economic columns. It is not dropped — a candidate that could not be
    backtested is a fact about the run, and hiding it would make the leaderboard
    read as though it never existed.

    Args:
        rows: Rows from `build_leaderboard`.
        economics: Per-model mapping of economic column name to value. Keys
            outside `ECONOMIC_COLUMNS` are rejected rather than ignored, since
            a misspelled key would otherwise leave the column empty and the
            row flagged `NO_BACKTEST` with no indication why.

    Returns:
        New rows, re-sorted best-first with dense 1-based ranks.

    Raises:
        ModelError: If `economics` names a model absent from `rows`, or carries
            a column outside `ECONOMIC_COLUMNS`.
    """
    known = {row["model"] for row in rows}
    unknown_models = sorted(set(economics) - known)
    if unknown_models:
        msg = (
            f"Economic results were supplied for {unknown_models}, which are not on the "
            f"leaderboard. Known models: {sorted(known)}."
        )
        raise ModelError(msg)

    updated: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        result = economics.get(str(row["model"]))
        if result is not None:
            unknown_columns = sorted(set(result) - set(ECONOMIC_COLUMNS))
            if unknown_columns:
                msg = (
                    f"Economic result for {row['model']!r} carries unknown columns "
                    f"{unknown_columns}. Known: {list(ECONOMIC_COLUMNS)}."
                )
                raise ModelError(msg)
            new_row.update({column: result.get(column) for column in ECONOMIC_COLUMNS})
            flags = [flag for flag in new_row.get("flags", []) if flag != "NO_BACKTEST"]
            dsr = new_row.get("dsr")
            if dsr is not None and dsr < WEAK_DSR:
                flags.append("LOW_DSR")
            new_row["flags"] = sorted(set(flags))
        updated.append(new_row)

    updated.sort(key=_sort_key)
    for position, row in enumerate(updated, start=1):
        row["rank"] = position

    logger.bind(phase="backtest").info(
        "economic columns attached to {} of {} leaderboard rows; best is {}",
        len(economics),
        len(updated),
        updated[0]["model"] if updated else "nothing",
    )
    return updated


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    """Rank rows, falling back down the metric hierarchy.

    Ranking is by net Sharpe once the backtester exists. Until then it is by
    the information coefficient's t-statistic, which is the closest available
    proxy for "does this rank opportunities better than chance" — and
    deliberately *not* accuracy (ADR 0007).

    Args:
        row: A leaderboard row.

    Returns:
        A descending sort key.
    """
    sharpe = row.get("oos_net_sharpe")
    tstat = row.get("ic_tstat")
    auc = row.get("auc")
    return (
        -(sharpe if sharpe is not None else -np.inf),
        -(abs(tstat) if tstat is not None else -np.inf),
        -(auc if auc is not None else -np.inf),
    )


def build_leaderboard(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: Sequence[Split],
    config: ModelsConfig,
    *,
    seed: int,
    n_trials: int = 1,
    prune: PruneConfig | None = None,
) -> list[dict[str, Any]]:
    """Fit every candidate and baseline out-of-fold and rank the results.

    Args:
        features: Feature frame.
        labels: Label frame.
        splits: Purged folds.
        config: The `models` section.
        seed: Run seed.
        n_trials: Configurations evaluated per model, for the DSR haircut.
        prune: Correlation pruning, applied per fold inside the OOF loop.

    Returns:
        Rows sorted best-first, each carrying a dense 1-based `rank`.

    Raises:
        ModelError: If no model could be scored.
    """
    rows, _ = score_models(
        features, labels, splits, config, seed=seed, n_trials=n_trials, prune=prune
    )
    return rows


def score_models(  # noqa: PLR0913 - each argument is a distinct scoring input
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: Sequence[Split],
    config: ModelsConfig,
    *,
    seed: int,
    n_trials: int = 1,
    prune: PruneConfig | None = None,
    n_trials_by_model: Mapping[str, int] | None = None,
    params_by_model: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, OOFPredictions]]:
    """Score every candidate and baseline, keeping the predictions.

    `build_leaderboard` is this function with the predictions discarded. The
    pipelines need to keep them: the backtest that fills the economic columns
    runs on the same out-of-fold probabilities the row was scored from, and
    computing them twice would risk the leaderboard and the equity curve
    describing different predictions.

    Args:
        features: Feature frame.
        labels: Label frame.
        splits: Purged folds.
        config: The `models` section.
        seed: Run seed.
        n_trials: Configurations evaluated per model, for the DSR haircut.
        prune: Correlation pruning, applied per fold inside the OOF loop.
        n_trials_by_model: Per-model trial counts from a tuning study, which
            override `n_trials`. Collected automatically because a hand-entered
            trial count is self-deception (§10.4).
        params_by_model: Winning hyperparameters from a tuning study, applied
            when the model is built. A model absent from it is built with the
            registry's defaults.

    Returns:
        The ranked rows, and the out-of-fold predictions by model name. Models
        that failed appear in the rows with a `FAILED` flag and are absent from
        the predictions.

    Raises:
        ModelError: If no model could be scored.
    """
    # Baselines are appended rather than substituted: a config that names one
    # explicitly should not cause it to be scored twice.
    names = list(dict.fromkeys([*config.candidates, *MANDATORY_BASELINES]))
    trials = dict(n_trials_by_model or {})
    tuned = dict(params_by_model or {})

    rows: list[dict[str, Any]] = []
    predictions: dict[str, OOFPredictions] = {}
    for name in names:
        model_trials = trials.get(name, n_trials)
        try:
            model = build_model(
                name, seed=seed, calibrate=config.calibrate, **dict(tuned.get(name, {}))
            )
            oof = fit_predict_oof(model, features, labels, splits, prune=prune)
            predictions[name] = oof
            rows.append(leaderboard_row(oof, n_trials=model_trials))
        except ModelError as exc:
            # One unavailable estimator must not take the whole leaderboard
            # down, but the gap is recorded rather than silently dropped.
            logger.bind(phase="train").warning("{} could not be scored: {}", name, exc)
            rows.append(
                {
                    "model": name,
                    "n_predictions": 0,
                    "n_folds": 0,
                    "n_trials": int(model_trials),
                    "rank": 0,
                    "flags": ["FAILED"],
                    "error": str(exc),
                    **dict.fromkeys(ECONOMIC_COLUMNS, None),
                }
            )

    if not any(row["flags"] != ["FAILED"] for row in rows):
        msg = "No model on the leaderboard could be scored; every candidate failed."
        raise ModelError(msg)

    rows.sort(key=_sort_key)
    for position, row in enumerate(rows, start=1):
        row["rank"] = position

    logger.bind(phase="train").info("leaderboard: {} models, best {}", len(rows), rows[0]["model"])
    return rows, predictions
