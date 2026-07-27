"""CSV and parquet exporters (§11.4, §16).

Large series live beside `tearsheet.json` rather than inside it. Parquet for
anything the UI streams or downsamples; CSV for the tables a human opens in a
spreadsheet.

Every writer goes through the run directory's write-once guard, so an export
cannot quietly overwrite a previous one (§4, I9).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from alphacore.errors import ArtifactError
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from alphacore.runtime.artifacts import RunDirectory

__all__ = ["export_frame", "export_series", "export_table", "lttb_downsample"]


def export_frame(run: RunDirectory, name: str, frame: pd.DataFrame) -> str:
    """Write a dataframe as parquet into the run directory.

    Args:
        run: The writable run directory.
        name: Artifact name, e.g. `'equity.parquet'`.
        frame: The frame to write.

    Returns:
        The artifact name, for recording in the tearsheet's `series_refs`.

    Raises:
        ArtifactError: If the artifact already exists, or the frame cannot be
            serialised.
    """
    path = run.artifact_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        msg = f"Artifact {name!r} already exists in run {run.run_id!r}; runs are immutable."
        raise ArtifactError(msg)
    try:
        frame.to_parquet(path, compression="zstd", index=True)
    except (OSError, ValueError) as exc:
        msg = f"Could not write {name!r}: {exc}"
        raise ArtifactError(msg) from exc
    logger.bind(phase="report").debug("wrote {} ({} rows)", name, len(frame))
    return name


def export_table(run: RunDirectory, name: str, frame: pd.DataFrame) -> str:
    """Write a dataframe as CSV into the run directory.

    Args:
        run: The writable run directory.
        name: Artifact name, e.g. `'leaderboard.csv'`.
        frame: The frame to write.

    Returns:
        The artifact name.

    Raises:
        ArtifactError: If the artifact already exists.
    """
    # Through write_text so the run's immutability guard and LF normalisation
    # both apply (§12.4 compares artifacts byte for byte).
    run.write_text(name, frame.to_csv(index=True, lineterminator="\n"))
    logger.bind(phase="report").debug("wrote {} ({} rows)", name, len(frame))
    return name


def export_series(run: RunDirectory, name: str, series: pd.Series, *, column: str = "value") -> str:
    """Write a single series as parquet.

    Args:
        run: The writable run directory.
        name: Artifact name.
        series: The series to write.
        column: Column name in the resulting frame.

    Returns:
        The artifact name.

    Raises:
        ArtifactError: If the artifact already exists.
    """
    return export_frame(run, name, series.to_frame(column))


def lttb_downsample(frame: pd.DataFrame, column: str, *, width: int) -> pd.DataFrame:
    """Downsample a series to a pixel width, preserving its visual shape.

    Largest-Triangle-Three-Buckets. Naive decimation drops the extremes, so a
    drawdown that lasted one bar vanishes from the chart entirely — which is
    exactly the bar a reviewer needs to see. LTTB keeps the points that define
    the silhouette.

    §14.1 has the API downsample server-side to the requested pixel width, with
    full resolution only on export.

    Args:
        frame: The source frame, indexed by timestamp.
        column: Column to preserve the shape of.
        width: Target point count, normally the chart's pixel width.

    Returns:
        A frame with at most `width` rows, always including the first and last.

    Raises:
        ArtifactError: If the column is absent or the width is below 3.
    """
    minimum = 3
    if column not in frame.columns:
        msg = f"Cannot downsample on missing column {column!r}."
        raise ArtifactError(msg)
    if width < minimum:
        msg = f"LTTB needs a width of at least {minimum}, got {width}."
        raise ArtifactError(msg)
    n = len(frame)
    if n <= width:
        return frame.copy()

    values = frame[column].to_numpy(dtype=float)
    positions = list(range(n))
    bucket_size = (n - 2) / (width - 2)

    picked = [0]
    previous = 0
    for i in range(width - 2):
        start = int(i * bucket_size) + 1
        end = min(int((i + 1) * bucket_size) + 1, n - 1)
        next_start = min(int((i + 1) * bucket_size) + 1, n - 1)
        next_end = min(int((i + 2) * bucket_size) + 1, n)

        avg_x = sum(positions[next_start:next_end]) / max(1, next_end - next_start)
        avg_y = float(values[next_start:next_end].mean()) if next_end > next_start else 0.0

        best_area, best = -1.0, start
        for candidate in range(start, max(start + 1, end)):
            area = abs(
                (previous - avg_x) * (values[candidate] - values[previous])
                - (previous - candidate) * (avg_y - values[previous])
            )
            if area > best_area:
                best_area, best = area, candidate
        picked.append(best)
        previous = best

    picked.append(n - 1)
    return frame.iloc[sorted(set(picked))].copy()


def export_run_artifacts(
    run: RunDirectory,
    *,
    equity: pd.DataFrame | None = None,
    drawdown: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    leaderboard: Sequence[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write the standard artifact set for a run (§16).

    Args:
        run: The writable run directory.
        equity: Per-bar equity curve.
        drawdown: Per-bar underwater series.
        trades: The blotter.
        predictions: Out-of-fold predictions.
        leaderboard: Leaderboard rows.

    Returns:
        A mapping of logical name to artifact name, for `series_refs`.

    Raises:
        ArtifactError: If any artifact already exists.
    """
    refs: dict[str, str] = {}
    for logical, name, frame in (
        ("equity", "equity.parquet", equity),
        ("drawdown", "drawdown.parquet", drawdown),
        ("trades", "trades.parquet", trades),
        ("predictions", "predictions.parquet", predictions),
    ):
        if frame is not None:
            refs[logical] = export_frame(run, name, frame)

    if trades is not None:
        refs["trades_csv"] = export_table(run, "trades.csv", trades)
    if leaderboard is not None:
        refs["leaderboard_csv"] = export_table(
            run, "leaderboard.csv", pd.DataFrame(list(leaderboard))
        )
    return refs
