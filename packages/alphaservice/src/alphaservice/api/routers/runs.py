"""Everything readable about a run (§14.1).

`tearsheet.json` is the only source of results (§11.4). The series endpoints
read the parquet beside it and downsample server-side with LTTB, because §13.5
requires a 100k-point equity chart to pan at 60fps and shipping 100k points to
do it is the wrong end of the problem.

Every representation here is immutable, so every one carries an `ETag` and
`Cache-Control: immutable`, and honours `If-None-Match`. That is not a
performance trick — it is invariant I9 restated in HTTP.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Annotated, Any, Final

import pandas as pd
from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from alphacore.cv import FOLDS_FILENAME
from alphacore.errors import RunNotFoundError
from alphacore.models import LeaderboardRow
from alphacore.report import (
    RESEARCH_FILENAME,
    TEARSHEET_FILENAME,
    Tearsheet,
    lttb_downsample,
    validate_tearsheet,
)
from alphaservice.api.deps import (
    RunDir,
    StoreDep,
    apply_immutable_headers,
    get_run,
    not_modified,
    run_etag,
)
from alphaservice.api.problems import Problem, problem_responses

if TYPE_CHECKING:
    from alphacore.runtime.artifacts import ArtifactStore, RunDirectory

__all__ = ["router"]

router = APIRouter(tags=["runs"])

_SERIES: Final[dict[str, str]] = {
    "equity": "equity.parquet",
    "drawdown": "drawdown.parquet",
    "rolling_sharpe": "rolling_sharpe.parquet",
    "trades": "trades.parquet",
    "predictions": "predictions.parquet",
    "ledger": "ledger.parquet",
}
"""Series a client may ask for, mapped to the artifact that holds each. A closed
set: an arbitrary filename parameter is a path traversal waiting to happen."""

_EXPORTABLE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".json", ".csv", ".md", ".parquet", ".yaml"}
)

_MIN_WIDTH: Final[int] = 3
_MAX_WIDTH: Final[int] = 20_000
_MAX_PAGE: Final[int] = 1_000

_NOT_MODIFIED: Final[dict[int | str, dict[str, Any]]] = {
    304: {"description": "The client's cached representation is still current."}
}
"""Documented on every immutable run resource, so the generated client knows a
304 is an expected outcome rather than an error."""


def _cacheable_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    """Document a cacheable route's 304 alongside its problem responses.

    Args:
        *codes: Error statuses the route can return.

    Returns:
        A `responses` mapping.
    """
    return {**_NOT_MODIFIED, **problem_responses(*codes)}


class RunSummaryRow(BaseModel):
    """One row of the run index — what `/` sorts and filters on (§13.4)."""

    run_id: str
    name: str
    status: str
    created_at: str
    duration_s: float | None = None
    seed: int
    config_hash: str
    data_hash: str | None = None
    code_version: str | None = None
    symbols: list[str] = Field(default_factory=list)
    interval: str | None = None
    span: list[str] = Field(default_factory=list)
    selected_model: str | None = None
    oos_net_sharpe: float | None = None
    dsr: float | None = None
    pbo: float | None = None
    warnings: list[str] = Field(default_factory=list)
    has_tearsheet: bool = Field(
        description="False for a run that failed before writing one. Those rows still appear."
    )


class RunPage(BaseModel):
    """A page of the run index."""

    runs: list[RunSummaryRow]
    next_cursor: str | None = Field(
        default=None, description="Pass as `cursor` for the next page. Null on the last page."
    )
    total: int = Field(ge=0, description="Runs matching the filter, across all pages.")


class SeriesResponse(BaseModel):
    """A downsampled series, ready to plot."""

    name: str
    columns: list[str]
    index: list[str]
    values: dict[str, list[float]]
    n_points: int = Field(ge=0, description="Points returned.")
    n_source_points: int = Field(ge=0, description="Points before downsampling.")
    downsampled: bool


class TablePage(BaseModel):
    """A page of a tabular artifact — trades, predictions."""

    rows: list[dict[str, Any]]
    next_cursor: str | None = None
    total: int = Field(ge=0)


class FoldsResponse(BaseModel):
    """Split definitions with their per-fold metrics."""

    scheme: str | None = None
    embargo_pct: float | None = None
    folds: list[dict[str, Any]]


class ConfigResponse(BaseModel):
    """The resolved config, and how it differs from the packaged defaults."""

    run_id: str
    config_yaml: str
    config_hash: str
    diff_vs_defaults: dict[str, Any] = Field(
        description="Flattened `path -> {run, default}` for every value that differs."
    )


class LeaderboardResponse(BaseModel):
    """The leaderboard, ranked."""

    run_id: str
    selected_model: str | None
    rows: list[LeaderboardRow]


class ShapSummaryResponse(BaseModel):
    """Mean absolute SHAP contribution per feature."""

    run_id: str
    model: str | None
    features: list[dict[str, Any]]


class CompareResponse(BaseModel):
    """Two runs side by side (§13.4, `/compare`)."""

    a: RunSummaryRow
    b: RunSummaryRow
    metric_deltas: dict[str, dict[str, float | None]]
    config_identical: bool
    data_identical: bool


def _tearsheet(run: RunDirectory) -> Tearsheet:
    """Read and validate a run's tearsheet.

    Args:
        run: The run directory.

    Returns:
        The validated tearsheet.

    Raises:
        Problem: With 404 and a message naming the phase that failed, when the
            run has no tearsheet. §13.5 requires the error to name the cause.
    """
    try:
        return validate_tearsheet(run.read_json(TEARSHEET_FILENAME))
    except RunNotFoundError as exc:
        status_text = _status_of(run)
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="no-tearsheet",
            title="Run has no result",
            detail=(
                f"Run {run.run_id!r} has no tearsheet.json. Its manifest says "
                f"{status_text!r} — see the job log for the phase that stopped it."
            ),
            extra={"run_id": run.run_id, "run_status": status_text},
        ) from exc


def _status_of(run: RunDirectory) -> str:
    """Read a run's status without raising.

    Args:
        run: The run directory.

    Returns:
        The status string, or `unknown` when the manifest cannot be read.
    """
    try:
        return run.read_manifest().status.value
    except (RunNotFoundError, ValueError):  # pragma: no cover - unreadable manifest
        return "unknown"


def _summary_row(store: ArtifactStore, run_id: str) -> RunSummaryRow | None:
    """Build one index row from a run directory.

    Args:
        store: The artifact store.
        run_id: The run id.

    Returns:
        The row, or None when the directory has no readable manifest — which
        means it is not a run, and listing it would be inventing one.
    """
    try:
        run = store.open_run(run_id)
        manifest = run.read_manifest()
    except (RunNotFoundError, ValueError):
        return None

    row = RunSummaryRow(
        run_id=manifest.run_id,
        name=manifest.name,
        status=manifest.status.value,
        created_at=manifest.created_at.isoformat(),
        duration_s=manifest.duration_s,
        seed=manifest.seed,
        config_hash=manifest.config_hash,
        data_hash=manifest.data_hash,
        code_version=manifest.code_version,
        has_tearsheet=run.exists(TEARSHEET_FILENAME),
    )
    if not row.has_tearsheet:
        return row

    sheet = validate_tearsheet(run.read_json(TEARSHEET_FILENAME))
    headline = sheet.headline
    return row.model_copy(
        update={
            "symbols": list(sheet.dataset.symbols),
            "interval": sheet.dataset.interval,
            "span": [sheet.dataset.span[0].isoformat(), sheet.dataset.span[1].isoformat()],
            "selected_model": sheet.selected_model,
            "oos_net_sharpe": headline["oos_net_sharpe"],
            "dsr": headline["dsr"],
            "pbo": sheet.overfitting.pbo,
            "warnings": list(sheet.warnings),
        }
    )


def _encode_cursor(offset: int) -> str:
    """Encode a pagination offset.

    Args:
        offset: Rows already returned.

    Returns:
        An opaque cursor. Opaque so the pagination strategy can change without
        breaking clients that hoarded one.
    """
    return base64.urlsafe_b64encode(f"o:{offset}".encode()).decode()


def _decode_cursor(cursor: str | None) -> int:
    """Decode a pagination cursor.

    Args:
        cursor: The cursor, or None for the first page.

    Returns:
        The offset.

    Raises:
        Problem: With 400 if the cursor is not one this service issued.
    """
    if not cursor:
        return 0
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode()).decode()
        prefix, _, value = decoded.partition(":")
        if prefix != "o":
            raise ValueError(decoded)
        return max(int(value), 0)
    except (ValueError, UnicodeDecodeError) as exc:
        raise Problem(
            status_code=status.HTTP_400_BAD_REQUEST,
            problem_type="invalid-cursor",
            title="Invalid cursor",
            detail="The `cursor` parameter was not issued by this service. Omit it to start over.",
        ) from exc


@router.get(
    "/runs",
    operation_id="listRuns",
    summary="Paginated run index",
    responses=problem_responses(400, 422),
)
def list_runs(
    store: StoreDep,
    symbol: Annotated[str | None, Query(description="Only runs trading this symbol.")] = None,
    run_status: Annotated[
        str | None, Query(alias="status", description="Only runs with this manifest status.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(description="Opaque cursor from a previous page.")] = None,
) -> RunPage:
    """List runs, newest first.

    Args:
        store: The artifact store.
        symbol: Optional symbol filter.
        run_status: Optional status filter.
        limit: Page size.
        cursor: Opaque cursor from a previous page.

    Returns:
        A page of the index. Runs that failed before writing a tearsheet are
        included with `has_tearsheet: false` — a failed run is a fact about the
        research, and hiding it would make the index a highlight reel.
    """
    offset = _decode_cursor(cursor)
    rows = [row for row in (_summary_row(store, r) for r in store.list_runs()) if row is not None]
    if symbol:
        rows = [row for row in rows if symbol in row.symbols]
    if run_status:
        rows = [row for row in rows if row.status == run_status]

    page = rows[offset : offset + limit]
    has_more = offset + limit < len(rows)
    return RunPage(
        runs=page,
        next_cursor=_encode_cursor(offset + limit) if has_more else None,
        total=len(rows),
    )


@router.get(
    "/runs/compare",
    operation_id="compareRuns",
    summary="Two runs side by side",
    responses=problem_responses(404, 422),
)
def compare_runs(
    store: StoreDep,
    a: Annotated[str, Query(description="First run id.")],
    b: Annotated[str, Query(description="Second run id.")],
) -> CompareResponse:
    """Compare two runs' headline metrics and provenance.

    Args:
        store: The artifact store.
        a: First run id.
        b: Second run id.

    Returns:
        Both index rows, the metric deltas, and whether the two share a config
        or data hash. Identical hashes are worth saying out loud: any difference
        between the runs then came from the code, not the configuration.

    Raises:
        Problem: With 404 if either run is missing.
    """
    left = _summary_row(store, a)
    right = _summary_row(store, b)
    missing = [name for name, row in ((a, left), (b, right)) if row is None]
    if missing or left is None or right is None:
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="run-not-found",
            title="Run not found",
            detail=f"No run {missing[0]!r} in {store.root.as_posix()}.",
        )

    deltas: dict[str, dict[str, float | None]] = {}
    for field_name in ("oos_net_sharpe", "dsr", "pbo", "duration_s"):
        first = getattr(left, field_name)
        second = getattr(right, field_name)
        deltas[field_name] = {
            "a": first,
            "b": second,
            "delta": (second - first) if first is not None and second is not None else None,
        }
    return CompareResponse(
        a=left,
        b=right,
        metric_deltas=deltas,
        config_identical=left.config_hash == right.config_hash,
        data_identical=left.data_hash == right.data_hash and left.data_hash is not None,
    )


@router.get(
    "/runs/{run_id}",
    operation_id="getRun",
    summary="The run's tearsheet",
    response_model=Tearsheet,
    responses=_cacheable_responses(404, 422),
)
def get_tearsheet(run: RunDir, request: Request, response: Response) -> Any:  # noqa: ANN401
    """Serve `tearsheet.json`, the canonical result artifact (§11.4).

    Args:
        run: The run directory.
        request: The request, for `If-None-Match`.
        response: The response, for cache headers.

    Returns:
        The validated tearsheet, or a 304.
    """
    etag = run_etag(run)
    cached = not_modified(request, etag)
    if cached is not None:
        return cached
    apply_immutable_headers(response, etag)
    return _tearsheet(run)


@router.get(
    "/runs/{run_id}/config",
    operation_id="getRunConfig",
    summary="Resolved configuration",
    response_model=ConfigResponse,
    responses=_cacheable_responses(404, 422),
)
def get_config(run: RunDir, request: Request, response: Response) -> Any:  # noqa: ANN401
    """Serve the resolved config and its diff against the packaged defaults.

    Args:
        run: The run directory.
        request: The request, for `If-None-Match`.
        response: The response, for cache headers.

    Returns:
        The config document, or a 304.

    Raises:
        Problem: With 404 if the run has no resolved config.
    """
    etag = run_etag(run, suffix="config")
    cached = not_modified(request, etag)
    if cached is not None:
        return cached

    try:
        text = run.read_bytes("config.resolved.yaml").decode("utf-8")
    except RunNotFoundError as exc:
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="no-config",
            title="Run has no resolved config",
            detail=f"Run {run.run_id!r} did not get far enough to write config.resolved.yaml.",
        ) from exc

    apply_immutable_headers(response, etag)
    return ConfigResponse(
        run_id=run.run_id,
        config_yaml=text,
        config_hash=run.read_manifest().config_hash,
        diff_vs_defaults=_diff_against_defaults(text),
    )


def _diff_against_defaults(config_yaml: str) -> dict[str, Any]:
    """Flatten the difference between a run's config and the packaged defaults.

    Args:
        config_yaml: The run's `config.resolved.yaml`.

    Returns:
        `path -> {run, default}` for every leaf that differs. Values present in
        the run and absent from the defaults appear with a null default.
    """
    import yaml  # noqa: PLC0415 - only needed on this route

    from alphacore.config.loader import load_defaults  # noqa: PLC0415

    run_tree = yaml.safe_load(config_yaml) or {}
    defaults = load_defaults()
    differences: dict[str, Any] = {}

    def walk(left: Any, right: Any, path: str) -> None:  # noqa: ANN401
        if isinstance(left, dict):
            for key, value in sorted(left.items()):
                walk(
                    value,
                    (right or {}).get(key) if isinstance(right, dict) else None,
                    f"{path}.{key}" if path else str(key),
                )
            return
        if left != right:
            differences[path] = {"run": left, "default": right}

    walk(run_tree, defaults, "")
    return differences


@router.get(
    "/runs/{run_id}/folds",
    operation_id="getRunFolds",
    summary="Split definitions",
    response_model=FoldsResponse,
    responses=_cacheable_responses(404, 422),
)
def get_folds(run: RunDir, request: Request, response: Response) -> Any:  # noqa: ANN401
    """Serve the fold definitions and their per-fold spans.

    Args:
        run: The run directory.
        request: The request, for `If-None-Match`.
        response: The response, for cache headers.

    Returns:
        The folds, or a 304.

    Raises:
        Problem: With 404 if the run has no folds.
    """
    etag = run_etag(run, suffix="folds")
    cached = not_modified(request, etag)
    if cached is not None:
        return cached

    try:
        folds = json.loads(run.read_bytes(FOLDS_FILENAME).decode("utf-8"))
    except RunNotFoundError as exc:
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="no-folds",
            title="Run has no folds",
            detail=f"Run {run.run_id!r} stopped before cross-validation.",
        ) from exc

    sheet = (
        validate_tearsheet(run.read_json(TEARSHEET_FILENAME))
        if run.exists(TEARSHEET_FILENAME)
        else None
    )
    apply_immutable_headers(response, etag)
    return FoldsResponse(
        scheme=sheet.cv.scheme if sheet and sheet.cv else None,
        embargo_pct=sheet.cv.embargo_pct if sheet and sheet.cv else None,
        folds=folds,
    )


@router.get(
    "/runs/{run_id}/leaderboard",
    operation_id="getRunLeaderboard",
    summary="Ranked leaderboard",
    response_model=LeaderboardResponse,
    responses=_cacheable_responses(404, 422),
)
def get_leaderboard(run: RunDir, request: Request, response: Response) -> Any:  # noqa: ANN401
    """Serve the leaderboard from the tearsheet.

    Args:
        run: The run directory.
        request: The request, for `If-None-Match`.
        response: The response, for cache headers.

    Returns:
        The leaderboard, or a 304.
    """
    etag = run_etag(run, suffix="leaderboard")
    cached = not_modified(request, etag)
    if cached is not None:
        return cached
    sheet = _tearsheet(run)
    apply_immutable_headers(response, etag)
    return LeaderboardResponse(
        run_id=run.run_id, selected_model=sheet.selected_model, rows=list(sheet.leaderboard)
    )


@router.get(
    "/runs/{run_id}/series/{name}",
    operation_id="getRunSeries",
    summary="Plottable series",
    response_model=SeriesResponse,
    responses=_cacheable_responses(404, 422),
)
def get_series(
    run: RunDir,
    name: str,
    request: Request,
    response: Response,
    width: Annotated[
        int,
        Query(
            ge=_MIN_WIDTH, le=_MAX_WIDTH, description="Target point count, normally chart pixels."
        ),
    ] = 1200,
) -> Any:  # noqa: ANN401
    """Serve a series, downsampled server-side with LTTB.

    Args:
        run: The run directory.
        name: One of `equity`, `drawdown`, `rolling_sharpe`, `trades`,
            `predictions`, `ledger`.
        request: The request, for `If-None-Match`.
        response: The response, for cache headers.
        width: Target point count.

    Returns:
        The series, or a 304.

    Raises:
        Problem: With 404 for an unknown series name or a missing artifact.
    """
    artifact = _SERIES.get(name)
    if artifact is None:
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="unknown-series",
            title="Unknown series",
            detail=f"{name!r} is not a series. Known: {sorted(_SERIES)}.",
        )

    etag = run_etag(run, suffix=f"{name}-{width}")
    cached = not_modified(request, etag)
    if cached is not None:
        return cached

    path = run.artifact_path(artifact)
    if not path.is_file():
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="no-series",
            title="Series not available for this run",
            detail=(
                f"Run {run.run_id!r} has no {artifact}. Its status is "
                f"{_status_of(run)!r}; a run that stopped before the backtest has no equity curve."
            ),
        )

    frame = pd.read_parquet(path)
    numeric = frame.select_dtypes("number")
    source_points = len(frame)
    if numeric.empty:
        reduced = frame.head(width)
    else:
        # LTTB preserves the silhouette of one column; the others are carried
        # along at the same rows so the series stay aligned.
        reduced = lttb_downsample(numeric, str(numeric.columns[0]), width=width)

    apply_immutable_headers(response, etag)
    return SeriesResponse(
        name=name,
        columns=[str(column) for column in reduced.columns],
        index=[str(value) for value in reduced.index],
        values={
            str(column): [float(value) for value in reduced[column].to_numpy()]
            for column in reduced.columns
            if pd.api.types.is_numeric_dtype(reduced[column])
        },
        n_points=len(reduced),
        n_source_points=source_points,
        downsampled=len(reduced) < source_points,
    )


def _table_page(
    run: RunDirectory, artifact: str, *, limit: int, cursor: str | None, label: str
) -> TablePage:
    """Serve a page of a tabular parquet artifact.

    Args:
        run: The run directory.
        artifact: Artifact name.
        limit: Page size.
        cursor: Opaque cursor.
        label: What to call it in an error message.

    Returns:
        The page.

    Raises:
        Problem: With 404 if the artifact is missing.
    """
    path = run.artifact_path(artifact)
    if not path.is_file():
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type=f"no-{label}",
            title=f"Run has no {label}",
            detail=(f"Run {run.run_id!r} has no {artifact}. Its status is {_status_of(run)!r}."),
        )
    offset = _decode_cursor(cursor)
    frame = pd.read_parquet(path)
    page = frame.iloc[offset : offset + limit]
    rows = json.loads(page.reset_index().to_json(orient="records", date_format="iso"))
    has_more = offset + limit < len(frame)
    return TablePage(
        rows=rows,
        next_cursor=_encode_cursor(offset + limit) if has_more else None,
        total=len(frame),
    )


@router.get(
    "/runs/{run_id}/trades",
    operation_id="getRunTrades",
    summary="Paginated trade blotter",
    responses=problem_responses(400, 404, 422),
)
def get_trades(
    run: RunDir,
    limit: Annotated[int, Query(ge=1, le=_MAX_PAGE)] = 200,
    cursor: Annotated[str | None, Query()] = None,
) -> TablePage:
    """Serve the trade blotter, paginated.

    Args:
        run: The run directory.
        limit: Page size.
        cursor: Opaque cursor.

    Returns:
        A page of trades.
    """
    return _table_page(run, "trades.parquet", limit=limit, cursor=cursor, label="trades")


@router.get(
    "/runs/{run_id}/predictions",
    operation_id="getRunPredictions",
    summary="Paginated out-of-fold predictions",
    responses=problem_responses(400, 404, 422),
)
def get_predictions(
    run: RunDir,
    fold: Annotated[int | None, Query(description="Restrict to one fold.")] = None,
    limit: Annotated[int, Query(ge=1, le=_MAX_PAGE)] = 200,
    cursor: Annotated[str | None, Query()] = None,
) -> TablePage:
    """Serve out-of-fold predictions, paginated.

    Args:
        run: The run directory.
        fold: Optional fold filter.
        limit: Page size.
        cursor: Opaque cursor.

    Returns:
        A page of predictions.

    Raises:
        Problem: With 404 if the run has no predictions.
    """
    if fold is None:
        return _table_page(
            run, "predictions.parquet", limit=limit, cursor=cursor, label="predictions"
        )

    path = run.artifact_path("predictions.parquet")
    if not path.is_file():
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="no-predictions",
            title="Run has no predictions",
            detail=f"Run {run.run_id!r} has no predictions.parquet.",
        )
    frame = pd.read_parquet(path)
    frame = frame[frame["fold"] == fold] if "fold" in frame.columns else frame.iloc[:0]
    offset = _decode_cursor(cursor)
    page = frame.iloc[offset : offset + limit]
    rows = json.loads(page.reset_index().to_json(orient="records", date_format="iso"))
    return TablePage(
        rows=rows,
        next_cursor=_encode_cursor(offset + limit) if offset + limit < len(frame) else None,
        total=len(frame),
    )


@router.get(
    "/runs/{run_id}/shap/summary",
    operation_id="getRunShapSummary",
    summary="SHAP summary",
    responses=problem_responses(404, 422),
)
def get_shap_summary(run: RunDir) -> ShapSummaryResponse:
    """Serve the SHAP summary frame.

    Args:
        run: The run directory.

    Returns:
        Mean absolute contribution per feature.

    Raises:
        Problem: With 404 when the run has no SHAP artifact, naming why — only
            tree models are explained (§9), and the smoke config selects a
            baseline.
    """
    path = run.artifact_path("shap_summary.parquet")
    if not path.is_file():
        sheet = (
            validate_tearsheet(run.read_json(TEARSHEET_FILENAME))
            if run.exists(TEARSHEET_FILENAME)
            else None
        )
        selected = sheet.selected_model if sheet else None
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="no-shap",
            title="Run has no SHAP explanation",
            detail=(
                f"Run {run.run_id!r} selected {selected!r}, which is not a tree model. "
                "TreeSHAP is only defined for the boosting models (§9); a linear model's "
                "coefficients already are its explanation."
            ),
            extra={"selected_model": selected},
        )
    frame = pd.read_parquet(path)
    sheet = validate_tearsheet(run.read_json(TEARSHEET_FILENAME))
    return ShapSummaryResponse(
        run_id=run.run_id,
        model=sheet.selected_model,
        features=json.loads(frame.reset_index().to_json(orient="records")),
    )


@router.get(
    "/runs/{run_id}/export/{artifact}",
    operation_id="exportRunArtifact",
    summary="Download one artifact",
    response_class=FileResponse,
    responses=problem_responses(400, 404, 422),
)
def export_artifact(run: RunDir, artifact: str) -> FileResponse:
    """Download a run artifact at full resolution.

    Args:
        run: The run directory.
        artifact: Artifact name, e.g. `tearsheet.json` or `equity.parquet`.

    Returns:
        The file.

    Raises:
        Problem: With 404 if the artifact does not exist, or 400 if the name is
            not an artifact this service will serve.
    """
    # Control characters are checked here as well as the traversal sequences,
    # because a NUL byte is not a name the filesystem can express at all:
    # `Path.resolve()` raises on one rather than reporting the file missing.
    # `ArtifactStore` refuses it too, but as "not a usable path", which maps to
    # a 409 whose title would be wrong. This is a bad request.
    if (
        any(part in artifact for part in ("..", "\\"))
        or artifact.startswith("/")
        or any(character < " " for character in artifact)
    ):
        raise Problem(
            status_code=status.HTTP_400_BAD_REQUEST,
            problem_type="invalid-artifact",
            title="Invalid artifact name",
            detail=(
                "Artifact names are relative to the run directory, may not escape it, "
                "and may not contain control characters."
            ),
        )
    path = run.artifact_path(artifact)
    if path.suffix not in _EXPORTABLE_SUFFIXES or not path.is_file():
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="no-artifact",
            title="Artifact not found",
            detail=(
                f"Run {run.run_id!r} has no {artifact!r}. Exportable suffixes: "
                f"{sorted(_EXPORTABLE_SUFFIXES)}."
            ),
        )
    return FileResponse(
        path,
        filename=f"{run.run_id}-{path.name}",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get(
    "/runs/{run_id}/research",
    operation_id="getRunResearchNote",
    summary="The generated research note",
    response_class=JSONResponse,
    responses=problem_responses(404, 422),
)
def get_research(run: RunDir) -> dict[str, str]:
    """Serve `research.md` as text.

    Args:
        run: The run directory.

    Returns:
        The markdown, under a `markdown` key so the response stays JSON and the
        generated client stays typed.

    Raises:
        Problem: With 404 if the run has no note.
    """
    try:
        return {"run_id": run.run_id, "markdown": run.read_bytes(RESEARCH_FILENAME).decode("utf-8")}
    except RunNotFoundError as exc:
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="no-research-note",
            title="Run has no research note",
            detail=f"Run {run.run_id!r} stopped before the report phase.",
        ) from exc


# Re-exported so `get_run` is importable from this module in tests.
__all__ += ["get_run"]
