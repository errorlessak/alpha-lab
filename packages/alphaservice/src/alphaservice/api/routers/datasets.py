"""The data catalog (§14.1, §6.3).

What has been ingested, over what span, with what content hash, and what the
quality report said about it. The quality report travels with the dataset rather
than being a separate concept in the UI: a dataset nobody has assessed is a
dataset nobody should model on.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from alphacore.config.schema import Interval
from alphacore.data import CATALOG_FILENAME, QUALITY_FILENAME, Catalog, DatasetRecord
from alphacore.data.lake import dataset_dir
from alphacore.errors import AlphaError
from alphaservice.api.deps import SettingsDep
from alphaservice.api.problems import Problem, problem_responses

__all__ = ["router"]

router = APIRouter(tags=["datasets"])


class DatasetList(BaseModel):
    """Every catalogued dataset."""

    datasets: list[DatasetRecord]
    total: int = Field(ge=0)


class DatasetDetail(BaseModel):
    """One dataset and the quality verdict on it."""

    dataset: DatasetRecord
    quality: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The stored quality report (§6.4), or null when the dataset predates one. "
            "Null is not a pass."
        ),
    )


def _catalog(settings: SettingsDep) -> Catalog:
    """Open the catalog.

    Args:
        settings: Service settings.

    Returns:
        The catalog.
    """
    return Catalog(settings.data_dir / CATALOG_FILENAME)


@router.get("/datasets", operation_id="listDatasets", summary="List catalogued datasets")
def list_datasets(settings: SettingsDep) -> DatasetList:
    """List every dataset in the lake.

    Args:
        settings: Service settings.

    Returns:
        The catalog rows.
    """
    records = _catalog(settings).list_datasets()
    return DatasetList(datasets=records, total=len(records))


@router.get(
    "/datasets/{dataset_id}",
    operation_id="getDataset",
    summary="One dataset with its quality report",
    responses=problem_responses(404, 422),
)
def get_dataset(dataset_id: str, settings: SettingsDep) -> DatasetDetail:
    """Read one dataset's catalog row and quality report.

    Args:
        dataset_id: Identifier such as `csv:SYNTH:1d`.
        settings: Service settings.

    Returns:
        The dataset and its quality report.

    Raises:
        Problem: With 404 if the dataset is not catalogued.
    """
    try:
        record = _catalog(settings).require(dataset_id)
    except AlphaError as exc:
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="dataset-not-found",
            title="Dataset not found",
            detail=(
                f"{dataset_id!r} is not in the catalog. Ingest it with "
                "`alpha data pull -c <config>`, or list what is there with GET /api/v1/datasets."
            ),
        ) from exc

    path = (
        dataset_dir(settings.data_dir, record.provider, record.symbol, Interval(record.interval))
        / QUALITY_FILENAME
    )
    quality = json.loads(path.read_text("utf-8")) if path.is_file() else None
    return DatasetDetail(dataset=record, quality=quality)
