"""End-to-end pipelines. Each is a thin composition of library pieces."""

from __future__ import annotations

from alphacore.pipelines.ingest import (
    IngestResult,
    ingest_dataset,
    ingest_symbols,
    resolve_calendar,
)

__all__ = ["IngestResult", "ingest_dataset", "ingest_symbols", "resolve_calendar"]
