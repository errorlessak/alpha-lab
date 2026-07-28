"""Result artifacts: the tearsheet, its exports, and the research note (§11.4)."""

from __future__ import annotations

from alphacore.report.exporters import (
    export_frame,
    export_run_artifacts,
    export_series,
    export_table,
    lttb_downsample,
)
from alphacore.report.research_md import RESEARCH_FILENAME, render_research_markdown
from alphacore.report.tearsheet import (
    TEARSHEET_FILENAME,
    TEARSHEET_SCHEMA_VERSION,
    WORSE_THAN_CHANCE_PBO,
    BacktestSection,
    CVSection,
    DatasetSection,
    ExplainSection,
    FeaturesSection,
    FoldSection,
    LabelsSection,
    ManifestSection,
    OverfittingSection,
    Tearsheet,
    validate_tearsheet,
)

__all__ = [
    "RESEARCH_FILENAME",
    "TEARSHEET_FILENAME",
    "TEARSHEET_SCHEMA_VERSION",
    "WORSE_THAN_CHANCE_PBO",
    "BacktestSection",
    "CVSection",
    "DatasetSection",
    "ExplainSection",
    "FeaturesSection",
    "FoldSection",
    "LabelsSection",
    "ManifestSection",
    "OverfittingSection",
    "Tearsheet",
    "export_frame",
    "export_run_artifacts",
    "export_series",
    "export_table",
    "lttb_downsample",
    "render_research_markdown",
    "validate_tearsheet",
]
