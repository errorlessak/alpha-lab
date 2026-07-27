"""Causal feature engineering (§7).

Importing this package registers every built-in feature, so
`registry.registered_names()` is populated from here on.
"""

from __future__ import annotations

from alphacore.features import primitives
from alphacore.features.pipeline import (
    ColumnProvenance,
    FeatureManifest,
    build_features,
    column_name,
    expand_params,
)
from alphacore.features.registry import (
    ALLOWED_INPUTS,
    FAMILIES,
    FORBIDDEN_INPUTS,
    FeatureDefinition,
    FeatureSpec,
    emit_data_dictionary,
    get,
    register,
    registered_names,
    resolve,
)
from alphacore.features.selection import (
    clustered_correlation_prune,
    correlation_prune,
    prune_correlated,
)
from alphacore.features.transforms import add_lags, drop_warmup, lag_column_name

__all__ = [
    "ALLOWED_INPUTS",
    "FAMILIES",
    "FORBIDDEN_INPUTS",
    "ColumnProvenance",
    "FeatureDefinition",
    "FeatureManifest",
    "FeatureSpec",
    "add_lags",
    "build_features",
    "clustered_correlation_prune",
    "column_name",
    "correlation_prune",
    "drop_warmup",
    "emit_data_dictionary",
    "expand_params",
    "get",
    "lag_column_name",
    "primitives",
    "prune_correlated",
    "register",
    "registered_names",
    "resolve",
]
