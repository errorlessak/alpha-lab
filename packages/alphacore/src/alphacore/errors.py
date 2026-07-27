"""Typed exception hierarchy for the whole library.

Every failure raised by `alphacore` derives from `AlphaError`. Bare `ValueError`
with a message string is banned (§18 of ARCHITECTURE.md): a caller must be able
to distinguish "your config is wrong" from "the data is dirty" from "the
pipeline just leaked the future" without string matching.
"""

from __future__ import annotations

__all__ = [
    "AlphaError",
    "ArtifactError",
    "ArtifactExistsError",
    "BacktestError",
    "CVError",
    "ConfigError",
    "DataError",
    "DataQualityError",
    "DeterminismError",
    "EvaluationError",
    "FeatureError",
    "LabelError",
    "LeakageError",
    "ModelError",
    "ProviderError",
    "RunNotFoundError",
    "SchemaValidationError",
    "SerialisationError",
]


class AlphaError(Exception):
    """Base class for every error raised by alphacore."""


class ConfigError(AlphaError):
    """A configuration value is missing, unknown, or internally inconsistent."""


class SerialisationError(AlphaError):
    """A value cannot be canonically serialised, so it cannot be hashed.

    Raised for `NaN`/`Infinity` in a config tree, or for a type that has no
    stable JSON representation. Both would silently break invariant I8
    (determinism), so they are hard errors rather than coercions.
    """


class DeterminismError(AlphaError):
    """The process is not in a state that can produce byte-identical artifacts."""


class ArtifactError(AlphaError):
    """The artifact store could not satisfy a request."""


class ArtifactExistsError(ArtifactError):
    """An attempt was made to overwrite an existing artifact.

    Run directories are immutable (§4, I9). Producing a different result means
    producing a new run, not editing an old one.
    """


class RunNotFoundError(ArtifactError):
    """No run directory exists for the requested run id."""


class DataError(AlphaError):
    """Something is wrong with the market data itself."""


class ProviderError(DataError):
    """A data provider failed to return usable bars."""


class SchemaValidationError(DataError):
    """A dataframe failed its declared schema contract."""


class DataQualityError(DataError):
    """A dataset failed its configured quality thresholds (§6.4)."""


class FeatureError(AlphaError):
    """A feature could not be specified or computed."""


class LeakageError(FeatureError):
    """A construction was rejected because it can see the future.

    This is the one error that must never be caught and downgraded to a warning:
    every number downstream of a leak is fiction.
    """


class LabelError(AlphaError):
    """A labelling method could not produce a valid label frame."""


class CVError(AlphaError):
    """A cross-validation scheme could not produce valid splits."""


class ModelError(AlphaError):
    """A model adapter failed to fit, predict, save, or load."""


class BacktestError(AlphaError):
    """The backtest engine reached an inconsistent state."""


class EvaluationError(AlphaError):
    """A metric could not be computed.

    Raised in preference to returning `NaN`: a silent `NaN` propagating into a
    Sharpe ratio is a critical bug, not an inconvenience (§0.7).
    """
