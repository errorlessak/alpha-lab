"""The typed exception hierarchy (§18)."""

from __future__ import annotations

import inspect

import pytest

from alphacore import errors
from alphacore.errors import (
    AlphaError,
    ArtifactError,
    ArtifactExistsError,
    DataError,
    DataQualityError,
    FeatureError,
    LeakageError,
    ProviderError,
    RunNotFoundError,
    SchemaValidationError,
)

PUBLIC_ERRORS = [
    getattr(errors, name)
    for name in errors.__all__
    if inspect.isclass(getattr(errors, name)) and issubclass(getattr(errors, name), BaseException)
]


def test_every_public_error_derives_from_alpha_error():
    assert PUBLIC_ERRORS
    for error in PUBLIC_ERRORS:
        assert issubclass(error, AlphaError)


def test_every_public_error_is_documented():
    for error in PUBLIC_ERRORS:
        assert error.__doc__, f"{error.__name__} has no docstring"


def test_all_is_sorted_and_complete():
    assert errors.__all__ == sorted(errors.__all__)
    assert set(errors.__all__) <= set(vars(errors))


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        (ArtifactExistsError, ArtifactError),
        (RunNotFoundError, ArtifactError),
        (ProviderError, DataError),
        (SchemaValidationError, DataError),
        (DataQualityError, DataError),
        (LeakageError, FeatureError),
    ],
)
def test_subhierarchies(child, parent):
    assert issubclass(child, parent)


def test_catching_alpha_error_catches_a_leak():
    with pytest.raises(AlphaError):
        raise LeakageError
