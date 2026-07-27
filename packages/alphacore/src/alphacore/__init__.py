"""Alpha Lab research core.

A pure research library for deciding whether a claimed short-horizon predictive
signal is real — meaning it survives purged validation, transaction costs, and
multiple-testing correction.

This package imports nothing web-shaped and nothing optional. In particular
`import alphacore` must succeed with PyTorch uninstalled; sequence models live
behind the `[dl]` extra and import torch lazily (§10.3).
"""

from __future__ import annotations

from importlib import metadata

from alphacore import errors, runtime
from alphacore.errors import AlphaError

__all__ = ["AlphaError", "__version__", "errors", "runtime"]


def _detect_version() -> str:
    """Read the installed distribution version.

    Returns:
        The version string, or `"0.0.0+unknown"` when running from a source
        tree that was never installed.
    """
    try:
        return metadata.version("alphacore")
    except metadata.PackageNotFoundError:  # pragma: no cover - only in a bare source tree
        return "0.0.0+unknown"


__version__: str = _detect_version()
