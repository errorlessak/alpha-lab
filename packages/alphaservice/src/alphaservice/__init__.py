"""Alpha Lab service layer: the `alpha` CLI and the FastAPI read model.

Depends on `alphacore`; `alphacore` never depends on this. The direction is
enforced by an import-linter contract in CI (§3).
"""

from __future__ import annotations

from importlib import metadata

__all__ = ["__version__"]


def _detect_version() -> str:
    """Read the installed distribution version.

    Returns:
        The version string, or `"0.0.0+unknown"` when running from a bare
        source tree.
    """
    try:
        return metadata.version("alphaservice")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "0.0.0+unknown"


__version__: str = _detect_version()
