"""Data providers and their registry (§6.1).

`binance` and `alphavantage` are named in §6.1 and are not yet implemented;
Phase 1's exit criteria cover `csv` and `yahoo`. Requesting one raises a
`ProviderError` that says which phase it arrives in, rather than falling back
to a different provider and producing a run whose manifest is a lie.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from alphacore.data.providers.base import retry
from alphacore.data.providers.csv import CsvProvider
from alphacore.data.providers.yahoo import YahooProvider
from alphacore.errors import ConfigError, ProviderError

if TYPE_CHECKING:
    from alphacore.config.schema import DataConfig
    from alphacore.data.protocol import BarProvider

__all__ = [
    "IMPLEMENTED_PROVIDERS",
    "PLANNED_PROVIDERS",
    "CsvProvider",
    "YahooProvider",
    "get_provider",
    "retry",
]

IMPLEMENTED_PROVIDERS: Final[frozenset[str]] = frozenset({"csv", "yahoo"})
PLANNED_PROVIDERS: Final[frozenset[str]] = frozenset({"binance", "alphavantage"})


def get_provider(config: DataConfig) -> BarProvider:
    """Build the provider named by a data config.

    Args:
        config: The resolved `data` section.

    Returns:
        A provider satisfying `BarProvider`.

    Raises:
        ConfigError: If the `csv` provider is selected without a `csv_dir`.
        ProviderError: If the provider is planned but not yet implemented, or
            is not recognised at all.
    """
    if config.provider == "csv":
        if not config.csv_dir:
            msg = "data.csv_dir is required when data.provider is 'csv'."
            raise ConfigError(msg)
        return CsvProvider(config.csv_dir)
    if config.provider == "yahoo":
        return YahooProvider()
    if config.provider in PLANNED_PROVIDERS:
        msg = (
            f"Provider {config.provider!r} is specified in §6.1 but is not implemented yet. "
            f"Phase 1 ships {sorted(IMPLEMENTED_PROVIDERS)}. Use one of those, or supply the "
            "data through the csv provider."
        )
        raise ProviderError(msg)
    msg = f"Unknown provider {config.provider!r}. Known: {sorted(IMPLEMENTED_PROVIDERS)}."
    raise ProviderError(msg)
