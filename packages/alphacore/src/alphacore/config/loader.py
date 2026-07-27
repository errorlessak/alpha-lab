"""Configuration loading and layering (§5).

Resolution order, lowest priority first:

1. `defaults.yaml`, shipped inside the package
2. the experiment YAML
3. environment variables, `ALPHA__SECTION__KEY=value`
4. CLI overrides, `--set section.key=value`

Layers are deep-merged: a mapping in a higher layer updates the corresponding
mapping below it rather than replacing it wholesale. Scalars and sequences do
replace — a partially overridden list is almost never what anyone means.

The result is validated against `AlphaConfig`, and because every model sets
`extra="forbid"`, a key that survives merging but matches nothing in the schema
is an error rather than a silently ignored typo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from alphacore.config.schema import AlphaConfig
from alphacore.errors import ConfigError
from alphacore.runtime.hashing import hash_json

__all__ = [
    "DEFAULTS_PATH",
    "ENV_PREFIX",
    "config_hash",
    "deep_merge",
    "env_overrides",
    "load_config",
    "load_defaults",
    "parse_override",
    "to_serialisable",
]

DEFAULTS_PATH: Final[Path] = Path(__file__).parent / "defaults.yaml"
ENV_PREFIX: Final[str] = "ALPHA__"
ENV_SEPARATOR: Final[str] = "__"

_NON_CONFIG_ENV_KEYS: Final[frozenset[str]] = frozenset(
    {"runs_dir", "data_dir", "log_level", "api_token", "alphavantage_api_key"}
)
"""`ALPHA__*` variables that configure the *process*, not the experiment. They
are deliberately excluded from the config tree so they cannot change
`config_hash` — where the artifact store lives has no bearing on what a run
computes."""


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge `overlay` onto `base`, recursing into nested mappings.

    Args:
        base: The lower-priority mapping. Not mutated.
        overlay: The higher-priority mapping.

    Returns:
        A new mapping. Nested dicts are merged key by key; every other type is
        replaced outright.
    """
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a mapping.

    Args:
        path: File to read.

    Returns:
        The parsed mapping, or an empty mapping for an empty file.

    Raises:
        ConfigError: If the file is missing, malformed, or not a mapping.
    """
    if not path.is_file():
        msg = f"Config file not found: {path}"
        raise ConfigError(msg)
    try:
        loaded = yaml.safe_load(path.read_text("utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Could not parse YAML in {path}: {exc}"
        raise ConfigError(msg) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = f"Config file {path} must contain a mapping at the top level, got {type(loaded)!r}."
        raise ConfigError(msg)
    return loaded


def load_defaults() -> dict[str, Any]:
    """Read the packaged baseline configuration.

    Returns:
        The contents of `defaults.yaml` as a mapping.

    Raises:
        ConfigError: If the packaged defaults are missing or malformed.
    """
    return _read_yaml(DEFAULTS_PATH)


def _coerce_scalar(raw: str) -> Any:  # noqa: ANN401 - returns any YAML scalar
    """Parse a string override into a typed value.

    YAML scalar rules are used, so `1.5` becomes a float, `true` a bool, `null`
    None, and `[1, 2]` a list. An unparseable value stays a string, which
    pydantic will then reject with a precise message.

    Args:
        raw: The raw string from the environment or the command line.

    Returns:
        The parsed value.
    """
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _nest(path: list[str], value: Any) -> dict[str, Any]:  # noqa: ANN401
    """Build a nested mapping from a dotted path and a leaf value.

    Args:
        path: Key segments, outermost first.
        value: The leaf value.

    Returns:
        A mapping such as `{"backtest": {"costs": {"commission_bps": 1.5}}}`.
    """
    nested: Any = value
    for key in reversed(path):
        nested = {key: nested}
    return dict(nested)


def env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Collect `ALPHA__SECTION__KEY` variables into a config fragment.

    Process-level variables (`ALPHA__RUNS_DIR`, `ALPHA__LOG_LEVEL`, and the
    provider keys) are ignored: they do not describe the experiment, so letting
    them into the tree would make `config_hash` depend on where the machine
    happens to keep its artifacts.

    Args:
        environ: Environment mapping. Defaults to `os.environ`.

    Returns:
        A nested mapping ready to merge.
    """
    source = os.environ if environ is None else environ
    overrides: dict[str, Any] = {}
    for name, raw in sorted(source.items()):
        if not name.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in name[len(ENV_PREFIX) :].split(ENV_SEPARATOR) if part]
        if not path or (len(path) == 1 and path[0] in _NON_CONFIG_ENV_KEYS):
            continue
        overrides = deep_merge(overrides, _nest(path, _coerce_scalar(raw)))
    return overrides


def parse_override(assignment: str) -> dict[str, Any]:
    """Parse one `section.key=value` CLI override.

    Args:
        assignment: The override, e.g. `"backtest.costs.commission_bps=1.5"`.

    Returns:
        A nested mapping ready to merge.

    Raises:
        ConfigError: If the assignment has no `=`, or an empty key path.
    """
    key, separator, raw = assignment.partition("=")
    if not separator:
        msg = f"Override {assignment!r} is not of the form section.key=value."
        raise ConfigError(msg)
    path = [part for part in key.strip().split(".") if part]
    if not path:
        msg = f"Override {assignment!r} has an empty key path."
        raise ConfigError(msg)
    return _nest(path, _coerce_scalar(raw.strip()))


def load_config(
    path: Path | str | None = None,
    *,
    overrides: list[str] | None = None,
    environ: dict[str, str] | None = None,
    use_env: bool = True,
) -> AlphaConfig:
    """Resolve and validate a configuration.

    Args:
        path: Experiment YAML. None loads the packaged defaults alone, which is
            only useful for tests — `meta` and `models.candidates` have no
            defaults, so a real run always supplies a file.
        overrides: CLI overrides, each `section.key=value`. Applied last.
        environ: Environment mapping for `ALPHA__*` overrides. Defaults to
            `os.environ`.
        use_env: Set False to ignore the environment entirely. Tests use this
            so an exported variable cannot silently change a fixture.

    Returns:
        The validated, frozen configuration.

    Raises:
        ConfigError: If a file is missing or malformed, an override is
            unparseable, or the merged result fails validation. Validation
            failures are re-raised with the offending field paths named.
    """
    merged = load_defaults()
    if path is not None:
        merged = deep_merge(merged, _read_yaml(Path(path)))
    if use_env:
        merged = deep_merge(merged, env_overrides(environ))
    for assignment in overrides or []:
        merged = deep_merge(merged, parse_override(assignment))

    try:
        return AlphaConfig.model_validate(merged)
    except ValidationError as exc:
        details = "\n".join(
            f"  {'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors()
        )
        source = f" from {path}" if path else ""
        msg = f"Invalid configuration{source}:\n{details}"
        raise ConfigError(msg) from exc


def to_serialisable(config: AlphaConfig) -> dict[str, Any]:
    """Render a config as a plain JSON-compatible mapping.

    This is what gets hashed and what gets written to
    `runs/{run_id}/config.resolved.yaml`.

    Args:
        config: The resolved configuration.

    Returns:
        A nested mapping of plain Python types.
    """
    return config.model_dump(mode="json")


def config_hash(config: AlphaConfig) -> str:
    """Compute `config_hash` for a resolved configuration (§5).

    Args:
        config: The resolved configuration.

    Returns:
        Lowercase blake2b hex digest over the canonical JSON of the config.

    Raises:
        SerialisationError: If the config contains a non-finite float.
    """
    return hash_json(to_serialisable(config))
