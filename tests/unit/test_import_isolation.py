"""`alphacore` stays a pure research library (§3, §10.3).

import-linter enforces this statically in CI. This test enforces it
dynamically, in a clean subprocess, because a lazy `import fastapi` inside a
function body is exactly the kind of thing a static contract can be argued
around and a runtime check cannot.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

FORBIDDEN_AT_IMPORT = (
    "fastapi",
    "starlette",
    "uvicorn",
    "typer",
    "click",
    "torch",
    "streamlit",
)

PROBE = """
import json, sys
import alphacore
print(json.dumps({
    "version": alphacore.__version__,
    "modules": sorted(m for m in sys.modules if "." not in m),
}))
"""


def _probe() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_alphacore_pulls_in_nothing_web_shaped_or_optional():
    loaded = set(_probe()["modules"])
    assert not loaded & set(FORBIDDEN_AT_IMPORT)


def test_alphacore_reports_a_version():
    assert _probe()["version"]
