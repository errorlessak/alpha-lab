"""Entry point for `python -m alphacore.features`.

Deliberately a separate module. Running `python -m alphacore.features.registry`
executes `registry.py` a *second* time under the name `__main__`, giving that
copy its own empty `_REGISTRY` — the built-ins registered into the first copy
are invisible to it. The symptom is a data dictionary containing the header and
no features, which looks like a formatting problem and is not.
"""

from __future__ import annotations

from alphacore.features.registry import _main

if __name__ == "__main__":
    raise SystemExit(_main())
