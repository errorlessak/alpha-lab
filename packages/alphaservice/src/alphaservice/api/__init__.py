"""The FastAPI service (§14).

`create_app()` builds it; `alphaservice.api.app:app` is what uvicorn serves.
"""

from __future__ import annotations

from alphaservice.api.app import create_app
from alphaservice.api.settings import API_PREFIX, Settings

__all__ = ["API_PREFIX", "Settings", "create_app"]
