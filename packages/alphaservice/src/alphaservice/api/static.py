"""Serving the built web application from the API's own origin.

The client in `web/src/lib/api.ts` requests `/api/v1/...` with no host, so in
production the bundle and the API have to answer on one origin — which is also
why the CORS allow-list in `app.py` names only the Vite dev server rather than
`*`. In development Vite proxies `/api` and this module does nothing. In the
container there is no proxy, so the service serves the bundle itself.

Mounting is opt-in through `ALPHA__WEB_DIST`. Auto-detecting `web/dist` would
mean `just serve` quietly started serving whatever bundle happened to be on
disk from an earlier build, alongside the dev server serving a fresher one.

The routes registered here are excluded from the schema. The OpenAPI document
is the contract the TypeScript client is generated from, and a catch-all that
returns `text/html` describes nothing a client should call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from fastapi.responses import FileResponse
from starlette.status import HTTP_404_NOT_FOUND

from alphacore.runtime import logger
from alphaservice.api.problems import Problem
from alphaservice.api.settings import API_PREFIX

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI

__all__ = ["mount_web_app"]

INDEX_FILENAME: Final[str] = "index.html"

_HASHED_PREFIX: Final[str] = "assets/"
"""Vite fingerprints everything it emits into `assets/` with a content hash, so
those URLs each name one specific build. Nothing else in `dist` is fingerprinted
— `index.html` is the document that points at the current hashes, and
`favicon.svg` keeps its name across builds."""

_IMMUTABLE: Final[str] = "public, max-age=31536000, immutable"
_REVALIDATE: Final[str] = "no-cache"


def mount_web_app(app: FastAPI, dist: Path) -> bool:
    """Serve a built web bundle from the application's root path.

    Args:
        app: The application. Routers must already be registered — the fallback
            added here matches any path, so anything registered after it is
            unreachable.
        dist: The Vite `dist` directory, containing `index.html`.

    Returns:
        True if the bundle was mounted, False if `dist` holds no `index.html`.
        A missing bundle is a misconfiguration rather than a crash: the API is
        still fully usable without a UI, and failing to start would take the
        working half down with the broken one.
    """
    root = dist.resolve()
    index = root / INDEX_FILENAME
    if not index.is_file():
        logger.warning(
            "web bundle not mounted: {path} has no {name}. The API is unaffected; "
            "build the app with `npm --prefix web run build` or unset ALPHA__WEB_DIST.",
            path=root,
            name=INDEX_FILENAME,
        )
        return False

    @app.get("/{asset_path:path}", include_in_schema=False)
    def web_app(asset_path: str) -> FileResponse:
        """Serve a built asset, falling back to the application shell.

        Args:
            asset_path: The request path, without its leading slash.

        Returns:
            The file if one exists at that path, else `index.html` so that a
            deep link such as `/runs/{id}/backtest` is resolved by the client
            router rather than 404ing on a path no server route knows about.

        Raises:
            Problem: 404 for an unmatched `/api/v1` path. Returning the HTML
                shell to a client expecting JSON would turn a wrong URL into a
                parse error somewhere far away from the cause.
        """
        if f"/{asset_path}".startswith(API_PREFIX):
            raise Problem(
                status_code=HTTP_404_NOT_FOUND,
                problem_type="not-found",
                title="No such API route",
                detail=(
                    f"/{asset_path} is not a route on this service. "
                    f"The routes are listed at {API_PREFIX}/docs."
                ),
            )

        candidate = (root / asset_path).resolve()
        # `resolve()` collapses `..`, so a traversal is rejected by where it
        # lands rather than by pattern-matching the path that produced it.
        if asset_path and root in candidate.parents and candidate.is_file():
            cache = _IMMUTABLE if asset_path.startswith(_HASHED_PREFIX) else _REVALIDATE
            return FileResponse(candidate, headers={"cache-control": cache})

        return FileResponse(index, headers={"cache-control": _REVALIDATE})

    logger.info("serving the web bundle from {path}", path=root)
    return True
