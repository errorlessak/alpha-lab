"""The FastAPI application (§14).

Read-only over the artifact store, plus a job runner that creates new runs. The
API never modifies a run directory: they are immutable (§4, I9), and everything
here either reads one or asks for a new one to be made.

The OpenAPI document is the contract the TypeScript client is generated from, so
every route carries an explicit `operation_id` and a response model. A route
whose schema says `{}` produces a client function that returns `any`, which
defeats the point of generating one.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import alphacore
import alphaservice
from alphaservice.api.problems import install_error_handlers
from alphaservice.api.routers import datasets, jobs, runs, system
from alphaservice.api.settings import API_PREFIX, Settings
from alphaservice.api.static import mount_web_app
from alphaservice.jobs.runner import JobRunner
from alphaservice.jobs.store import JobStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["app", "create_app"]

_DESCRIPTION = """
Read-only access to the immutable artifact store, and a job runner that creates
new runs.

`tearsheet.json` is the canonical result artifact: every number this API serves
about a run comes from one, and the run it came from records the config hash,
data hash, code revision and seed that produced it.

**Research output. Not investment advice.**
"""

_DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
"""The Vite dev server (§17). Production serves the built assets from the same
origin, so this list is deliberately not `*`."""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Where to read state from. Defaults to the environment, which
            is the same set of variables the CLI honours.

    Returns:
        The configured application.
    """
    resolved = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Own the job runner for the life of the process.

        Args:
            app: The application.

        Yields:
            Nothing. The pool is shut down on the way out, so a reload does not
            leave orphaned workers holding the job database open.
        """
        store = JobStore(resolved.jobs_db)
        runner = JobRunner(
            store=store,
            data_dir=resolved.data_dir,
            runs_dir=resolved.runs_dir,
            max_workers=resolved.max_workers,
        )
        app.state.settings = resolved
        app.state.job_store = store
        app.state.job_runner = runner
        try:
            yield
        finally:
            runner.shutdown(wait=False)

    app = FastAPI(
        title="Alpha Lab",
        version=alphaservice.__version__,
        description=_DESCRIPTION,
        summary=f"alphacore {alphacore.__version__}",
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        lifespan=lifespan,
        separate_input_output_schemas=False,
    )
    # Settings are also set outside the lifespan so a router can read them in
    # tests that construct the app without entering it.
    app.state.settings = resolved

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_DEV_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["ETag"],
    )

    install_error_handlers(app)
    for router in (system.router, datasets.router, runs.router, jobs.router):
        app.include_router(router, prefix=API_PREFIX)

    # Last, and only when asked: the fallback matches every path, so anything
    # registered after it would never be reached.
    if resolved.web_dist is not None:
        mount_web_app(app, resolved.web_dist)
    return app


app = create_app()
"""The module-level application uvicorn serves.

`alphaservice.api.app:app` is what `just serve` and the Dockerfile point at.
Built from the environment, which is the same set of variables the CLI reads."""
