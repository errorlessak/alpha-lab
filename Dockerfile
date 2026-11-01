# syntax=docker/dockerfile:1.7
#
# Alpha Lab — one image, one process, one origin.
#
# The web client requests `/api/v1/...` with no host (`web/src/lib/api.ts`), so
# the bundle and the API have to answer on the same origin. In development Vite
# proxies `/api`; here there is no proxy, so the service serves the bundle
# itself through `ALPHA__WEB_DIST`. Splitting this into an nginx container and
# an API container would add a second thing to configure in exchange for
# nothing at this scale.
#
# The image carries the `gbm` extra and not `dl`: LightGBM, XGBoost and CatBoost
# are what the leaderboard's baselines and TreeSHAP need, while torch adds ~2 GB
# for a model family no packaged config uses. Build `--build-arg EXTRAS="--extra
# gbm --extra dl"` if you want the sequence models in the image.

# --------------------------------------------------------------------------
# 1. The web bundle.
# --------------------------------------------------------------------------
FROM node:22-slim AS web

WORKDIR /web

# Dependencies first, so a source edit does not reinstall node_modules.
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
# `build` is `tsc --noEmit && vite build`: a type error fails the image rather
# than shipping a bundle that only works because nobody checked.
RUN npm run build

# --------------------------------------------------------------------------
# 2. The Python environment.
# --------------------------------------------------------------------------
FROM python:3.15-slim-bookworm AS python-build

# Pinned rather than `:latest`: the resolver is part of what makes `uv.lock`
# reproducible, and an image that silently changes resolver is an image whose
# builds are not comparable.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ARG EXTRAS="--extra gbm"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# The lockfile and the three manifests, then a dependency-only sync. This layer
# is invalidated by a dependency change and by nothing else, which is the whole
# reason the source is copied afterwards.
COPY pyproject.toml uv.lock ./
COPY packages/alphacore/pyproject.toml packages/alphacore/pyproject.toml
COPY packages/alphaservice/pyproject.toml packages/alphaservice/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-packages --no-install-workspace ${EXTRAS}

COPY packages/ packages/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-packages ${EXTRAS}

# --------------------------------------------------------------------------
# 3. The runtime.
# --------------------------------------------------------------------------
FROM python:3.15-slim-bookworm AS runtime

# libgomp is LightGBM's and XGBoost's OpenMP runtime. Without it the import
# fails at first use, which is after a run has already started.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Determinism (§4, I8): CPython reads PYTHONHASHSEED at startup, so it belongs
# in the image rather than in an entrypoint that has already started.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    ALPHA__DATA_DIR=/data \
    ALPHA__RUNS_DIR=/runs \
    ALPHA__JOBS_DB=/runs/jobs.sqlite \
    ALPHA__WEB_DIST=/app/web/dist

WORKDIR /app

# The virtual environment installs the workspace packages in editable mode, so
# `packages/` has to be at the same path it was built at.
COPY --from=python-build /app/.venv /app/.venv
COPY packages/ packages/
COPY configs/ configs/
COPY samples/ samples/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --from=web /web/dist /app/web/dist

# The artifact store is immutable and append-only, and it is the one thing in
# here worth keeping, so both state directories are volumes owned by a user
# that is not root.
RUN useradd --create-home --uid 10001 alpha \
    && mkdir -p /data /runs \
    && chown -R alpha:alpha /data /runs \
    && chmod +x /usr/local/bin/entrypoint.sh
USER alpha
VOLUME ["/data", "/runs"]

EXPOSE 8000

# `/api/v1/health` reports whether the service can read its artifact store,
# which is the only thing "healthy" can usefully mean here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as r; r.urlopen('http://127.0.0.1:8000/api/v1/health').read()"

ENTRYPOINT ["entrypoint.sh"]
# 0.0.0.0 inside a container is the container's own interface, not the host's.
# The compose file publishes it to loopback only — this service has no
# authentication and can start jobs.
CMD ["uvicorn", "alphaservice.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
