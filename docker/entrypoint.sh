#!/bin/sh
# Seed the artifact store on first start, then hand over to the real command.
#
# The exit criterion for §15 Phase 14 is that `docker compose up` serves the
# app *with the sample dataset* on a clean machine. An empty store would serve
# a correct but empty index whose call to action is a `uv run` command that
# does not exist inside this container.
#
# The run is produced only when the store has none. Runs are immutable (§4,
# I9), so this can never overwrite anything; it either adds the first run or
# does nothing.

set -e

RUNS_DIR="${ALPHA__RUNS_DIR:-/runs}"
CONFIG="${ALPHA_SEED_CONFIG:-configs/experiments/smoke.yaml}"

if [ "${ALPHA_SEED_SMOKE:-1}" = "1" ]; then
    # jobs.sqlite is created by the service, not by a run, so it does not count
    # as a populated store.
    existing="$(ls -A "$RUNS_DIR" 2>/dev/null | grep -v '^jobs\.sqlite' || true)"
    if [ -z "$existing" ]; then
        echo "alpha-lab: no runs in $RUNS_DIR — producing one from $CONFIG" >&2
        alpha run -c "$CONFIG"
    fi
fi

exec "$@"
