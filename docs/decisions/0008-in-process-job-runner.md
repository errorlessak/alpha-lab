# ADR 0008 — An in-process job runner instead of Celery and Redis

- **Status:** Accepted
- **Date:** 2026-07-28
- **Phase:** 11

## Context

§14.2 specifies a `ProcessPoolExecutor` with one or two workers and SQLite for
job state and a rolling log table, and says explicitly: no Celery, no Redis. It
also says the interesting part of this decision is *what you would change at
scale*. This ADR is the answer to that, reserved since Phase 0 and written now
that the runner exists.

The workload is one researcher submitting one run at a time. A run is minutes of
CPU-bound numpy and LightGBM, and it produces a directory of artifacts on the
local filesystem. There is no fan-out, no multi-tenancy, and no requirement that
a job survive the machine it was submitted on — the run directory is the durable
record, not the job.

Celery plus Redis would add a broker process, a result backend, a worker
supervisor, a serialisation contract between them, and a deployment story with
four moving parts. In exchange it would offer distribution, retries, and
scheduling. None of the three is needed by a tool whose bottleneck is the
attention of the person reading the tearsheet.

## Decision

`alphaservice.jobs` is a `ProcessPoolExecutor` (one worker by default) and a
SQLite database holding job rows and a rolling per-job log table. The API process
reads that database; the worker writes it. Nothing else is involved.

Three consequences were decided rather than inherited:

1. **Processes, not threads.** A run holds the GIL for minutes inside numpy and
   the boosting libraries, and it calls `set_global_seed`, which mutates
   interpreter-wide state. Two runs sharing an interpreter would contend on the
   first and *corrupt each other's determinism* through the second — the second
   run's results would depend on whether the first had already seeded the RNG.
   That is the one failure this project cannot tolerate, so isolation is by
   process.

2. **Cancellation is cooperative.** A `ProcessPoolExecutor` cannot kill an
   individual worker: `Future.cancel` only works before a job starts, and there
   is no supported way to signal one member of the pool. `POST /jobs/{id}/cancel`
   therefore sets a flag in SQLite which the worker observes at each phase
   boundary, through `run_experiment(should_cancel=...)`.

   The cost is latency: a cancel lands at the end of the current phase, not
   instantly. The benefit is that §14.2's actual requirement is met — the run
   directory is finalised as `cancelled` with everything it managed to write
   intact, rather than being killed mid-parquet-write and left looking valid. A
   hard kill would produce exactly the half-written directory the specification
   forbids.

3. **SQLite holds the live tail, not the log.** The worker attaches a loguru sink
   that appends to a rolling table capped per job; the SSE endpoint polls it. The
   complete, authoritative log is still `runs/{run_id}/logs.jsonl`, written by the
   run itself. A run that logs for an hour cannot fill the disk through the job
   database, and the log that matters is never the one that rolled.

## Consequences

**Benefit.** The deployment is one process. `uv run alpha serve` is the whole
story; there is no broker to start, no worker to supervise, and no state that
outlives the filesystem. A researcher can read the job database with the
`sqlite3` shell during an incident.

**Cost — jobs do not survive a restart.** A job running when the API stops is
lost, and its run directory is left in whatever state the worker had reached.
The manifest will not say `completed`, so it cannot be mistaken for a result,
but nothing resumes it. Acceptable for a single-user tool; unacceptable for a
shared one.

**Cost — no retries, no scheduling, no priority.** Jobs run in submission order,
once. A failed job is failed.

**Cost — cancellation latency.** Bounded by the longest phase, which on a large
config is the training phase and can be minutes.

## What would change at scale

The interesting question, in order of when each becomes the binding constraint:

1. **More than one concurrent user** → raise `max_workers` and move the job
   database off SQLite's single-writer model. SQLite in WAL mode handles many
   readers and one writer, which is exactly the current shape; several
   concurrently *writing* workers would start to contend on the log table first.

2. **Jobs that must survive a restart** → the job row needs a lease and a
   heartbeat, and start-up needs to reconcile: any job marked `running` with an
   expired lease is either resumed or marked `failed`. This is the first change
   that genuinely needs a design, and it is the point at which SQLite starts
   feeling load-bearing rather than convenient.

3. **More than one machine** → *now* a broker earns its place, and the artifact
   store has to stop being a local directory. The run directory contract (§16)
   is already content-addressed by config and data hash, so object storage is a
   change of backend rather than of meaning; `ArtifactStore` is the single seam.

4. **Hard cancellation** → give each job its own `multiprocessing.Process`
   instead of a pool slot, so it can be terminated. That trades the pool's
   worker reuse (and its process start-up amortisation) for the ability to kill,
   and it reintroduces the half-written-directory problem that cooperative
   cancellation avoids. Worth doing only when a phase gets long enough that
   waiting for a boundary is genuinely unacceptable.

Notably, none of these is a reason to start with Celery. Each is a change with a
clear trigger, and the current design is the one with the fewest parts that meets
the requirement in front of it.

## A note on the numbering

§2 assigns ADRs 0001–0007 to the seven stack deviations; §14.2 separately calls
the job runner "ADR 0006". Both cannot hold. §2 states its range explicitly and
covers exactly seven items, so the job runner took the next free number and has
been reserved as 0008 since Phase 0. See `docs/decisions/README.md`.
