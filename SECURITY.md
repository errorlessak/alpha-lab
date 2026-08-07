# Security

## The posture, stated plainly

Alpha Lab is **single-user research software with no authentication of any
kind.** That is a design decision, not an oversight, and it has consequences
you should know before you run it anywhere but your own machine.

The API service:

- has **no authentication and no authorisation**. Every route is open to
  anyone who can reach the port.
- **can start jobs.** `POST /api/v1/jobs` runs a pipeline in a subprocess on
  the host, driven by a config file path the caller chooses.
- **reads the artifact store and the data lake** and will serve any run in
  them.

It therefore binds `127.0.0.1` by default, `alpha serve` prints a warning if
told to bind anything else, and the compose file publishes the container port
to loopback only.

**Do not expose this service to a network you do not control.** If you need to,
put it behind something that authenticates — a reverse proxy with basic auth, an
SSH tunnel, or a private network — and treat job submission as equivalent to
shell access.

## Supported versions

The project is pre-1.0. Only `main` is supported; fixes land there and nowhere
else.

## Reporting a vulnerability

Please **do not open a public issue.**

Use GitHub's private reporting on this repository —
[Security → Report a vulnerability](https://github.com/errorlessak/alpha-lab/security/advisories/new)
— which opens a private advisory visible only to the maintainer.

Include what you would want to receive: the version or commit, what you did,
what happened, and what an attacker gets out of it. A proof of concept helps
and a patch helps more.

Expect an acknowledgement within a week. This is a personal project, so that is
a best effort rather than a service level.

## What is in scope

- Anything that lets a request escape the artifact store or the data lake —
  path traversal through a run id, a dataset id, an artifact name, or the
  static bundle.
- Anything that turns a job submission into arbitrary code execution beyond the
  pipeline it names.
- Anything in the container image that runs as root, or that ships credentials.

## What is not in scope

- **The absence of authentication.** It is documented above and on every path
  that matters. A report that the API is unauthenticated is a report of the
  design.
- **Denial of service by submitting expensive jobs.** The runner is a small
  process pool with no quota; a caller who can submit jobs can saturate it. See
  the point about not exposing the service.
- **Findings that require an attacker to already have local access** to the
  machine, the artifact store, or the config files.

## Things worth knowing

- **No credentials are needed or stored.** The `csv` and `yahoo` providers take
  none; `alphavantage` and `binance` are not implemented. `.env.example` lists
  every variable the code reads, and none of them is a secret.
- **Run directories are immutable** (§4, I9) and written once. The API never
  writes to one.
- **Artifacts are pickled models.** `runs/{id}/models/{model}/model.pkl` is a
  Python pickle, which is code. Do not load a run directory you did not
  produce, for the same reason you would not run a script somebody emailed you.
