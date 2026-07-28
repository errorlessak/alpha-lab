# web

The React application (§13). **Not built yet — that is Phase 12.**

What is here now is the one thing Phase 11 produced:

```
web/src/api/generated/schema.d.ts
```

Types generated from `docs/openapi.json` by `just client`, which runs
`openapi-typescript` over the document `alpha openapi` writes. Both files are
committed, and `tests/api/test_openapi_document.py` fails if either drifts from
the routes — so a change to the API that nobody regenerated the client for turns
the build red rather than surfacing as a runtime type error months later.

Regenerate after any route change:

```bash
just client
```

Phase 12 adds the Vite project around this. §15 is explicit that the design pass
comes first: `docs/design-plan.md`, with the token system from §13.2, ASCII
wireframes for `/` and `/runs/$id`, and a written self-critique — before any
component exists.
