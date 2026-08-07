# web

The React application (§13) — a lab notebook for a run, with a margin that
argues with its own results.

Vite · React 19 · TypeScript strict · TanStack Router (file-based) · TanStack
Query · Tailwind v4, with §13.2's tokens as `@theme` variables in
`src/styles/tokens.css`. The design is worked through in
[`../docs/design-plan.md`](../docs/design-plan.md), including the self-critique
against the three looks AI-generated design tends to produce.

## Running it

The app reads the API, so start that first — from the repository root:

```bash
just serve
```

```bash
npm --prefix web run dev
```

Development server on `http://localhost:5173`; its origin is already in the
API's CORS allow-list. `npm --prefix web run build && npm --prefix web run
preview` serves the production bundle on `4173`, which is what the end-to-end
suite drives.

## What is worth knowing before changing it

- **`src/api/generated/schema.d.ts` is generated, not written.** `just client`
  regenerates it from `docs/openapi.json`, and
  `tests/api/test_openapi_document.py` fails if either drifts from the routes —
  so an API change nobody regenerated the client for turns the build red rather
  than surfacing as a runtime type error months later. Nothing in `src/` should
  declare an API shape by hand.
- **The Fold Strip (`components/FoldStrip.tsx`) is hand-written SVG**, and it is
  three things at once: the hero, the global fold filter, and the honest
  disclosure of how the model was validated. One lane per fold — folds overlap
  in time, and drawing them on a single row says nothing.
- **Every plate takes a provenance stamp as a required prop**, so no chart can be
  screenshotted out of context and misread.
- **`lib/critique.ts` is the margin.** Its rules make factual claims about the
  run beside the numbers they refer to, so a rule that assumes a sign or a model
  name is a bug of the worst kind here — the gutter is what a reader trusts to
  contradict the headline. Every rule has a test in `critique.test.ts`.
- **A React context created inside a code-split route file may not be the
  context you think it is.** `autoCodeSplitting` will happily put a second copy
  of that module in a child chunk. Shared contexts live in `lib/`.
- **Text takes the `-text` colour token, not the accent.** `--color-sulphur` is
  2.31:1 as light-mode text and `--color-oxblood` is 3.92:1 on the striped row
  fill in dark; both are fine as fills. `--color-caution-text` and
  `--color-danger-text` exist for the text case
  ([ADR 0014](../docs/decisions/0014-text-only-colour-variants.md)), and the
  end-to-end suite measures the contrast of every text node in both schemes, so
  reaching for the accent because it has the obvious name turns the build red.
- **Wide tables go in `TableScroll`.** Ten columns of figures are wider than a
  tablet viewport, and without a container it is the *page* that scrolls
  sideways — taking the masthead and the disclaimer with it. The container takes
  focus only while it actually overflows.

## Checks

```bash
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run test
npm --prefix web run e2e
npm --prefix web run budget
```

`e2e` starts its own API and preview server (`playwright.config.ts`) and walks
the golden path plus the §13.5 quality bar: keyboard-only navigation, a visible
focus ring, the margin collapsing below 1280px, and the disclaimer on every
page. `budget` fails the build if initial JavaScript exceeds 200 KB gzipped —
currently 99 KB. `npm --prefix web run lighthouse` takes the median of three
runs, because a single Lighthouse score is not evidence.

All of these run from the repository root as part of `just verify`.
