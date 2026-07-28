# Design plan — the web application

Required by §15 before any component exists, and by §13 before any code does.
This is the plan the build follows; where it and instinct disagree during
implementation, this document wins, and if it turns out to be wrong it gets
amended here first.

---

## 1. The brief, restated in one line

**A quantitative researcher's evidence file for one hypothesis**, read by the
researcher and by a sceptic over their shoulder, whose single job is to make it
fast to decide whether an edge is real and *impossible to accidentally overlook
the reason it isn't* (§13.1).

The second half of that sentence is the whole design problem. Making a number
easy to find is ordinary. Making its disqualifying context impossible to skip is
not, and it is the only thing this interface has to be good at.

**Metaphor:** a lab notebook and its margin. Not a trading terminal. §13.1 is
explicit that terminals are built for speed of action and this tool is built for
slowness of judgement.

---

## 2. Tokens

§13.2 pins these down completely, including hex values. They are implemented
verbatim in `web/src/styles/tokens.css` as Tailwind v4 `@theme` variables. This
section records them so the file has a specification to be checked against, not
because there was a decision to make.

### Colour — "plotter on cold stock"

| Token | Value | Role |
|---|---|---|
| `--paper` | `#EFF0EA` | page ground, a cold oat that is not cream |
| `--paper-sunk` | `#E4E6DE` | recessed panels, table stripes |
| `--rule` | `#C8CBC0` | hairlines, grid |
| `--ink` | `#191B18` | primary text |
| `--ink-muted` | `#6B7069` | secondary text, captions |
| `--prussian` | `#1E3A5F` | in-sample, train bands, primary structure, focus ring |
| `--plum` | `#6C2B54` | out-of-sample, test bands, the model's own line |
| `--sulphur` | `#C29A1F` | purge and embargo hatching, warnings |
| `--oxblood` | `#7A2418` | losses, drawdown fill, failures |
| `--sage` | `#3F6B57` | gains — never a saturated green |

Dark mode inverts the ground to `--ink #101210` and lifts the four accents by
12% lightness. The accents keep their identity; only the paper flips. This is
the rule that stops dark mode becoming a different product.

**Colour is never the only encoder** (§13.2). Long and short carry ▲ and ▼,
purge and embargo carry diagonal hatching, and pass/warn/fail carry a text
label. Every one of those three has a test.

### Type

| Role | Face | Use |
|---|---|---|
| Display | Bricolage Grotesque Variable | headings only, `wdth` axis to condense section titles. **Max three uses per page.** |
| Body | Public Sans 400/500 | prose, labels, controls |
| Data | IBM Plex Mono 400/500 | **every number, without exception**, `tabular-nums` everywhere |

Self-hosted through `@fontsource-variable`. Not Google Fonts at runtime: a
network round trip for a typeface is a layout shift the quality bar forbids, and
a research tool that will not render without an internet connection is a
research tool that fails in the one place it is used.

Scale (1.25 ratio, 16px base): 12 / 14 / 16 / 20 / 25 / 31 / 39.
Spacing (4px base): 4 8 12 16 24 32 48 64.
Radius: **2px only.** Shadows: **none** — depth comes from hairline rules and
`--paper-sunk` fills, which is how a printed plate does it.

---

## 3. Layout concept

> **A single ruled column of figure plates in pipeline order, each stamped with
> its provenance, with the tool's own objections running down the right margin.**

A lab notebook is a ruled page with a margin, and the margin is where the person
reading it argues with what is written. That is the entire layout: a content
column of plates, and a 280px right gutter that talks back.

Three consequences follow, and each is a decision rather than a default:

1. **No sidebar navigation.** The sub-routes of a run are the pipeline stages in
   order — data, features, folds, models, backtest, explain — so they are
   presented as a horizontal **stage rail** under the run header, in pipeline
   sequence, each carrying its own verdict label. The order encodes something
   true: it is the order in which the run could have gone wrong. A left sidebar
   would encode nothing except a habit.

2. **No cards.** Content sits in plates separated by hairline rules on the same
   paper, the way figures sit on a report page. Cards with shadows would be
   floating panes in a terminal; §13.2 has already banned the shadows, and
   floating is the wrong metaphor anyway.

3. **The margin is not a sidebar either.** It holds generated marginalia tied to
   specific plates, and it is empty when the tool has nothing to object to. An
   empty margin is information: it means this run's numbers survived the checks
   the tool knows how to make.

### Structural devices, and whether they earn their place

The `frontend-design` skill warns that numbered markers (01 / 02 / 03) are only
appropriate when the content genuinely is a sequence. Here it is: the run
**is** a pipeline, and a failure at stage three makes stages four onward
meaningless. So the stage rail is numbered — but the numbers are the *stage
index*, and a stage the run never reached is struck through rather than
hidden. The numbering is load-bearing: it tells a reader how far the run got.

Eyebrows, dividers and labels otherwise carry the artefact name they came from
(`tearsheet.json`, `folds.json`), which is the vernacular of this subject.

---

## 4. Signature elements

### 4.1 The Fold Strip — the one loud thing

Specified in §13.2 and implemented as hand-written SVG. A full-width horizontal
timeline of the walk-forward split: prussian bands for train, plum for test,
sulphur diagonal hatching for purge and embargo, with the price series as a
hairline behind it.

It is three things at once, and the third is the point:

- the hero of the run page
- the global fold selector — click a band and every chart below filters to it
- **the honest disclosure of exactly how the model was validated**

Nothing else on the page is allowed to be this loud (§13.2). It persists as a
sticky element across every sub-route of a run, so the validation scheme is
never off screen while a number is being read.

### 4.2 The Provenance Stamp — the detail a reviewer remembers

A one-line monospace footer at 12px in `--ink-muted` under every chart and
table:

```
run a3f91c · data 7e2b · costs 1.5bps+1.0sp · net of costs · lag 1 bar
```

It looks like a plate stamp on a lab photograph, and it means **no chart can be
screenshotted out of context and misread**. Implemented as one component that
every plate is required to render; a plate without one fails a test.

### 4.3 The margin — the tool arguing with itself

280px right gutter, collapsing to inline callouts below 1280px. Notes are
generated from the tearsheet's `warnings` array plus rules in
`web/src/lib/critique.ts`, in the researcher's own vocabulary:

```
PBO 0.62 — the selection process performs worse than chance
break-even 6.4 bps — your assumed cost is 1.5
```

Each note aligns to the top edge of the plate it refers to and connects to it
with a 16px hairline leader. The leader is the risk described in §6.

---

## 5. Instead of a hero stat row: the tolerance rail

§13.4 asks `/runs/$id` for "headline metrics with CIs". The template answer is a
row of big numbers with small labels, and the `frontend-design` skill names that
as the default to avoid. It is also *wrong here*: a big number is exactly the
thing this interface exists to stop a reader taking at face value.

A calibration certificate does not print a big number. It prints a measured
value against the tolerance it had to fall inside. So each headline metric is a
**tolerance rail** — a 220px hairline scale carrying:

```
  net Sharpe            ├────────●━━━━━━━━━━┤        0.96
                        0      0.22   1.94    threshold 0
  deflated Sharpe       ├──────────────●─────┤       0.52
                        0            ╎0.5     1.00
                                      └ must clear
```

the point estimate as a filled tick, its confidence interval as a heavier
segment, and the threshold it must clear as a notch. A Sharpe of 0.96 whose
interval crosses zero *looks* like what it is, at a glance, without the reader
having to compare two numbers in their head.

This is deliberately quiet — hairlines, 12px mono labels, no fill. The boldness
budget is spent on the Fold Strip (§4.1); this device wins by being legible, not
by being loud.

---

## 6. The risk, and its justification

**Marginalia with leader lines.** Web interfaces put annotations in tooltips,
popovers, or a stacked sidebar. Typesetting them as true marginalia — aligned to
their subject's top edge, hanging in the gutter, connected by a hairline leader —
is a print convention that almost no web app attempts, because it needs a
`ResizeObserver` per note and it degrades badly if done carelessly.

It is worth the effort because it is the only treatment that makes the objection
*coordinate in space* with the thing being objected to. A tooltip hides the
criticism behind an interaction; a stacked sidebar makes the reader match note
to chart by reading. The brief's single job is that the disqualifying context is
impossible to overlook. Marginalia is the typographic answer to exactly that.

**How it degrades:** below 1280px the gutter collapses and each note becomes an
inline callout directly beneath its plate — same content, same wording, no
leaders. Below 768px the layout is a single column and the callouts stay. The
leader is an enhancement on wide screens, never load-bearing.

---

## 7. Wireframes

### 7.1 `/` — the run index

The notebook's contents page. Dense, ruled, sorted newest first. No hero: the
runs are the content, and a marketing-shaped hero on an index of one's own work
would be absurd.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ALPHA LAB                                          runs   compare   config  │  ← Bricolage, use 1 of 3
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Evidence file                                            14 runs · 3 failed │  ← Bricolage, use 2 of 3
│  Research output. Not investment advice.                                     │
│                                                                              │
│  ┌ filter ─────────────────────────────────────────────────────────────────┐ │
│  │ symbol [ all ▾ ]   status [ all ▾ ]   flag [ any ▾ ]        1 of 14 hidden│ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  run          config    symbol  span            model      net SR   DSR  PBO │  ← IBM Plex Mono, tabular
│  ──────────────────────────────────────────────────────────────────────────  │
│  5ad0f1  ●    smoke     SYNTH   2015–2024       buyhold     0.960  0.52 0.08 │
│               ⚑ LOW_DSR                                                      │
│  625d30  ●    smoke     SYNTH   2015–2024       buyhold     0.960  0.52 0.10 │
│  a3f91c  ✕    spy_tb    SPY     2005–2025       —               —     —    — │  ← failed: struck, not hidden
│               failed during backtest · see job log                           │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  read from runs/ · alphacore 0.1.0 · 3a7aecb                                 │  ← provenance stamp
└──────────────────────────────────────────────────────────────────────────────┘
```

**Empty state** is an invitation with the exact command, copyable (§13.5):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   No runs yet.                                                               │
│                                                                              │
│   The first one takes about ten seconds and needs no network:                │
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────┐  copy   │
│   │ uv run alpha run -c configs/experiments/smoke.yaml              │   ⧉     │
│   └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│   It runs on a committed synthetic fixture, and it should not find an edge.   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 `/runs/$id` — the overview

```
┌───────────────────────────────────────────────────────────┬──────────────────┐
│  ← runs      2026-07-28T05-40-19Z_smoke_5ad0f1            │   MARGIN         │
│  ─────────────────────────────────────────────────────────│                  │
│  smoke · SYNTH · 1d · XNYS · 2015-04-15 → 2024-12-26      │                  │
│  config 5ad0f1… · data 87bc16… · 3a7aecb · seed 20260727  │                  │
│                                                           │                  │
│  ① data  ② features  ③ folds  ④ models  ⑤ backtest  ⑥ explain               │  ← stage rail, pipeline order
│  ─────────────────────────────────────────────────────────│                  │
│                                                           │                  │
│  ┏━━━━━━━━━━━━━━━ THE FOLD STRIP ━━━━━━━━━━━━━━━━━━━━━━━┓ │                  │
│  ┃ ▁▂▃▂▁▃▄▃▂▄▅▄▃▅▆▅▄▆▇▆▅▇█▇▆  ← price, hairline        ┃ │                  │
│  ┃ ███████████▨▨░░░░░  ███████████▨▨░░░░░  ██████▨▨░░░ ┃ │──╴ fold 2 is the  │
│  ┃ └─ train ──┘└p┘└test┘                                ┃ │    only one that │
│  ┃ 2015        2018      2020        2022       2024    ┃ │    made money    │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ │                  │
│  walkforward · 3 folds · 2,847 purged · 128 embargoed     │                  │
│                                                           │                  │
│  ─────────────────────────────────────────────────────────│                  │
│  Verdict                                          buyhold │                  │
│                                                           │                  │
│  net Sharpe        ├────●━━━━━━━━━━┤            0.960     │──╴ the interval  │
│                    0  0.16       1.94  ╎0                 │    crosses zero  │
│  deflated Sharpe   ├─────────────●──┤          0.524      │                  │
│                    0            ╎0.5   1                  │                  │
│  break-even cost   ├──────────────●─┤       20.0 bps      │──╴ break-even    │
│                    0        ╎3.0 assumed                  │    20.0 against  │
│  PBO               ├──●────────────┤          0.080       │    3.0 assumed   │
│                    0    ╎0.5 worse than chance            │                  │
│  ─────────────────────────────────────────────────────────│                  │
│                                                           │                  │
│  Equity, net of costs                                     │                  │
│  ┌───────────────────────────────────────────────────┐   │                  │
│  │            ╱╲      ╱╲╱╲                            │   │                  │
│  │      ╱╲╱╲╱╱  ╲╱╲╱╱                                 │   │                  │
│  │  ╱╲╱╱                                              │   │                  │
│  └───────────────────────────────────────────────────┘   │                  │
│  ┌───────────────────────────────────────────────────┐   │                  │
│  │▓▓▓▒▒░░░▒▒▓▓▓▓▒▒░░  underwater, oxblood fill        │   │                  │
│  └───────────────────────────────────────────────────┘   │                  │
│  run 5ad0f1 · data 87bc16 · costs 0.5bps+1.0sp · net · lag 1  ← STAMP        │
│                                                           │                  │
└───────────────────────────────────────────────────────────┴──────────────────┘
```

Below 1280px the margin collapses and each note becomes an inline callout under
its plate. Below 768px everything is one column; the Fold Strip stays full
width and stays sticky, because it is the one thing that must never scroll out
of reach.

---

## 8. Motion

**One orchestrated page-load** (§13.2): the Fold Strip draws left to right over
600ms, charts fade in after it. Hover states are instant. **Nothing else
animates.** `prefers-reduced-motion` disables the draw entirely — not shortens
it, disables it.

The restraint is the point. A page that animates on every interaction reads as
AI-generated; a page with exactly one orchestrated moment reads as designed.

---

## 9. Self-critique against the defaults

The `frontend-design` skill names three clusters that AI-generated design falls
into. Working the brief honestly, here is where this plan would have landed
without the check, and what changed.

**Default 1 — warm cream, high-contrast serif, terracotta accent.**
Avoided, but only just. `--paper #EFF0EA` is one hue away from the cream, and my
first instinct for a "printed statistical report" was a Times-like serif display
face. §13.2 stops both: the paper is specified as *cold oat, explicitly not
cream*, and the display face is a grotesque with a width axis, not a serif. The
lesson is that "printed report" leads straight to the default unless something
pins it. **Verified in implementation** by checking `--paper` against `#F4F1EA`;
they are different hues, not different tints, and the plan is to keep the cold
cast visible by never placing a genuinely warm colour on the ground.

**Default 2 — near-black with an acid accent.**
Explicitly forbidden by §13.1 as both the AI default *and* the wrong mood. The
trap here is dark mode: the obvious dark theme for a finance tool is exactly
this. The rule in §2 above — invert the ground, lift the accents 12%, never
change their identity — exists to stop dark mode drifting into a terminal.

**Default 3 — broadsheet, hairline rules, zero radius, dense columns.**
This one is uncomfortably close to the plan, and honesty requires admitting it:
hairline rules, 2px radius, dense tabular data, no shadows. The difference is
that here it is *specified by the brief* rather than reached for, and the skill
is explicit that the brief's own words win. But specification is not a licence
to stop thinking, so two things were changed to keep it from reading as a
newspaper pastiche:

- **The margin is not a newspaper column.** Broadsheet layouts use multiple
  columns of the same content. This gutter holds a *different kind of thing* —
  generated criticism — and is often empty. That asymmetry is what stops the
  page reading as a newspaper.
- **The Fold Strip is not a broadsheet device at all.** It is a piece of
  scientific instrumentation dropped into a typeset page, and the contrast
  between the two is the identity.

**What I cut.** The first plan gave the run index a miniature Fold Strip per row
as a sparkline. It was attractive and it was wrong: §13.2 says nothing else may
be as loud as the Fold Strip, and putting fourteen of them on the index would
have made the real one on the run page feel like a repeat rather than the hero.
Chanel's rule, applied — this was the accessory removed before leaving the
house.

**What I am still unsure about.** The tolerance rail (§5) is a fourth device in
a brief that already specifies three. It earns its place by replacing something
worse — the template stat row — and by being quiet enough not to compete. If it
reads as clutter in the built page, it is the first thing to go, and the
fallback is a plain two-column definition list of value and interval.

---

## 10. Quality floor

Not announced in the interface, just met (§13.5).

- Initial JS **≤ 200 KB gzipped**, route-level code splitting, enforced by a
  build-time budget check that fails CI rather than warning.
- The equity chart pans at 60fps at 100k points, because the **server**
  downsamples with LTTB to the requested pixel width. The browser never receives
  100k points.
- Every interactive element is keyboard-reachable with a visible focus ring in
  `--prussian`. There is a Playwright test that walks every route by keyboard.
- Loading states are **skeletons matched to the final layout**. No spinners, no
  layout shift.
- Error states name the cause and the fix. The API already returns RFC 9457
  problem documents whose `detail` does exactly this — the UI renders that text
  rather than inventing a friendlier one. "Something went wrong" appears nowhere.
- Responsive to 768px.
- Vitest for `lib/` logic; a Playwright smoke test covering the golden path.

---

## 11. Build order

Per §15, and in this order because each stage is the foundation of the next:

1. Scaffold — Vite, React 19, TS strict, Tailwind v4, TanStack Router and Query
2. `tokens.css` and the type system
3. Shell, routing, the API client over the generated types
4. **The Fold Strip**
5. Charts — uPlot for series, `lightweight-charts` for OHLC
6. `critique.ts` and the margin
7. Every remaining route
8. Polish: skeletons, focus, reduced motion, responsive
9. Playwright golden path, Lighthouse, bundle budget
