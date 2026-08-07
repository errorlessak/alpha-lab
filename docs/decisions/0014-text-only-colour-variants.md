# ADR 0014 — Text-only variants of the accent colours

- **Status:** Accepted
- **Date:** 2026-08-07
- **Phase:** 14

## Context

§13.2 specifies the palette by value: prussian `#1e3a5f`, plum `#6c2b54`,
sulphur `#c29a1f`, oxblood `#7a2418`, sage `#3f6b57`, ink-muted `#6b7069`, and a
dark mode that inverts the ground and lifts the accents by 12% lightness. §13.5
sets an accessibility bar that is not described as negotiable.

Those two requirements conflict, and the conflict was already known: the tokens
file carries a note explaining that `--color-link` had to be lifted further than
§13.2's +12% in dark mode, because at `#4f7cb0` a 14px link measures 4.33:1
against the 4.5:1 AA needs.

Computing the rest of the palette rather than eyeballing it found that the same
problem existed in three more places, and that one of them was much worse than
the link:

| Token | As text | Measured | Where it shows |
|---|---|---|---|
| `--color-sulphur` | light mode | **2.31:1** on the ground, 2.10:1 on the striped row fill | the `⚑ HIGH_PBO · LOW_DSR` warning flags on the run index |
| `--color-oxblood` | dark mode | 4.34:1 / **3.92:1** | the status of a run that did not complete, and `no result — see the job log` |
| `--color-ink-muted` | light mode | 4.42:1 / **4.02:1** | most of the secondary text on every screen |

Lighthouse reported accessibility 96 and named only the second of these, because
it ran in one colour scheme. The first is the worst failure in the palette and
no automated check in the repository was looking for it.

The same colours are also used as fills and strokes — the sulphur hatch on the
Fold Strip's purge band, the oxblood leader in the margin, ink-muted axis ticks.
Those are graphics, measured against a 3:1 bar, and all of them pass.

## Decision

Split the tokens by role rather than tinting the accents.

- `--color-caution-text` — `#7c6314` in light (4.57:1), `#e0bc55` in dark, which
  is the accent unchanged because it already measures 9.32:1 there.
- `--color-danger-text` — `#7a2418` in light, which is the accent unchanged at
  7.97:1, and `#c86b58` in dark (4.62:1).
- `--color-ink-muted` is darkened in place, `#6b7069` → `#626761` (4.59:1 on the
  striped fill). It is a neutral rather than an accent, it has no identity to
  preserve, and it is used at 12px in sixty-odd places.

The structural colours keep the values §13.2 specifies. This follows the
precedent `--color-link` set: **only text moves.**

`e2e/golden-path.spec.ts` computes the contrast ratio of every text node under
`main` against its nearest painting ancestor, in both `light` and `dark`, using
WCAG's large-text allowance where it applies. A future edit that reaches for
`--color-sulphur` because it is the token with the obvious name turns the build
red.

## Consequences

**Benefit.** Lighthouse accessibility is 100 in both schemes, and the reason it
is 100 is a check in this repository rather than a score from a tool that
happened to audit one of them.

**Benefit.** The warning flags on the run index are legible. They were the least
readable text in the application and they carry `HIGH_PBO` and `LOW_DSR` — the
two flags §19 exists to make people look at.

**Cost.** The caution text in light mode is a dark olive-gold rather than the
specified sulphur. Against the ground at 12px there is no value that is both
sulphur and legible; the hatch and the highlighter keep the real colour, so the
identity survives where it is a graphic.

**Cost.** Six tokens where there were four, and a rule a contributor has to know:
text takes the `-text` variant. The contrast check enforces it, which is the
only reason a rule like this survives contact with a codebase.

**Cost.** The dark palette is declared twice — once under
`prefers-color-scheme`, once under `[data-theme='dark']` — so each new token is
two declarations. `light-dark()` would collapse them and is well supported;
rewriting the token layer during a packaging phase was judged out of scope, and
the duplication is flagged in the file itself.
