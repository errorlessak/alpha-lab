/**
 * The margin — the third device (§13.2).
 *
 * A persistent 280px right gutter holding generated notes tied to what is on
 * screen. The app argues with itself, in the margin, in the researcher's own
 * vocabulary.
 *
 * **The risk, per docs/design-plan.md §6.** These are typeset as true
 * marginalia: each note aligns to the top edge of the plate it refers to and
 * connects to it with a hairline leader. Web interfaces normally put this in a
 * tooltip or a stacked sidebar. A tooltip hides the criticism behind an
 * interaction; a stack makes the reader match note to chart by reading. The
 * brief's single job is that the disqualifying context is impossible to
 * overlook, and coordinating the objection in space with its subject is the
 * typographic answer to exactly that.
 *
 * **How it degrades.** Below 1280px the gutter collapses and every note
 * renders as an inline callout directly beneath its plate — same content, same
 * wording, no leaders. The leader is an enhancement on wide screens and never
 * load-bearing.
 */

import { useEffect, useState } from 'react'

import type { CritiqueSeverity, MarginNote } from '../lib/critique'

const SEVERITY_LABEL: Record<CritiqueSeverity, string> = {
  blocking: 'disqualifying',
  caution: 'caution',
  note: 'note',
}

/** The leader hairline: a graphic, measured against the 3:1 bar. */
const SEVERITY_COLOUR: Record<CritiqueSeverity, string> = {
  blocking: 'var(--color-oxblood)',
  caution: 'var(--color-sulphur)',
  note: 'var(--color-ink-muted)',
}

/** The label above the note: 12px text, measured against the 4.5:1 bar. See
 *  the text-only variants in `styles/tokens.css`. */
const SEVERITY_TEXT_COLOUR: Record<CritiqueSeverity, string> = {
  blocking: 'var(--color-danger-text)',
  caution: 'var(--color-caution-text)',
  note: 'var(--color-ink-muted)',
}

/**
 * The wide-screen gutter. Notes are absolutely positioned against the top of
 * the plate they belong to, measured from the DOM.
 */
export function Margin({ notes }: { notes: MarginNote[] }) {
  const offsets = useAnchorOffsets(notes)

  if (notes.length === 0) {
    return (
      <aside
        aria-label="Margin notes"
        className="relative hidden w-[280px] shrink-0 border-l border-[color:var(--color-rule)] pl-5 xl:block"
      >
        <p className="sticky top-6 text-2xs text-[color:var(--color-ink-muted)]">
          Nothing to flag. This run passed every check this interface knows how to make — which is
          not the same as being right.
        </p>
      </aside>
    )
  }

  return (
    <aside
      aria-label="Margin notes"
      className="relative hidden w-[280px] shrink-0 border-l border-[color:var(--color-rule)] pl-5 xl:block"
    >
      {notes.map((note) => (
        <article
          key={note.id}
          data-testid="margin-note"
          className="absolute left-5 right-0 pr-1"
          style={{ top: offsets[note.id] ?? 0 }}
        >
          {/* the leader: a hairline from the gutter edge back toward the plate */}
          <span
            aria-hidden
            className="absolute -left-5 top-[0.6rem] block h-px w-4"
            style={{ background: SEVERITY_COLOUR[note.severity] }}
          />
          <NoteBody note={note} />
        </article>
      ))}
    </aside>
  )
}

/**
 * The narrow-screen form: the same notes, inline, under their plate.
 *
 * Rendered by each plate rather than by the gutter, so the note stays adjacent
 * to its subject when there is no gutter to hang it in.
 */
export function InlineNotes({ notes }: { notes: MarginNote[] }) {
  if (notes.length === 0) return null
  return (
    <div className="mt-4 space-y-3 border-l-2 border-[color:var(--color-rule)] pl-3 xl:hidden">
      {notes.map((note) => (
        <article key={note.id} data-testid="margin-note-inline">
          <NoteBody note={note} />
        </article>
      ))}
    </div>
  )
}

function NoteBody({ note }: { note: MarginNote }) {
  return (
    <>
      <p
        className="tabular text-2xs uppercase tracking-wide"
        style={{ color: SEVERITY_TEXT_COLOUR[note.severity] }}
      >
        {SEVERITY_LABEL[note.severity]}
      </p>
      <p className="mt-1 text-xs leading-snug text-[color:var(--color-ink)]">
        <span className="tabular font-medium">{note.headline}</span>{' '}
        <span className="text-[color:var(--color-ink-muted)]">{note.body}</span>
      </p>
    </>
  )
}

/**
 * Measure where each note's plate begins, so the note can sit beside it.
 *
 * A ResizeObserver rather than a one-shot measurement: charts settle after
 * their data arrives, and a note pinned to a stale offset is worse than no
 * leader at all.
 */
function useAnchorOffsets(notes: MarginNote[]): Record<string, number> {
  const [offsets, setOffsets] = useState<Record<string, number>>({})
  const anchors = notes.map((note) => note.anchor).join(',')
  const ids = notes.map((note) => note.id).join(',')

  useEffect(() => {
    const container = document.querySelector('[data-plates]')
    if (!container) return

    const measure = () => {
      const containerTop = container.getBoundingClientRect().top
      const next: Record<string, number> = {}
      const used: Record<string, number> = {}

      for (const note of notes) {
        const plate = document.querySelector(`[data-plate="${note.anchor}"]`)
        if (!plate) continue
        const top = plate.getBoundingClientRect().top - containerTop
        // Two notes on one plate stack rather than overlap.
        const stacked = used[note.anchor] ?? 0
        next[note.id] = Math.max(top, 0) + stacked
        used[note.anchor] = stacked + 76
      }
      setOffsets(next)
    }

    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(container)
    for (const note of notes) {
      const plate = document.querySelector(`[data-plate="${note.anchor}"]`)
      if (plate) observer.observe(plate)
    }
    window.addEventListener('resize', measure)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
    }
    // `notes` is rebuilt on every render; its identity is not meaningful, the
    // set of anchors and ids is.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anchors, ids])

  return offsets
}
