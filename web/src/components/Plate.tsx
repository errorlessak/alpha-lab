/**
 * A plate is a figure on the page.
 *
 * Hairline above, never a card, never a shadow (§13.2). Depth on this page
 * comes from rules and recessed fills, the way it does on a printed report.
 *
 * Every plate carries an `id` because the margin anchors its notes to one, and
 * a heading because a figure without a caption is not a figure.
 */

import type { ReactNode } from 'react'

import { ProvenanceStamp } from './ProvenanceStamp'
import type { StampFacts } from './ProvenanceStamp'

interface PlateProps {
  /** Anchor id — the margin aligns notes to this element. */
  id: string
  title: string
  /** Shown beside the title in the data face: counts, spans, units. */
  meta?: ReactNode
  /** Provenance facts. A plate that shows numbers must pass these (§13.2). */
  stamp?: StampFacts
  children: ReactNode
}

export function Plate({ id, title, meta, stamp, children }: PlateProps) {
  return (
    <section id={id} data-plate={id} className="plate" aria-labelledby={`${id}-title`}>
      <header className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2
          id={`${id}-title`}
          className="font-display text-[color:var(--color-ink)] text-base font-semibold"
          style={{ fontStretch: '92%' }}
        >
          {title}
        </h2>
        {meta ? (
          <p className="tabular text-2xs text-[color:var(--color-ink-muted)]">{meta}</p>
        ) : null}
      </header>

      {children}

      {stamp ? <ProvenanceStamp facts={stamp} /> : null}
    </section>
  )
}
