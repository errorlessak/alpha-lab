/**
 * The Fold Strip — the signature element (§13.2).
 *
 * A full-width horizontal timeline of the walk-forward split: prussian bands
 * for train, plum for test, sulphur diagonal hatching for purge and embargo,
 * with the price series drawn as a hairline behind it.
 *
 * It is three things at once, and the third is why it is the hero:
 *
 *   1. the hero of the run page
 *   2. the global fold selector — click a band and every chart below filters
 *   3. **the honest disclosure of exactly how the model was validated**
 *
 * Nothing else on the page is allowed to be this loud (§13.2).
 *
 * Hand-written SVG rather than a charting library. The shapes are rectangles
 * and one polyline; a library would add kilobytes to draw them and would fight
 * every one of the accessibility affordances below.
 *
 * **Colour is never the only encoder.** Purge and embargo carry diagonal
 * hatching as well as their hue, every band is a real `<button>` with a text
 * label for screen readers, and the legend names each region in words.
 */

import { useId, useMemo } from 'react'

import type { FoldsResponse, SeriesResponse } from '../lib/api'
import { fmtDate } from '../lib/format'

interface Fold {
  fold: number
  train_span: [string, string]
  test_span: [string, string]
  purged: number
  embargoed: number
  n_train: number
  n_test: number
}

interface FoldStripProps {
  folds: FoldsResponse | undefined
  /** Price series drawn as a hairline behind the bands. Optional. */
  price?: SeriesResponse | undefined
  /** Currently filtered fold, or null for all. */
  selected: number | null
  onSelect: (fold: number | null) => void
  /** Disables the draw animation. Set from `prefers-reduced-motion`. */
  reduceMotion: boolean
}

const PRICE_HEIGHT = 34
const LANE_HEIGHT = 22
const LANE_GAP = 4
const VIEW_WIDTH = 1000

/**
 * One lane per fold, stacked.
 *
 * Walk-forward windows **overlap in time** — fold 1's training span covers most
 * of fold 0's. Drawing them on a single row paints them over each other and the
 * strip collapses into one long band that says nothing. Stacked, the sliding
 * window is the shape you see, which is the entire point of the element.
 */
function laneTop(index: number): number {
  return PRICE_HEIGHT + index * (LANE_HEIGHT + LANE_GAP)
}

export function FoldStrip({ folds, price, selected, onSelect, reduceMotion }: FoldStripProps) {
  const hatchId = useId().replaceAll(':', '')
  const parsed = useMemo(() => normalise(folds), [folds])

  if (!parsed || parsed.folds.length === 0) {
    return (
      <div
        data-testid="fold-strip-empty"
        className="border border-dashed border-[color:var(--color-rule)] p-6 text-sm text-[color:var(--color-ink-muted)]"
      >
        This run has no folds. It stopped before cross-validation, so nothing below was validated.
      </div>
    )
  }

  const { folds: rows, start, end } = parsed
  const span = Math.max(end - start, 1)
  const x = (time: number) => ((time - start) / span) * VIEW_WIDTH
  const height = laneTop(rows.length) + 14

  return (
    <div data-testid="fold-strip">
      <svg
        viewBox={`0 0 ${VIEW_WIDTH} ${height}`}
        width="100%"
        height={height}
        role="group"
        aria-label={`Walk-forward validation: ${rows.length} folds`}
        className="block overflow-visible"
        preserveAspectRatio="none"
      >
        <defs>
          <pattern
            id={`hatch-${hatchId}`}
            width="6"
            height="6"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--color-sulphur)" strokeWidth="2.5" />
          </pattern>
        </defs>

        {/* Price, as a hairline behind everything. It is context, not data to
            be read off — there is no axis, deliberately. */}
        {price ? <PriceLine price={price} width={VIEW_WIDTH} height={PRICE_HEIGHT} /> : null}

        {rows.map((fold, index) => {
          const isSelected = selected === fold.fold
          const dim = selected !== null && !isSelected
          const y = laneTop(index)
          const trainX = x(fold.trainStart)
          const trainW = Math.max(x(fold.trainEnd) - trainX, 1)
          const testX = x(fold.testStart)
          const testW = Math.max(x(fold.testEnd) - testX, 1)
          const gapX = x(fold.trainEnd)
          const gapW = Math.max(testX - gapX, 0)

          return (
            <g
              key={fold.fold}
              opacity={dim ? 0.28 : 1}
              style={
                reduceMotion
                  ? undefined
                  : {
                      animation: `fold-draw 600ms ease-out both`,
                      animationDelay: `${index * 90}ms`,
                      transformOrigin: 'left center',
                    }
              }
            >
              {/* the unused span, so each lane reads as a full timeline */}
              <rect
                x={0}
                y={y}
                width={VIEW_WIDTH}
                height={LANE_HEIGHT}
                fill="var(--color-paper-sunk)"
              />
              <rect
                x={trainX}
                y={y}
                width={trainW}
                height={LANE_HEIGHT}
                fill="var(--color-prussian)"
              />
              {gapW > 0 ? (
                <rect
                  x={gapX}
                  y={y}
                  width={gapW}
                  height={LANE_HEIGHT}
                  fill={`url(#hatch-${hatchId})`}
                />
              ) : null}
              <rect x={testX} y={y} width={testW} height={LANE_HEIGHT} fill="var(--color-plum)" />
              {/* The lane label sits on prussian for the first fold and on the
                  recessed ground for later ones. A paper halo under the glyph
                  keeps it legible on both without a second colour rule. */}
              <text
                x={5}
                y={y + LANE_HEIGHT / 2 + 4}
                className="tabular"
                fontSize="10"
                fill="var(--color-ink)"
                stroke="var(--color-paper)"
                strokeWidth="2.5"
                paintOrder="stroke"
              >
                {fold.fold}
              </text>
            </g>
          )
        })}
      </svg>

      {/* The bands are drawn in SVG and operated here. Real buttons rather
          than SVG click handlers: keyboard reachability, focus rings and
          screen-reader labels all come free, and none of them come free on a
          <rect>. */}
      <div className="mt-3 flex flex-wrap gap-2">
        <FoldButton
          label="All folds"
          description="Show every fold"
          active={selected === null}
          onClick={() => onSelect(null)}
        />
        {rows.map((fold) => (
          <FoldButton
            key={fold.fold}
            label={`Fold ${fold.fold}`}
            description={`Train ${fmtDate(fold.trainSpan[0])} to ${fmtDate(fold.trainSpan[1])}, test ${fmtDate(fold.testSpan[0])} to ${fmtDate(fold.testSpan[1])}, ${fold.purged} purged and ${fold.embargoed} embargoed`}
            active={selected === fold.fold}
            onClick={() => onSelect(selected === fold.fold ? null : fold.fold)}
          />
        ))}
      </div>

      <Legend />
    </div>
  )
}

function FoldButton({
  label,
  description,
  active,
  onClick,
}: {
  label: string
  description: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      title={description}
      className={[
        'tabular rounded-[2px] border px-2 py-1 text-2xs transition-colors',
        active
          ? 'border-[color:var(--color-ink)] bg-[color:var(--color-ink)] text-[color:var(--color-paper)]'
          : 'border-[color:var(--color-rule)] text-[color:var(--color-ink-muted)] hover:border-[color:var(--color-ink)] hover:text-[color:var(--color-ink)]',
      ].join(' ')}
    >
      {label}
      <span className="sr-only"> — {description}</span>
    </button>
  )
}

function Legend() {
  return (
    <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-2xs text-[color:var(--color-ink-muted)]">
      <li className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="inline-block h-2.5 w-4"
          style={{ background: 'var(--color-prussian)' }}
        />
        train — in sample
      </li>
      <li className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="inline-block h-2.5 w-4"
          style={{ background: 'var(--color-plum)' }}
        />
        test — out of sample
      </li>
      <li className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="inline-block h-2.5 w-4"
          style={{
            backgroundImage:
              'repeating-linear-gradient(45deg, var(--color-sulphur) 0 2px, transparent 2px 5px)',
          }}
        />
        purge and embargo — removed from training
      </li>
    </ul>
  )
}

function PriceLine({
  price,
  width,
  height,
}: {
  price: SeriesResponse
  width: number
  height: number
}) {
  const column = price.columns.find((name) => name in price.values)
  const values = column ? (price.values[column] ?? []) : []
  if (values.length < 2) return null

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = Math.max(max - min, Number.EPSILON)
  const step = width / (values.length - 1)

  const d = values
    .map((value, index) => {
      const y = height - ((value - min) / range) * (height - 8) - 4
      return `${index === 0 ? 'M' : 'L'}${(index * step).toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')

  return (
    <path
      d={d}
      fill="none"
      stroke="var(--color-ink-muted)"
      strokeWidth="1"
      opacity="0.55"
      aria-hidden
    />
  )
}

interface NormalisedFold {
  fold: number
  trainStart: number
  trainEnd: number
  testStart: number
  testEnd: number
  trainSpan: [string, string]
  testSpan: [string, string]
  purged: number
  embargoed: number
}

function normalise(
  folds: FoldsResponse | undefined,
): { folds: NormalisedFold[]; start: number; end: number } | null {
  if (!folds?.folds?.length) return null

  const rows = folds.folds
    .map((raw) => {
      const fold = raw as unknown as Fold
      const trainSpan = fold.train_span
      const testSpan = fold.test_span
      if (!trainSpan || !testSpan) return null
      return {
        fold: fold.fold,
        trainStart: Date.parse(trainSpan[0]),
        trainEnd: Date.parse(trainSpan[1]),
        testStart: Date.parse(testSpan[0]),
        testEnd: Date.parse(testSpan[1]),
        trainSpan,
        testSpan,
        purged: fold.purged ?? 0,
        embargoed: fold.embargoed ?? 0,
      }
    })
    .filter((row): row is NormalisedFold => row !== null && Number.isFinite(row.trainStart))

  if (rows.length === 0) return null

  return {
    folds: rows,
    start: Math.min(...rows.map((row) => row.trainStart)),
    end: Math.max(...rows.map((row) => row.testEnd)),
  }
}
