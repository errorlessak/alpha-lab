/**
 * The tolerance rail.
 *
 * §13.4 asks the run overview for "headline metrics with CIs". The template
 * answer is a row of big numbers with small labels — and here that is not just
 * generic, it is *wrong*: a big number is exactly the thing this interface
 * exists to stop a reader taking at face value.
 *
 * A calibration certificate does not print a big number. It prints a measured
 * value against the tolerance it had to fall inside. So each headline metric is
 * a hairline scale carrying the point estimate, its confidence interval, and
 * the threshold it must clear. A Sharpe of 0.96 whose interval crosses zero
 * *looks* like what it is, without the reader comparing two numbers in their
 * head.
 *
 * Deliberately quiet. The boldness budget is spent on the Fold Strip; this
 * device wins by being legible.
 */

import { fmtInterval, fmtNum } from '../lib/format'

interface ToleranceRailProps {
  label: string
  value: number | null | undefined
  /** Inclusive scale bounds. */
  domain: [number, number]
  /** The line the value has to clear, and which way clears it. */
  threshold?: { at: number; label: string; direction: 'above' | 'below' } | undefined
  interval?: readonly number[] | null | undefined
  /** How to render the value; the rail is unitless. */
  format?: (value: number | null | undefined) => string
}

export function ToleranceRail({
  label,
  value,
  domain,
  threshold,
  interval,
  format = (v) => fmtNum(v, 3),
}: ToleranceRailProps) {
  const [min, max] = domain
  const span = Math.max(max - min, Number.EPSILON)
  const pos = (n: number) => `${(((clamp(n, min, max) - min) / span) * 100).toFixed(2)}%`

  const hasValue = typeof value === 'number' && Number.isFinite(value)
  const bounds =
    interval && interval.length >= 2 && interval.every((n) => Number.isFinite(n))
      ? ([interval[0] as number, interval[1] as number] as const)
      : null

  const clears =
    hasValue && threshold
      ? threshold.direction === 'above'
        ? value > threshold.at
        : value < threshold.at
      : null

  return (
    <div className="grid grid-cols-[minmax(9rem,auto)_1fr_minmax(5rem,auto)] items-center gap-x-4 gap-y-1 py-2">
      <span className="text-xs text-[color:var(--color-ink)]">{label}</span>

      <div className="relative h-6" aria-hidden>
        {/* the rail */}
        <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-[color:var(--color-rule)]" />
        {/* end caps, so the scale reads as bounded rather than clipped */}
        <div className="absolute left-0 top-1/2 h-2.5 w-px -translate-y-1/2 bg-[color:var(--color-rule)]" />
        <div className="absolute right-0 top-1/2 h-2.5 w-px -translate-y-1/2 bg-[color:var(--color-rule)]" />

        {threshold ? (
          <div
            className="absolute top-1/2 h-4 w-px -translate-y-1/2 bg-[color:var(--color-ink-muted)]"
            style={{ left: pos(threshold.at) }}
          />
        ) : null}

        {bounds ? (
          <div
            className="absolute top-1/2 h-[3px] -translate-y-1/2 bg-[color:var(--color-rule)]"
            style={{
              left: pos(bounds[0]),
              width: `${Math.max(
                ((clamp(bounds[1], min, max) - clamp(bounds[0], min, max)) / span) * 100,
                0.5,
              ).toFixed(2)}%`,
            }}
          />
        ) : null}

        {hasValue ? (
          <div
            className="absolute top-1/2 h-3.5 w-[3px] -translate-x-[1px] -translate-y-1/2"
            style={{
              left: pos(value),
              background: clears === false ? 'var(--color-oxblood)' : 'var(--color-prussian)',
            }}
          />
        ) : null}
      </div>

      <span className="tabular text-right text-xs text-[color:var(--color-ink)]">
        {format(value)}
      </span>

      {/* The scale's own labels, and the verdict in words — colour is never the
          only encoder (§13.2). */}
      <span />
      <span className="tabular text-2xs text-[color:var(--color-ink-muted)]">
        {threshold ? (
          <>
            must be {threshold.direction} {fmtNum(threshold.at, 2)} · {threshold.label}
            {clears === false ? ' · does not clear' : clears === true ? ' · clears' : ''}
          </>
        ) : (
          <>
            scale {fmtNum(min, 2)} to {fmtNum(max, 2)}
          </>
        )}
      </span>
      <span className="tabular text-right text-2xs text-[color:var(--color-ink-muted)]">
        {bounds ? fmtInterval(bounds) : ''}
      </span>
    </div>
  )
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}
