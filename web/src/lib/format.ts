/**
 * Number formatting for the data face.
 *
 * §0.7 makes the library raise rather than return NaN when a metric cannot be
 * computed. The interface owes the same honesty one layer out: an absent
 * number renders as an em dash — never `NaN`, never `0`, and never an empty
 * cell, because a scanning eye reads a blank as a zero.
 *
 * Everything here emits a real minus sign (U+2212) rather than a hyphen. A
 * hyphen is narrower than a digit, so a column of negative numbers in a
 * `tabular-nums` face stops aligning on the decimal point, which is the one
 * thing the mono face was chosen for.
 */

const MISSING = '—'
const MINUS = '−'

type Maybe = number | null | undefined

function usable(value: Maybe): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

/** Fixed-decimal number, or an em dash when there is nothing to show. */
export function fmtNum(value: Maybe, decimals = 2): string {
  if (!usable(value)) return MISSING
  return value.toFixed(decimals).replace('-', MINUS)
}

/** Number with its sign always visible, for deltas and returns. */
export function fmtSigned(value: Maybe, decimals = 2): string {
  if (!usable(value)) return MISSING
  if (value === 0) return (0).toFixed(decimals)
  const rendered = Math.abs(value).toFixed(decimals)
  return value > 0 ? `+${rendered}` : `${MINUS}${rendered}`
}

/** A fraction rendered as a percentage. */
export function fmtPct(value: Maybe, decimals = 2): string {
  if (!usable(value)) return MISSING
  return `${(value * 100).toFixed(decimals).replace('-', MINUS)}%`
}

/** Basis points, carrying their unit so a bare number cannot be misread. */
export function fmtBps(value: Maybe, decimals = 1): string {
  if (!usable(value)) return MISSING
  return `${value.toFixed(decimals).replace('-', MINUS)} bps`
}

/**
 * A confidence interval.
 *
 * Absent renders as "not computed" rather than an em dash: an interval that
 * was never estimated is a different fact from a value that is missing, and
 * §11.2 asks for the interval to be reported rather than only the point.
 */
export function fmtInterval(interval: readonly number[] | null | undefined, decimals = 2): string {
  if (!interval || interval.length < 2) return 'not computed'
  const [lower, upper] = interval
  if (!usable(lower) || !usable(upper)) return 'not computed'
  return `[${fmtNum(lower, decimals)}, ${fmtNum(upper, decimals)}]`
}

/**
 * The trailing hash of a run id.
 *
 * Run ids are `2026-07-28T05-40-19Z_smoke_5ad0f1`. The hash is what a
 * researcher says out loud and types into `alpha runs show`.
 */
export function shortId(runId: string): string {
  if (!runId) return ''
  const parts = runId.split('_')
  return parts[parts.length - 1] ?? runId
}

/** An ISO timestamp as a calendar date, or an em dash if it will not parse. */
export function fmtDate(value: string | null | undefined): string {
  if (!value) return MISSING
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return MISSING
  return parsed.toISOString().slice(0, 10)
}

/** An ISO timestamp as date and minute, for job timelines. */
export function fmtDateTime(value: string | null | undefined): string {
  if (!value) return MISSING
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return MISSING
  return parsed.toISOString().slice(0, 16).replace('T', ' ')
}

/** A duration in seconds, at the precision a human cares about. */
export function fmtDuration(seconds: Maybe): string {
  if (!usable(seconds)) return MISSING
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${Math.round(seconds - minutes * 60)}s`
}
