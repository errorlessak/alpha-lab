/**
 * Number formatting for the data face.
 *
 * §0.7 says a metric that cannot be computed raises rather than returning NaN.
 * The interface has the same duty one layer out: an absent number renders as
 * an em dash, never as `NaN`, never as `0`, and never as a blank cell that
 * reads as zero to a scanning eye.
 */

import { describe, expect, it } from 'vitest'

import {
  fmtBps,
  fmtDate,
  fmtInterval,
  fmtNum,
  fmtPct,
  fmtSigned,
  runPickerLabel,
  shortId,
} from './format'

describe('fmtNum', () => {
  it('renders to a fixed number of decimals', () => {
    expect(fmtNum(0.9604, 3)).toBe('0.960')
  })

  it('renders an absent value as an em dash, never as zero', () => {
    expect(fmtNum(null)).toBe('—')
    expect(fmtNum(undefined)).toBe('—')
  })

  it('refuses to render a non-finite number as a number', () => {
    // A NaN that reaches the screen is a bug that looks like a measurement.
    expect(fmtNum(Number.NaN)).toBe('—')
    expect(fmtNum(Number.POSITIVE_INFINITY)).toBe('—')
  })
})

describe('fmtSigned', () => {
  it('keeps the sign visible on positive numbers', () => {
    expect(fmtSigned(0.42, 2)).toBe('+0.42')
    expect(fmtSigned(-0.42, 2)).toBe('−0.42')
  })

  it('uses a real minus sign, not a hyphen', () => {
    // A hyphen is narrower than a digit and breaks decimal alignment in a
    // tabular-nums column.
    expect(fmtSigned(-1, 0)).toContain('−')
    expect(fmtSigned(-1, 0)).not.toContain('-')
  })

  it('renders zero without a sign', () => {
    expect(fmtSigned(0, 2)).toBe('0.00')
  })

  it('renders an absent value as an em dash', () => {
    expect(fmtSigned(null)).toBe('—')
  })
})

describe('fmtPct', () => {
  it('renders a fraction as a percentage', () => {
    expect(fmtPct(-0.1576, 2)).toBe('−15.76%')
  })

  it('renders an absent value as an em dash', () => {
    expect(fmtPct(null)).toBe('—')
  })
})

describe('fmtBps', () => {
  it('carries its unit', () => {
    expect(fmtBps(20)).toBe('20.0 bps')
  })

  it('renders an absent value as an em dash without a stray unit', () => {
    expect(fmtBps(null)).toBe('—')
  })
})

describe('fmtInterval', () => {
  it('renders a confidence interval as a closed range', () => {
    expect(fmtInterval([0.16, 1.94])).toBe('[0.16, 1.94]')
  })

  it('renders an absent interval as not computed', () => {
    // Distinct from an em dash: the interval was not estimated, which is a
    // different fact from a value being missing.
    expect(fmtInterval(null)).toBe('not computed')
  })
})

describe('shortId', () => {
  it('takes the trailing hash a run id ends with', () => {
    expect(shortId('2026-07-28T05-40-19Z_smoke_5ad0f1')).toBe('5ad0f1')
  })

  it('leaves a short id alone', () => {
    expect(shortId('abc123')).toBe('abc123')
  })

  it('handles an empty id without throwing', () => {
    expect(shortId('')).toBe('')
  })
})

describe('fmtDate', () => {
  it('renders an ISO timestamp as a calendar date', () => {
    expect(fmtDate('2026-07-28T05:40:19Z')).toBe('2026-07-28')
  })

  it('renders an unparseable value as an em dash rather than Invalid Date', () => {
    expect(fmtDate('not a date')).toBe('—')
    expect(fmtDate(null)).toBe('—')
  })
})

describe('runPickerLabel', () => {
  const run = {
    run_id: '2026-07-28T05-40-19Z_smoke_5ad0f1',
    name: 'smoke',
    created_at: '2026-07-28T05:40:19Z',
  }

  it('carries what tells two runs of the same config apart', () => {
    // The config hash is deliberately shared by every run of a config, so a
    // picker that shows only the hash offers thirteen identical options.
    const label = runPickerLabel(run)
    expect(label).toContain('5ad0f1')
    expect(label).toContain('smoke')
    expect(label).toContain('2026-07-28 05:40')
  })

  it('renders a run with no name or timestamp without printing undefined', () => {
    const label = runPickerLabel({ run_id: 'x_y_abc123', name: null, created_at: null })
    expect(label).toContain('abc123')
    expect(label).not.toContain('undefined')
    expect(label).not.toContain('null')
  })
})
