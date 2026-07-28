/**
 * The margin critique engine (§13.2).
 *
 * This is the code that makes the tool argue with itself. Every rule here
 * exists to stop a reader accidentally overlooking the reason a result is not
 * real, which §13.1 names as the single job of the whole interface.
 *
 * The tests are written against the shape the API actually returns, so a note
 * that fires in a test fires on a real run.
 */

import { describe, expect, it } from 'vitest'

import { critique, severityRank } from './critique'
import type { Tearsheet } from './api'

function tearsheet(overrides: Partial<Tearsheet> = {}): Tearsheet {
  return {
    schema_version: '1.0',
    run_id: '2026-07-28T00-00-00Z_smoke_abc123',
    created_at: '2026-07-28T00:00:00Z',
    manifest: {
      config_hash: 'abc123',
      data_hash: 'def456',
      code_version: '3a7aecb',
      seed: 20260727,
      python: '3.12.13',
      packages: {},
      duration_s: 10,
      host: 'test',
    },
    dataset: {
      symbols: ['SYNTH'],
      interval: '1d',
      calendar: 'XNYS',
      span: ['2015-04-15', '2024-12-26'],
      n_bars: 1411,
      quality: 'PASS',
      quality_report_ref: 'quality.json',
      periods_per_year: 251.6,
    },
    features: null,
    labels: null,
    cv: null,
    leaderboard: [],
    selected_model: 'lightgbm',
    backtest: null,
    overfitting: {
      pbo: null,
      median_logit: null,
      n_configs_evaluated: 6,
      n_splits: 0,
      shuffled_label_sharpe: null,
      shuffled_label_ic: null,
    },
    explain: null,
    warnings: [],
    disclaimer: 'Research output. Not investment advice.',
    ...overrides,
  } as Tearsheet
}

function overfitting(overrides: Partial<Tearsheet['overfitting']> = {}): Tearsheet['overfitting'] {
  return {
    pbo: null,
    median_logit: null,
    n_configs_evaluated: 6,
    n_splits: 0,
    shuffled_label_sharpe: null,
    shuffled_label_ic: null,
    ...overrides,
  }
}

function row(overrides: Partial<Tearsheet['leaderboard'][number]> = {}) {
  return {
    model: 'lightgbm',
    rank: 1,
    n_predictions: 700,
    n_folds: 3,
    n_trials: 6,
    flags: [],
    error: null,
    accuracy: null,
    balanced_accuracy: null,
    precision: null,
    recall: null,
    f1: null,
    auc: null,
    pr_auc: null,
    brier: null,
    log_loss: null,
    ic: null,
    ic_tstat: null,
    oos_net_sharpe: 1.2,
    sharpe_ci: [0.4, 2.0],
    dsr: 0.8,
    psr: 0.95,
    break_even_bps: 20,
    max_drawdown: -0.15,
    ...overrides,
  } as Tearsheet['leaderboard'][number]
}

describe('critique', () => {
  it('says nothing when there is nothing to object to', () => {
    const notes = critique(
      tearsheet({
        leaderboard: [row()],
        overfitting: overfitting({ pbo: 0.1 }),
        backtest: { break_even_bps: 20, round_trip_bps: 3 } as Tearsheet['backtest'],
      }),
    )
    // An empty margin is information: this run survived the checks the tool
    // knows how to make.
    expect(notes).toHaveLength(0)
  })

  it('objects when the probability of backtest overfitting is worse than chance', () => {
    const notes = critique(
      tearsheet({
        leaderboard: [row()],
        overfitting: overfitting({ pbo: 0.62 }),
      }),
    )
    const note = notes.find((n) => n.id === 'pbo')
    expect(note).toBeDefined()
    expect(note?.severity).toBe('blocking')
    expect(note?.headline).toContain('0.62')
    expect(note?.body).toContain('worse than chance')
    expect(note?.anchor).toBe('leaderboard')
  })

  it('does not object to a probability of overfitting that holds up', () => {
    const notes = critique(
      tearsheet({ leaderboard: [row()], overfitting: overfitting({ pbo: 0.2 }) }),
    )
    expect(notes.find((n) => n.id === 'pbo')).toBeUndefined()
  })

  it('objects when the deflated Sharpe does not clear the line', () => {
    const notes = critique(tearsheet({ leaderboard: [row({ dsr: 0.31 })] }))
    const note = notes.find((n) => n.id === 'dsr')
    expect(note?.severity).toBe('blocking')
    expect(note?.headline).toContain('0.31')
    expect(note?.anchor).toBe('verdict')
  })

  it('objects when the Sharpe interval contains zero', () => {
    const notes = critique(tearsheet({ leaderboard: [row({ sharpe_ci: [-0.3, 1.9] })] }))
    const note = notes.find((n) => n.id === 'sharpe-ci')
    expect(note?.severity).toBe('blocking')
    expect(note?.body).toContain('zero')
  })

  it('objects when the strategy breaks even below the cost it assumed', () => {
    const notes = critique(
      tearsheet({
        leaderboard: [row()],
        backtest: { break_even_bps: 1.2, round_trip_bps: 3.0 } as Tearsheet['backtest'],
      }),
    )
    const note = notes.find((n) => n.id === 'break-even')
    expect(note?.severity).toBe('blocking')
    expect(note?.headline).toContain('1.2')
    expect(note?.body).toContain('3.0')
    expect(note?.anchor).toBe('equity')
  })

  it('accepts a break-even cost with real margin over the assumption', () => {
    const notes = critique(
      tearsheet({
        leaderboard: [row()],
        backtest: { break_even_bps: 20, round_trip_bps: 3 } as Tearsheet['backtest'],
      }),
    )
    expect(notes.find((n) => n.id === 'break-even')).toBeUndefined()
  })

  it('turns every tearsheet warning into a note in the researcher’s vocabulary', () => {
    const notes = critique(tearsheet({ leaderboard: [row()], warnings: ['UNCOSTED'] }))
    const note = notes.find((n) => n.id === 'warning:UNCOSTED')
    expect(note).toBeDefined()
    expect(note?.body).toContain('no costs')
  })

  it('does not repeat itself when a warning duplicates a computed rule', () => {
    // The tearsheet raises LOW_DSR and the DSR rule fires on the same fact.
    // Saying it twice in a 280px gutter is noise.
    const notes = critique(tearsheet({ leaderboard: [row({ dsr: 0.2 })], warnings: ['LOW_DSR'] }))
    // Searches the whole rendered note: a reader sees headline and body
    // together, so "stated once" is a property of the pair, not of one field.
    const mentions = notes.filter((n) =>
      `${n.headline} ${n.body}`.toLowerCase().includes('deflated'),
    )
    expect(mentions).toHaveLength(1)
  })

  it('names an unrecognised warning rather than dropping it', () => {
    // A warning the UI has no copy for is still a warning. Silently swallowing
    // it is how a reviewer misses the one that mattered.
    const notes = critique(tearsheet({ leaderboard: [row()], warnings: ['SOMETHING_NEW'] }))
    const note = notes.find((n) => n.id === 'warning:SOMETHING_NEW')
    expect(note).toBeDefined()
    expect(note?.headline).toContain('SOMETHING_NEW')
  })

  it('points out when nothing beat buy-and-hold', () => {
    const notes = critique(
      tearsheet({
        selected_model: 'baseline_buyhold',
        leaderboard: [row({ model: 'baseline_buyhold' })],
      }),
    )
    const note = notes.find((n) => n.id === 'baseline-wins')
    expect(note).toBeDefined()
    expect(note?.body).toContain('buy-and-hold')
  })

  it('points out when a model is barely ahead of shuffled labels', () => {
    const notes = critique(
      tearsheet({
        leaderboard: [row({ ic: 0.012 }), row({ model: 'baseline_shuffled', rank: 2, ic: 0.011 })],
      }),
    )
    const note = notes.find((n) => n.id === 'shuffled-margin')
    expect(note?.severity).toBe('blocking')
    expect(note?.body).toContain('shuffled')
  })

  it('says nothing about shuffled labels when the margin is real', () => {
    const notes = critique(
      tearsheet({
        leaderboard: [row({ ic: 0.09 }), row({ model: 'baseline_shuffled', rank: 2, ic: 0.001 })],
      }),
    )
    expect(notes.find((n) => n.id === 'shuffled-margin')).toBeUndefined()
  })

  it('objects when a run has no leaderboard at all', () => {
    const notes = critique(tearsheet({ leaderboard: [], selected_model: null }))
    expect(notes.find((n) => n.id === 'no-result')).toBeDefined()
  })

  it('sorts blocking objections above cautions and notes', () => {
    const notes = critique(
      tearsheet({
        leaderboard: [row({ dsr: 0.2 })],
        warnings: ['UNSTOPPED_WARMUP'],
        overfitting: overfitting({ pbo: 0.62 }),
      }),
    )
    const ranks = notes.map((n) => severityRank(n.severity))
    expect(ranks).toEqual([...ranks].sort((a, b) => a - b))
    expect(notes[0]?.severity).toBe('blocking')
  })

  it('groups notes by the plate they belong beside', () => {
    const notes = critique(
      tearsheet({
        leaderboard: [row({ dsr: 0.2 })],
        overfitting: overfitting({ pbo: 0.62 }),
      }),
    )
    expect(new Set(notes.map((n) => n.anchor))).toEqual(new Set(['verdict', 'leaderboard']))
  })
})
