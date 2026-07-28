/**
 * The margin critique engine (§13.2).
 *
 * The app arguing with itself, in the margin, in the researcher's own
 * vocabulary. Notes come from the tearsheet's `warnings` array plus the rules
 * below, and each one is anchored to the plate it is about so the objection
 * sits beside the number it disqualifies.
 *
 * Two rules govern what belongs here:
 *
 * 1. **Every note names a threshold and where the run landed against it.**
 *    "PBO 0.62 — the selection process performs worse than chance" is useful;
 *    "possible overfitting" is not.
 * 2. **A fact is stated once.** The tearsheet raises `LOW_DSR` and the DSR rule
 *    fires on the same number; saying it twice in a 280px gutter is noise, and
 *    noise is how the note that mattered gets skipped.
 *
 * An empty margin is information. It means this run survived the checks the
 * tool knows how to make — not that nobody looked.
 */

import type { Tearsheet } from './api'

/** How much a note should interrupt the reader. */
export type CritiqueSeverity = 'blocking' | 'caution' | 'note'

/** Which plate a note belongs beside. */
export type CritiqueAnchor = 'verdict' | 'leaderboard' | 'equity' | 'folds' | 'data'

export interface MarginNote {
  /** Stable id, so a note keeps its identity across refetches. */
  id: string
  anchor: CritiqueAnchor
  severity: CritiqueSeverity
  /** The measurement, e.g. `PBO 0.62`. Set in the data face. */
  headline: string
  /** What it means, in the researcher's vocabulary. */
  body: string
}

/** Sort order for severities: blocking first. */
export function severityRank(severity: CritiqueSeverity): number {
  return severity === 'blocking' ? 0 : severity === 'caution' ? 1 : 2
}

/** §11.3: above this the selection process performs worse than chance. */
const WORSE_THAN_CHANCE_PBO = 0.5

/** §11.2: below this the deflated Sharpe cannot distinguish the result from
 *  the best of a set of coin flips. */
const WEAK_DSR = 0.5

/** An information coefficient this close to the shuffled-label baseline's is
 *  not a margin, it is rounding. */
const SHUFFLED_MARGIN = 0.005

/**
 * Copy for each flag the pipeline can raise. Written as the sentence a
 * reviewer would say out loud, not as the flag name with spaces in it.
 */
const WARNING_COPY: Record<
  string,
  { severity: CritiqueSeverity; anchor: CritiqueAnchor; body: string }
> = {
  UNCOSTED: {
    severity: 'blocking',
    anchor: 'equity',
    body: 'This backtest charged no costs — no commission, no spread, no impact. The equity curve describes an execution nobody can obtain.',
  },
  NO_BACKTEST: {
    severity: 'blocking',
    anchor: 'leaderboard',
    body: 'No costed backtest ran, so the leaderboard is ranked on the information coefficient rather than on money.',
  },
  LOW_DSR: {
    severity: 'blocking',
    anchor: 'verdict',
    body: 'The deflated Sharpe is low. Corrected for how many configurations were tried, this is not distinguishable from the best of a set of coin flips.',
  },
  HIGH_PBO: {
    severity: 'blocking',
    anchor: 'leaderboard',
    body: 'The selection process performs worse than chance. The configuration chosen here is more likely than not to underperform the median out of sample.',
  },
  SHORT_SAMPLE: {
    severity: 'caution',
    anchor: 'verdict',
    body: 'The sample is shorter than the minimum track record length, so the Sharpe has a wide interval whatever its point estimate.',
  },
  KILL_SWITCH: {
    severity: 'caution',
    anchor: 'equity',
    body: 'The drawdown kill switch tripped and trading stopped before the end of the sample.',
  },
  UNSTOPPED_WARMUP: {
    severity: 'note',
    anchor: 'equity',
    body: 'A position was held before the ATR warm-up finished, so the configured stops were not active for part of the run.',
  },
  SUSPICIOUS_ACCURACY: {
    severity: 'blocking',
    anchor: 'leaderboard',
    body: 'Directional accuracy is above 0.55. Real short-horizon edges live at 0.51 to 0.54, so this usually means leakage rather than skill.',
  },
  UNTRUSTED_DATA: {
    severity: 'blocking',
    anchor: 'data',
    body: 'The data failed its quality thresholds and was let through on an override. Every number downstream inherits that.',
  },
  SHORT_THRESHOLD_UNUSED: {
    severity: 'note',
    anchor: 'verdict',
    body: 'Shorting is disabled while a short threshold is configured, so that threshold does nothing.',
  },
}

/**
 * Build the margin notes for a tearsheet.
 *
 * Returns them sorted with blocking objections first, because the gutter is
 * read top to bottom and the thing that disqualifies the result should not be
 * below the thing that merely qualifies it.
 */
export function critique(tearsheet: Tearsheet): MarginNote[] {
  const notes: MarginNote[] = []
  const top = tearsheet.leaderboard?.[0] ?? null

  if (!top) {
    notes.push({
      id: 'no-result',
      anchor: 'verdict',
      severity: 'blocking',
      headline: 'No result',
      body: 'This run produced no leaderboard, so there is nothing here to judge. Check the job log for the phase that stopped it.',
    })
  }

  const pbo = tearsheet.overfitting?.pbo ?? null
  if (pbo !== null && pbo > WORSE_THAN_CHANCE_PBO) {
    notes.push({
      id: 'pbo',
      anchor: 'leaderboard',
      severity: 'blocking',
      headline: `PBO ${pbo.toFixed(2)}`,
      body: 'the selection process performs worse than chance. The configuration chosen here is more likely than not to underperform the median out of sample.',
    })
  }

  const dsr = top?.dsr ?? null
  if (dsr !== null && dsr < WEAK_DSR) {
    notes.push({
      id: 'dsr',
      anchor: 'verdict',
      severity: 'blocking',
      headline: `deflated Sharpe ${dsr.toFixed(2)}`,
      body: `corrected for the ${tearsheet.overfitting?.n_configs_evaluated ?? top?.n_trials ?? 1} configurations tried, this result is not distinguishable from the best of a set of coin flips.`,
    })
  }

  const interval = top?.sharpe_ci ?? null
  const lower = interval?.[0]
  const upper = interval?.[1]
  if (typeof lower === 'number' && typeof upper === 'number') {
    if (lower <= 0 && upper >= 0) {
      notes.push({
        id: 'sharpe-ci',
        anchor: 'verdict',
        severity: 'blocking',
        headline: `Sharpe interval [${lower.toFixed(2)}, ${upper.toFixed(2)}]`,
        body: 'contains zero. The point estimate is positive but the interval does not exclude no edge at all.',
      })
    }
  }

  const breakEven = tearsheet.backtest?.break_even_bps ?? null
  const assumed = tearsheet.backtest?.round_trip_bps ?? null
  if (breakEven !== null && assumed !== null && breakEven <= assumed) {
    notes.push({
      id: 'break-even',
      anchor: 'equity',
      severity: 'blocking',
      headline: `break-even ${breakEven.toFixed(1)} bps`,
      body: `your assumed round-trip cost is ${assumed.toFixed(1)}. The strategy does not survive its own cost model.`,
    })
  }

  // No length guard: a baseline ranking first is worth saying whether it beat
  // four candidates or none. Every leaderboard carries the five mandatory
  // baselines anyway, so "there were no models to beat it" is not a real case.
  if (top && isBaseline(top.model)) {
    notes.push({
      id: 'baseline-wins',
      anchor: 'leaderboard',
      severity: 'blocking',
      headline: `${top.model} ranks first`,
      body: 'no model beat buy-and-hold. Whatever the candidates learned, it did not pay for the turnover it created.',
    })
  }

  const shuffled = tearsheet.leaderboard?.find((row) => row.model === 'baseline_shuffled') ?? null
  const topIc = top?.ic
  const shuffledIc = shuffled?.ic
  if (
    top &&
    shuffled &&
    typeof topIc === 'number' &&
    typeof shuffledIc === 'number' &&
    top.model !== shuffled.model
  ) {
    const margin = Math.abs(topIc) - Math.abs(shuffledIc)
    if (margin < SHUFFLED_MARGIN) {
      notes.push({
        id: 'shuffled-margin',
        anchor: 'leaderboard',
        severity: 'blocking',
        headline: `IC margin ${margin.toFixed(4)}`,
        body: 'over the shuffled-label baseline — the same estimator trained on labels whose relationship to the features was destroyed. This model has not distinguished itself from noise.',
      })
    }
  }

  for (const warning of tearsheet.warnings ?? []) {
    const copy = WARNING_COPY[warning]
    if (copy) {
      notes.push({
        id: `warning:${warning}`,
        anchor: copy.anchor,
        severity: copy.severity,
        headline: warning.toLowerCase().replaceAll('_', ' '),
        body: copy.body,
      })
    } else {
      // A flag the UI has no copy for is still a flag. Swallowing it silently
      // is how a reviewer misses the one that mattered.
      notes.push({
        id: `warning:${warning}`,
        anchor: 'verdict',
        severity: 'caution',
        headline: warning,
        body: 'was raised by the pipeline. This interface has no explanation for it yet — see the run log.',
      })
    }
  }

  return dedupe(notes).sort((a, b) => severityRank(a.severity) - severityRank(b.severity))
}

/**
 * Drop a warning-derived note when a computed rule already said the same
 * thing. The computed one wins because it carries the measurement.
 */
function dedupe(notes: MarginNote[]): MarginNote[] {
  const computed = new Set(notes.filter((note) => !note.id.startsWith('warning:')).map((n) => n.id))
  const duplicates: Record<string, string> = {
    'warning:LOW_DSR': 'dsr',
    'warning:HIGH_PBO': 'pbo',
  }

  const seen = new Set<string>()
  return notes.filter((note) => {
    const shadows = duplicates[note.id]
    if (shadows && computed.has(shadows)) return false
    if (seen.has(note.id)) return false
    seen.add(note.id)
    return true
  })
}

function isBaseline(model: string): boolean {
  return model.startsWith('baseline_')
}
