/**
 * The Provenance Stamp — the second signature element (§13.2).
 *
 * A one-line monospace footer under every chart and table:
 *
 *     run a3f91c · data 7e2b · costs 1.5bps+1.0sp · net of costs · lag 1 bar
 *
 * It looks like a plate stamp on a lab photograph, and it exists so that **no
 * chart can be screenshotted out of context and misread**. A Sharpe ratio in a
 * slide deck with no cost assumption attached is how a backtest becomes a
 * claim; this is the sentence that travels with the pixels.
 *
 * Facts that are genuinely unknown are omitted rather than guessed. A stamp
 * saying "costs 0.0bps" when nobody recorded them would be worse than a stamp
 * that does not mention costs.
 */

import { shortId } from '../lib/format'
import type { Tearsheet } from '../lib/api'

export interface StampFacts {
  runId: string
  dataHash?: string | null
  commissionBps?: number | null
  halfSpreadBps?: number | null
  executionLagBars?: number | null
  netOfCosts?: boolean
  /** Anything else worth stamping on this particular plate. */
  extra?: string[]
}

/** Build the stamp facts a run's plates share. */
export function stampFrom(tearsheet: Tearsheet, extra: string[] = []): StampFacts {
  const costs = tearsheet.backtest?.round_trip_bps ?? null
  return {
    runId: tearsheet.run_id,
    dataHash: tearsheet.manifest.data_hash,
    // The tearsheet records the round trip; the split into commission and
    // spread lives in the resolved config, so the stamp shows the total it can
    // actually vouch for.
    commissionBps: costs,
    halfSpreadBps: null,
    executionLagBars: 1,
    netOfCosts: tearsheet.backtest !== null,
    extra,
  }
}

export function ProvenanceStamp({ facts }: { facts: StampFacts }) {
  const parts: string[] = [`run ${shortId(facts.runId)}`]

  if (facts.dataHash) parts.push(`data ${facts.dataHash.slice(0, 6)}`)

  if (typeof facts.commissionBps === 'number') {
    const spread =
      typeof facts.halfSpreadBps === 'number' ? `+${facts.halfSpreadBps.toFixed(1)}sp` : ''
    parts.push(`costs ${facts.commissionBps.toFixed(1)}bps${spread} round trip`)
  }

  if (facts.netOfCosts) parts.push('net of costs')
  if (typeof facts.executionLagBars === 'number') {
    parts.push(`lag ${facts.executionLagBars} bar${facts.executionLagBars === 1 ? '' : 's'}`)
  }

  parts.push(...(facts.extra ?? []))

  return (
    <p
      data-testid="provenance-stamp"
      className="tabular mt-3 border-t border-dotted border-[color:var(--color-rule)] pt-2 text-2xs text-[color:var(--color-ink-muted)]"
    >
      {parts.join(' · ')}
    </p>
  )
}
