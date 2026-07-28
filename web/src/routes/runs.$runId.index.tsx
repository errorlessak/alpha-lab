/**
 * `/runs/$id` — the overview.
 *
 * Fold Strip (in the layout above) → the verdict as tolerance rails → equity
 * and underwater → the margin arguing with all of it.
 *
 * The verdict plate deliberately does not lead with a big number. §13.4 asks
 * for "headline metrics with CIs", and a big number is exactly what this
 * interface exists to stop a reader taking at face value. See
 * docs/design-plan.md §5.
 */

import { useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'

import { InlineNotes, Margin } from '../components/Margin'
import { Plate } from '../components/Plate'
import { SeriesChart } from '../components/SeriesChart'
import { stampFrom } from '../components/ProvenanceStamp'
import { ToleranceRail } from '../components/ToleranceRail'
import { ErrorState, Skeleton, SkeletonLines } from '../components/states'
import { api } from '../lib/api'
import { critique } from '../lib/critique'
import type { MarginNote } from '../lib/critique'
import { fmtBps, fmtNum, fmtPct } from '../lib/format'
import { useRun } from '../lib/run-context'

export const Route = createFileRoute('/runs/$runId/')({
  component: Overview,
})

function Overview() {
  const { runId, tearsheet } = useRun()

  const equity = useQuery({
    queryKey: ['series', runId, 'equity', 1200],
    queryFn: () => api.getSeries(runId, 'equity', 1200),
    retry: false,
  })
  const drawdown = useQuery({
    queryKey: ['series', runId, 'drawdown', 1200],
    queryFn: () => api.getSeries(runId, 'drawdown', 1200),
    retry: false,
  })

  if (!tearsheet) {
    return (
      <div className="space-y-6">
        <SkeletonLines rows={5} />
        <Skeleton height={220} />
      </div>
    )
  }

  const notes = critique(tearsheet)
  const top = tearsheet.leaderboard[0]
  const stamp = stampFrom(tearsheet)

  return (
    <div className="flex gap-8">
      <div data-plates className="min-w-0 flex-1 space-y-0">
        <Plate
          id="verdict"
          title="Verdict"
          meta={top ? `selected ${top.model}` : 'no model selected'}
          stamp={stamp}
        >
          {top ? (
            <>
              <div className="max-w-2xl divide-y divide-[color:var(--color-rule)]">
                <ToleranceRail
                  label="net Sharpe"
                  value={top.oos_net_sharpe}
                  interval={top.sharpe_ci}
                  domain={[-1, 2.5]}
                  threshold={{ at: 0, label: 'positive after costs', direction: 'above' }}
                  format={(v) => fmtNum(v, 3)}
                />
                <ToleranceRail
                  label="deflated Sharpe"
                  value={top.dsr}
                  domain={[0, 1]}
                  threshold={{ at: 0.5, label: 'survives the search', direction: 'above' }}
                  format={(v) => fmtNum(v, 3)}
                />
                <ToleranceRail
                  label="break-even cost"
                  value={tearsheet.backtest?.break_even_bps}
                  domain={[0, 30]}
                  threshold={{
                    at: tearsheet.backtest?.round_trip_bps ?? 0,
                    label: 'the cost you assumed',
                    direction: 'above',
                  }}
                  format={(v) => fmtBps(v, 1)}
                />
                <ToleranceRail
                  label="overfitting (PBO)"
                  value={tearsheet.overfitting?.pbo}
                  domain={[0, 1]}
                  threshold={{ at: 0.5, label: 'worse than chance', direction: 'below' }}
                  format={(v) => fmtNum(v, 3)}
                />
                <ToleranceRail
                  label="max drawdown"
                  value={top.max_drawdown}
                  domain={[-0.6, 0]}
                  format={(v) => fmtPct(v, 2)}
                />
              </div>
              {tearsheet.overfitting?.pbo === null ? (
                <p className="mt-4 max-w-prose text-xs leading-relaxed text-[color:var(--color-ink-muted)]">
                  Probability of backtest overfitting was not computed for this run — there were too
                  few configurations or too short a sample for the statistic to mean anything.
                </p>
              ) : null}
            </>
          ) : (
            <p className="text-sm text-[color:var(--color-ink-muted)]">
              This run has no leaderboard, so there is nothing to judge.
            </p>
          )}
          <InlineNotes notes={notesFor(notes, 'verdict')} />
        </Plate>

        <Plate
          id="equity"
          title="Equity, net of costs"
          meta={equity.data ? `${equity.data.n_source_points.toLocaleString()} bars` : undefined}
          stamp={stamp}
        >
          {equity.isPending ? <Skeleton height={220} /> : null}
          {equity.isError ? (
            <ErrorState error={equity.error} context={`GET /api/v1/runs/${runId}/series/equity`} />
          ) : null}
          {equity.data ? (
            <SeriesChart
              series={equity.data}
              column="equity"
              label="Equity"
              stroke="var(--color-plum)"
            />
          ) : null}

          {drawdown.data ? (
            <div className="mt-4">
              <SeriesChart
                series={drawdown.data}
                column="drawdown"
                label="Underwater"
                height={120}
                stroke="var(--color-oxblood)"
                fill="rgba(122, 36, 24, 0.18)"
                zeroLine
              />
            </div>
          ) : null}
          <InlineNotes notes={notesFor(notes, 'equity')} />
        </Plate>

        <Plate
          id="leaderboard"
          title="Leaderboard"
          meta={`${tearsheet.leaderboard.length} models · ${tearsheet.overfitting?.n_configs_evaluated ?? 0} configurations evaluated`}
          stamp={stamp}
        >
          <table className="w-full border-collapse text-xs">
            <caption className="sr-only">
              Every candidate and every mandatory baseline, ranked by net Sharpe
            </caption>
            <thead>
              <tr className="border-b border-[color:var(--color-ink)] text-left">
                <th scope="col" className="py-2 pr-3 font-medium">
                  #
                </th>
                <th scope="col" className="py-2 pr-3 font-medium">
                  model
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  net SR
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  DSR
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  PSR
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  IC t
                </th>
                <th scope="col" className="py-2 font-medium">
                  flags
                </th>
              </tr>
            </thead>
            <tbody>
              {tearsheet.leaderboard.map((row) => (
                <tr
                  key={row.model}
                  className={`border-b border-[color:var(--color-rule)] ${
                    row.model === 'baseline_shuffled' ? 'bg-[color:var(--color-paper-sunk)]' : ''
                  }`}
                >
                  <td className="tabular py-1.5 pr-3">{row.rank}</td>
                  <td className="py-1.5 pr-3">{row.model}</td>
                  <td className="tabular py-1.5 pr-3 text-right">
                    {fmtNum(row.oos_net_sharpe, 3)}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.dsr, 2)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.psr, 2)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.ic_tstat, 2)}</td>
                  <td className="py-1.5 text-2xs text-[color:var(--color-ink-muted)]">
                    {(row.flags ?? []).join(' · ') || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 max-w-prose text-2xs text-[color:var(--color-ink-muted)]">
            `baseline_shuffled` is the same estimator trained on labels whose relationship to the
            features has been destroyed. Any model above it by less than its own margin has not
            distinguished itself from noise.
          </p>
          <InlineNotes notes={notesFor(notes, 'leaderboard')} />
        </Plate>
      </div>

      <Margin notes={notes} />
    </div>
  )
}

function notesFor(notes: MarginNote[], anchor: string): MarginNote[] {
  return notes.filter((note) => note.anchor === anchor)
}
