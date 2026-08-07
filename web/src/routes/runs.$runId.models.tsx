/** `/runs/$id/models` — the leaderboard in full, against the shuffled baseline. */

import { useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'

import { Plate } from '../components/Plate'
import { stampFrom } from '../components/ProvenanceStamp'
import { TableScroll } from '../components/TableScroll'
import { ErrorState, SkeletonLines } from '../components/states'
import { api } from '../lib/api'
import { fmtNum } from '../lib/format'
import { useRun } from '../lib/run-context'

export const Route = createFileRoute('/runs/$runId/models')({
  component: ModelsStage,
})

const COLUMNS = [
  ['rank', '#', false],
  ['model', 'model', false],
  ['oos_net_sharpe', 'net SR', true],
  ['sharpe_ci', '95% interval', true],
  ['dsr', 'DSR', true],
  ['psr', 'PSR', true],
  ['break_even_bps', 'break-even', true],
  ['max_drawdown', 'max DD', true],
  ['ic', 'IC', true],
  ['ic_tstat', 'IC t', true],
  ['auc', 'AUC', true],
  ['brier', 'Brier', true],
  ['n_trials', 'trials', true],
] as const

function ModelsStage() {
  const { runId, tearsheet } = useRun()
  const leaderboard = useQuery({
    queryKey: ['leaderboard', runId],
    queryFn: () => api.getLeaderboard(runId),
    retry: false,
  })

  if (!tearsheet) return <SkeletonLines rows={6} />

  const rows = leaderboard.data?.rows ?? tearsheet.leaderboard
  const shuffled = rows.find((row) => row.model === 'baseline_shuffled')

  return (
    <div data-plates className="max-w-[1400px]">
      <Plate
        id="leaderboard"
        title="Leaderboard"
        meta={`ranked by out-of-fold net Sharpe · ${tearsheet.overfitting?.n_configs_evaluated ?? 0} configurations evaluated`}
        stamp={stampFrom(tearsheet)}
      >
        <p className="mb-4 max-w-prose text-xs text-[color:var(--color-ink-muted)]">
          Ranked on money, not accuracy (ADR 0007). Accuracy on price direction ignores magnitude
          and ignores turnover, and optimising it produces models that are right about small moves
          and wrong about large ones.
        </p>

        {leaderboard.isError ? (
          <ErrorState error={leaderboard.error} context={`GET /api/v1/runs/${runId}/leaderboard`} />
        ) : null}

        <TableScroll label="Leaderboard">
          <table className="w-full border-collapse text-xs">
            <caption className="sr-only">Every candidate and every mandatory baseline</caption>
            <thead>
              <tr className="border-b border-[color:var(--color-ink)] text-left">
                {COLUMNS.map(([key, label, numeric]) => (
                  <th
                    key={key}
                    scope="col"
                    className={`py-2 pr-3 font-medium whitespace-nowrap ${numeric ? 'text-right' : ''}`}
                  >
                    {label}
                  </th>
                ))}
                <th scope="col" className="py-2 font-medium">
                  flags
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.model}
                  className={`border-b border-[color:var(--color-rule)] ${
                    row.model === 'baseline_shuffled' ? 'bg-[color:var(--color-paper-sunk)]' : ''
                  }`}
                >
                  <td className="tabular py-1.5 pr-3">{row.rank}</td>
                  <td className="py-1.5 pr-3 whitespace-nowrap">{row.model}</td>
                  <td className="tabular py-1.5 pr-3 text-right">
                    {fmtNum(row.oos_net_sharpe, 3)}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right text-[color:var(--color-ink-muted)]">
                    {row.sharpe_ci
                      ? `[${fmtNum(row.sharpe_ci[0], 2)}, ${fmtNum(row.sharpe_ci[1], 2)}]`
                      : '—'}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.dsr, 3)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.psr, 2)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">
                    {fmtNum(row.break_even_bps, 1)}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.max_drawdown, 3)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.ic, 4)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.ic_tstat, 2)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.auc, 3)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(row.brier, 3)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{row.n_trials}</td>
                  <td className="py-1.5 text-2xs text-[color:var(--color-ink-muted)]">
                    {(row.flags ?? []).join(' · ') || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableScroll>

        {shuffled ? (
          <p className="mt-4 max-w-prose text-xs leading-relaxed text-[color:var(--color-ink-muted)]">
            The shuffled-label baseline — the same estimator trained on labels whose relationship to
            the features has been destroyed — scored an information coefficient of{' '}
            <span className="tabular">{fmtNum(shuffled.ic, 4)}</span> and a net Sharpe of{' '}
            <span className="tabular">{fmtNum(shuffled.oos_net_sharpe, 3)}</span>. That is the scale
            on this axis. A model above it by less than that margin has found nothing.
          </p>
        ) : null}
      </Plate>
    </div>
  )
}
