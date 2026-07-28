/** `/runs/$id/data` — the quality report behind everything downstream. */

import { useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'

import { Plate } from '../components/Plate'
import { stampFrom } from '../components/ProvenanceStamp'
import { ErrorState, SkeletonLines } from '../components/states'
import { api } from '../lib/api'
import { fmtNum } from '../lib/format'
import { useRun } from '../lib/run-context'

export const Route = createFileRoute('/runs/$runId/data')({
  component: DataStage,
})

const MEASURES = [
  ['n_rows', 'bars stored'],
  ['n_expected_bars', 'bars the calendar expects'],
  ['n_missing_bars', 'missing'],
  ['missing_bar_pct', 'missing %'],
  ['longest_gap_bars', 'longest gap, bars'],
  ['n_unexpected_bars', 'bars outside the session grid'],
  ['n_duplicate_timestamps', 'duplicate timestamps'],
  ['n_zero_volume', 'zero-volume bars'],
  ['n_ohlc_violations', 'OHLC violations'],
  ['n_extreme_returns', 'extreme returns'],
] as const

function DataStage() {
  const { tearsheet } = useRun()
  const datasetId = tearsheet
    ? `${inferProvider(tearsheet.run_id)}:${tearsheet.dataset.symbols[0]}:${tearsheet.dataset.interval}`
    : ''

  const dataset = useQuery({
    queryKey: ['dataset', datasetId],
    queryFn: () => api.getDataset(datasetId),
    enabled: Boolean(datasetId),
    retry: false,
  })

  if (!tearsheet) return <SkeletonLines rows={6} />

  const quality = dataset.data?.quality as Record<string, unknown> | null | undefined

  return (
    <div data-plates className="max-w-[1000px]">
      <Plate
        id="data"
        title="Data quality"
        meta={`${tearsheet.dataset.quality} · ${tearsheet.dataset.n_bars.toLocaleString()} bars`}
        stamp={stampFrom(tearsheet, [`quality ${tearsheet.dataset.quality}`])}
      >
        <p className="mb-4 max-w-prose text-xs text-[color:var(--color-ink-muted)]">
          Measured at ingestion against the {tearsheet.dataset.calendar} session calendar. A dataset
          nobody assessed is a dataset nobody should model on, so this verdict travels with every
          number downstream of it.
        </p>

        {dataset.isPending ? <SkeletonLines rows={6} /> : null}
        {dataset.isError ? (
          <ErrorState
            error={dataset.error}
            context={`GET /api/v1/datasets/${datasetId} — the run recorded its own verdict as ${tearsheet.dataset.quality}`}
          />
        ) : null}

        {quality ? (
          <table className="w-full max-w-lg border-collapse text-xs">
            <caption className="sr-only">Data quality measures</caption>
            <tbody>
              {MEASURES.map(([key, label]) => (
                <tr key={key} className="border-b border-[color:var(--color-rule)]">
                  <th scope="row" className="py-1.5 pr-4 text-left font-normal">
                    {label}
                  </th>
                  <td className="tabular py-1.5 text-right">
                    {typeof quality[key] === 'number' ? fmtNum(quality[key] as number, 2) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}

        {Array.isArray(quality?.['reasons']) && (quality['reasons'] as string[]).length > 0 ? (
          <div className="mt-4">
            <p className="text-xs font-medium">Why it was flagged</p>
            <ul className="mt-1 list-inside list-disc text-xs text-[color:var(--color-ink-muted)]">
              {(quality['reasons'] as string[]).map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </Plate>
    </div>
  )
}

/** Dataset ids are `provider:symbol:interval`; the provider is in the config. */
function inferProvider(_runId: string): string {
  // The tearsheet does not carry the provider, and guessing wrongly produces a
  // 404 whose message already says exactly that. `csv` is the offline default
  // used by the smoke config.
  return 'csv'
}
