/** `/runs/$id/folds` — every split, and what purging cost it. */

import { useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'

import { Plate } from '../components/Plate'
import { stampFrom } from '../components/ProvenanceStamp'
import { ErrorState, SkeletonLines } from '../components/states'
import { api } from '../lib/api'
import { fmtDate } from '../lib/format'
import { useRun } from '../lib/run-context'

export const Route = createFileRoute('/runs/$runId/folds')({
  component: FoldsStage,
})

interface FoldRow {
  fold: number
  train_span: [string, string]
  test_span: [string, string]
  n_train: number
  n_test: number
  purged: number
  embargoed: number
}

function FoldsStage() {
  const { runId, tearsheet } = useRun()
  const folds = useQuery({
    queryKey: ['folds', runId],
    queryFn: () => api.getFolds(runId),
    retry: false,
  })

  if (!tearsheet) return <SkeletonLines rows={6} />

  const rows = (folds.data?.folds ?? []) as unknown as FoldRow[]

  return (
    <div data-plates className="max-w-[1100px]">
      <Plate
        id="folds"
        title="Splits"
        meta={folds.data ? `${folds.data.scheme} · ${rows.length} folds` : undefined}
        stamp={stampFrom(tearsheet)}
      >
        <p className="mb-4 max-w-prose text-xs text-[color:var(--color-ink-muted)]">
          Purging removes training samples whose label span overlaps the test window; the embargo
          removes a further stretch immediately after it, because features are built from trailing
          windows and a sample starting one bar later still shares most of its inputs. Neither is
          optional (§4, I4).
        </p>

        {folds.isPending ? <SkeletonLines rows={5} /> : null}
        {folds.isError ? (
          <ErrorState error={folds.error} context={`GET /api/v1/runs/${runId}/folds`} />
        ) : null}

        {rows.length > 0 ? (
          <table className="w-full border-collapse text-xs">
            <caption className="sr-only">Per-fold spans and purge counts</caption>
            <thead>
              <tr className="border-b border-[color:var(--color-ink)] text-left">
                <th scope="col" className="py-2 pr-3 font-medium">
                  fold
                </th>
                <th scope="col" className="py-2 pr-3 font-medium">
                  train span
                </th>
                <th scope="col" className="py-2 pr-3 font-medium">
                  test span
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  train
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  test
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  purged
                </th>
                <th scope="col" className="py-2 text-right font-medium">
                  embargoed
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((fold) => (
                <tr key={fold.fold} className="border-b border-[color:var(--color-rule)]">
                  <td className="tabular py-1.5 pr-3">{fold.fold}</td>
                  <td className="tabular py-1.5 pr-3 text-[color:var(--color-ink-muted)]">
                    {fmtDate(fold.train_span?.[0])} → {fmtDate(fold.train_span?.[1])}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-[color:var(--color-ink-muted)]">
                    {fmtDate(fold.test_span?.[0])} → {fmtDate(fold.test_span?.[1])}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right">
                    {fold.n_train?.toLocaleString() ?? '—'}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right">
                    {fold.n_test?.toLocaleString() ?? '—'}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right">
                    {fold.purged?.toLocaleString() ?? '—'}
                  </td>
                  <td className="tabular py-1.5 text-right">
                    {fold.embargoed?.toLocaleString() ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Plate>
    </div>
  )
}
