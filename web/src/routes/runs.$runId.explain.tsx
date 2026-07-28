/** `/runs/$id/explain` — SHAP, when the selected model is one TreeSHAP defines. */

import { useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'

import { Plate } from '../components/Plate'
import { stampFrom } from '../components/ProvenanceStamp'
import { ErrorState, SkeletonLines } from '../components/states'
import { api } from '../lib/api'
import { fmtNum } from '../lib/format'
import { useRun } from '../lib/run-context'

export const Route = createFileRoute('/runs/$runId/explain')({
  component: ExplainStage,
})

function ExplainStage() {
  const { runId, tearsheet } = useRun()
  const shap = useQuery({
    queryKey: ['shap', runId],
    queryFn: () => api.getShapSummary(runId),
    retry: false,
  })

  if (!tearsheet) return <SkeletonLines rows={6} />

  const features = (shap.data?.features ?? []) as Array<Record<string, unknown>>
  const peak = features.reduce(
    (max, row) => Math.max(max, Math.abs(Number(row['mean_abs_shap'] ?? 0))),
    0,
  )

  return (
    <div data-plates className="max-w-[1000px]">
      <Plate
        id="shap"
        title="Feature attribution"
        meta={shap.data ? `TreeSHAP · ${shap.data.model}` : undefined}
        stamp={stampFrom(tearsheet)}
      >
        {shap.isPending ? <SkeletonLines rows={8} /> : null}

        {/* The 404 here is informative rather than a failure: §9 scopes SHAP to
            tree models, and the API's message says which model was selected and
            why it has no explanation. */}
        {shap.isError ? (
          <ErrorState error={shap.error} context={`GET /api/v1/runs/${runId}/shap/summary`} />
        ) : null}

        {features.length > 0 ? (
          <>
            <p className="mb-4 max-w-prose text-xs text-[color:var(--color-ink-muted)]">
              Mean absolute contribution per feature, in raw-score space. The direction column is
              the correlation between a feature's value and its own contribution: positive means
              high values push the prediction up.
            </p>
            <table className="w-full border-collapse text-xs">
              <caption className="sr-only">Mean absolute SHAP contribution per feature</caption>
              <thead>
                <tr className="border-b border-[color:var(--color-ink)] text-left">
                  <th scope="col" className="py-2 pr-3 font-medium">
                    feature
                  </th>
                  <th scope="col" className="py-2 pr-3 text-right font-medium">
                    mean |SHAP|
                  </th>
                  <th scope="col" className="py-2 pr-3 font-medium">
                    &nbsp;
                  </th>
                  <th scope="col" className="py-2 text-right font-medium">
                    direction
                  </th>
                </tr>
              </thead>
              <tbody>
                {features.map((row) => {
                  const magnitude = Number(row['mean_abs_shap'] ?? 0)
                  const direction = Number(row['direction'] ?? 0)
                  return (
                    <tr
                      key={String(row['feature'])}
                      className="border-b border-[color:var(--color-rule)]"
                    >
                      <td className="py-1.5 pr-3">{String(row['feature'])}</td>
                      <td className="tabular py-1.5 pr-3 text-right">{fmtNum(magnitude, 4)}</td>
                      <td className="w-1/3 py-1.5 pr-3">
                        <span
                          aria-hidden
                          className="block h-2"
                          style={{
                            width: peak > 0 ? `${(magnitude / peak) * 100}%` : '0%',
                            background: 'var(--color-prussian)',
                          }}
                        />
                      </td>
                      <td className="tabular py-1.5 text-right">
                        {direction >= 0 ? '▲' : '▼'} {fmtNum(Math.abs(direction), 2)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </>
        ) : null}
      </Plate>
    </div>
  )
}
