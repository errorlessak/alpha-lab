/** `/runs/$id/features` — what the model was allowed to see. */

import { createFileRoute } from '@tanstack/react-router'

import { Plate } from '../components/Plate'
import { stampFrom } from '../components/ProvenanceStamp'
import { SkeletonLines } from '../components/states'
import { useRun } from '../lib/run-context'

export const Route = createFileRoute('/runs/$runId/features')({
  component: FeaturesStage,
})

function FeaturesStage() {
  const { tearsheet } = useRun()
  if (!tearsheet) return <SkeletonLines rows={6} />

  const features = tearsheet.features

  return (
    <div data-plates className="max-w-[1000px]">
      <Plate
        id="features"
        title="Feature manifest"
        meta={
          features
            ? `${features.n_built} built · ${features.n_after_prune} after pruning`
            : 'not built'
        }
        stamp={stampFrom(tearsheet)}
      >
        {features ? (
          <>
            <dl className="grid max-w-lg grid-cols-2 gap-x-6 border-b border-[color:var(--color-rule)] pb-3 text-xs">
              <Row label="columns built">{features.n_built}</Row>
              <Row label="after pruning">{features.n_after_prune}</Row>
              <Row label="warm-up dropped">{features.max_lookback} bars</Row>
            </dl>

            <p className="mt-3 max-w-prose text-2xs text-[color:var(--color-ink-muted)]">
              Pruning is fitted, so under invariant I3 it runs inside each fold on that fold's
              training rows only. The count above is the last fold's — the selection the model a
              reader would carry forward was trained under.
            </p>

            <h3 className="mt-6 text-xs font-medium">By family</h3>
            <table className="mt-2 w-full max-w-sm border-collapse text-xs">
              <thead>
                <tr className="border-b border-[color:var(--color-ink)] text-left">
                  <th scope="col" className="py-1.5 font-medium">
                    family
                  </th>
                  <th scope="col" className="py-1.5 text-right font-medium">
                    columns
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(features.families ?? {}).map(([family, count]) => (
                  <tr key={family} className="border-b border-[color:var(--color-rule)]">
                    <td className="py-1.5">{family}</td>
                    <td className="tabular py-1.5 text-right">{count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <p className="text-sm text-[color:var(--color-ink-muted)]">
            This run stopped before features were built.
          </p>
        )}
      </Plate>

      {tearsheet.labels ? (
        <Plate
          id="labels"
          title="Labels"
          meta={tearsheet.labels.method}
          stamp={stampFrom(tearsheet)}
        >
          <dl className="grid max-w-lg grid-cols-2 gap-x-6 text-xs">
            <Row label="method">{tearsheet.labels.method}</Row>
            <Row label="horizon">{tearsheet.labels.horizon_bars} bars</Row>
            <Row label="reduced to binary by">{tearsheet.labels.binarised ?? 'not required'}</Row>
            <Row label="mean uniqueness">{tearsheet.labels.mean_uniqueness?.toFixed(3) ?? '—'}</Row>
          </dl>

          <h3 className="mt-4 text-xs font-medium">Class balance, before reduction</h3>
          <ul className="tabular mt-1 text-xs text-[color:var(--color-ink-muted)]">
            {Object.entries(tearsheet.labels.class_balance ?? {}).map(([label, share]) => (
              <li key={label}>
                {label}: {(share * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </Plate>
      ) : null}
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="py-1 text-[color:var(--color-ink-muted)]">{label}</dt>
      <dd className="tabular py-1 text-right">{children}</dd>
    </>
  )
}
