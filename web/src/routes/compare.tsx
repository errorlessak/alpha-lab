/**
 * `/compare?a=&b=` — two runs side by side.
 *
 * The most useful thing this page does is say when two runs share a config
 * hash. At that point any difference between them came from the data or the
 * code, and a reader hunting for a configuration difference is hunting for
 * something that does not exist.
 */

import { useQuery } from '@tanstack/react-query'
import { createFileRoute, useNavigate } from '@tanstack/react-router'

import { EmptyState, ErrorState, SkeletonLines } from '../components/states'
import { api } from '../lib/api'
import { fmtNum, fmtSigned, runPickerLabel, shortId } from '../lib/format'

interface CompareSearch {
  a?: string
  b?: string
}

export const Route = createFileRoute('/compare')({
  validateSearch: (search: Record<string, unknown>): CompareSearch => ({
    a: typeof search['a'] === 'string' ? search['a'] : undefined,
    b: typeof search['b'] === 'string' ? search['b'] : undefined,
  }),
  component: Compare,
})

function Compare() {
  const { a, b } = Route.useSearch()
  const navigate = useNavigate({ from: '/compare' })

  const runs = useQuery({ queryKey: ['runs', '', ''], queryFn: () => api.listRuns({ limit: 100 }) })
  const comparison = useQuery({
    queryKey: ['compare', a, b],
    queryFn: () => api.compare(a as string, b as string),
    enabled: Boolean(a && b),
    retry: false,
  })

  const options = runs.data?.runs ?? []

  return (
    <div className="max-w-[1000px]">
      <h1
        className="font-display text-xl font-semibold text-[color:var(--color-ink)]"
        style={{ fontStretch: '88%' }}
      >
        Compare
      </h1>

      <div className="mt-4 flex flex-wrap gap-4 border-y border-[color:var(--color-rule)] py-3">
        <Picker
          label="run A"
          value={a ?? ''}
          options={options}
          onChange={(next) => void navigate({ search: (prev) => ({ ...prev, a: next }) })}
        />
        <Picker
          label="run B"
          value={b ?? ''}
          options={options}
          onChange={(next) => void navigate({ search: (prev) => ({ ...prev, b: next }) })}
        />
      </div>

      {!a || !b ? (
        <div className="mt-6">
          <EmptyState
            headline="Pick two runs"
            body="Comparing needs two. Metric deltas and provenance appear once both are chosen."
          />
        </div>
      ) : null}

      {comparison.isPending && a && b ? <SkeletonLines rows={6} /> : null}
      {comparison.isError ? (
        <div className="mt-6">
          <ErrorState error={comparison.error} context="GET /api/v1/runs/compare" />
        </div>
      ) : null}

      {comparison.data ? (
        <div className="mt-6">
          <table className="w-full border-collapse text-xs">
            <caption className="sr-only">Headline metrics for both runs, with deltas</caption>
            <thead>
              <tr className="border-b border-[color:var(--color-ink)] text-left">
                <th scope="col" className="py-2 pr-3 font-medium">
                  measure
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  {shortId(comparison.data.a.run_id)}
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  {shortId(comparison.data.b.run_id)}
                </th>
                <th scope="col" className="py-2 text-right font-medium">
                  delta
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison.data.metric_deltas).map(([measure, values]) => (
                <tr key={measure} className="border-b border-[color:var(--color-rule)]">
                  <td className="py-1.5 pr-3">{measure.replaceAll('_', ' ')}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(values['a'], 3)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(values['b'], 3)}</td>
                  <td className="tabular py-1.5 text-right">{fmtSigned(values['delta'], 3)}</td>
                </tr>
              ))}
              <tr className="border-b border-[color:var(--color-rule)]">
                <td className="py-1.5 pr-3">selected model</td>
                <td className="py-1.5 pr-3 text-right">
                  {comparison.data.a.selected_model ?? '—'}
                </td>
                <td className="py-1.5 pr-3 text-right">
                  {comparison.data.b.selected_model ?? '—'}
                </td>
                <td className="py-1.5 text-right">
                  {comparison.data.a.selected_model === comparison.data.b.selected_model
                    ? 'same'
                    : 'differs'}
                </td>
              </tr>
            </tbody>
          </table>

          {comparison.data.config_identical ? (
            <p className="mt-4 max-w-prose border-l-2 border-[color:var(--color-sulphur)] pl-3 text-xs leading-relaxed text-[color:var(--color-ink)]">
              These runs share a config hash. Any difference between them came from the data or the
              code version, not from the configuration — there is no setting to find.
            </p>
          ) : (
            <p className="tabular mt-4 text-2xs text-[color:var(--color-ink-muted)]">
              config {comparison.data.a.config_hash.slice(0, 10)} vs{' '}
              {comparison.data.b.config_hash.slice(0, 10)} · data{' '}
              {comparison.data.data_identical ? 'identical' : 'differs'}
            </p>
          )}
        </div>
      ) : null}
    </div>
  )
}

function Picker({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: readonly { run_id: string; name?: string | null; created_at?: string | null }[]
  onChange: (next: string) => void
}) {
  const id = `compare-${label.replace(/\s/g, '-')}`
  return (
    <span className="flex items-center gap-2">
      <label htmlFor={id} className="text-2xs text-[color:var(--color-ink-muted)]">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="tabular border border-[color:var(--color-rule)] bg-[color:var(--color-paper)] px-2 py-1 text-2xs"
      >
        <option value="">choose…</option>
        {options.map((option) => (
          <option key={option.run_id} value={option.run_id}>
            {runPickerLabel(option)}
          </option>
        ))}
      </select>
    </span>
  )
}
