/**
 * `/` — the run index. The notebook's contents page.
 *
 * Dense, ruled, newest first. No hero: the runs are the content, and a
 * marketing-shaped hero on an index of one's own work would be absurd.
 *
 * Failed runs are shown, struck through, with the phase that stopped them. A
 * research index that hides its failures is a highlight reel, and the whole
 * point of §19 is that a run which found nothing is the valuable one.
 */

import { useQuery } from '@tanstack/react-query'
import { Link, createFileRoute } from '@tanstack/react-router'
import { useState } from 'react'

import { TableScroll } from '../components/TableScroll'
import { EmptyState, ErrorState, Skeleton } from '../components/states'
import { api } from '../lib/api'
import type { RunSummaryRow } from '../lib/api'
import { fmtDate, fmtDateTime, fmtNum, shortId } from '../lib/format'

export const Route = createFileRoute('/')({
  component: RunIndex,
})

const FIRST_RUN_COMMAND = 'uv run alpha run -c configs/experiments/smoke.yaml'

function RunIndex() {
  const [symbol, setSymbol] = useState('')
  const [status, setStatus] = useState('')

  const runs = useQuery({
    queryKey: ['runs', symbol, status],
    queryFn: () => api.listRuns({ symbol: symbol || undefined, status: status || undefined }),
  })

  const rows = runs.data?.runs ?? []
  const symbols = [...new Set(rows.flatMap((row) => row.symbols))]
    .filter((symbol): symbol is string => Boolean(symbol))
    .sort()
  const failed = rows.filter((row) => row.status !== 'completed').length

  return (
    <div className="max-w-[1200px]">
      <header className="mb-6">
        <h1
          className="font-display text-xl font-semibold text-[color:var(--color-ink)]"
          style={{ fontStretch: '88%' }}
        >
          Evidence file
        </h1>
        <p className="tabular mt-1 text-2xs text-[color:var(--color-ink-muted)]">
          {runs.isPending
            ? 'reading runs/'
            : `${runs.data?.total ?? 0} runs${failed ? ` · ${failed} did not complete` : ''}`}
        </p>
        <p className="mt-1 max-w-prose text-2xs text-[color:var(--color-ink-muted)]">
          The short code is the <em>config</em> hash — two runs of the same configuration share it,
          which is the point. Sort by when they were made to tell them apart.
        </p>
      </header>

      {rows.length > 0 || symbol || status ? (
        <div className="mb-4 flex flex-wrap items-center gap-4 border-y border-[color:var(--color-rule)] py-2">
          <Filter label="symbol" value={symbol} onChange={setSymbol} options={symbols} />
          <Filter
            label="status"
            value={status}
            onChange={setStatus}
            options={['completed', 'failed', 'cancelled', 'running']}
          />
        </div>
      ) : null}

      {runs.isPending ? <IndexSkeleton /> : null}

      {runs.isError ? <ErrorState error={runs.error} context="GET /api/v1/runs" /> : null}

      {runs.isSuccess && rows.length === 0 ? (
        symbol || status ? (
          <EmptyState
            headline="No runs match that filter"
            body="Clear the filter to see everything in the store."
          />
        ) : (
          <EmptyState
            headline="No runs yet."
            body="The first one takes about ten seconds and needs no network:"
            command={FIRST_RUN_COMMAND}
            aside="It runs on a committed synthetic fixture — a seeded random walk — and it should not find an edge. A run that claims one on this data means something upstream is leaking."
          />
        )
      ) : null}

      {rows.length > 0 ? <RunTable rows={rows} /> : null}
    </div>
  )
}

function RunTable({ rows }: { rows: RunSummaryRow[] }) {
  return (
    <TableScroll label="Runs">
      <table className="w-full min-w-[52rem] border-collapse text-xs">
        <caption className="sr-only">
          Completed and failed runs, newest first, with their headline metrics
        </caption>
        <thead>
          <tr className="border-b border-[color:var(--color-ink)] text-left">
            <Th>run</Th>
            <Th>created</Th>
            <Th>config</Th>
            <Th>symbol</Th>
            <Th>span</Th>
            <Th>model</Th>
            <Th numeric>net SR</Th>
            <Th numeric>DSR</Th>
            <Th numeric>PBO</Th>
            <Th>flags</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.run_id}
              className="border-b border-[color:var(--color-rule)] align-top even:bg-[color:var(--color-paper-sunk)]"
            >
              <td className="tabular py-2 pr-3">
                <Link
                  to="/runs/$runId"
                  params={{ runId: row.run_id }}
                  className="text-[color:var(--color-link)] underline underline-offset-2"
                >
                  {shortId(row.run_id)}
                </Link>
                {row.status !== 'completed' ? (
                  <span className="ml-2 text-2xs text-[color:var(--color-danger-text)]">
                    {row.status}
                  </span>
                ) : null}
              </td>
              <td className="tabular py-2 pr-3 text-[color:var(--color-ink-muted)]">
                {fmtDateTime(row.created_at)}
              </td>
              <td className="py-2 pr-3">{row.name}</td>
              <td className="py-2 pr-3">{(row.symbols ?? []).join(', ') || '—'}</td>
              <td className="tabular py-2 pr-3 text-[color:var(--color-ink-muted)]">
                {row.span?.length === 2 ? `${fmtDate(row.span[0])} → ${fmtDate(row.span[1])}` : '—'}
              </td>
              <td className="py-2 pr-3">{row.selected_model ?? '—'}</td>
              <Td>{fmtNum(row.oos_net_sharpe, 3)}</Td>
              <Td>{fmtNum(row.dsr, 2)}</Td>
              <Td>{fmtNum(row.pbo, 2)}</Td>
              <td className="py-2">
                {row.has_tearsheet ? (
                  (row.warnings?.length ?? 0) > 0 ? (
                    <span className="tabular text-2xs text-[color:var(--color-caution-text)]">
                      ⚑ {(row.warnings ?? []).join(' · ')}
                    </span>
                  ) : (
                    <span className="text-2xs text-[color:var(--color-ink-muted)]">clean</span>
                  )
                ) : (
                  <span className="text-2xs text-[color:var(--color-danger-text)]">
                    no result — see the job log
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableScroll>
  )
}

function Th({ children, numeric }: { children: string; numeric?: boolean }) {
  return (
    <th
      scope="col"
      className={`py-2 pr-3 font-medium text-[color:var(--color-ink-muted)] ${numeric ? 'text-right' : ''}`}
    >
      {children}
    </th>
  )
}

function Td({ children }: { children: string }) {
  return <td className="tabular py-2 pr-3 text-right">{children}</td>
}

function Filter({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (next: string) => void
  options: string[]
}) {
  const id = `filter-${label}`
  return (
    <span className="flex items-center gap-2">
      <label htmlFor={id} className="text-2xs text-[color:var(--color-ink-muted)]">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="tabular border border-[color:var(--color-rule)] bg-[color:var(--color-paper)] px-2 py-1 text-2xs text-[color:var(--color-ink)]"
      >
        <option value="">all</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </span>
  )
}

function IndexSkeleton() {
  return (
    <div className="space-y-2">
      <Skeleton height={28} />
      {Array.from({ length: 6 }, (_, index) => (
        <Skeleton key={index} height={34} />
      ))}
    </div>
  )
}
